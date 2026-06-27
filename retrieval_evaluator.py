"""Retrieval evaluation used during embedder fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from sentence_transformers.evaluation import SentenceEvaluator
except ModuleNotFoundError:
    SentenceEvaluator = object

from config import Config
from dataset_io import PROJECT_ROOT, load_facts, load_query_fact_ids, validate_relevance
from embedder_model_registry import EmbedderModelWrapper

DEFAULT_EVAL_KS = [1, 3, 5, 10, 20, 50, 100, 150, 200, 400]
DEFAULT_EVAL_MRR_K = 10
# Cumulative character budgets over the ranked documents, for recall/hit "@chars" curves.
DEFAULT_EVAL_CHAR_BUDGETS = [
    1000, 2500, 5000, 10000, 25000, 50000, 75000, 100000, 125000, 150000, 200000
]

# Series are differentiated by color first; once the color palette is exhausted,
# the next group reuses the colors with a different line/marker shape.
CURVE_LINE_SHAPES = [
    {"mode": "lines", "dash": "solid", "symbol": None},          # solid line
    {"mode": "lines", "dash": "dash", "symbol": None},           # dashed line
    {"mode": "lines+markers", "dash": "solid", "symbol": "star"},  # line with stars
    {"mode": "lines+markers", "dash": "dot", "symbol": "circle"},  # dotted line with circles
]

# Available eval datasets. Pick which ones to run via Config.eval_dataset_keys.
EVAL_DATASET_REGISTRY = {
    "qaitest500-500": {
        "query_fact_ids_path": PROJECT_ROOT / "datasets/qaitest500_500_query_fact_ids.json",
        "facts_path": PROJECT_ROOT / "datasets/qaitest500_500_non_summary_dbelements.json",
    },
    "qaitest100-500": {
        "query_fact_ids_path": PROJECT_ROOT / "datasets/qaitest100_500_query_fact_ids.json",
        "facts_path": PROJECT_ROOT / "datasets/qaitest100_500_non_summary_dbelements.json",
    },
    "qaitest100-500-2": {
        "query_fact_ids_path": PROJECT_ROOT / "datasets/qaitest100_500_2_query_fact_ids.json",
        "facts_path": PROJECT_ROOT / "datasets/qaitest100_500_2_non_summary_dbelements.json",
    },
}


def available_eval_dataset_keys() -> list[str]:
    return sorted(EVAL_DATASET_REGISTRY)


def eval_evaluator_name(eval_dataset_key: str) -> str:
    return "eval_" + eval_dataset_key.replace("-", "_")


def encode_kwargs(batch_size: int) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def top_k_indices(scores: Any, max_k: int):
    import numpy as np

    if max_k == scores.shape[1]:
        return np.argsort(-scores, axis=1)

    unsorted_top_k = np.argpartition(-scores, kth=max_k - 1, axis=1)[:, :max_k]
    top_k_scores = np.take_along_axis(scores, unsorted_top_k, axis=1)
    sorted_order = np.argsort(-top_k_scores, axis=1)
    return np.take_along_axis(unsorted_top_k, sorted_order, axis=1)


def chunked_top_k_indices(
    query_embeddings: Any,
    corpus_embeddings: Any,
    max_k: int,
    corpus_chunk_size: int,
):
    import numpy as np

    top_scores = None
    top_indices = None

    for chunk_start in range(0, corpus_embeddings.shape[0], corpus_chunk_size):
        chunk_end = min(chunk_start + corpus_chunk_size, corpus_embeddings.shape[0])
        chunk_scores = query_embeddings @ corpus_embeddings[chunk_start:chunk_end].T
        chunk_k = min(max_k, chunk_scores.shape[1])
        chunk_indices = top_k_indices(chunk_scores, chunk_k)
        chunk_top_scores = np.take_along_axis(chunk_scores, chunk_indices, axis=1)
        chunk_global_indices = chunk_indices + chunk_start

        if top_scores is None:
            top_scores = chunk_top_scores
            top_indices = chunk_global_indices
            continue

        combined_scores = np.concatenate([top_scores, chunk_top_scores], axis=1)
        combined_indices = np.concatenate([top_indices, chunk_global_indices], axis=1)
        keep_k = min(max_k, combined_scores.shape[1])
        keep_indices = top_k_indices(combined_scores, keep_k)
        top_scores = np.take_along_axis(combined_scores, keep_indices, axis=1)
        top_indices = np.take_along_axis(combined_indices, keep_indices, axis=1)

    return top_indices


class RegistryRetrievalEvaluator(SentenceEvaluator):
    def __init__(
        self,
        model_config: EmbedderModelWrapper,
        queries: dict[str, str],
        corpus: dict[str, str],
        relevant_docs: dict[str, set[str]],
        batch_size: int,
        corpus_chunk_size: int,
        write_metrics: bool,
        name: str = "uthereal_eval",
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.write_metrics = write_metrics
        self.name = name
        self.greater_is_better = True
        self.primary_metric = f"{name}_cosine_mrr@{DEFAULT_EVAL_MRR_K}"
        self.eval_history: list[dict[str, Any]] = []
        # Prefix added to curve series labels so multi-stage chains stay distinguishable
        # (e.g. "oqa-v1 epoch 1"). Empty for single-stage runs.
        self.stage_label = ""
        # Added to the trainer's per-stage global step before logging to wandb. Each stage
        # restarts global_step at 0, so a chain that shares one wandb run offsets later
        # stages by the cumulative step count to keep the run's step axis monotonic.
        self.step_offset = 0

    def __call__(self, model: Any, output_path: str | None = None, epoch: int = -1, steps: int = -1):
        import json
        import time

        import numpy as np

        started_at = time.perf_counter()
        query_ids = list(self.queries)
        corpus_ids = list(self.corpus)
        query_texts = [self.queries[query_id] for query_id in query_ids]
        corpus_texts = [self.corpus[corpus_id] for corpus_id in corpus_ids]

        corpus_embeddings = np.asarray(
            self.model_config.encode_document(
                model,
                corpus_texts,
                **encode_kwargs(self.batch_size),
            ),
            dtype=np.float32,
        )
        query_embeddings = np.asarray(
            self.model_config.encode_query(
                model,
                query_texts,
                **encode_kwargs(self.batch_size),
            ),
            dtype=np.float32,
        )

        corpus_lengths = np.array([len(text) for text in corpus_texts], dtype=np.int64)
        char_budgets = DEFAULT_EVAL_CHAR_BUDGETS

        # Rank deep enough to cover both the largest element-k and the largest character
        # budget. The worst case for the char budget is that the top-ranked documents are
        # the shortest ones in the corpus, so size max_k from the smallest documents whose
        # lengths sum to the largest budget. That guarantees the ranked list always spans
        # every budget without ranking the entire corpus.
        sorted_cumulative_lengths = np.cumsum(np.sort(corpus_lengths))
        char_cover_k = (
            int(np.searchsorted(sorted_cumulative_lengths, max(char_budgets), side="left")) + 1
        )
        element_max_k = max(max(DEFAULT_EVAL_KS), DEFAULT_EVAL_MRR_K)
        max_k = min(len(corpus_ids), max(element_max_k, char_cover_k))

        ranked_indices = chunked_top_k_indices(
            query_embeddings=query_embeddings,
            corpus_embeddings=corpus_embeddings,
            max_k=max_k,
            corpus_chunk_size=self.corpus_chunk_size,
        )

        metrics: dict[str, float] = {}
        eval_ks = [k for k in DEFAULT_EVAL_KS if k <= len(corpus_ids)]
        recall_values = {k: [] for k in eval_ks}
        accuracy_hits = {k: 0 for k in eval_ks}
        char_recall_values = {budget: [] for budget in char_budgets}
        char_accuracy_hits = {budget: 0 for budget in char_budgets}
        reciprocal_ranks = []

        for query_index, query_id in enumerate(query_ids):
            relevant_ids = self.relevant_docs[query_id]
            ranked_for_query = ranked_indices[query_index]
            ranked_doc_ids = [corpus_ids[index] for index in ranked_for_query]
            hit_mask = np.array(
                [doc_id in relevant_ids for doc_id in ranked_doc_ids],
                dtype=np.int32,
            )
            cumulative_hits = np.cumsum(hit_mask)
            mrr_hit_positions = np.flatnonzero(hit_mask[:DEFAULT_EVAL_MRR_K])
            reciprocal_ranks.append(
                1 / (int(mrr_hit_positions[0]) + 1)
                if mrr_hit_positions.size > 0
                else 0.0
            )

            for k in eval_ks:
                found = int(cumulative_hits[k - 1])
                recall_values[k].append(found / len(relevant_ids))
                accuracy_hits[k] += int(found > 0)

            # Walk the ranked docs, accumulating their character lengths; at each budget,
            # count only the documents that fully fit within it.
            cumulative_chars = np.cumsum(corpus_lengths[ranked_for_query])
            for budget in char_budgets:
                count = int(np.searchsorted(cumulative_chars, budget, side="right"))
                found = int(cumulative_hits[count - 1]) if count > 0 else 0
                char_recall_values[budget].append(found / len(relevant_ids))
                char_accuracy_hits[budget] += int(found > 0)

        query_count = len(query_ids)
        for k in eval_ks:
            metrics[f"{self.name}_cosine_accuracy@{k}"] = accuracy_hits[k] / query_count
            metrics[f"{self.name}_cosine_recall@{k}"] = float(np.mean(recall_values[k]))
        for budget in char_budgets:
            metrics[f"{self.name}_cosine_char_accuracy@{budget}"] = (
                char_accuracy_hits[budget] / query_count
            )
            metrics[f"{self.name}_cosine_char_recall@{budget}"] = float(
                np.mean(char_recall_values[budget])
            )
        metrics[f"{self.name}_cosine_mrr@{DEFAULT_EVAL_MRR_K}"] = float(
            np.mean(reciprocal_ranks)
        )
        metrics[f"{self.name}_runtime"] = time.perf_counter() - started_at
        self.log_wandb_curves(
            eval_ks=eval_ks,
            char_budgets=char_budgets,
            metrics=metrics,
            epoch=epoch,
            steps=steps,
        )

        if output_path and self.write_metrics:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            metrics_path = output_dir / f"{self.name}_metrics.jsonl"
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "steps": steps,
                            "metrics": metrics,
                        }
                    )
                    + "\n"
                )

        return metrics

    def log_wandb_curves(
        self,
        eval_ks: list[int],
        char_budgets: list[int],
        metrics: dict[str, float],
        epoch: int,
        steps: int,
    ) -> None:
        label = self.eval_label(epoch=epoch, steps=steps)
        self.eval_history.append(
            {
                "label": label,
                "recalls": [metrics[f"{self.name}_cosine_recall@{k}"] for k in eval_ks],
                "hits": [metrics[f"{self.name}_cosine_accuracy@{k}"] for k in eval_ks],
                "char_recalls": [
                    metrics[f"{self.name}_cosine_char_recall@{b}"] for b in char_budgets
                ],
                "char_hits": [
                    metrics[f"{self.name}_cosine_char_accuracy@{b}"] for b in char_budgets
                ],
            }
        )

        try:
            import wandb
        except ModuleNotFoundError:
            return

        if wandb.run is None:
            return

        # Split into "recall_curves/" and "hit_curves/" sections so recall and hit figures
        # group separately in the wandb workspace (and both sort before the eval/ and
        # train/ scalar sections).
        charts = {
            f"recall_curves/{self.name}_recall_at_k_curve": self.build_curve_figure(
                eval_ks, "recalls", "recall@k", "k"
            ),
            f"recall_curves/{self.name}_recall_at_chars_curve": self.build_curve_figure(
                char_budgets, "char_recalls", "recall@chars", "cumulative characters"
            ),
            f"hit_curves/{self.name}_hit_at_k_curve": self.build_curve_figure(
                eval_ks, "hits", "hit@k", "k"
            ),
            f"hit_curves/{self.name}_hit_at_chars_curve": self.build_curve_figure(
                char_budgets, "char_hits", "hit@chars", "cumulative characters"
            ),
        }
        # Log both charts in one call pinned to the trainer's global step, so recall
        # and hit always share the same step. Logging them in separate calls (or with
        # commit=False) can land them on different steps, which makes the wandb media
        # panel show only one chart at a given step.
        if steps is not None and steps >= 0:
            wandb.log(charts, step=steps + self.step_offset)
        else:
            wandb.log(charts)

    def series_style(self, index: int) -> dict[str, Any]:
        """Differentiate series by color first, then by line/marker shape."""
        import plotly.express as px

        colors = px.colors.qualitative.Plotly
        color = colors[index % len(colors)]
        shape = CURVE_LINE_SHAPES[(index // len(colors)) % len(CURVE_LINE_SHAPES)]
        return {"color": color, **shape}

    def build_curve_figure(
        self,
        x_values: list[int],
        value_key: str,
        ylabel: str,
        xlabel: str = "k",
    ):
        import plotly.graph_objects as go

        fig = go.Figure()
        for index, row in enumerate(self.eval_history):
            style = self.series_style(index)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=row[value_key],
                    name=row["label"],
                    mode=style["mode"],
                    line=dict(color=style["color"], dash=style["dash"], width=2),
                    marker=dict(color=style["color"], symbol=style["symbol"], size=8),
                )
            )

        fig.update_layout(
            title=f"{self.name} {ylabel}",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template="plotly_white",
            hovermode="x unified",
            legend=dict(title="checkpoint"),
        )
        fig.update_xaxes(tickmode="array", tickvals=x_values)
        fig.update_yaxes(range=[0, 1.02])
        return fig

    def eval_label(self, epoch: int, steps: int) -> str:
        if steps == 0 and not self.eval_history:
            base_label = "baseline"
        elif steps >= 0:
            base_label = f"step {steps}"
        elif epoch >= 0:
            base_label = f"epoch {epoch}"
        else:
            base_label = f"eval {len(self.eval_history) + 1}"

        if self.stage_label:
            base_label = f"{self.stage_label} {base_label}"

        existing_labels = {row["label"] for row in self.eval_history}
        if base_label not in existing_labels:
            return base_label

        duplicate_count = sum(
            row["label"].startswith(base_label) for row in self.eval_history
        )
        return f"{base_label} #{duplicate_count + 1}"

    @property
    def description(self) -> str:
        return "Registry Retrieval"


def set_evaluator_stage_label(evaluator: Any, stage_label: str) -> None:
    """Set the stage label on a RegistryRetrievalEvaluator or every sub-evaluator inside
    a SequentialEvaluator, so accumulated curve series are tagged with the current stage."""
    if evaluator is None:
        return
    if isinstance(evaluator, RegistryRetrievalEvaluator):
        evaluator.stage_label = stage_label
        return
    for sub_evaluator in getattr(evaluator, "evaluators", []):
        if isinstance(sub_evaluator, RegistryRetrievalEvaluator):
            sub_evaluator.stage_label = stage_label


def set_evaluator_step_offset(evaluator: Any, step_offset: int) -> None:
    """Set the wandb step offset on a RegistryRetrievalEvaluator or every sub-evaluator
    inside a SequentialEvaluator, so a chain sharing one wandb run keeps a monotonic step
    axis across stages (each stage's trainer restarts global_step at 0)."""
    if evaluator is None:
        return
    if isinstance(evaluator, RegistryRetrievalEvaluator):
        evaluator.step_offset = step_offset
        return
    for sub_evaluator in getattr(evaluator, "evaluators", []):
        if isinstance(sub_evaluator, RegistryRetrievalEvaluator):
            sub_evaluator.step_offset = step_offset


def build_single_retrieval_evaluator(
    eval_dataset_key: str,
    config: Config,
    model_config: EmbedderModelWrapper,
) -> RegistryRetrievalEvaluator:
    try:
        spec = EVAL_DATASET_REGISTRY[eval_dataset_key]
    except KeyError as exc:
        available = ", ".join(available_eval_dataset_keys())
        raise ValueError(
            f"Unknown eval dataset key {eval_dataset_key!r}. Available: {available}"
        ) from exc

    query_fact_ids = load_query_fact_ids(Path(spec["query_fact_ids_path"]))
    facts = load_facts(Path(spec["facts_path"]))
    validate_relevance(query_fact_ids, facts)

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for query_index, (query, relevant_ids) in enumerate(query_fact_ids.items()):
        query_id = f"q{query_index}"
        queries[query_id] = query
        relevant_docs[query_id] = relevant_ids

    print(
        f"Prepared retrieval evaluator [{eval_dataset_key}]: "
        f"{len(queries)} queries, {len(facts)} facts, "
        f"{sum(len(ids) for ids in relevant_docs.values())} relevance labels"
    )

    return RegistryRetrievalEvaluator(
        model_config=model_config,
        queries=queries,
        corpus=facts,
        relevant_docs=relevant_docs,
        batch_size=config.eval_batch_size,
        corpus_chunk_size=config.eval_corpus_chunk_size,
        write_metrics=not config.no_eval_csv,
        name=eval_evaluator_name(eval_dataset_key),
    )


def build_retrieval_evaluator(
    config: Config, model_config: EmbedderModelWrapper
):
    if config.eval_strategy == "no" or not config.eval_dataset_keys:
        return None

    evaluators = [
        build_single_retrieval_evaluator(eval_dataset_key, config, model_config)
        for eval_dataset_key in config.eval_dataset_keys
    ]
    if len(evaluators) == 1:
        return evaluators[0]

    from sentence_transformers.evaluation import SequentialEvaluator

    return SequentialEvaluator(evaluators)

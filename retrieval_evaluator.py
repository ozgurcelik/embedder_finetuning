"""Retrieval evaluation used during embedder fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ModuleNotFoundError:
    SentenceEvaluator = object

from config import Config
from dataset_io import PROJECT_ROOT, load_facts, load_query_fact_ids, validate_relevance
from embedder_model_registry import EmbedderModelWrapper

DEFAULT_EVAL_KS = [1, 3, 5, 10, 20, 50, 100, 150, 200, 400]
DEFAULT_EVAL_MRR_K = 10

# Available eval datasets. Pick which ones to run via Config.eval_dataset_keys.
EVAL_DATASET_REGISTRY = {
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
        self.recall_curve_history: list[dict[str, Any]] = []

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

        max_k = min(max(max(DEFAULT_EVAL_KS), DEFAULT_EVAL_MRR_K), len(corpus_ids))
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
        reciprocal_ranks = []

        for query_index, query_id in enumerate(query_ids):
            relevant_ids = self.relevant_docs[query_id]
            ranked_doc_ids = [corpus_ids[index] for index in ranked_indices[query_index]]
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

        query_count = len(query_ids)
        for k in eval_ks:
            metrics[f"{self.name}_cosine_accuracy@{k}"] = accuracy_hits[k] / query_count
            metrics[f"{self.name}_cosine_recall@{k}"] = float(np.mean(recall_values[k]))
        metrics[f"{self.name}_cosine_mrr@{DEFAULT_EVAL_MRR_K}"] = float(
            np.mean(reciprocal_ranks)
        )
        metrics[f"{self.name}_runtime"] = time.perf_counter() - started_at
        self.log_wandb_recall_curve(
            eval_ks=eval_ks,
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

    def log_wandb_recall_curve(
        self,
        eval_ks: list[int],
        metrics: dict[str, float],
        epoch: int,
        steps: int,
    ) -> None:
        label = self.eval_label(epoch=epoch, steps=steps)
        recalls = [metrics[f"{self.name}_cosine_recall@{k}"] for k in eval_ks]
        self.recall_curve_history.append({"label": label, "recalls": recalls})

        try:
            import wandb
        except ModuleNotFoundError:
            return

        if wandb.run is None:
            return

        chart = wandb.plot.line_series(
            xs=eval_ks,
            ys=[row["recalls"] for row in self.recall_curve_history],
            keys=[row["label"] for row in self.recall_curve_history],
            title=f"{self.name} recall@k",
            xname="k",
        )
        wandb.log({f"eval/{self.name}_recall_at_k_curve": chart})

    def eval_label(self, epoch: int, steps: int) -> str:
        if steps == 0 and not self.recall_curve_history:
            base_label = "baseline"
        elif steps >= 0:
            base_label = f"step {steps}"
        elif epoch >= 0:
            base_label = f"epoch {epoch}"
        else:
            base_label = f"eval {len(self.recall_curve_history) + 1}"

        existing_labels = {row["label"] for row in self.recall_curve_history}
        if base_label not in existing_labels:
            return base_label

        duplicate_count = sum(
            row["label"].startswith(base_label) for row in self.recall_curve_history
        )
        return f"{base_label} #{duplicate_count + 1}"

    @property
    def description(self) -> str:
        return "Registry Retrieval"


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

    from sentence_transformers.sentence_transformer.evaluation import SequentialEvaluator

    return SequentialEvaluator(evaluators)

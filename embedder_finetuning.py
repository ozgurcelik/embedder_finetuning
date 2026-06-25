#!/usr/bin/env python3
"""Fine-tune embedding models on question/context retrieval pairs."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

try:
    from sentence_transformers.sentence_transformer.evaluation import SentenceEvaluator
except ModuleNotFoundError:
    SentenceEvaluator = object

from embedder_model_registry import (
    DEFAULT_MODEL_KEY,
    EmbedderModelWrapper,
    available_model_keys,
    resolve_model_wrapper,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_DATASET_KEY = "oqa-v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models"
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 1.0
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SEED = 42
DEFAULT_WANDB_ENTITY = "dirtem1998"
DEFAULT_WANDB_PROJECT = "embedder_finetuning"
DEFAULT_EVAL_QUERY_FACT_IDS_PATH = PROJECT_ROOT / "datasets/qaitest100_500_query_fact_ids.json"
DEFAULT_EVAL_FACTS_PATH = PROJECT_ROOT / "datasets/qaitest100_500_non_summary_dbelements.json"
DEFAULT_EVAL_KS = [1, 3, 5, 10, 20, 50, 100, 150, 200, 400]
DEFAULT_EVAL_MRR_K = 10

TRAIN_DATASET_REGISTRY = {
    "oqa-v1": {
        "name": "m-rousseau/oqa-v1",
        "config": None,
        "splits": ("train", "validation", "test"),
        "query_column": "question",
        "positive_column": "context",
        "output_suffix": "oqa-v1",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune an embedding model with MultipleNegativesRankingLoss "
            "using query/document pairs and in-batch negatives."
        )
    )
    parser.add_argument(
        "--model-key",
        choices=available_model_keys(),
        default=DEFAULT_MODEL_KEY,
        help=f"Registered model wrapper. Default: {DEFAULT_MODEL_KEY}",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional model name or local path override. The selected --model-key "
            "still controls query/document encoding behavior."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints and final model. Defaults to models/<model>-<dataset>.",
    )
    parser.add_argument(
        "--train-dataset-key",
        choices=sorted(TRAIN_DATASET_REGISTRY),
        default=DEFAULT_TRAIN_DATASET_KEY,
        help=f"Registered training dataset. Default: {DEFAULT_TRAIN_DATASET_KEY}",
    )
    parser.add_argument(
        "--train-dataset-name",
        default=None,
        help="Optional Hugging Face dataset name override.",
    )
    parser.add_argument(
        "--train-dataset-config",
        default=None,
        help="Optional Hugging Face dataset config override.",
    )
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=None,
        help="Splits to combine. Use 'all' for every split. Defaults to the registry value.",
    )
    parser.add_argument(
        "--query-column",
        default=None,
        help="Dataset column containing the query/question text.",
    )
    parser.add_argument(
        "--positive-column",
        default=None,
        help="Dataset column containing the positive document/context text.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs. Default: 1",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional max optimizer steps. Use -1 to train for --epochs. Default: -1",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Per-device train batch size. Default: 8",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Default: 1",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate. Default: 2e-5",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=DEFAULT_WARMUP_RATIO,
        help="Warmup ratio. Default: 0.1",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Optional max sequence length override. If omitted, the model default is used.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use FP16 mixed precision training/eval when supported. Default: false.",
    )
    parser.add_argument(
        "--bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use BF16 mixed precision training/eval when supported. Default: false.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable TF32 matmul on supported NVIDIA GPUs. Default: trainer default.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap after combining splits.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for row shuffling and trainer setup. Default: 42",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Checkpoint save interval if --save-strategy steps is used. Default: 1000",
    )
    parser.add_argument(
        "--save-strategy",
        choices=("epoch", "steps", "no"),
        default="steps",
        help="When to save trainer checkpoints. Default: steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Training log interval in optimizer steps. Default: 50",
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log trainer and evaluator metrics to Weights & Biases. Default: true",
    )
    parser.add_argument(
        "--wandb-entity",
        default=DEFAULT_WANDB_ENTITY,
        help=f"Weights & Biases entity. Default: {DEFAULT_WANDB_ENTITY}",
    )
    parser.add_argument(
        "--wandb-project",
        default=DEFAULT_WANDB_PROJECT,
        help=f"Weights & Biases project. Default: {DEFAULT_WANDB_PROJECT}",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Optional Weights & Biases run name. Defaults to the output directory name.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline"),
        default=None,
        help="Optional Weights & Biases mode override.",
    )
    parser.add_argument(
        "--eval-strategy",
        choices=("steps", "epoch", "no"),
        default="steps",
        help="When to run retrieval evaluation during training. Default: steps",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=1000,
        help="Evaluation interval if --eval-strategy steps is used. Default: 1000",
    )
    parser.add_argument(
        "--eval-query-fact-ids",
        type=Path,
        default=DEFAULT_EVAL_QUERY_FACT_IDS_PATH,
        help=(
            "JSON mapping eval queries to relevant fact IDs. "
            f"Default: {DEFAULT_EVAL_QUERY_FACT_IDS_PATH}"
        ),
    )
    parser.add_argument(
        "--eval-facts",
        type=Path,
        default=DEFAULT_EVAL_FACTS_PATH,
        help=(
            "JSON mapping eval fact IDs to fact text. "
            f"Default: {DEFAULT_EVAL_FACTS_PATH}"
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batch size for retrieval evaluator encoding. Default: 32",
    )
    parser.add_argument(
        "--eval-corpus-chunk-size",
        type=int,
        default=50000,
        help="Corpus chunk size for retrieval evaluator scoring. Default: 50000",
    )
    parser.add_argument(
        "--eval-at-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run retrieval evaluation once before training starts. Default: true",
    )
    parser.add_argument(
        "--no-eval-csv",
        action="store_true",
        help="Do not write evaluator metric files under the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and prepare data, print a summary, then exit before training.",
    )
    parser.add_argument(
        "--use-cached-mnrl",
        action="store_true",
        help="Use CachedMultipleNegativesRankingLoss for larger effective batches.",
    )
    parser.add_argument(
        "--cached-mini-batch-size",
        type=int,
        default=16,
        help="Mini-batch size for CachedMultipleNegativesRankingLoss. Default: 16",
    )
    return parser.parse_args()


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_args(args: argparse.Namespace) -> None:
    require_positive("--batch-size", args.batch_size)
    require_positive(
        "--gradient-accumulation-steps", args.gradient_accumulation_steps
    )
    if args.save_strategy == "steps":
        require_positive("--save-steps", args.save_steps)
    require_positive("--logging-steps", args.logging_steps)
    if args.eval_strategy == "steps":
        require_positive("--eval-steps", args.eval_steps)
    if args.eval_strategy != "no":
        require_positive("--eval-batch-size", args.eval_batch_size)
        require_positive("--eval-corpus-chunk-size", args.eval_corpus_chunk_size)
    if args.max_train_samples is not None:
        require_positive("--max-train-samples", args.max_train_samples)
    if args.max_seq_length is not None:
        require_positive("--max-seq-length", args.max_seq_length)
    if args.fp16 and args.bf16:
        raise ValueError("Use only one mixed precision mode: --fp16 or --bf16")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError(f"--max-steps must be -1 or positive, got {args.max_steps}")
    if args.learning_rate <= 0:
        raise ValueError(f"--learning-rate must be positive, got {args.learning_rate}")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError(f"--warmup-ratio must be in [0, 1), got {args.warmup_ratio}")
    if args.wandb and not args.wandb_entity:
        raise ValueError("--wandb-entity cannot be empty when W&B logging is enabled")
    if args.wandb and not args.wandb_project:
        raise ValueError("--wandb-project cannot be empty when W&B logging is enabled")


def dataset_spec_from_args(args: argparse.Namespace) -> dict[str, Any]:
    spec = dict(TRAIN_DATASET_REGISTRY[args.train_dataset_key])
    if args.train_dataset_name:
        spec["name"] = args.train_dataset_name
    if args.train_dataset_config:
        spec["config"] = args.train_dataset_config
    if args.train_splits:
        spec["splits"] = tuple(args.train_splits)
    if args.query_column:
        spec["query_column"] = args.query_column
    if args.positive_column:
        spec["positive_column"] = args.positive_column
    return spec


def default_output_dir(
    model_config: EmbedderModelWrapper, dataset_spec: dict[str, Any]
) -> Path:
    dataset_suffix = dataset_spec.get("output_suffix") or dataset_spec["name"].split("/")[-1]
    return DEFAULT_OUTPUT_ROOT / f"{model_config.output_dir_name}-{dataset_suffix}"


def first_present(row: dict[str, Any], column: str) -> str:
    value = row.get(column)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"Column {column!r} is missing or empty")


def load_source_dataset(dataset_spec: dict[str, Any]):
    try:
        from datasets import concatenate_datasets, load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install it in embeddervenv with:\n"
            "  conda run -n embeddervenv python -m pip install -U datasets"
        ) from exc

    name = dataset_spec["name"]
    config = dataset_spec.get("config")
    splits = tuple(dataset_spec["splits"])
    dataset_dict = load_dataset(name, config) if config else load_dataset(name)
    selected_splits = tuple(dataset_dict.keys()) if splits == ("all",) else splits

    missing_splits = sorted(set(selected_splits) - set(dataset_dict.keys()))
    if missing_splits:
        available = ", ".join(dataset_dict.keys())
        raise ValueError(
            f"Dataset {name!r} does not have split(s) {missing_splits}. "
            f"Available: {available}"
        )

    datasets = [dataset_dict[split] for split in selected_splits]
    combined = concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]
    split_sizes = ", ".join(f"{split}={len(dataset_dict[split])}" for split in selected_splits)
    print(f"Loaded {name}: {split_sizes}; combined={len(combined)}")
    return combined


def read_json_value(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_json_object(path: Path) -> dict[str, Any]:
    data = read_json_value(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_eval_query_fact_ids(path: Path) -> dict[str, set[str]]:
    raw_data = read_json_value(path)
    if isinstance(raw_data, list):
        return load_eval_query_fact_id_records(path, raw_data)
    if not isinstance(raw_data, dict):
        raise ValueError(f"{path} must contain a JSON object or a JSON list")

    query_fact_ids: dict[str, set[str]] = {}

    for query, fact_ids in raw_data.items():
        if not isinstance(query, str):
            raise ValueError(f"{path} contains a non-string query key: {query!r}")
        if not isinstance(fact_ids, list) or not fact_ids:
            raise ValueError(f"Relevant IDs for query {query!r} must be a non-empty list")
        query_fact_ids[query] = {str(fact_id) for fact_id in fact_ids}

    if not query_fact_ids:
        raise ValueError(f"{path} contains no queries")
    return query_fact_ids


def load_eval_query_fact_id_records(
    path: Path,
    records: list[Any],
) -> dict[str, set[str]]:
    query_fact_ids: dict[str, set[str]] = {}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}[{index}] must be a JSON object")

        query = record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}[{index}] must contain a non-empty query")

        positive_fact_ids = record.get("positive_fact_ids")
        if not isinstance(positive_fact_ids, list) or not positive_fact_ids:
            raise ValueError(
                f"{path}[{index}] must contain a non-empty positive_fact_ids list"
            )

        query_fact_ids.setdefault(query.strip(), set()).update(
            str(fact_id) for fact_id in positive_fact_ids
        )

    if not query_fact_ids:
        raise ValueError(f"{path} contains no queries")
    return query_fact_ids


def load_eval_facts(path: Path) -> dict[str, str]:
    raw_data = read_json_object(path)
    facts: dict[str, str] = {}

    for fact_id, text in raw_data.items():
        if not isinstance(fact_id, str):
            raise ValueError(f"{path} contains a non-string fact ID: {fact_id!r}")
        if not isinstance(text, str):
            raise ValueError(f"Text for fact ID {fact_id!r} must be a string")
        facts[fact_id] = text

    if not facts:
        raise ValueError(f"{path} contains no facts")
    return facts


def validate_eval_relevance(
    query_fact_ids: dict[str, set[str]], facts: dict[str, str]
) -> None:
    fact_ids = set(facts)
    missing_ids = sorted(
        {
            fact_id
            for relevant_ids in query_fact_ids.values()
            for fact_id in relevant_ids
            if fact_id not in fact_ids
        }
    )

    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        if len(missing_ids) > 5:
            preview += ", ..."
        raise ValueError(
            f"{len(missing_ids)} eval relevant fact IDs are missing from "
            f"the facts file: {preview}"
        )


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


def build_retrieval_evaluator(
    args: argparse.Namespace, model_config: EmbedderModelWrapper
):
    if args.eval_strategy == "no":
        return None

    query_fact_ids = load_eval_query_fact_ids(args.eval_query_fact_ids)
    facts = load_eval_facts(args.eval_facts)
    validate_eval_relevance(query_fact_ids, facts)

    queries: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    for query_index, (query, relevant_ids) in enumerate(query_fact_ids.items()):
        query_id = f"q{query_index}"
        queries[query_id] = query
        relevant_docs[query_id] = relevant_ids

    print(
        "Prepared retrieval evaluator: "
        f"{len(queries)} queries, {len(facts)} facts, "
        f"{sum(len(ids) for ids in relevant_docs.values())} relevance labels"
    )

    return RegistryRetrievalEvaluator(
        model_config=model_config,
        queries=queries,
        corpus=facts,
        relevant_docs=relevant_docs,
        batch_size=args.eval_batch_size,
        corpus_chunk_size=args.eval_corpus_chunk_size,
        write_metrics=not args.no_eval_csv,
        name="uthereal_eval",
    )


def build_training_rows(
    args: argparse.Namespace,
    dataset_spec: dict[str, Any],
    model_config: EmbedderModelWrapper,
) -> list[dict[str, str]]:
    dataset = load_source_dataset(dataset_spec)
    if args.max_train_samples is not None:
        sample_count = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(sample_count))

    rows: list[dict[str, str]] = []
    seen_pairs = set()
    skipped = 0
    query_column = dataset_spec["query_column"]
    positive_column = dataset_spec["positive_column"]

    for row in dataset:
        row_dict = dict(row)
        try:
            query = first_present(row_dict, query_column)
            positive = first_present(row_dict, positive_column)
        except ValueError:
            skipped += 1
            continue

        pair_key = (query, positive)
        if pair_key in seen_pairs:
            skipped += 1
            continue
        seen_pairs.add(pair_key)

        rows.append(
            {
                "anchor": model_config.format_query(query),
                "positive": model_config.format_document(positive),
            }
        )

    if not rows:
        raise ValueError("No training rows were produced")

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    print(f"Prepared {len(rows)} training pairs; skipped {skipped} empty/duplicate rows")
    return rows


def rows_to_dataset(rows: list[dict[str, str]]):
    try:
        from datasets import Dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: datasets. Install it in embeddervenv with:\n"
            "  conda run -n embeddervenv python -m pip install -U datasets"
        ) from exc

    return Dataset.from_list(rows)


def print_dataset_summary(rows: list[dict[str, str]]) -> None:
    columns = list(rows[0])
    unique_positives = len({row["positive"] for row in rows})
    print(f"Prepared training rows: {len(rows)}")
    print(f"Unique positives: {unique_positives}")
    print("Batch sampler: NO_DUPLICATES, to avoid duplicate positives in one batch")
    print(f"Columns: {', '.join(columns)}")
    print("First row text lengths:")
    for column in columns:
        print(f"  {column}: {len(rows[0][column])} chars")
    print("First row prefixes:")
    for column in columns:
        print(f"  {column}: {rows[0][column][:40]!r}")


def build_loss(model: Any, use_cached_mnrl: bool, cached_mini_batch_size: int):
    from sentence_transformers.sentence_transformer import losses

    if use_cached_mnrl:
        return losses.CachedMultipleNegativesRankingLoss(
            model,
            mini_batch_size=cached_mini_batch_size,
        )
    return losses.MultipleNegativesRankingLoss(model)


def configure_wandb(args: argparse.Namespace) -> str:
    if not args.wandb:
        return "none"

    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: wandb. Install it in embeddervenv with:\n"
            "  conda run -n embeddervenv python -m pip install -U wandb"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_ENTITY"] = args.wandb_entity
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_DIR"] = str(args.output_dir)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    print(
        "Weights & Biases logging enabled: "
        f"entity={args.wandb_entity}, project={args.wandb_project}"
    )
    return "wandb"


def select_device() -> str:
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"

    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.device_count()} device(s)")
        print(f"Default CUDA device: {torch.cuda.get_device_name(0)}")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("MPS available")
        return "mps"
    print("CUDA not available; using CPU")
    return "cpu"


def train(args: argparse.Namespace) -> None:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.training_args import BatchSamplers

    model_config = resolve_model_wrapper(args.model_key, args.model)
    dataset_spec = dataset_spec_from_args(args)
    if args.output_dir is None:
        args.output_dir = default_output_dir(model_config, dataset_spec)

    rows = build_training_rows(args, dataset_spec, model_config)
    print_dataset_summary(rows)

    if args.dry_run:
        print("Dry run complete; no model was trained.")
        return

    report_to = configure_wandb(args)
    train_dataset = rows_to_dataset(rows)
    device = select_device()
    model_kwargs = model_config.sentence_transformer_kwargs()
    model_kwargs["device"] = device
    model = SentenceTransformer(
        model_config.model_name,
        **model_kwargs,
    )
    print(f"Model device: {model.device}")
    if args.max_seq_length is not None:
        model.max_seq_length = args.max_seq_length
    train_loss = build_loss(
        model=model,
        use_cached_mnrl=args.use_cached_mnrl,
        cached_mini_batch_size=args.cached_mini_batch_size,
    )
    evaluator = build_retrieval_evaluator(args, model_config)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=None,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        eval_on_start=args.eval_strategy != "no" and args.eval_at_start,
        do_eval=args.eval_strategy != "no",
        fp16=args.fp16,
        bf16=args.bf16,
        fp16_full_eval=args.fp16,
        bf16_full_eval=args.bf16,
        tf32=args.tf32,
        use_cpu=device == "cpu",
        logging_steps=args.logging_steps,
        report_to=report_to,
        run_name=args.wandb_run_name or args.output_dir.name,
        seed=args.seed,
        data_seed=args.seed,
        dataloader_pin_memory=False,
        disable_tqdm=False,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    final_model_dir = args.output_dir / "final"
    model.save_pretrained(str(final_model_dir))
    print(f"Saved final model to {final_model_dir}")


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
        train(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

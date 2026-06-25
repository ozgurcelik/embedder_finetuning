#!/usr/bin/env python3
"""Evaluate fact retrieval with a sentence-transformers embedding model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from dataset_io import (
    PROJECT_ROOT,
    load_facts,
    load_query_fact_ids,
    select_device,
    validate_relevance,
)
from embedder_model_registry import (
    DEFAULT_MODEL_KEY,
    EmbedderModelWrapper,
    available_model_keys,
    resolve_model_wrapper,
)
from retrieval_evaluator import top_k_indices

DEFAULT_QUERY_FACT_IDS_PATH = PROJECT_ROOT / "datasets/qaitest100_500_query_fact_ids.json"
DEFAULT_FACTS_PATH = PROJECT_ROOT / "datasets/qaitest100_500_non_summary_dbelements.json"
DEFAULT_KS = [1, 3, 5, 10, 20, 50, 100, 150, 200, 400]
DEFAULT_MRR_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Embed every query and fact, rank facts by cosine similarity, "
            "and report retrieval recall@k."
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
            "Optional SentenceTransformer model name or local path override. "
            "The selected --model-key still controls query/document encoding behavior."
        ),
    )
    parser.add_argument(
        "--query-fact-ids",
        type=Path,
        default=DEFAULT_QUERY_FACT_IDS_PATH,
        help=f"JSON mapping query text to relevant fact IDs. Default: {DEFAULT_QUERY_FACT_IDS_PATH}",
    )
    parser.add_argument(
        "--facts",
        type=Path,
        default=DEFAULT_FACTS_PATH,
        help=f"JSON mapping fact ID to fact text. Default: {DEFAULT_FACTS_PATH}",
    )
    parser.add_argument(
        "--ks",
        nargs="+",
        default=[str(k) for k in DEFAULT_KS],
        help="k values to evaluate. Accepts space or comma separated values.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model.encode(). Default: 32",
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=None,
        help="Optional path for aggregate metrics and run metadata as JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable sentence-transformers progress bars.",
    )
    return parser.parse_args()


def parse_ks(raw_values: list[str], fact_count: int) -> list[int]:
    values: list[int] = []
    for raw_value in raw_values:
        for part in raw_value.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                k = int(part)
            except ValueError as exc:
                raise ValueError(f"Invalid k value: {part!r}") from exc
            if k <= 0:
                raise ValueError(f"k must be positive, got {k}")
            if k > fact_count:
                raise ValueError(
                    f"k={k} is larger than the number of facts ({fact_count})"
                )
            values.append(k)

    if not values:
        raise ValueError("At least one k value is required")
    return sorted(set(values))


def load_sentence_transformer(model_config: EmbedderModelWrapper):
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install it with:\n"
            "  python3 -m pip install -U sentence-transformers\n"
            "The first real evaluation run will also download the Hugging Face model "
            "unless it is already cached."
        ) from exc

    kwargs = model_config.sentence_transformer_kwargs()
    kwargs["device"] = select_device()
    model = SentenceTransformer(model_config.model_name, **kwargs)
    print(f"Model device: {model.device}")
    return model


def encode_kwargs(batch_size: int, show_progress: bool) -> dict[str, Any]:
    return {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": show_progress,
    }


def compute_metrics(
    query_texts: list[str],
    query_fact_ids: dict[str, set[str]],
    fact_ids: list[str],
    scores: np.ndarray,
    ks: list[int],
) -> tuple[list[dict[str, float]], float]:
    mrr_depth = min(DEFAULT_MRR_K, len(fact_ids))
    max_k = max(max(ks), mrr_depth)
    ranked_indices = top_k_indices(scores, max_k)

    macro_recalls = {k: [] for k in ks}
    reciprocal_ranks = []
    query_hits = {k: 0 for k in ks}

    for query_index, query in enumerate(query_texts):
        relevant_ids = query_fact_ids[query]
        ranked_fact_ids = [fact_ids[index] for index in ranked_indices[query_index]]
        hit_mask = np.array(
            [fact_id in relevant_ids for fact_id in ranked_fact_ids],
            dtype=np.int32,
        )
        cumulative_hits = np.cumsum(hit_mask)
        mrr_hit_positions = np.flatnonzero(hit_mask[:mrr_depth])
        reciprocal_rank = (
            1 / (int(mrr_hit_positions[0]) + 1)
            if mrr_hit_positions.size > 0
            else 0.0
        )
        reciprocal_ranks.append(reciprocal_rank)

        for k in ks:
            found = int(cumulative_hits[k - 1])
            recall = found / len(relevant_ids)

            macro_recalls[k].append(recall)
            query_hits[k] += int(found > 0)

    metric_rows = []
    for k in ks:
        metric_rows.append(
            {
                "k": k,
                "recall_at_k": float(np.mean(macro_recalls[k])),
                "hit_rate_at_k": query_hits[k] / len(query_texts),
            }
        )

    return metric_rows, float(np.mean(reciprocal_ranks))


def print_metrics(
    model_name: str,
    query_count: int,
    fact_count: int,
    relevant_count: int,
    metric_rows: list[dict[str, float]],
    mrr_at_10: float,
) -> None:
    print(f"Model: {model_name}")
    print(f"Queries: {query_count}")
    print(f"Facts: {fact_count}")
    print(f"Relevant query/fact labels: {relevant_count}")
    print()
    print("k\trecall@k\thit_rate@k")
    for row in metric_rows:
        print(
            f"{int(row['k'])}\t"
            f"{row['recall_at_k']:.4f}\t"
            f"{row['hit_rate_at_k']:.4f}"
        )
    print(f"\nMRR@10: {mrr_at_10:.4f}")


def write_details(
    output_path: Path,
    model_name: str,
    query_fact_ids_path: Path,
    facts_path: Path,
    metric_rows: list[dict[str, float]],
    mrr_at_10: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "query_fact_ids_path": str(query_fact_ids_path),
        "facts_path": str(facts_path),
        "metrics": metric_rows,
        "mrr_at_10": mrr_at_10,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote details to {output_path}")


def main() -> int:
    args = parse_args()

    try:
        model_config = resolve_model_wrapper(args.model_key, args.model)
        query_fact_ids = load_query_fact_ids(args.query_fact_ids)
        facts = load_facts(args.facts)
        validate_relevance(query_fact_ids, facts)
        ks = parse_ks(args.ks, fact_count=len(facts))
    except (OSError, ValueError) as exc:
        print(f"Input error: {exc}", file=sys.stderr)
        return 2

    query_texts = list(query_fact_ids)
    fact_items = list(facts.items())
    fact_ids = [fact_id for fact_id, _ in fact_items]
    fact_texts = [text for _, text in fact_items]

    try:
        model = load_sentence_transformer(model_config)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    show_progress = not args.no_progress
    print(f"Embedding {len(fact_texts)} facts...")
    fact_embeddings = np.asarray(
        model_config.encode_document(
            model,
            fact_texts,
            **encode_kwargs(args.batch_size, show_progress),
        ),
        dtype=np.float32,
    )

    print(f"Embedding {len(query_texts)} queries...")
    query_embeddings = np.asarray(
        model_config.encode_query(
            model,
            query_texts,
            **encode_kwargs(args.batch_size, show_progress),
        ),
        dtype=np.float32,
    )

    scores = query_embeddings @ fact_embeddings.T
    metric_rows, mrr_at_10 = compute_metrics(
        query_texts=query_texts,
        query_fact_ids=query_fact_ids,
        fact_ids=fact_ids,
        scores=scores,
        ks=ks,
    )

    print_metrics(
        model_name=model_config.model_name,
        query_count=len(query_texts),
        fact_count=len(fact_ids),
        relevant_count=sum(len(ids) for ids in query_fact_ids.values()),
        metric_rows=metric_rows,
        mrr_at_10=mrr_at_10,
    )

    if args.details_output:
        write_details(
            output_path=args.details_output,
            model_name=model_config.model_name,
            query_fact_ids_path=args.query_fact_ids,
            facts_path=args.facts,
            metric_rows=metric_rows,
            mrr_at_10=mrr_at_10,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build training rows from registered training datasets."""

from __future__ import annotations

import random
from dataclasses import replace
from pathlib import Path
from typing import Any

from config import Config
from dataset_io import (
    PROJECT_ROOT,
    load_facts,
    read_json_list,
)
from embedder_model_registry import EmbedderModelWrapper

DEFAULT_TRAIN_DATASET_KEY = "qaitrain500-500"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models"
DEFAULT_NEGATIVES_PER_QUERY = 4
DEFAULT_TRAIN_QUERY_FACT_IDS_PATH = (
    PROJECT_ROOT / "datasets/qaitrain500_500_query_fact_ids.json"
)
DEFAULT_TRAIN_FACTS_PATH = (
    PROJECT_ROOT / "datasets/qaitrain500_500_non_summary_dbelements.json"
)

TRAIN_DATASET_REGISTRY = {
    "qaitrain500-500": {
        "source": "local_query_fact_ids",
        "query_fact_ids_path": DEFAULT_TRAIN_QUERY_FACT_IDS_PATH,
        "facts_path": DEFAULT_TRAIN_FACTS_PATH,
        "output_suffix": "qaitrain500-500",
    },
    "qaitrain500-2-500": {
        "source": "local_query_fact_ids",
        "query_fact_ids_path": PROJECT_ROOT / "datasets/qaitrain500_2_500_query_fact_ids.json",
        "facts_path": PROJECT_ROOT / "datasets/qaitrain500_2_500_non_summary_dbelements.json",
        "output_suffix": "qaitrain500-2-500",
    },
    # 1 positive per query, no curated negatives: train with in-batch negatives only
    # (set negatives_per_query=0 for the stage that uses it).
    "qaitrain500-3-5000-fast": {
        "source": "local_query_fact_ids",
        "query_fact_ids_path": PROJECT_ROOT
        / "datasets/qaitrain500_3_5000_fast_query_fact_ids.json",
        "facts_path": PROJECT_ROOT
        / "datasets/qaitrain500_3_5000_fast_non_summary_dbelements.json",
        "output_suffix": "qaitrain500-3-5000-fast",
    },
    "qaitrain500-5000-fast": {
        "source": "local_query_fact_ids",
        "query_fact_ids_path": PROJECT_ROOT
        / "datasets/qaitrain500_5000_fast_query_fact_ids.json",
        "facts_path": PROJECT_ROOT
        / "datasets/qaitrain500_5000_fast_non_summary_dbelements.json",
        "output_suffix": "qaitrain500-5000-fast",
    },
    "qaitrain500-2-5000-fast": {
        "source": "local_query_fact_ids",
        "query_fact_ids_path": PROJECT_ROOT
        / "datasets/qaitrain500_2_5000_fast_query_fact_ids.json",
        "facts_path": PROJECT_ROOT
        / "datasets/qaitrain500_2_5000_fast_non_summary_dbelements.json",
        "output_suffix": "qaitrain500-2-5000-fast",
    },
    "oqa-v1": {
        "source": "huggingface",
        "name": "m-rousseau/oqa-v1",
        "config": None,
        "splits": ("train", "validation", "test"),
        "query_column": "question",
        "positive_column": "context",
        "answer_column": "answers",  # SQuAD-style {answer_start: [...], text: [...]}
        "output_suffix": "oqa-v1",
    },
}


def available_train_dataset_keys() -> list[str]:
    return sorted(TRAIN_DATASET_REGISTRY)


def resolve_train_dataset_keys(args: Config) -> list[str]:
    """The dataset key(s) a stage trains on. `train_dataset_keys` (a list) takes precedence
    and mixes several registered datasets into one stage; otherwise the single
    `train_dataset_key` is used."""
    if args.train_dataset_keys:
        return list(args.train_dataset_keys)
    return [args.train_dataset_key]


def dataset_spec_from_args(args: Config) -> dict[str, Any]:
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
    if args.train_query_fact_ids:
        spec["query_fact_ids_path"] = args.train_query_fact_ids
    if args.train_facts:
        spec["facts_path"] = args.train_facts
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


def answer_span_bounds(answers: Any) -> tuple[int, int] | None:
    """Return (start, end) char offsets of the first answer span, if available."""
    if not isinstance(answers, dict):
        return None
    starts = answers.get("answer_start")
    texts = answers.get("text")
    if not isinstance(starts, list) or not isinstance(texts, list):
        return None
    if not starts or not texts:
        return None
    start = starts[0]
    text = texts[0]
    if not isinstance(start, int) or not isinstance(text, str):
        return None
    return start, start + len(text)


def reduce_context(context: str, answers: Any, max_chars: int) -> str:
    """Shorten context to <= max_chars, keeping the answer span when known.

    Without answer info we head-truncate. With an answer span we center a window on
    the answer so the relevant text survives (the tokenizer would otherwise cut the
    tail of a long context, potentially dropping the answer entirely).
    """
    if max_chars <= 0 or len(context) <= max_chars:
        return context

    bounds = answer_span_bounds(answers)
    if bounds is None:
        return context[:max_chars].strip()

    start, end = bounds
    center = (start + end) // 2
    lo = max(0, center - max_chars // 2)
    hi = min(len(context), lo + max_chars)
    lo = max(0, hi - max_chars)
    return context[lo:hi].strip()


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


def unique_string_values(values: list[Any], path: Path, index: int, column: str) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        string_value = str(value).strip()
        if not string_value:
            raise ValueError(f"{path}[{index}].{column} contains an empty fact ID")
        if string_value in seen:
            continue
        seen.add(string_value)
        result.append(string_value)

    return result


def record_fact_ids(
    path: Path,
    index: int,
    record: dict[str, Any],
    column: str,
    require_non_empty: bool,
) -> list[str]:
    values = record.get(column)
    if not isinstance(values, list):
        raise ValueError(f"{path}[{index}].{column} must be a list")
    if require_non_empty and not values:
        raise ValueError(f"{path}[{index}].{column} must be a non-empty list")
    return unique_string_values(values, path, index, column)


def load_train_query_fact_id_records(path: Path) -> list[dict[str, Any]]:
    raw_records = read_json_list(path)
    records: list[dict[str, Any]] = []

    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"{path}[{index}] must be a JSON object")

        query = raw_record.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}[{index}] must contain a non-empty query")

        records.append(
            {
                "query": query.strip(),
                "positive_fact_ids": record_fact_ids(
                    path,
                    index,
                    raw_record,
                    "positive_fact_ids",
                    require_non_empty=True,
                ),
                "hard_negative_fact_ids": record_fact_ids(
                    path,
                    index,
                    raw_record,
                    "hard_negative_fact_ids",
                    require_non_empty=False,
                ),
                "soft_negative_fact_ids": record_fact_ids(
                    path,
                    index,
                    raw_record,
                    "soft_negative_fact_ids",
                    require_non_empty=False,
                ),
            }
        )

    if not records:
        raise ValueError(f"{path} contains no training records")
    return records


def validate_train_record_fact_ids(
    records: list[dict[str, Any]],
    facts: dict[str, str],
    path: Path,
) -> None:
    missing_ids = sorted(
        {
            fact_id
            for record in records
            for column in (
                "positive_fact_ids",
                "hard_negative_fact_ids",
                "soft_negative_fact_ids",
            )
            for fact_id in record[column]
            if fact_id not in facts
        }
    )

    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        if len(missing_ids) > 5:
            preview += ", ..."
        raise ValueError(
            f"{len(missing_ids)} train fact IDs from {path} are missing from "
            f"the facts file: {preview}"
        )


def sample_negative_fact_ids(
    record: dict[str, Any],
    positive_ids: set[str],
    facts: dict[str, str],
    sample_count: int,
    rng: random.Random,
) -> list[str]:
    candidate_ids: list[str] = []
    seen: set[str] = set(positive_ids)

    for column in ("hard_negative_fact_ids", "soft_negative_fact_ids"):
        for fact_id in record[column]:
            if fact_id in seen or not facts[fact_id].strip():
                continue
            seen.add(fact_id)
            candidate_ids.append(fact_id)

    if len(candidate_ids) < sample_count:
        return []
    return rng.sample(candidate_ids, sample_count)


def build_training_rows(
    args: Config,
    dataset_spec: dict[str, Any],
    model_config: EmbedderModelWrapper,
) -> list[dict[str, str]]:
    source = dataset_spec.get("source", "huggingface")
    if source == "local_query_fact_ids":
        return build_local_fact_training_rows(args, dataset_spec, model_config)
    if source == "huggingface":
        return build_huggingface_training_rows(args, dataset_spec, model_config)
    raise ValueError(f"Unknown training dataset source: {source!r}")


def build_mixed_training_rows(
    args: Config,
    model_config: EmbedderModelWrapper,
) -> list[dict[str, str]]:
    """Build the training rows for one stage, mixing several registered datasets into a
    single pool when `train_dataset_keys` lists more than one. Datasets are combined at the
    row level (each row already carries its own text), then shuffled together and capped, so
    one stage trains on an interleaved mix instead of running them as separate stages.

    Mixed datasets must produce rows with the same columns (e.g. all qai local datasets share
    anchor/positive/negative_1..N); mixing sources with different shapes (a local dataset that
    has explicit negatives with a plain anchor/positive HF dataset) is rejected, since the
    trainer needs a single consistent schema across all rows."""
    keys = resolve_train_dataset_keys(args)

    # Single dataset: keep the original path so per-dataset overrides (HF name/splits/columns,
    # local path overrides) still apply exactly as before.
    if not args.train_dataset_keys:
        return build_training_rows(args, dataset_spec_from_args(args), model_config)
    if len(keys) == 1:
        return build_training_rows(
            args, dict(TRAIN_DATASET_REGISTRY[keys[0]]), model_config
        )

    # Build each dataset without the per-dataset cap so max_train_samples applies to the
    # combined pool, then concatenate, shuffle together, and cap once.
    per_dataset_args = replace(args, max_train_samples=None)
    combined: list[dict[str, str]] = []
    per_key_counts: list[str] = []
    for key in keys:
        rows = build_training_rows(
            per_dataset_args, dict(TRAIN_DATASET_REGISTRY[key]), model_config
        )
        per_key_counts.append(f"{key}={len(rows)}")
        combined.extend(rows)

    if not combined:
        raise ValueError("No training rows were produced from the mixed datasets")

    column_sets = {frozenset(row) for row in combined}
    if len(column_sets) > 1:
        shapes = sorted(sorted(columns) for columns in column_sets)
        raise ValueError(
            "Cannot mix training datasets with different columns "
            f"(found {len(column_sets)} distinct row shapes: {shapes}). "
            "Mix only datasets that produce the same columns, e.g. local qai datasets that "
            "all use the same negatives_per_query."
        )

    rng = random.Random(args.seed)
    rng.shuffle(combined)
    if args.max_train_samples is not None:
        combined = combined[: args.max_train_samples]

    print(
        f"Mixed {len(keys)} datasets into {len(combined)} combined training rows "
        f"({', '.join(per_key_counts)})"
    )
    return combined


def build_huggingface_training_rows(
    args: Config,
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
    num_reduced = 0
    query_column = dataset_spec["query_column"]
    positive_column = dataset_spec["positive_column"]
    answer_column = dataset_spec.get("answer_column")
    max_chars = model_config.context_max_chars

    for row in dataset:
        row_dict = dict(row)
        try:
            query = first_present(row_dict, query_column)
            positive = first_present(row_dict, positive_column)
        except ValueError:
            skipped += 1
            continue

        if max_chars is not None:
            answers = row_dict.get(answer_column) if answer_column else None
            reduced_positive = reduce_context(positive, answers, max_chars)
            if len(reduced_positive) < len(positive):
                num_reduced += 1
            positive = reduced_positive

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
    if max_chars is not None:
        print(
            f"Context reduction: capped positives to {max_chars} chars "
            f"(answer-centered when available); {num_reduced} contexts shortened"
        )
    return rows


def build_local_fact_training_rows(
    args: Config,
    dataset_spec: dict[str, Any],
    model_config: EmbedderModelWrapper,
) -> list[dict[str, str]]:
    query_fact_ids_path = Path(dataset_spec["query_fact_ids_path"])
    facts_path = Path(dataset_spec["facts_path"])
    records = load_train_query_fact_id_records(query_fact_ids_path)
    facts = load_facts(facts_path)
    validate_train_record_fact_ids(records, facts, query_fact_ids_path)

    rows: list[dict[str, str]] = []
    seen_pairs = set()
    skipped_duplicate_pairs = 0
    skipped_empty_positive = 0
    skipped_insufficient_negatives = 0
    rng = random.Random(args.seed)

    for record in records:
        query = record["query"]
        positive_ids = set(record["positive_fact_ids"])

        for positive_fact_id in record["positive_fact_ids"]:
            positive = facts[positive_fact_id].strip()
            if not positive:
                skipped_empty_positive += 1
                continue

            pair_key = (query, positive_fact_id)
            if pair_key in seen_pairs:
                skipped_duplicate_pairs += 1
                continue
            seen_pairs.add(pair_key)

            # negatives_per_query == 0 means "no explicit negatives": emit anchor/positive
            # rows and rely on in-batch negatives (MultipleNegativesRankingLoss treats the
            # other rows' positives in the batch as negatives). With > 0 we require a full
            # set of curated negatives and skip rows that cannot supply them.
            if args.negatives_per_query > 0:
                negative_fact_ids = sample_negative_fact_ids(
                    record=record,
                    positive_ids=positive_ids,
                    facts=facts,
                    sample_count=args.negatives_per_query,
                    rng=rng,
                )
                if not negative_fact_ids:
                    skipped_insufficient_negatives += 1
                    continue
            else:
                negative_fact_ids = []

            row = {
                "anchor": model_config.format_query(query),
                "positive": model_config.format_document(positive),
            }
            for negative_index, negative_fact_id in enumerate(negative_fact_ids, start=1):
                row[f"negative_{negative_index}"] = model_config.format_document(
                    facts[negative_fact_id]
                )
            rows.append(row)

    if not rows:
        raise ValueError("No training rows were produced")

    prepared_count = len(rows)
    rng.shuffle(rows)
    if args.max_train_samples is not None:
        rows = rows[: args.max_train_samples]

    skipped = (
        skipped_duplicate_pairs
        + skipped_empty_positive
        + skipped_insufficient_negatives
    )
    print(
        "Prepared local training rows: "
        f"{len(rows)} from {prepared_count} candidates, "
        f"{len(records)} query records, {len(facts)} facts"
    )
    if args.negatives_per_query > 0:
        print(
            "Sampled explicit negatives: "
            f"{args.negatives_per_query} per row from hard+soft negative IDs"
        )
    else:
        print("Explicit negatives: 0 per row (in-batch negatives only)")
    if skipped:
        print(
            "Skipped local training rows: "
            f"duplicates={skipped_duplicate_pairs}, "
            f"empty positives={skipped_empty_positive}, "
            f"insufficient negatives={skipped_insufficient_negatives}"
        )
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
    negative_columns = [column for column in columns if column.startswith("negative_")]
    unique_positives = len({row["positive"] for row in rows})
    print(f"Prepared training rows: {len(rows)}")
    print(f"Unique positives: {unique_positives}")
    if negative_columns:
        print(f"Explicit negatives per row: {len(negative_columns)}")
    print("Batch sampler: NO_DUPLICATES, to avoid duplicate texts in one batch")
    print(f"Columns: {', '.join(columns)}")
    print("First row text lengths:")
    for column in columns:
        print(f"  {column}: {len(rows[0][column])} chars")
    print("First row prefixes:")
    for column in columns:
        print(f"  {column}: {rows[0][column][:40]!r}")

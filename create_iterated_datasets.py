#!/usr/bin/env python3
"""Build an "iterated" (harder) training dataset from a fine-tuned model's retrieval.

For every query in a source *_query_fact_ids.json file, this ranks the full fact corpus
with the given model, then keeps only the queries whose gold facts the model buries deep,
and mines depth-banded negatives around them:

  - positives  : the query's gold facts that fall *outside* the first POSITIVE_CHAR_BUDGET
                 characters of the ranked list (i.e. recall@chars@50k misses them).
  - hard negs  : HARD_NEG_COUNT random non-positive facts whose cumulative-character
                 position in the ranked list lies in [HARD_NEG_LO, HARD_NEG_HI).
  - soft negs  : SOFT_NEG_COUNT random non-positive facts in [SOFT_NEG_LO, SOFT_NEG_HI).

Queries with no "deep" positive are dropped. The cumulative-character accounting matches the
recall@chars metric in retrieval_evaluator.py (it walks the ranked docs summing full fact
lengths). Output mirrors the source *_query_fact_ids.json schema and reuses the same facts
file, so it can be registered as a local training dataset.

Edit the constants below, then run: python3 create_iterated_datasets.py
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from dataset_io import PROJECT_ROOT, load_facts, read_json_list, select_device
from embedder_model_registry import resolve_model_wrapper

# --- Configuration -----------------------------------------------------------------
MODEL_KEY = "all-mpnet-base-v2"  # controls query/document formatting (prefixes)
MODEL_PATH = PROJECT_ROOT / "models/all-mpnet-base-v2-qaitrain500-5000-fast/final"
# Fall back to the run root if the "final" export is absent.
MODEL_PATH_FALLBACK = PROJECT_ROOT / "models/all-mpnet-base-v2-qaitrain500-5000-fast"

SOURCE_QUERY_FACT_IDS = PROJECT_ROOT / "datasets/qaitrain500_2_500_query_fact_ids.json"
FACTS_PATH = PROJECT_ROOT / "datasets/qaitrain500_2_500_non_summary_dbelements.json"
OUTPUT_PATH = PROJECT_ROOT / "datasets/qaitrain500_2_500_iterated_query_fact_ids.json"

# A positive counts as "deep" (kept) if it does not fit within this many characters.
POSITIVE_CHAR_BUDGET = 50_000
# Character bands (over the ranked list) to mine negatives from.
HARD_NEG_LO, HARD_NEG_HI = 100_000, 200_000
SOFT_NEG_LO, SOFT_NEG_HI = 200_000, 300_000
HARD_NEG_COUNT = 4
SOFT_NEG_COUNT = 4

ENCODE_BATCH_SIZE = 64
SEED = 42
# -----------------------------------------------------------------------------------


def resolve_model_path() -> str:
    if MODEL_PATH.exists():
        return str(MODEL_PATH)
    if MODEL_PATH_FALLBACK.exists():
        print(f"'{MODEL_PATH}' not found; falling back to {MODEL_PATH_FALLBACK}")
        return str(MODEL_PATH_FALLBACK)
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH} or {MODEL_PATH_FALLBACK}"
    )


def load_source_records(path: Path, facts: dict[str, str]) -> list[dict[str, Any]]:
    """Return [{query, positive_fact_ids}] from the source file, keeping only positives
    that exist in the facts corpus and dropping queries left with none."""
    raw_records = read_json_list(path)
    records: list[dict[str, Any]] = []
    dropped_missing = 0

    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"{path}[{index}] must be a JSON object")
        query = raw.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{path}[{index}] must contain a non-empty query")
        positives = [
            str(fact_id)
            for fact_id in raw.get("positive_fact_ids", [])
            if str(fact_id) in facts
        ]
        # Preserve order while de-duplicating.
        positives = list(dict.fromkeys(positives))
        if not positives:
            dropped_missing += 1
            continue
        records.append({"query": query.strip(), "positive_fact_ids": positives})

    if dropped_missing:
        print(f"Dropped {dropped_missing} source queries with no positive present in facts")
    if not records:
        raise ValueError(f"{path} produced no usable query records")
    return records


def encode_corpus(model: Any, model_config: Any, corpus_texts: list[str]):
    import numpy as np

    print(f"Encoding {len(corpus_texts)} facts ...")
    embeddings = model_config.encode_document(
        model,
        corpus_texts,
        batch_size=ENCODE_BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def sample_band(
    order: Any,
    cumulative_chars: Any,
    corpus_ids: list[str],
    char_lo: int,
    char_hi: int,
    exclude: set[str],
    count: int,
    rng: random.Random,
) -> list[str]:
    """Randomly pick `count` fact IDs whose cumulative-character position in the ranked
    list falls in [char_lo, char_hi), excluding `exclude`. Returns fewer than `count` only
    when the band does not contain enough eligible facts."""
    import numpy as np

    # Ranks whose cumulative chars land inside the band; cumulative_chars is sorted by rank.
    lo_rank = int(np.searchsorted(cumulative_chars, char_lo, side="left"))
    hi_rank = int(np.searchsorted(cumulative_chars, char_hi, side="left"))
    candidate_ids = [
        corpus_ids[order[rank]]
        for rank in range(lo_rank, hi_rank)
        if corpus_ids[order[rank]] not in exclude
    ]
    if len(candidate_ids) <= count:
        return candidate_ids
    return rng.sample(candidate_ids, count)


def build_iterated_records(
    records: list[dict[str, Any]],
    corpus_ids: list[str],
    corpus_embeddings: Any,
    corpus_lengths: Any,
    model: Any,
    model_config: Any,
) -> list[dict[str, Any]]:
    import numpy as np

    rng = random.Random(SEED)
    fact_index = {fact_id: index for index, fact_id in enumerate(corpus_ids)}

    output: list[dict[str, Any]] = []
    short_hard = 0
    short_soft = 0

    query_texts = [record["query"] for record in records]
    print(f"Encoding {len(query_texts)} queries ...")
    query_embeddings = np.asarray(
        model_config.encode_query(
            model,
            query_texts,
            batch_size=ENCODE_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )

    for record, query_embedding in zip(records, query_embeddings):
        positives = record["positive_fact_ids"]
        positives_set = set(positives)

        # Cosine scores (embeddings are normalized) -> rank all facts, best first.
        scores = corpus_embeddings @ query_embedding
        order = np.argsort(-scores)
        cumulative_chars = np.cumsum(corpus_lengths[order])
        rank_of_index = np.empty(len(corpus_ids), dtype=np.int64)
        rank_of_index[order] = np.arange(len(corpus_ids))

        # A positive is "deep" if including it pushes the cumulative chars past the budget,
        # i.e. it is not retrieved within the first POSITIVE_CHAR_BUDGET characters.
        deep_positives = [
            fact_id
            for fact_id in positives
            if cumulative_chars[rank_of_index[fact_index[fact_id]]] > POSITIVE_CHAR_BUDGET
        ]
        if not deep_positives:
            continue

        hard_negatives = sample_band(
            order, cumulative_chars, corpus_ids,
            HARD_NEG_LO, HARD_NEG_HI, positives_set, HARD_NEG_COUNT, rng,
        )
        soft_negatives = sample_band(
            order, cumulative_chars, corpus_ids,
            SOFT_NEG_LO, SOFT_NEG_HI, positives_set, SOFT_NEG_COUNT, rng,
        )
        if len(hard_negatives) < HARD_NEG_COUNT:
            short_hard += 1
        if len(soft_negatives) < SOFT_NEG_COUNT:
            short_soft += 1

        output.append(
            {
                "query": record["query"],
                "positive_fact_ids": deep_positives,
                "hard_negative_fact_ids": hard_negatives,
                "soft_negative_fact_ids": soft_negatives,
            }
        )

    print(
        f"Kept {len(output)}/{len(records)} queries with a positive beyond "
        f"{POSITIVE_CHAR_BUDGET} chars"
    )
    if short_hard or short_soft:
        print(
            f"Note: {short_hard} queries had < {HARD_NEG_COUNT} hard-negative candidates, "
            f"{short_soft} had < {SOFT_NEG_COUNT} soft-negative candidates "
            "(corpus too short to fill the band); used what was available."
        )
    return output


def main() -> int:
    import json

    import numpy as np
    from sentence_transformers import SentenceTransformer

    facts = load_facts(FACTS_PATH)
    records = load_source_records(SOURCE_QUERY_FACT_IDS, facts)

    corpus_ids = list(facts)
    corpus_texts = [facts[fact_id] for fact_id in corpus_ids]
    corpus_lengths = np.array([len(text) for text in corpus_texts], dtype=np.int64)
    print(
        f"Loaded {len(corpus_ids)} facts ({corpus_lengths.sum():,} chars total), "
        f"{len(records)} source queries"
    )

    model_path = resolve_model_path()
    model_config = resolve_model_wrapper(MODEL_KEY, model_path)
    device = select_device()
    model = SentenceTransformer(
        model_path, device=device, **model_config.sentence_transformer_kwargs()
    )
    print(f"Loaded model from {model_path} (max_seq_length={model.max_seq_length})")

    corpus_embeddings = encode_corpus(model, model_config, corpus_texts)
    output = build_iterated_records(
        records, corpus_ids, corpus_embeddings, corpus_lengths, model, model_config
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
    print(f"Wrote {len(output)} records to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

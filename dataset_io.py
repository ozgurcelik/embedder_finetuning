"""Shared dataset loading and validation for embedder training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json_value(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_json_object(path: Path) -> dict[str, Any]:
    data = read_json_value(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def read_json_list(path: Path) -> list[Any]:
    data = read_json_value(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    return data


def load_query_fact_ids(path: Path) -> dict[str, set[str]]:
    raw_data = read_json_value(path)
    if isinstance(raw_data, list):
        return load_query_fact_id_records(path, raw_data)
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


def load_query_fact_id_records(
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


def load_facts(path: Path) -> dict[str, str]:
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


def validate_relevance(
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
            f"{len(missing_ids)} relevant fact IDs are missing from the facts file: "
            f"{preview}"
        )


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

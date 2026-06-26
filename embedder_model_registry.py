"""Model-specific query/document encoding wrappers."""

from __future__ import annotations

from typing import Any


QWEN_RETRIEVAL_TASK = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


class EmbedderModelWrapper:
    key = ""
    default_model_name = ""
    output_dir_name = ""
    trust_remote_code = False
    query_prefix = ""
    document_prefix = ""

    def __init__(self, model_name_or_path: str | None = None) -> None:
        self.model_name = model_name_or_path or self.default_model_name

    def sentence_transformer_kwargs(self) -> dict[str, Any]:
        return {"trust_remote_code": self.trust_remote_code}

    def format_query(self, text: str) -> str:
        return apply_prefix(text, self.query_prefix)

    def format_document(self, text: str) -> str:
        return apply_prefix(text, self.document_prefix)

    def encode_query(self, model: Any, texts: list[str], **encode_kwargs: Any):
        return model.encode(
            [self.format_query(text) for text in texts],
            **encode_kwargs,
        )

    def encode_document(self, model: Any, texts: list[str], **encode_kwargs: Any):
        return model.encode(
            [self.format_document(text) for text in texts],
            **encode_kwargs,
        )


class NomicEmbedTextV15(EmbedderModelWrapper):
    key = "nomic-embed-text-v1.5"
    default_model_name = "nomic-ai/nomic-embed-text-v1.5"
    output_dir_name = "nomic-embed-text-v1_5"
    trust_remote_code = True
    query_prefix = "search_query: "
    document_prefix = "search_document: "


class E5BaseV2(EmbedderModelWrapper):
    key = "e5-base-v2"
    default_model_name = "intfloat/e5-base-v2"
    output_dir_name = "e5-base-v2"
    query_prefix = "query: "
    document_prefix = "passage: "


class BgeM3(EmbedderModelWrapper):
    key = "bge-m3"
    default_model_name = "BAAI/bge-m3"
    output_dir_name = "bge-m3"


class EmbeddingGemma300M(EmbedderModelWrapper):
    key = "embeddinggemma-300m"
    default_model_name = "google/embeddinggemma-300m"
    output_dir_name = "embeddinggemma-300m"
    query_prefix = "task: search result | query: "
    document_prefix = "title: none | text: "

    def encode_query(self, model: Any, texts: list[str], **encode_kwargs: Any):
        return model.encode_query(
            [text.strip() for text in texts],
            **encode_kwargs,
        )

    def encode_document(self, model: Any, texts: list[str], **encode_kwargs: Any):
        return model.encode_document(
            [text.strip() for text in texts],
            **encode_kwargs,
        )


class Qwen3Embedding06B(EmbedderModelWrapper):
    key = "qwen3-embedding-0.6b"
    default_model_name = "Qwen/Qwen3-Embedding-0.6B"
    output_dir_name = "qwen3-embedding-0_6b"
    query_prefix = f"Instruct: {QWEN_RETRIEVAL_TASK}\nQuery: "

    def encode_query(self, model: Any, texts: list[str], **encode_kwargs: Any):
        kwargs = dict(encode_kwargs)
        kwargs.setdefault("prompt_name", "query")
        return model.encode(
            [text.strip() for text in texts],
            **kwargs,
        )

    def encode_document(self, model: Any, texts: list[str], **encode_kwargs: Any):
        return model.encode(
            [text.strip() for text in texts],
            **encode_kwargs,
        )


class AllMpnetBaseV2(EmbedderModelWrapper):
    key = "all-mpnet-base-v2"
    default_model_name = "sentence-transformers/all-mpnet-base-v2"
    output_dir_name = "all-mpnet-base-v2"


MODEL_WRAPPERS = (
    NomicEmbedTextV15,
    E5BaseV2,
    BgeM3,
    EmbeddingGemma300M,
    Qwen3Embedding06B,
    AllMpnetBaseV2,
)
MODEL_REGISTRY: dict[str, type[EmbedderModelWrapper]] = {
    wrapper.key: wrapper for wrapper in MODEL_WRAPPERS
}
DEFAULT_MODEL_KEY = "all-mpnet-base-v2"


def model_max_supported_tokens(model: Any) -> int | None:
    """Largest sequence length the loaded model can encode without overflowing
    its position-embedding table. Returns None if it cannot be determined."""
    limits: list[int] = []

    tokenizer_max = getattr(getattr(model, "tokenizer", None), "model_max_length", None)
    # Some tokenizers use a huge sentinel value to mean "no limit"; ignore those.
    if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 100_000:
        limits.append(tokenizer_max)

    try:
        hf_config = model[0].auto_model.config
    except (AttributeError, IndexError, KeyError, TypeError):
        hf_config = None
    if hf_config is not None:
        max_pos = getattr(hf_config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            # MPNet/RoBERTa-style models offset position ids past the padding
            # index, so the usable length is smaller than the table size.
            pad_offset_types = {"mpnet", "roberta", "xlm-roberta", "camembert"}
            pad_idx = getattr(hf_config, "pad_token_id", None)
            if (
                getattr(hf_config, "model_type", "") in pad_offset_types
                and isinstance(pad_idx, int)
                and pad_idx >= 0
            ):
                max_pos = max_pos - pad_idx - 1
            limits.append(max_pos)

    return min(limits) if limits else None


def resolve_max_seq_length(model: Any, requested: int | None) -> int | None:
    """Clamp the requested max_seq_length to what the model actually supports."""
    supported = model_max_supported_tokens(model)
    if supported is None:
        return requested
    if requested is None:
        return supported
    if requested > supported:
        print(
            f"Warning: requested max_seq_length={requested} exceeds the model's "
            f"supported limit of {supported} tokens; clamping to {supported}."
        )
        return supported
    return requested


def apply_prefix(text: str, prefix: str) -> str:
    stripped = text.strip()
    if not prefix or stripped.startswith(prefix):
        return stripped
    return f"{prefix}{stripped}"


def available_model_keys() -> list[str]:
    return sorted(MODEL_REGISTRY)


def resolve_model_wrapper(
    model_key: str = DEFAULT_MODEL_KEY,
    model_name_or_path: str | None = None,
) -> EmbedderModelWrapper:
    try:
        wrapper = MODEL_REGISTRY[model_key]
    except KeyError as exc:
        available = ", ".join(available_model_keys())
        raise ValueError(f"Unknown model key {model_key!r}. Available: {available}") from exc

    return wrapper(model_name_or_path)


def resolve_model_config(
    model_key: str = DEFAULT_MODEL_KEY,
    model_name_or_path: str | None = None,
) -> EmbedderModelWrapper:
    return resolve_model_wrapper(model_key, model_name_or_path)

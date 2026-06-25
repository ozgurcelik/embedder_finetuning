"""Training and evaluation settings. Edit the values below to configure a run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class Config:
    # --- Model ---
    model_key: str = "nomic-embed-text-v1.5"
    model: str | None = None  # optional name/path override; model_key still controls encoding
    output_dir: Path | None = None  # defaults to models/<model>-<dataset>

    # --- Training dataset ---
    train_dataset_key: str = "qaitrain500-500"
    train_dataset_name: str | None = None  # Hugging Face dataset name override
    train_dataset_config: str | None = None  # Hugging Face dataset config override
    train_splits: list[str] | None = None  # splits to combine; ["all"] for every split
    query_column: str | None = None
    positive_column: str | None = None
    train_query_fact_ids: Path | None = None  # local training JSON override
    train_facts: Path | None = None  # local fact-text JSON override
    negatives_per_query: int = 4

    # --- Optimization ---
    epochs: int = 2
    max_steps: int = -1  # -1 trains for `epochs`
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int | None = 1024  # None uses the model default (8192 for nomic -> OOM)
    seed: int = 42

    # --- Precision ---
    fp16: bool = False
    bf16: bool = True  # A5000 supports bf16; ~halves memory vs fp32
    tf32: bool | None = None  # None uses the trainer default

    max_train_samples: int | None = None  # cap after combining splits

    # --- Checkpointing & logging ---
    save_strategy: str = "steps"  # "epoch" | "steps" | "no"
    save_steps: int = 1000
    logging_steps: int = 50

    # --- Weights & Biases ---
    wandb: bool = True
    wandb_entity: str = "uthereal"
    wandb_project: str = "embedder_finetuning"
    wandb_run_name: str | None = None  # defaults to the output directory name
    wandb_mode: str | None = None  # "online" | "offline"

    # --- Evaluation ---
    eval_strategy: str = "steps"  # "steps" | "epoch" | "no"
    eval_steps: int = 100
    # Which eval datasets to run; keys come from EVAL_DATASET_REGISTRY in retrieval_evaluator.py
    eval_dataset_keys: list[str] = field(
        default_factory=lambda: ["qaitest100-500", "qaitest100-500-2"]
    )
    eval_batch_size: int = 32
    eval_corpus_chunk_size: int = 50000
    eval_at_start: bool = True
    no_eval_csv: bool = False  # set True to skip writing evaluator metric files

    # --- Misc ---
    dry_run: bool = False  # prepare data, print a summary, then exit before training
    use_cached_mnrl: bool = False  # CachedMultipleNegativesRankingLoss for bigger batches
    cached_mini_batch_size: int = 16

    def validate(self) -> None:
        require_positive("batch_size", self.batch_size)
        require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        if self.save_strategy == "steps":
            require_positive("save_steps", self.save_steps)
        require_positive("logging_steps", self.logging_steps)
        if self.eval_strategy == "steps":
            require_positive("eval_steps", self.eval_steps)
        if self.eval_strategy != "no":
            require_positive("eval_batch_size", self.eval_batch_size)
            require_positive("eval_corpus_chunk_size", self.eval_corpus_chunk_size)
            # Imported lazily to avoid a circular import (retrieval_evaluator imports config).
            from retrieval_evaluator import available_eval_dataset_keys

            if not self.eval_dataset_keys:
                raise ValueError(
                    "eval_dataset_keys cannot be empty when eval_strategy is not 'no'"
                )
            unknown_keys = sorted(
                set(self.eval_dataset_keys) - set(available_eval_dataset_keys())
            )
            if unknown_keys:
                available = ", ".join(available_eval_dataset_keys())
                raise ValueError(
                    f"Unknown eval_dataset_keys {unknown_keys}. Available: {available}"
                )
        if self.max_train_samples is not None:
            require_positive("max_train_samples", self.max_train_samples)
        require_positive("negatives_per_query", self.negatives_per_query)
        if self.max_seq_length is not None:
            require_positive("max_seq_length", self.max_seq_length)
        if self.fp16 and self.bf16:
            raise ValueError("Use only one mixed precision mode: fp16 or bf16")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.max_steps == 0 or self.max_steps < -1:
            raise ValueError(f"max_steps must be -1 or positive, got {self.max_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")
        if self.wandb and not self.wandb_entity:
            raise ValueError("wandb_entity cannot be empty when W&B logging is enabled")
        if self.wandb and not self.wandb_project:
            raise ValueError("wandb_project cannot be empty when W&B logging is enabled")

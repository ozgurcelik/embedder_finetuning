"""Training and evaluation settings. Edit the values below to configure a run."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path


def require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass
class StageConfig:
    """One training stage in a chain. Every field except `name` defaults to None,
    meaning "inherit from the base Config". Set only what should differ for this stage.

    Stages run sequentially on the same model, so weights carry over from one to the
    next. Global settings (model, precision, evaluation, W&B, Hub, output root) always
    come from the base Config; only the fields below can vary per stage."""

    name: str = "stage"

    # --- Training dataset (None inherits from Config) ---
    train_dataset_key: str | None = None
    # Mix several registered datasets into this one stage (overrides train_dataset_key).
    train_dataset_keys: list[str] | None = None
    train_dataset_name: str | None = None
    train_dataset_config: str | None = None
    train_splits: list[str] | None = None
    query_column: str | None = None
    positive_column: str | None = None
    train_query_fact_ids: Path | None = None
    train_facts: Path | None = None
    negatives_per_query: int | None = None
    max_train_samples: int | None = None

    # --- Optimization (None inherits from Config) ---
    epochs: int | None = None
    max_steps: int | None = None
    batch_size: int | None = None
    gradient_accumulation_steps: int | None = None
    learning_rate: float | None = None
    warmup_ratio: float | None = None
    seed: int | None = None

    # --- Loss (None inherits from Config) ---
    use_cached_mnrl: bool | None = None
    cached_mini_batch_size: int | None = None

    # --- Evaluation (None inherits from Config) ---
    eval_strategy: str | None = None  # "steps" | "epoch" | "no"
    eval_steps: int | None = None  # absolute steps between evals (eval_strategy == "steps")
    # Evaluate this many times per epoch; overrides eval_steps and forces step-based eval.
    # e.g. 4 -> evaluate every 1/4 epoch, 2 -> every half epoch.
    evals_per_epoch: int | None = None

    # --- Checkpointing (None inherits from Config) ---
    save_strategy: str | None = None
    save_steps: int | None = None


@dataclass
class Config:
    # --- Model ---
    model_key: str = "all-mpnet-base-v2"
    # Optional name/path override; model_key still controls encoding/formatting. Point this
    # at a locally saved model to continue fine-tuning from it (e.g. the saved fast model).
    model: str | None = "/root/models/all-mpnet-base-v2-qaitrain500-5000-fast/final"
    output_dir: Path | None = None  # defaults to models/<model>-<dataset>

    # --- Training dataset ---
    train_dataset_key: str = "qaitrain500-500"
    # Mix several registered datasets into a single stage (keys come from
    # TRAIN_DATASET_REGISTRY in training_data.py). When set, this overrides train_dataset_key
    # and the listed datasets are combined at the row level, shuffled together, and trained
    # as one pool. Example: ["qaitrain500-500", "qaitrain500-2-500"].
    train_dataset_keys: list[str] | None = None
    train_dataset_name: str | None = None  # Hugging Face dataset name override
    train_dataset_config: str | None = None  # Hugging Face dataset config override
    train_splits: list[str] | None = None  # splits to combine; ["all"] for every split
    query_column: str | None = None
    positive_column: str | None = None
    train_query_fact_ids: Path | None = None  # local training JSON override
    train_facts: Path | None = None  # local fact-text JSON override
    negatives_per_query: int = 4  # explicit negatives per row; pool is 20 hard + 20 soft (=40)

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
    bf16: bool = False  # A5000 supports bf16; ~halves memory vs fp32
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
    eval_strategy: str = "epoch"  # "steps" | "epoch" | "no"
    eval_steps: int = 100  # only used when eval_strategy == "steps"
    # Evaluate this many times per epoch; overrides eval_steps and forces step-based eval
    # (e.g. 4 -> every 1/4 epoch). None keeps the eval_strategy/eval_steps above.
    evals_per_epoch: int | None = None
    # Which eval datasets to run; keys come from EVAL_DATASET_REGISTRY in retrieval_evaluator.py
    eval_dataset_keys: list[str] = field(
        default_factory=lambda: ["qaitest500-500", "qaitest100-500-2"]
    )
    eval_batch_size: int = 32
    eval_corpus_chunk_size: int = 50000
    eval_at_start: bool = True
    no_eval_csv: bool = False  # set True to skip writing evaluator metric files

    # --- Saving & Hugging Face Hub ---
    save_model_locally: bool = True  # save the final model under output_dir/final
    push_to_hub: bool = False  # push the trained model to the Hugging Face Hub (off by default)
    hub_account: str = "ozgur-celik"  # HF user/org the model is pushed under
    hub_model_id: str | None = None  # full repo id; defaults to <hub_account>/<output_dir name>
    hub_version: str | None = None  # optional version suffix, e.g. "v2" -> <repo>-v2
    push_every_save_step: bool = False  # push at every save step, not only at the end
    hub_private: bool = False  # create the hub repo as private

    # --- Multi-stage training ---
    # Optional chain of training stages run sequentially on the same model (weights
    # carry over). Each StageConfig overrides only the fields it sets; everything else
    # inherits from this Config. Leave empty for a single-stage run using the fields
    # above. Example:
    #   stages = [
    #       StageConfig(name="oqa-v1", train_dataset_key="oqa-v1", epochs=1),
    #       StageConfig(name="qaitrain500-500", train_dataset_key="qaitrain500-500", epochs=2),
    #   ]
    stages: list[StageConfig] = field(
        default_factory=lambda: [
            # StageConfig(name="oqa-v1", train_dataset_key="oqa-v1", epochs=1),
            # StageConfig(
            #     name="qaitrain500-500", train_dataset_key="qaitrain500-500", epochs=1
            # ),
            # StageConfig(
            #     name="qaitrain500-2-500", train_dataset_key="qaitrain500-2-500", epochs=1
            # ),
            # qai-mixed runs first (explicit hard/soft negatives sharpen the top ranks), then
            # the broad in-batch stage runs last so it has the final say on the embedding
            # geometry -- the last stage dominates, and the in-batch objective is what
            # maximizes deep recall@chars.
            # StageConfig(
            #     name="qai-mixed",
            #     train_dataset_keys=["qaitrain500-500", "qaitrain500-2-500"],
            #     epochs=1,
            #     learning_rate=5e-6,
            # ),
            # Three "fast" datasets (1 positive per query, no curated negatives) mixed into
            # one in-batch stage: negatives_per_query=0 + batch size 32 (= 31 in-batch
            # negatives), rows from all three are combined and shuffled together.
            # StageConfig(
            #     name="qai-fast-mixed",
            #     train_dataset_keys=[
            #         "qaitrain500-3-5000-fast",
            #         "qaitrain500-5000-fast",
            #         "qaitrain500-2-5000-fast",
            #     ],
            #     negatives_per_query=0,
            #     batch_size=64,
            #     epochs=1,
            #     # Eval at ~0.25 / 0.5 / 0.75 epoch and at the end (plus the step-0 baseline,
            #     # since this can be the first stage).
            #     evals_per_epoch=1,
            # ),
            # Single in-batch stage on qaitrain500-5000-fast (1 positive per query, no
            # curated negatives -> in-batch negatives only).
            # StageConfig(
            #     name="qaitrain500-5000-fast",
            #     train_dataset_key="qaitrain500-5000-fast",
            #     negatives_per_query=0,
            #     batch_size=32,
            #     epochs=1,
            # ),
            # Continue fine-tuning the saved fast model (Config.model above) on the iterated
            # hard set: deep positives + 4 hard/4 soft negatives per query (pool of 8, sample
            # 4). Small + gentle LR to refine without forgetting.
            StageConfig(
                name="qaitrain500-2-500-iterated",
                train_dataset_key="qaitrain500-2-500-iterated",
                negatives_per_query=4,
                batch_size=16,
                epochs=3,
                learning_rate=5e-6,
            ),
        ]
    )

    # --- Misc ---
    dry_run: bool = False  # prepare data, print a summary, then exit before training
    use_cached_mnrl: bool = True  # CachedMultipleNegativesRankingLoss for bigger batches
    cached_mini_batch_size: int = 16

    def validate(self) -> None:
        require_positive("batch_size", self.batch_size)
        require_positive("gradient_accumulation_steps", self.gradient_accumulation_steps)
        if self.save_strategy == "steps":
            require_positive("save_steps", self.save_steps)
        require_positive("logging_steps", self.logging_steps)
        if self.eval_strategy == "steps":
            require_positive("eval_steps", self.eval_steps)
        if self.evals_per_epoch is not None:
            require_positive("evals_per_epoch", self.evals_per_epoch)
            if self.eval_strategy == "no":
                raise ValueError("evals_per_epoch requires eval_strategy to not be 'no'")
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
        # Imported lazily to avoid a circular import (training_data imports config).
        from training_data import available_train_dataset_keys, resolve_train_dataset_keys

        if self.train_dataset_keys is not None and not self.train_dataset_keys:
            raise ValueError("train_dataset_keys cannot be an empty list; use None instead")
        unknown_train_keys = sorted(
            set(resolve_train_dataset_keys(self)) - set(available_train_dataset_keys())
        )
        if unknown_train_keys:
            available = ", ".join(available_train_dataset_keys())
            raise ValueError(
                f"Unknown train dataset key(s) {unknown_train_keys}. Available: {available}"
            )
        if self.max_train_samples is not None:
            require_positive("max_train_samples", self.max_train_samples)
        # 0 is allowed: it means "no explicit negatives, use in-batch negatives only".
        if self.negatives_per_query < 0:
            raise ValueError(
                f"negatives_per_query must be >= 0, got {self.negatives_per_query}"
            )
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
        if self.push_to_hub:
            if not self.hub_model_id and not self.hub_account:
                raise ValueError("Set hub_account or hub_model_id to push to the Hub")
            if self.push_every_save_step and self.save_strategy == "no":
                raise ValueError(
                    "push_every_save_step requires save_strategy to be 'steps' or 'epoch'"
                )
        # Validate each stage by resolving it against this base config and checking the
        # merged result (resolve clears `stages`, so this does not recurse infinitely).
        for stage in self.stages:
            resolve_stage_config(self, stage).validate()


def resolve_stage_config(base: Config, stage: StageConfig) -> Config:
    """Merge a StageConfig onto a base Config: every non-None stage field overrides the
    base value; everything else is inherited. The returned Config has `stages` cleared."""
    overrides = {
        f.name: getattr(stage, f.name)
        for f in fields(stage)
        if f.name != "name" and getattr(stage, f.name) is not None
    }
    return replace(base, stages=[], **overrides)

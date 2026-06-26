#!/usr/bin/env python3
"""Fine-tune embedding models on question/context retrieval pairs.

Edit run settings in config.py, then run this file: python3 embedder_finetuning.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from config import Config
from dataset_io import select_device
from embedder_model_registry import resolve_max_seq_length, resolve_model_wrapper
from retrieval_evaluator import build_retrieval_evaluator
from training_data import (
    build_training_rows,
    dataset_spec_from_args,
    default_output_dir,
    print_dataset_summary,
    rows_to_dataset,
)


def build_loss(model: Any, use_cached_mnrl: bool, cached_mini_batch_size: int):
    from sentence_transformers import losses

    if use_cached_mnrl:
        return losses.CachedMultipleNegativesRankingLoss(
            model,
            mini_batch_size=cached_mini_batch_size,
        )
    return losses.MultipleNegativesRankingLoss(model)


def configure_wandb(config: Config) -> str:
    if not config.wandb:
        return "none"

    try:
        import wandb  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency: wandb. Install it in embeddervenv with:\n"
            "  conda run -n embeddervenv python -m pip install -U wandb"
        ) from exc

    config.output_dir.mkdir(parents=True, exist_ok=True)

    os.environ["WANDB_ENTITY"] = config.wandb_entity
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_DIR"] = str(config.output_dir)
    if config.wandb_mode:
        os.environ["WANDB_MODE"] = config.wandb_mode

    print(
        "Weights & Biases logging enabled: "
        f"entity={config.wandb_entity}, project={config.wandb_project}"
    )
    return "wandb"


def train(config: Config) -> None:
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.training_args import BatchSamplers

    model_config = resolve_model_wrapper(config.model_key, config.model)
    dataset_spec = dataset_spec_from_args(config)
    if config.output_dir is None:
        config.output_dir = default_output_dir(model_config, dataset_spec)

    hub_model_id = config.hub_model_id or f"{config.hub_account}/{config.output_dir.name}"

    rows = build_training_rows(config, dataset_spec, model_config)
    print_dataset_summary(rows)

    if config.dry_run:
        print("Dry run complete; no model was trained.")
        return

    report_to = configure_wandb(config)
    train_dataset = rows_to_dataset(rows)
    device = select_device()
    model_kwargs = model_config.sentence_transformer_kwargs()
    model_kwargs["device"] = device
    model = SentenceTransformer(
        model_config.model_name,
        **model_kwargs,
    )
    print(f"Model device: {model.device}")
    effective_max_seq_length = resolve_max_seq_length(model, config.max_seq_length)
    if effective_max_seq_length is not None:
        model.max_seq_length = effective_max_seq_length
    print(f"Model max_seq_length: {model.max_seq_length}")
    train_loss = build_loss(
        model=model,
        use_cached_mnrl=config.use_cached_mnrl,
        cached_mini_batch_size=config.cached_mini_batch_size,
    )
    evaluator = build_retrieval_evaluator(config, model_config)

    # The trainer's push_to_hub only drives intermediate, per-save-step uploads. The
    # final end-of-training push is done explicitly after trainer.train() and is gated
    # on config.push_to_hub alone, so end-only pushing works regardless of this flag.
    push_checkpoints_every_save = config.push_to_hub and config.push_every_save_step

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=None,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps if config.eval_strategy == "steps" else None,
        eval_on_start=config.eval_strategy != "no" and config.eval_at_start,
        do_eval=config.eval_strategy != "no",
        fp16=config.fp16,
        bf16=config.bf16,
        fp16_full_eval=config.fp16,
        bf16_full_eval=config.bf16,
        tf32=config.tf32,
        use_cpu=device == "cpu",
        logging_steps=config.logging_steps,
        report_to=report_to,
        run_name=config.wandb_run_name or config.output_dir.name,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_pin_memory=False,
        disable_tqdm=False,
        push_to_hub=push_checkpoints_every_save,
        hub_model_id=hub_model_id,
        hub_strategy="every_save",
        hub_private_repo=config.hub_private,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    if config.save_model_locally:
        final_model_dir = config.output_dir / "final"
        model.save_pretrained(str(final_model_dir))
        print(f"Saved final model to {final_model_dir}")

    if config.push_to_hub:
        # Explicitly upload the trained model. The trainer's hub_strategy only handles
        # intermediate (every_save) pushes and repo creation; the final model must be
        # pushed here, otherwise the repo ends up containing only .gitattributes.
        print(f"Pushing model to https://huggingface.co/{hub_model_id} ...")
        model.push_to_hub(
            hub_model_id,
            private=config.hub_private,
            exist_ok=True,
        )
        print(f"Pushed model to https://huggingface.co/{hub_model_id}")
    elif not config.save_model_locally:
        print(
            "Note: save_model_locally and push_to_hub are both False; "
            "the trained model was not exported (only trainer checkpoints, if any, remain)."
        )


def main() -> int:
    config = Config()
    try:
        config.validate()
        train(config)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Training error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

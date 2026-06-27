#!/usr/bin/env python3
"""Fine-tune embedding models on question/context retrieval pairs.

Edit run settings in config.py, then run this file: python3 embedder_finetuning.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

from config import Config, StageConfig, resolve_stage_config
from dataset_io import select_device
from embedder_model_registry import resolve_max_seq_length, resolve_model_wrapper
from retrieval_evaluator import build_retrieval_evaluator, set_evaluator_stage_label
from training_data import (
    DEFAULT_OUTPUT_ROOT,
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


def run_final_eval(trainer: Any, evaluator: Any, model: Any, config: Config) -> None:
    """Evaluate once at the final global step so the last checkpoint shows up on the
    eval curves. Skipped when the final step already coincided with a scheduled eval
    (i.e. an "epoch" strategy, or a "steps" strategy where the step is a multiple of
    eval_steps), to avoid logging a duplicate point."""
    if evaluator is None or config.eval_strategy == "no":
        return

    global_step = trainer.state.global_step
    if config.eval_strategy == "epoch":
        return
    if config.eval_strategy == "steps" and global_step % config.eval_steps == 0:
        return

    epoch = int(trainer.state.epoch) if trainer.state.epoch is not None else -1
    print(f"Running end-of-run evaluation at step {global_step} ...")
    evaluator(
        model,
        output_path=str(config.output_dir),
        epoch=epoch,
        steps=global_step,
    )


def resolve_run_root(config: Config, model_config: Any, stages: list[StageConfig]) -> Any:
    """Top-level output directory for the run. For a single stage this matches the old
    models/<model>-<dataset> layout; for a chain it is models/<model>-chain-<stages>."""
    if config.output_dir is not None:
        return config.output_dir
    if len(stages) == 1:
        stage_config = resolve_stage_config(config, stages[0])
        return default_output_dir(model_config, dataset_spec_from_args(stage_config))
    suffix = "-".join(stage.name for stage in stages)
    return DEFAULT_OUTPUT_ROOT / f"{model_config.output_dir_name}-chain-{suffix}"


def resolve_hub_model_id(config: Config, run_root: Any) -> str:
    if config.hub_model_id:
        return config.hub_model_id
    repo_name = run_root.name
    if config.hub_version:
        repo_name = f"{repo_name}-{config.hub_version}"
    return f"{config.hub_account}/{repo_name}"


def build_training_args(
    config: Config,
    device: str,
    report_to: str,
    hub_model_id: str,
    push_checkpoints_every_save: bool,
    is_first_stage: bool = True,
):
    from sentence_transformers import SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers

    # Only evaluate at the start of the very first stage. For later stages the start
    # state is identical to the previous stage's final (already-evaluated) checkpoint,
    # so a start-of-stage eval would just duplicate that point.
    eval_on_start = (
        config.eval_strategy != "no" and config.eval_at_start and is_first_stage
    )

    return SentenceTransformerTrainingArguments(
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
        eval_on_start=eval_on_start,
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


def save_and_push_model(config: Config, model: Any, hub_model_id: str) -> None:
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


def run_stage_dry_run(config: Config, model_config: Any, stages: list[StageConfig]) -> None:
    multistage = len(stages) > 1
    for index, stage in enumerate(stages, start=1):
        stage_config = resolve_stage_config(config, stage)
        if multistage:
            print(f"\n=== Stage {index}/{len(stages)}: {stage.name} ===")
        rows = build_training_rows(
            stage_config, dataset_spec_from_args(stage_config), model_config
        )
        print_dataset_summary(rows)
    print("Dry run complete; no model was trained.")


def train(config: Config) -> None:
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer

    model_config = resolve_model_wrapper(config.model_key, config.model)
    # An empty stages list means a single implicit stage built from the top-level fields,
    # which keeps single-stage runs behaving exactly as before.
    stages = config.stages or [StageConfig(name=config.train_dataset_key)]
    multistage = len(stages) > 1

    run_root = resolve_run_root(config, model_config, stages)
    config.output_dir = run_root
    hub_model_id = resolve_hub_model_id(config, run_root)

    if config.dry_run:
        run_stage_dry_run(config, model_config, stages)
        return

    report_to = configure_wandb(config)
    device = select_device()
    model_kwargs = model_config.sentence_transformer_kwargs()
    model_kwargs["device"] = device
    model = SentenceTransformer(model_config.model_name, **model_kwargs)
    print(f"Model device: {model.device}")
    effective_max_seq_length = resolve_max_seq_length(model, config.max_seq_length)
    if effective_max_seq_length is not None:
        model.max_seq_length = effective_max_seq_length
    print(f"Model max_seq_length: {model.max_seq_length}")

    # Built once and reused across stages so the eval curves accumulate over the chain.
    evaluator = build_retrieval_evaluator(config, model_config)
    push_checkpoints_every_save = config.push_to_hub and config.push_every_save_step
    base_run_name = config.wandb_run_name or run_root.name

    for index, stage in enumerate(stages, start=1):
        stage_config = resolve_stage_config(config, stage)
        if multistage:
            stage_config.output_dir = run_root / f"stage{index}_{stage.name}"
            stage_config.wandb_run_name = f"{base_run_name}-stage{index}-{stage.name}"
            set_evaluator_stage_label(evaluator, stage.name)
            print(f"\n=== Stage {index}/{len(stages)}: {stage.name} ===")
        else:
            stage_config.output_dir = run_root
            stage_config.wandb_run_name = base_run_name
            set_evaluator_stage_label(evaluator, "")

        rows = build_training_rows(
            stage_config, dataset_spec_from_args(stage_config), model_config
        )
        print_dataset_summary(rows)
        train_dataset = rows_to_dataset(rows)
        train_loss = build_loss(
            model=model,
            use_cached_mnrl=stage_config.use_cached_mnrl,
            cached_mini_batch_size=stage_config.cached_mini_batch_size,
        )
        training_args = build_training_args(
            stage_config,
            device,
            report_to,
            hub_model_id,
            push_checkpoints_every_save,
            is_first_stage=index == 1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
        )
        trainer.train()
        run_final_eval(trainer, evaluator, model, stage_config)

        # Finish the per-stage W&B run so the next stage starts a fresh run with its own
        # monotonic step axis. The evaluator keeps its history, so each subsequent stage's
        # run still shows the full accumulated curve across all stages so far.
        if report_to == "wandb" and multistage:
            import wandb

            wandb.finish()

    save_and_push_model(config, model, hub_model_id)


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

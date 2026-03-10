#!/usr/bin/env python3
"""Generate evaluation images from the *base* model only, using existing repo logic.

This script does NOT implement a new generation pipeline.
It only wires together the repo's existing components:
- src.config.schema.load_config
- src.training.trainer.DreamBoothLoRATrainer
- src.eval.generator.generate_eval_images
- src.utils.logging.setup_logging
- src.utils.seed.set_global_seed

Example:
  python test_base_model_preview.py \
    --repo-root /workspace/DreamBooth-CLoRA \
    --config configs/tasks_5char.yaml \
    --task-index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview what the base model generates for one configured task",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Path to the DreamBooth-CLoRA repository root",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML config (relative to --repo-root or absolute)",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=0,
        help="Task index to preview (ignored if --task-name is given)",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Optional task name to preview instead of using --task-index",
    )
    parser.add_argument(
        "--prompt-mode",
        type=str,
        choices=["auto", "dreambooth", "clora"],
        default="auto",
        help=(
            "Prompt mode passed to src.eval.generator.generate_eval_images. "
            "'auto' follows the repo config for faithful_c_lora, otherwise uses dreambooth."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional explicit output directory for generated images",
    )
    return parser.parse_args()


def resolve_config_path(repo_root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    return config_path.resolve()


def pick_task(tasks, task_name: str | None, task_index: int):
    if task_name is not None:
        for task in tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task name not found in config: {task_name}")

    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task-index out of range: {task_index} (num_tasks={len(tasks)})")
    return tasks[task_index]


def resolve_prompt_mode(config, prompt_mode_arg: str) -> str:
    if prompt_mode_arg != "auto":
        return prompt_mode_arg
    if config.experiment.method == "faithful_c_lora":
        return config.c_lora.prompt_mode
    return "dreambooth"


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from src.config.schema import load_config
    from src.eval.generator import generate_eval_images
    from src.training.trainer import DreamBoothLoRATrainer
    from src.utils.logging import setup_logging
    from src.utils.seed import set_global_seed

    config_path = resolve_config_path(repo_root, args.config)
    config = load_config(str(config_path))
    task = pick_task(config.tasks, args.task_name, args.task_index)
    prompt_mode = resolve_prompt_mode(config, args.prompt_mode)

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else Path(config.experiment.output_dir)
        / "debug"
        / "base_model_preview"
        / task.name
    )

    logger = setup_logging(output_dir=str(output_dir))
    set_global_seed(config.experiment.seed)

    logger.info("Repo root: %s", repo_root)
    logger.info("Config: %s", config_path)
    logger.info("Selected task: %s", task.name)
    logger.info("Prompt mode: %s", prompt_mode)
    logger.info("Output dir: %s", output_dir)

    if prompt_mode == "clora":
        logger.warning(
            "You are previewing the BASE model with prompt_mode='clora'. "
            "That means prompts will contain the configured trigger token, but no trained token embedding is loaded."
        )

    trainer = DreamBoothLoRATrainer(config.model, config.training)
    trainer.load_models()
    pipe = trainer.build_inference_pipeline()

    generated_paths = generate_eval_images(
        pipeline=pipe,
        task=task,
        eval_config=config.evaluation,
        output_dir=str(output_dir),
        seed=config.experiment.seed,
        prompt_mode=prompt_mode,
    )

    print("=" * 80)
    print("BASE MODEL PREVIEW COMPLETE")
    print(f"Task          : {task.name}")
    print(f"Prompt mode   : {prompt_mode}")
    print(f"Images saved  : {len(generated_paths)}")
    print(f"Output folder : {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()

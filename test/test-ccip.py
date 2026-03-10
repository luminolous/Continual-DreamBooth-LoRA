#!/usr/bin/env python3
"""Test the repo's CCIP evaluation logic on train-vs-ref folders.

This script does NOT re-implement CCIP.
It only calls the repo's existing evaluation function:
- src.eval.metrics.compute_ccip_score

By default it scores every task in the config using:
  generated_dir = task.data_dir
  ref_dir       = task.ref_dir

Example:
  python test_ccip_train_vs_ref.py \
    --repo-root /workspace/DreamBooth-CLoRA \
    --config configs/tasks_5char.yaml
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the repo CCIP metric on train-vs-ref directories",
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
        default=None,
        help="Optional single task index to score",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default=None,
        help="Optional single task name to score",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for logs",
    )
    return parser.parse_args()


def resolve_config_path(repo_root: Path, config_arg: str) -> Path:
    config_path = Path(config_arg)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    return config_path.resolve()


def pick_tasks(tasks, task_name: str | None, task_index: int | None):
    if task_name is not None:
        for task in tasks:
            if task.name == task_name:
                return [task]
        raise ValueError(f"Task name not found in config: {task_name}")

    if task_index is not None:
        if task_index < 0 or task_index >= len(tasks):
            raise IndexError(f"task-index out of range: {task_index} (num_tasks={len(tasks)})")
        return [tasks[task_index]]

    return list(tasks)


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from src.config.schema import load_config
    from src.eval.metrics import compute_ccip_score
    from src.utils.logging import setup_logging

    config_path = resolve_config_path(repo_root, args.config)
    config = load_config(str(config_path))

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else Path(config.experiment.output_dir) / "debug" / "ccip_train_vs_ref"
    )

    logger = setup_logging(output_dir=str(output_dir), log_filename="ccip_train_vs_ref.log")
    logger.info("Repo root: %s", repo_root)
    logger.info("Config: %s", config_path)

    selected_tasks = pick_tasks(config.tasks, args.task_name, args.task_index)
    scores = []

    print("=" * 80)
    print("CCIP TRAIN-vs-REF TEST")
    print("generated_dir = task.data_dir")
    print("ref_dir       = task.ref_dir")
    print("=" * 80)

    for task in selected_tasks:
        train_dir = Path(task.data_dir)
        ref_dir = Path(task.ref_dir)

        if not train_dir.exists():
            raise FileNotFoundError(f"Train dir not found for task '{task.name}': {train_dir}")
        if not ref_dir.exists():
            raise FileNotFoundError(f"Ref dir not found for task '{task.name}': {ref_dir}")

        score = compute_ccip_score(str(train_dir), str(ref_dir))
        scores.append(score)

        logger.info("Task=%s | train=%s | ref=%s | ccip=%.6f", task.name, train_dir, ref_dir, score)
        print(f"{task.name:20s}  CCIP={score:.6f}")
        print(f"  train: {train_dir}")
        print(f"  ref  : {ref_dir}")

    if scores:
        mean_score = statistics.mean(scores)
        print("-" * 80)
        print(f"Num tasks scored : {len(scores)}")
        print(f"Mean CCIP        : {mean_score:.6f}")
        print(f"Log dir          : {output_dir}")
        print("-" * 80)


if __name__ == "__main__":
    main()

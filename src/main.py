"""
CLI entry point for Continual DreamBooth-CLoRA.

Usage:
    # Legacy methods (naive baseline / scaffold)
    python -m src.main --config configs/tasks_5char.yaml

    # Faithful C-LoRA (shared continual model with occupancy regularization)
    python -m src.main --config configs/tasks_5char_faithful.yaml

    # Eval-only from faithful checkpoint (restores full continual state)
    python -m src.main --config configs/tasks_5char_faithful.yaml --eval-only outputs/5char_faithful

    # Eval-only from legacy checkpoint
    python -m src.main --config configs/tasks_5char.yaml --eval-only outputs/5char_naive/checkpoints/task_04_char_05/lora_weights

    # Resume faithful training from checkpoint
    python -m src.main --config configs/tasks_5char_faithful.yaml --resume
"""

from __future__ import annotations

import argparse
import sys
import time

from src.config.schema import load_config
from src.orchestrator.pipeline import ContinualPipeline
from src.utils.logging import setup_logging
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continual DreamBooth-LoRA for Sequential Anime Character Personalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main --config configs/tasks_5char.yaml
  python -m src.main --config configs/tasks_5char.yaml --dry-run
  python -m src.main --config configs/tasks_5char_faithful.yaml --eval-only outputs/5char_faithful
  python -m src.main --config configs/tasks_5char_faithful.yaml --resume
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and datasets without running training",
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="CHECKPOINT_DIR",
        help=(
            "Skip training and run evaluation only. "
            "For faithful_c_lora: pass the base experiment output dir. "
            "For legacy: pass the LoRA checkpoint directory."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume faithful_c_lora training from the last completed task. "
            "Requires faithful_c_lora method and existing checkpoint state."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup logging (to console + output dir)
    logger = setup_logging(output_dir=config.experiment.output_dir)
    logger.info("Loaded config from: %s", args.config)
    logger.info("Experiment: %s", config.experiment.name)
    logger.info("Method: %s", config.experiment.method)
    logger.info("Tasks: %d", len(config.tasks))
    logger.info("Output: %s", config.experiment.output_dir)

    # Set global seed
    set_global_seed(config.experiment.seed)
    logger.info("Global seed set to %d", config.experiment.seed)

    # Initialize pipeline
    pipeline = ContinualPipeline(config)

    if args.dry_run:
        logger.info("DRY RUN: validating config and datasets only")
        pipeline.validate_all_tasks()
        logger.info("DRY RUN complete — all validations passed")
        return

    # Run the full pipeline
    start_time = time.time()
    try:
        if args.eval_only:
            logger.info("EVAL-ONLY mode: checkpoint=%s", args.eval_only)
            pipeline.run_eval_only(args.eval_only)
        elif args.resume:
            logger.info("RESUME mode")
            if config.experiment.method != "faithful_c_lora":
                print(
                    "ERROR: --resume is only supported for faithful_c_lora method",
                    file=sys.stderr,
                )
                sys.exit(1)
            pipeline.resume_faithful()
        else:
            pipeline.run()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info("Total elapsed time: %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()

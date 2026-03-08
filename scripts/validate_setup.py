#!/usr/bin/env python3
"""
Quick sanity check script for DreamBooth-CLoRA setup.

Verifies:
- Python version
- Required packages importable
- Config file loadable
- Dataset directories exist (if config provided)

Usage:
    python scripts/validate_setup.py
    python scripts/validate_setup.py --config configs/tasks_5char.yaml
"""

from __future__ import annotations

import argparse
import sys


def check_python_version() -> bool:
    v = sys.version_info
    ok = v.major == 3 and v.minor >= 9
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Python version: {v.major}.{v.minor}.{v.micro} (need >=3.9)")
    return ok


def check_import(name: str, package: str | None = None) -> bool:
    try:
        __import__(name)
        print(f"  [OK]   {package or name}")
        return True
    except ImportError as e:
        print(f"  [FAIL] {package or name}: {e}")
        return False


def check_config(config_path: str) -> bool:
    try:
        from src.config.schema import load_config
        config = load_config(config_path)
        print(f"  [OK]   Config loaded: {config.experiment.name}")
        print(f"         Method: {config.experiment.method}")
        print(f"         Tasks: {len(config.tasks)}")
        return True
    except Exception as e:
        print(f"  [FAIL] Config loading: {e}")
        return False


def check_dataset(config_path: str) -> bool:
    try:
        from src.config.schema import load_config
        from src.data.dataset import validate_task_data
        config = load_config(config_path)
        all_ok = True
        for task in config.tasks:
            try:
                validate_task_data(task)
                print(f"  [OK]   Dataset: {task.name}")
            except (FileNotFoundError, ValueError) as e:
                print(f"  [WARN] Dataset: {task.name}: {e}")
                all_ok = False
        return all_ok
    except Exception as e:
        print(f"  [FAIL] Dataset validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate DreamBooth-CLoRA setup")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config file to validate")
    args = parser.parse_args()

    print("=" * 50)
    print("DreamBooth-CLoRA Setup Validation")
    print("=" * 50)

    all_ok = True

    print("\n1. Python Version")
    all_ok &= check_python_version()

    print("\n2. Core Dependencies")
    for pkg in ["torch", "torchvision", "diffusers", "transformers",
                "accelerate", "peft", "safetensors"]:
        all_ok &= check_import(pkg)

    print("\n3. Evaluation Dependencies")
    all_ok &= check_import("sdeval", "sdeval (CCIP metrics)")
    all_ok &= check_import("PIL", "Pillow")
    all_ok &= check_import("yaml", "pyyaml")
    all_ok &= check_import("pandas")
    all_ok &= check_import("matplotlib")

    print("\n4. Optional Dependencies")
    check_import("bitsandbytes", "bitsandbytes (8-bit Adam)")
    check_import("xformers", "xformers (memory efficient attention)")

    print("\n5. Internal Modules")
    all_ok &= check_import("src.config.schema")
    all_ok &= check_import("src.data.dataset")
    all_ok &= check_import("src.training.trainer")
    all_ok &= check_import("src.methods.naive_sequential")
    all_ok &= check_import("src.methods.c_lora_scaffold")
    all_ok &= check_import("src.eval.generator")
    all_ok &= check_import("src.eval.metrics")
    all_ok &= check_import("src.eval.report")
    all_ok &= check_import("src.orchestrator.pipeline")

    if args.config:
        print(f"\n6. Config File: {args.config}")
        all_ok &= check_config(args.config)

        print(f"\n7. Dataset Validation")
        check_dataset(args.config)  # Warnings only, don't fail

    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed!")
    else:
        print("Some checks failed — see above for details.")
    print("=" * 50)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

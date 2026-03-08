#!/usr/bin/env bash
set -e

# cd DreamBooth-CLoRA

export PYTHONPATH="$PWD"

python src/main.py \
  --config configs/tasks_5char.yaml \
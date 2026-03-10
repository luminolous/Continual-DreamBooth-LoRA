#!/usr/bin/env bash
# set -e

# cd Continual-DreamBooth-LoRA

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python main.py \
  --config configs/tasks_5char_scaffold.yaml
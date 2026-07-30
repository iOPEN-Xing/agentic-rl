#!/bin/bash
set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"

cd "$PROJECT_ROOT"
mkdir -p experiments/turn_discount

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=turn_discount

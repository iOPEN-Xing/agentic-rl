#!/bin/bash
set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"

cd "$PROJECT_ROOT"
mkdir -p experiments/prm_lite

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=prm_lite 2>&1 | tee experiments/prm_lite/training.log

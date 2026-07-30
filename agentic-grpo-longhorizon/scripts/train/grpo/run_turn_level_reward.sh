#!/bin/bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"

cd "$PROJECT_ROOT"
mkdir -p experiments/turn_level_reward

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=turn_level_reward

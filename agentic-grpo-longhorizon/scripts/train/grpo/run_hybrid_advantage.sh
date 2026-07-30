#!/bin/bash
set -euo pipefail

source "$(dirname -- "$BASH_SOURCE")/common_env.sh"

cd "$PROJECT_ROOT"
mkdir -p experiments/hybrid_advantage

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=hybrid_advantage

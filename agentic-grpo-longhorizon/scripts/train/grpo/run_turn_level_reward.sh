#!/bin/bash
set -euo pipefail

source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"

export AGENTIC_RL_VANILLA_DATA_ROOT="${AGENTIC_RL_VANILLA_DATA_ROOT:-/data/xjz/agentic-grpo-longhorizon-main/agentic-grpo-longhorizon/experiments/vanilla}"
for DATA_FILE in train.parquet val.parquet; do
    if [[ ! -r "$AGENTIC_RL_VANILLA_DATA_ROOT/$DATA_FILE" ]]; then
        echo "ERROR: missing readable dataset: $AGENTIC_RL_VANILLA_DATA_ROOT/$DATA_FILE" >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
mkdir -p experiments/turn_ppo

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=turn_level_reward

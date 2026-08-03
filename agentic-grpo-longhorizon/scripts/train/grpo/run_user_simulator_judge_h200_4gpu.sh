#!/bin/bash
# Active bounded Judge shaping on 4x H200: GPU0 serves user+judge, GPU1-3 train.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/h200_4gpu_common.sh"

H200_METHOD_SLUG="user_simulator_judge"
H200_CONFIG_NAME="user_simulator_judge"
VANILLA_DATA_ROOT="${AGENTIC_RL_VANILLA_DATA_ROOT:-$H200_PROJECT_ROOT/experiments/vanilla}"
H200_METHOD_OVERRIDES=(
    "data.train_files=$VANILLA_DATA_ROOT/train.parquet"
    "data.val_files=$VANILLA_DATA_ROOT/val.parquet"
    "algorithm.judge_turn_reward.enabled=${AGENTIC_RL_ENABLE_JUDGE_REWARD:-true}"
    "algorithm.judge_turn_reward.turn_weight=${AGENTIC_RL_JUDGE_TURN_WEIGHT:-0.05}"
)

h200_prepare_method() {
    [[ -r "$VANILLA_DATA_ROOT/train.parquet" ]] || {
        echo "ERROR: missing train parquet: $VANILLA_DATA_ROOT/train.parquet" >&2
        return 1
    }
    [[ -r "$VANILLA_DATA_ROOT/val.parquet" ]] || {
        echo "ERROR: missing validation parquet: $VANILLA_DATA_ROOT/val.parquet" >&2
        return 1
    }
}

h200_run_training

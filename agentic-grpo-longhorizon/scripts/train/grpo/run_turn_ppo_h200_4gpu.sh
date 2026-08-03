#!/bin/bash
# Strict Turn-PPO on 4x H200 141GB: GPU0 user sim, GPU1-3 actor/ref/critic.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/h200_4gpu_common.sh"

H200_METHOD_SLUG="turn_ppo"
H200_CONFIG_NAME="turn_level_reward"
VANILLA_DATA_ROOT="${AGENTIC_RL_VANILLA_DATA_ROOT:-$H200_PROJECT_ROOT/experiments/vanilla}"
H200_METHOD_OVERRIDES=(
    "data.train_files=$VANILLA_DATA_ROOT/train.parquet"
    "data.val_files=$VANILLA_DATA_ROOT/val.parquet"
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

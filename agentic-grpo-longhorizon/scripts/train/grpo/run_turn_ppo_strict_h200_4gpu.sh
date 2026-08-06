#!/bin/bash
# Strict Turn-PPO on 4x H200 141GB. Mirrors run_turn_ppo_h200_4gpu.sh but targets the
# configs/train/grpo/turn_ppo_strict.yaml config and writes artifacts under
# experiments/h200_4gpu/turn_ppo_strict/<run-tag>.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/h200_4gpu_common.sh"

H200_METHOD_SLUG="turn_ppo_strict"
H200_CONFIG_NAME="turn_ppo_strict"
VANILLA_DATA_ROOT="${AGENTIC_RL_VANILLA_DATA_ROOT:-$H200_PROJECT_ROOT/experiments/vanilla}"
H200_METHOD_OVERRIDES=(
    "data.train_files=$VANILLA_DATA_ROOT/train.parquet"
    "data.val_files=$VANILLA_DATA_ROOT/val.parquet"
    "actor_rollout_ref.actor.policy_loss.loss_mode=turn_ppo"
    "algorithm.adv_estimator=turn_gae"
    "algorithm.gamma=0.99"
    "algorithm.lam=0.9"
    "algorithm.use_kl_in_reward=false"
    "algorithm.turn_level_reward.enabled=false"
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

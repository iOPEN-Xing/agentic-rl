#!/bin/bash
# Evaluate strict Turn-PPO checkpoints under the shared airline task/sampling contract.
# Mirrors scripts/eval/eval_turn_ppo.sh but targets configs/eval/turn_ppo_strict/.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_CHECKPOINT_ROOT="${TURN_PPO_STRICT_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/turn_ppo_strict}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    turn_ppo_strict eval_turn_ppo_strict turn_ppo_strict

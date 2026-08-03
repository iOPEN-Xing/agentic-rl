#!/bin/bash
# Evaluate Turn-PPO checkpoints under the shared airline task/sampling contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_CHECKPOINT_ROOT="${TURN_PPO_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/turn_ppo}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    turn_ppo eval_turn_ppo turn_ppo

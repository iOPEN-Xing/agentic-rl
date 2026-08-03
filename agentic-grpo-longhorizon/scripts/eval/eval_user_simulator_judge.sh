#!/bin/bash
# Evaluate User-Simulator/Judge checkpoints under the shared airline contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_CHECKPOINT_ROOT="${USER_JUDGE_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/user_simulator_judge}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    user_simulator_judge eval_user_simulator_judge user_simulator_judge

#!/bin/bash
# Evaluate legacy turn-discount checkpoints under the shared airline contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_EVAL_STEPS="${AGENTIC_RL_EVAL_STEPS:-200 250 300}"
export AGENTIC_RL_CHECKPOINT_ROOT="${TURN_DISCOUNT_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/turn_discount}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    tda eval_turn_discount turn_discount

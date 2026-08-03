#!/bin/bash
# Evaluate legacy LATA checkpoints under the shared airline contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_EVAL_STEPS="${AGENTIC_RL_EVAL_STEPS:-200 250 300}"
export AGENTIC_RL_CHECKPOINT_ROOT="${LATA_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/lata}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    lata eval_lata lata

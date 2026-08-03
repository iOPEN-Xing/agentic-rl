#!/bin/bash
# Evaluate TRACE-style Hybrid Advantage checkpoints under the shared airline contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_CHECKPOINT_ROOT="${HYBRID_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/hybrid_advantage}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    hybrid_advantage eval_hybrid_advantage hybrid_advantage

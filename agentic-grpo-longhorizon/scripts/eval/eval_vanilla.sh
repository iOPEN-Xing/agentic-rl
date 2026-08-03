#!/bin/bash
# Evaluate vanilla GRPO checkpoints under the shared airline task/sampling contract.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
export AGENTIC_RL_CHECKPOINT_ROOT="${VANILLA_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/vanilla}"
exec bash "$PROJECT_ROOT/scripts/eval/eval_checkpoint_series.sh" \
    vanilla_grpo eval_vanilla vanilla

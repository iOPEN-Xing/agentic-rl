#!/bin/bash
set -euo pipefail

source "$(dirname -- "$BASH_SOURCE")/common_env.sh"

cd "$PROJECT_ROOT"
mkdir -p experiments/hybrid_advantage

# TRACE needs a non-empty, training-only gold target for every task. Rebuild
# these small index files deterministically instead of reusing vanilla parquet
# files whose reward_model.ground_truth is intentionally empty.
python scripts/train/grpo/build_grpo_parquet.py \
    --seen-task-ids-from experiments/sft_collect_airline/split.json \
    --output-train experiments/hybrid_advantage/train.parquet \
    --output-val experiments/hybrid_advantage/val.parquet

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=hybrid_advantage

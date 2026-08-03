#!/bin/bash
# TRACE-style Hybrid Advantage on 4x H200: GPU0 user sim, GPU1-3 trainer/scorer.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/h200_4gpu_common.sh"

H200_METHOD_SLUG="hybrid_advantage"
H200_CONFIG_NAME="hybrid_advantage"
TRACE_SCORING_LENGTH="${AGENTIC_RL_TRACE_MAX_SCORING_LENGTH:-${AGENTIC_RL_POLICY_MAX_MODEL_LEN:-32768}}"
H200_METHOD_OVERRIDES=(
    "algorithm.hybrid_advantage.max_scoring_length=$TRACE_SCORING_LENGTH"
)

h200_prepare_method() {
    local split_file="${AGENTIC_RL_SPLIT_FILE:-$H200_PROJECT_ROOT/experiments/sft_collect_airline/split.json}"
    local data_dir="$H200_RUN_ROOT/data"
    [[ -r "$split_file" ]] || {
        echo "ERROR: missing split file: $split_file" >&2
        return 1
    }
    mkdir -p "$data_dir"
    python "$H200_PROJECT_ROOT/scripts/train/grpo/build_grpo_parquet.py" \
        --seen-task-ids-from "$split_file" \
        --output-train "$data_dir/train.parquet" \
        --output-val "$data_dir/val.parquet"
    H200_METHOD_OVERRIDES+=(
        "data.train_files=$data_dir/train.parquet"
        "data.val_files=$data_dir/val.parquet"
    )
}

h200_run_training

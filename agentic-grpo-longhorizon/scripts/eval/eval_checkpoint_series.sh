#!/bin/bash
# Serve and evaluate a checkpoint series under the repository-wide airline contract.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
    echo "Usage: $0 <config-dir> <config-prefix> <experiment-dir>" >&2
    exit 2
fi

CONFIG_DIR="$1"
CONFIG_PREFIX="$2"
EXPERIMENT_DIR="$3"
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/train/grpo/common_env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CHECKPOINT_ROOT="${AGENTIC_RL_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/$EXPERIMENT_DIR}"
SPLIT_FILE="${AGENTIC_RL_EVAL_SPLIT_FILE:-$PROJECT_ROOT/experiments/sft_collect_airline/split.json}"
EVAL_GPU="${AGENTIC_RL_EVAL_GPU:-0}"
# Space-separated override, e.g. AGENTIC_RL_EVAL_STEPS="200 250 300".
read -r -a STEPS <<< "${AGENTIC_RL_EVAL_STEPS:-50 100 150 200}"

if [[ "${#STEPS[@]}" -eq 0 ]]; then
    echo "ERROR: AGENTIC_RL_EVAL_STEPS resolved to an empty checkpoint list" >&2
    exit 2
fi

if [[ ! -r "$SPLIT_FILE" ]]; then
    echo "ERROR: missing readable evaluation split: $SPLIT_FILE" >&2
    exit 1
fi
if ! curl --fail --silent --show-error http://localhost:8001/v1/models >/dev/null; then
    echo "ERROR: fixed Qwen3-14B user simulator is not ready at http://localhost:8001/v1" >&2
    exit 1
fi
if curl --fail --silent http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "ERROR: port 8000 is already serving a model; refusing to replace it" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
for STEP in "${STEPS[@]}"; do
    MODEL_PATH="$CHECKPOINT_ROOT/hf_step_${STEP}"
    OUTPUT_DIR="$PROJECT_ROOT/experiments/$EXPERIMENT_DIR/eval_step_${STEP}"
    CONFIG="$PROJECT_ROOT/configs/eval/$CONFIG_DIR/${CONFIG_PREFIX}_step${STEP}.yaml"
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: missing exported Hugging Face checkpoint: $MODEL_PATH" >&2
        exit 1
    fi
    if [[ ! -r "$CONFIG" ]]; then
        echo "ERROR: missing evaluation config: $CONFIG" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "agentic-rl-policy" \
        --port 8000 \
        --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.82 \
        --max-model-len 16384 \
        --max-num-seqs 8 \
        --enable-prefix-caching \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        --trust-remote-code \
        > "$OUTPUT_DIR/vllm_server.log" 2>&1 &
    SERVER_PID=$!

    cleanup_server() {
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    }
    trap cleanup_server EXIT INT TERM

    READY=false
    for _ in {1..60}; do
        if curl --fail --silent http://localhost:8000/v1/models >/dev/null 2>&1; then
            READY=true
            break
        fi
        sleep 5
    done
    if [[ "$READY" != true ]]; then
        echo "ERROR: policy server did not become ready; see $OUTPUT_DIR/vllm_server.log" >&2
        exit 1
    fi

    python scripts/eval/eval_sft.py --config "$CONFIG" --split-file "$SPLIT_FILE"
    cleanup_server
    trap - EXIT INT TERM
done

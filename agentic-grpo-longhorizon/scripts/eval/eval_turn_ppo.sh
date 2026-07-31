#!/bin/bash
# Evaluate Turn-PPO checkpoints under the exact vanilla-GRPO task/sampling budget.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$PROJECT_ROOT/scripts/train/grpo/common_env.sh"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
TURN_PPO_CHECKPOINT_ROOT="${TURN_PPO_CHECKPOINT_ROOT:-$PROJECT_ROOT/experiments/turn_ppo}"
SPLIT_FILE="${AGENTIC_RL_EVAL_SPLIT_FILE:-/data/xjz/agentic-grpo-longhorizon-main/agentic-grpo-longhorizon/experiments/sft_collect_airline/split.json}"
STEPS=(50 100 150 200)

if [[ ! -r "$SPLIT_FILE" ]]; then
    echo "ERROR: missing readable evaluation split: $SPLIT_FILE" >&2
    exit 1
fi
if ! curl --fail --silent --show-error http://localhost:8001/v1/models >/dev/null; then
    echo "ERROR: τ-bench user simulator is not ready at http://localhost:8001/v1" >&2
    exit 1
fi
if curl --fail --silent http://localhost:8000/v1/models >/dev/null 2>&1; then
    echo "ERROR: port 8000 is already serving a model; refusing to replace it" >&2
    exit 1
fi

cd "$PROJECT_ROOT"
for STEP in "${STEPS[@]}"; do
    MODEL_PATH="$TURN_PPO_CHECKPOINT_ROOT/hf_step_${STEP}"
    OUTPUT_DIR="$PROJECT_ROOT/experiments/turn_ppo/eval_step_${STEP}"
    CONFIG="$PROJECT_ROOT/configs/eval/turn_ppo/eval_turn_ppo_step${STEP}.yaml"
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: missing exported Hugging Face checkpoint: $MODEL_PATH" >&2
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"
    CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --served-model-name "Qwen/Qwen2.5-7B-Instruct" \
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
    trap cleanup_server EXIT

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
    trap - EXIT
done

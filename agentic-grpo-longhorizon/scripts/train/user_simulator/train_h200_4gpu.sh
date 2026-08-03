#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${AGENTIC_RL_USER_SIM_SFT_CONFIG:-$PROJECT_ROOT/configs/train/sft/sft_user_simulator_qwen3_14b_lora.yaml}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_ENTITY="${WANDB_ENTITY:-jiezhengxing-aaaa}"
export WANDB_PROJECT="${WANDB_PROJECT:-agentic-grpo-longhorizon}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "$PROJECT_ROOT"
exec torchrun \
  --standalone \
  --nproc_per_node=4 \
  scripts/train/sft/sft_train.py \
  --config "$CONFIG"

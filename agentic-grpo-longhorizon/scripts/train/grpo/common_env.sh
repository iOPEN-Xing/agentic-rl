#!/bin/bash

# Shared runtime for all four-GPU GRPO experiments on this server.
source /data/xjz/miniconda3/etc/profile.d/conda.sh
conda activate cu12.8

export CUDA_HOME=/usr/local/cuda-12.8
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
export CUDA_VISIBLE_DEVICES=0,1,4,5
export DS_SKIP_TRITON=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_V1=1
export HF_ENDPOINT=https://hf-mirror.com
export OPENAI_API_KEY=dummy
export LITELLM_LOCAL_MODEL_COST_MAP="True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_LOGGING_LEVEL=ERROR
export WANDB_MODE=online
export WANDB_PROJECT=agentic-grpo-longhorizon

# Resolve the project from this file so launch scripts work from any directory.
GRPO_SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"
PROJECT_ROOT="$(cd -- "$GRPO_SCRIPT_DIR/../../.." && pwd)"
BUNDLE_ROOT="$(cd -- "$PROJECT_ROOT/.." && pwd)"
export PYTHONPATH="$BUNDLE_ROOT/verl:$BUNDLE_ROOT/tau-bench:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PROJECT_ROOT
export BUNDLE_ROOT

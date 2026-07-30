#!/bin/bash
set -e

source "$(dirname -- "${BASH_SOURCE[0]}")/common_env.sh"

# expandable_segments disabled: incompatible with vLLM memory pool
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

cd "$PROJECT_ROOT"
mkdir -p experiments/vanilla

# nohup python -m verl.trainer.main_ppo \
#     --config-path=$(pwd)/configs \
#     --config-name=vanilla_grpo \
#     > experiments/vanilla/training.log 2>&1 &
# echo "Training PID: $!"

python -m verl.trainer.main_ppo \
    --config-path="$PROJECT_ROOT/configs/train/grpo" \
    --config-name=vanilla_grpo
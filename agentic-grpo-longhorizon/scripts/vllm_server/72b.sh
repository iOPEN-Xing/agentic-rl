#!/bin/bash
# Qwen3-14B user simulator. The historical filename is retained for
# compatibility with existing launch documentation.

set -euo pipefail


source /data/xjz/miniconda3/etc/profile.d/conda.sh
conda activate cu12.8

MODEL_PATH=${MODEL_PATH:-"/data/xjz/model/qwen3-14b"}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"/data/xjz/model/qwen3-14b"}
PORT=${PORT:-8001}
TP_SIZE=${TP_SIZE:-2}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
CUDA_DEVICES=${CUDA_DEVICES:-2,3}

export CUDA_HOME=/usr/local/cuda-12.8
export TRITON_PTXAS_PATH=/usr/local/cuda-12.8/bin/ptxas
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_V1=1

echo "Starting Qwen3-14B user simulator..."
echo "Model:     $MODEL_PATH"
echo "Served as: $SERVED_MODEL_NAME"
echo "Port:      $PORT"
echo "TP:        $TP_SIZE"
echo "GPUs:      $CUDA_DEVICES"
echo "MaxLen:    $MAX_MODEL_LEN"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --trust-remote-code

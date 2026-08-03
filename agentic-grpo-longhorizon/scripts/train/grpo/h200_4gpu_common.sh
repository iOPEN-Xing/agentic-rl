#!/bin/bash
# Shared one-node H200 launcher. Source this file from one method wrapper.

H200_GRPO_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
H200_PROJECT_ROOT="$(cd -- "$H200_GRPO_SCRIPT_DIR/../../.." && pwd)"
H200_BUNDLE_ROOT="$(cd -- "$H200_PROJECT_ROOT/.." && pwd)"

_h200_die() {
    echo "ERROR: $*" >&2
    return 1
}

_h200_validate_gpu_allocation() {
    local train_devices="$1"
    local user_device="$2"
    local -a train_gpu_list

    [[ "$train_devices" =~ ^[0-9]+(,[0-9]+){2}$ ]] || {
        _h200_die "TRAIN_CUDA_DEVICES must contain exactly three GPU indices"
        return 1
    }
    [[ "$user_device" =~ ^[0-9]+$ ]] || {
        _h200_die "USER_SIM_CUDA_DEVICES must contain exactly one GPU index"
        return 1
    }
    IFS=',' read -r -a train_gpu_list <<< "$train_devices"
    local left right
    for ((left = 0; left < ${#train_gpu_list[@]}; left++)); do
        for ((right = left + 1; right < ${#train_gpu_list[@]}; right++)); do
            [[ "${train_gpu_list[$left]}" != "${train_gpu_list[$right]}" ]] || {
                _h200_die "duplicate trainer GPU index: ${train_gpu_list[$left]}"
                return 1
            }
        done
    done
    for gpu in "${train_gpu_list[@]}"; do
        [[ "$gpu" != "$user_device" ]] || {
            _h200_die "user simulator and trainer GPU sets overlap at GPU $gpu"
            return 1
        }
    done
}

_h200_validate_context_budget() {
    local prompt_length="$1"
    local response_length="$2"
    local model_length="$3"

    [[ "$prompt_length" =~ ^[1-9][0-9]*$ ]] || {
        _h200_die "max prompt length must be a positive integer"
        return 1
    }
    [[ "$response_length" =~ ^[1-9][0-9]*$ ]] || {
        _h200_die "max response length must be a positive integer"
        return 1
    }
    [[ "$model_length" =~ ^[1-9][0-9]*$ ]] || {
        _h200_die "policy max model length must be a positive integer"
        return 1
    }
    ((prompt_length + response_length <= model_length)) || {
        _h200_die \
            "prompt/response context budget ($prompt_length + $response_length) exceeds model context $model_length"
        return 1
    }
}

_h200_user_simulator_matches() {
    local endpoint="$1"
    local expected_model="$2"
    local payload

    payload="$(curl --fail --silent --show-error "$endpoint/v1/models")" || return 1
    python -c '
import json, sys
payload = json.load(sys.stdin)
model_ids = {str(item.get("id", "")) for item in payload.get("data", [])}
raise SystemExit(0 if sys.argv[1] in model_ids else 1)
' "$expected_model" <<< "$payload"
}

_h200_cleanup_user_simulator() {
    if [[ -n "${H200_USER_SIM_PID:-}" && "${AGENTIC_RL_KEEP_USER_SIMULATOR:-0}" != "1" ]]; then
        kill "$H200_USER_SIM_PID" 2>/dev/null || true
        wait "$H200_USER_SIM_PID" 2>/dev/null || true
        H200_USER_SIM_PID=""
    fi
}

_h200_start_user_simulator() {
    local user_device="$1"
    local user_model="$2"
    local user_endpoint="$3"
    local user_port="$4"
    local user_log="$5"

    if curl --fail --silent "$user_endpoint/v1/models" >/dev/null 2>&1; then
        _h200_user_simulator_matches "$user_endpoint" "$user_model" || {
            _h200_die "port $user_port is serving a different model"
            return 1
        }
        [[ "${AGENTIC_RL_REUSE_USER_SIMULATOR:-0}" == "1" ]] || {
            _h200_die \
                "a compatible service already uses port $user_port, but its GPU placement is unknown; " \
                "stop it or set AGENTIC_RL_REUSE_USER_SIMULATOR=1 after verifying it uses only GPU $user_device"
            return 1
        }
        echo "Reusing compatible user simulator at $user_endpoint"
        return 0
    fi

    echo "Starting Qwen3-14B user simulator on physical GPU $user_device"
    CUDA_DEVICES="$user_device" \
    MODEL_PATH="$user_model" \
    SERVED_MODEL_NAME="$user_model" \
    PORT="$user_port" \
    TP_SIZE=1 \
    GPU_MEM_UTIL="${AGENTIC_RL_USER_GPU_MEMORY_UTILIZATION:-0.85}" \
    MAX_MODEL_LEN="${AGENTIC_RL_USER_MAX_MODEL_LEN:-32768}" \
    MAX_NUM_SEQS="${AGENTIC_RL_USER_MAX_NUM_SEQS:-24}" \
        bash "$H200_PROJECT_ROOT/scripts/vllm_server/72b.sh" >>"$user_log" 2>&1 &
    H200_USER_SIM_PID=$!

    local attempt
    for attempt in {1..120}; do
        if _h200_user_simulator_matches "$user_endpoint" "$user_model"; then
            echo "User simulator is ready (PID $H200_USER_SIM_PID)"
            return 0
        fi
        if ! kill -0 "$H200_USER_SIM_PID" 2>/dev/null; then
            _h200_die "user simulator exited early; inspect $user_log"
            return 1
        fi
        sleep 5
    done
    _h200_die "user simulator did not become ready; inspect $user_log"
}

h200_run_training() {
    : "${H200_METHOD_SLUG:?wrapper must set H200_METHOD_SLUG}"
    : "${H200_CONFIG_NAME:?wrapper must set H200_CONFIG_NAME}"

    local train_devices="${TRAIN_CUDA_DEVICES:-1,2,3}"
    local user_device="${USER_SIM_CUDA_DEVICES:-0}"
    local policy_model="${AGENTIC_RL_POLICY_MODEL:-/data/xjz/model/qwen3-8b}"
    local user_model="${AGENTIC_RL_USER_SIM_MODEL:-/data/xjz/model/qwen3-14b}"
    local user_port="${AGENTIC_RL_USER_SIM_PORT:-8001}"
    local user_endpoint="http://localhost:$user_port"
    local run_tag="${AGENTIC_RL_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
    local run_name="${H200_METHOD_SLUG}_h200_4gpu_${run_tag}"
    local wandb_entity="${WANDB_ENTITY:-jiezhengxing-aaaa}"
    local wandb_project="${WANDB_PROJECT:-agentic-grpo-longhorizon}"
    local max_prompt_length="${AGENTIC_RL_MAX_PROMPT_LENGTH:-12288}"
    local max_response_length="${AGENTIC_RL_MAX_RESPONSE_LENGTH:-16384}"
    local policy_max_model_len="${AGENTIC_RL_POLICY_MAX_MODEL_LEN:-32768}"
    local reuse_user_simulator="${AGENTIC_RL_REUSE_USER_SIMULATOR:-0}"
    local -a common_overrides method_overrides command

    [[ "$run_tag" =~ ^[A-Za-z0-9._-]+$ ]] || {
        _h200_die "AGENTIC_RL_RUN_TAG may contain only letters, numbers, dot, underscore and dash"
        return 1
    }
    _h200_validate_gpu_allocation "$train_devices" "$user_device" || return 1
    _h200_validate_context_budget \
        "$max_prompt_length" "$max_response_length" "$policy_max_model_len" || return 1
    [[ "$reuse_user_simulator" =~ ^[01]$ ]] || {
        _h200_die "AGENTIC_RL_REUSE_USER_SIMULATOR must be 0 or 1"
        return 1
    }

    H200_RUN_ROOT="${AGENTIC_RL_RUN_ROOT:-$H200_PROJECT_ROOT/experiments/h200_4gpu/$H200_METHOD_SLUG/$run_tag}"
    export H200_RUN_ROOT
    common_overrides=(
        "data.max_prompt_length=$max_prompt_length"
        "data.max_response_length=$max_response_length"
        "actor_rollout_ref.model.path=$policy_model"
        "actor_rollout_ref.ref.model.path=$policy_model"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${AGENTIC_RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
        "actor_rollout_ref.rollout.max_model_len=$policy_max_model_len"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${AGENTIC_RL_MAX_BATCHED_TOKENS:-32768}"
        "actor_rollout_ref.rollout.max_num_seqs=${AGENTIC_RL_ROLLOUT_MAX_NUM_SEQS:-16}"
        "trainer.n_gpus_per_node=3"
        "trainer.nnodes=1"
        "trainer.project_name=$wandb_project"
        "trainer.experiment_name=$run_name"
        "trainer.default_local_dir=$H200_RUN_ROOT/checkpoints"
        "trainer.resume_mode=auto"
    )
    method_overrides=("${H200_METHOD_OVERRIDES[@]:-}")

    if [[ "${AGENTIC_RL_DRY_RUN:-0}" == "1" ]]; then
        echo "METHOD=$H200_METHOD_SLUG"
        echo "CONFIG_NAME=$H200_CONFIG_NAME"
        echo "USER_SIM_CUDA_DEVICES=$user_device"
        echo "REUSE_USER_SIMULATOR=$reuse_user_simulator"
        echo "TRAIN_CUDA_DEVICES=$train_devices"
        echo "TRAIN_GPU_COUNT=3"
        echo "POLICY_MODEL=$policy_model"
        echo "USER_SIM_MODEL=$user_model"
        echo "WANDB_ENTITY=$wandb_entity"
        echo "WANDB_PROJECT=$wandb_project"
        echo "RUN_ROOT=$H200_RUN_ROOT"
        printf 'OVERRIDE=%s\n' "${common_overrides[@]}" "${method_overrides[@]}"
        return 0
    fi

    source "$H200_GRPO_SCRIPT_DIR/common_env.sh"
    [[ "$PROJECT_ROOT" == "$H200_PROJECT_ROOT" ]] || {
        _h200_die "common_env.sh resolved an unexpected project root: $PROJECT_ROOT"
        return 1
    }
    export CUDA_VISIBLE_DEVICES="$train_devices"
    export WANDB_MODE=online
    export WANDB_ENTITY="$wandb_entity"
    export WANDB_PROJECT="$wandb_project"
    export WANDB_NAME="$run_name"
    export WANDB_RUN_ID="${AGENTIC_RL_WANDB_RUN_ID:-$run_name}"
    export WANDB_RESUME=allow
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-h200_4gpu_agentic_rl}"
    export WANDB_TAGS="${WANDB_TAGS:-h200-141gb,4gpu,$H200_METHOD_SLUG}"
    export WANDB_DIR="$H200_RUN_ROOT/wandb"

    [[ -d "$policy_model" ]] || {
        _h200_die "missing policy model: $policy_model"
        return 1
    }
    [[ -d "$user_model" ]] || {
        _h200_die "missing user simulator model: $user_model"
        return 1
    }
    mkdir -p "$H200_RUN_ROOT/checkpoints" "$WANDB_DIR"
    if declare -F h200_prepare_method >/dev/null; then
        h200_prepare_method
        method_overrides=("${H200_METHOD_OVERRIDES[@]:-}")
    fi

    printf '%s\n' \
        "method=$H200_METHOD_SLUG" \
        "config=$H200_CONFIG_NAME" \
        "run_name=$run_name" \
        "run_root=$H200_RUN_ROOT" \
        "policy_model=$policy_model" \
        "user_simulator_model=$user_model" \
        "user_simulator_gpu=$user_device" \
        "trainer_gpus=$train_devices" \
        "wandb_url=https://wandb.ai/$wandb_entity/$wandb_project" \
        "git_commit=$(git -C "$H200_BUNDLE_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)" \
        >"$H200_RUN_ROOT/launch_manifest.env"

    H200_USER_SIM_PID=""
    trap _h200_cleanup_user_simulator EXIT
    trap '_h200_cleanup_user_simulator; exit 130' INT
    trap '_h200_cleanup_user_simulator; exit 143' TERM
    _h200_start_user_simulator \
        "$user_device" "$user_model" "$user_endpoint" "$user_port" \
        "$H200_RUN_ROOT/user_simulator.log"

    command=(
        python -m verl.trainer.main_ppo
        "--config-path=$H200_PROJECT_ROOT/configs/train/grpo"
        "--config-name=$H200_CONFIG_NAME"
        "${common_overrides[@]}"
        "${method_overrides[@]}"
    )
    printf '%q ' "${command[@]}" >"$H200_RUN_ROOT/resolved_command.sh"
    printf '\n' >>"$H200_RUN_ROOT/resolved_command.sh"

    echo "Training GPUs: $train_devices; user simulator GPU: $user_device"
    echo "Results:       $H200_RUN_ROOT"
    echo "W&B:           https://wandb.ai/$wandb_entity/$wandb_project"
    cd "$H200_PROJECT_ROOT"
    set +e
    "${command[@]}" 2>&1 | tee -a "$H200_RUN_ROOT/train.log"
    local training_status=${PIPESTATUS[0]}
    set -e
    return "$training_status"
}

#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  serve.sh <baseline|lora> <eager|graph> [NPU_ID ... | all]
  Explicit IDs set TP to their count; all uses every NPU. Default TP is 1.
  Without IDs, all NPUs are visible. This script does not clean up processes.

Required environment:
  MODEL_PATH       BF16, unquantized Qwen3-30B-A3B checkpoint.

Required for the lora server:
  ADAPTER_A_PATH   First true per-expert MoE adapter.
  ADAPTER_B_PATH   Second true per-expert MoE adapter.

Optional environment:
  SGLANG_DIR          SGLang checkout (default: <repo>/sglang/ascend_moe_lora)
  SERVER_HOST         Listen address (default: 127.0.0.1)
  SERVER_PORT         Listen port (default: VLLM_PORT or 6969)
  CONTEXT_LENGTH      Maximum context length (default: 4096)
  MEM_FRACTION_STATIC Static memory fraction (default: 0.80)
  GRAPH_BATCH_SIZE    Decode NPU Graph bucket (default: 3)
  SGLANG_EXTRA_ARGS   Extra launch arguments, split on shell whitespace.

Examples:
  MODEL_PATH=/weights/Qwen3-30B-A3B \
    ./llm/ascend_moe_lora/serve.sh baseline eager 0 1

  MODEL_PATH=/weights/Qwen3-30B-A3B \
  ADAPTER_A_PATH=/weights/lora-a ADAPTER_B_PATH=/weights/lora-b \
    ./llm/ascend_moe_lora/serve.sh lora graph 0 1
EOF
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

server_kind=$1
execution_mode=$2
shift 2

case "$server_kind" in
    baseline|lora) ;;
    *)
        echo "Unknown server kind: $server_kind" >&2
        usage >&2
        exit 2
        ;;
esac

case "$execution_mode" in
    eager|graph) ;;
    *)
        echo "Unknown execution mode: $execution_mode" >&2
        usage >&2
        exit 2
        ;;
esac

: "${MODEL_PATH:?Set MODEL_PATH to the BF16 Qwen3-30B-A3B checkpoint}"
if [[ "$server_kind" == lora ]]; then
    : "${ADAPTER_A_PATH:?Set ADAPTER_A_PATH to a true per-expert adapter}"
    : "${ADAPTER_B_PATH:?Set ADAPTER_B_PATH to a true per-expert adapter}"
fi

script_dir=$(dirname "$(readlink -f "$0")")
repo_root=$(dirname "$(dirname "$script_dir")")
sglang_dir=${SGLANG_DIR:-${repo_root}/sglang/ascend_moe_lora}
if [[ ! -d "$sglang_dir/python/sglang" ]]; then
    echo "SGLang checkout not found: $sglang_dir" >&2
    echo "Set SGLANG_DIR to the junlin-ascend-moe-lora checkout." >&2
    exit 2
fi

export PYTHONPATH="${sglang_dir}/python${PYTHONPATH:+:${PYTHONPATH}}"

source "$script_dir/../common/npu-launch.sh"
npu_resolve_devices 1 "$@"

host=${SERVER_HOST:-127.0.0.1}
port=${SERVER_PORT:-${VLLM_PORT:-6969}}
context_length=${CONTEXT_LENGTH:-4096}
mem_fraction_static=${MEM_FRACTION_STATIC:-0.80}
graph_batch_size=${GRAPH_BATCH_SIZE:-3}

cmd=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name qwen3-moe-lora
    --host "$host"
    --port "$port"
    --device npu
    --dtype bfloat16
    --tp-size "$NPU_TP_SIZE"
    --context-length "$context_length"
    --mem-fraction-static "$mem_fraction_static"
    --moe-a2a-backend none
    --moe-runner-backend auto
    --trust-remote-code
)

if [[ "$execution_mode" == eager ]]; then
    cmd+=(
        --cuda-graph-backend-decode disabled
        --cuda-graph-backend-prefill disabled
    )
else
    cmd+=(
        --cuda-graph-backend-decode full
        --cuda-graph-bs-decode "$graph_batch_size"
        --cuda-graph-backend-prefill disabled
    )
fi

if [[ "$server_kind" == lora ]]; then
    cmd+=(
        --enable-lora
        --lora-backend ascend
        --lora-paths
        "adapter_a=${ADAPTER_A_PATH}"
        "adapter_b=${ADAPTER_B_PATH}"
        --max-loras-per-batch 3
        --max-loaded-loras 3
    )
fi

if [[ -n "${SGLANG_EXTRA_ARGS:-}" ]]; then
    read -r -a extra_args <<<"$SGLANG_EXTRA_ARGS"
    cmd+=("${extra_args[@]}")
fi

echo "server_kind=$server_kind"
echo "execution_mode=$execution_mode"
echo "model_path=$MODEL_PATH"
echo "tp_size=$NPU_TP_SIZE"
echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "endpoint=http://${host}:${port}"
printf 'command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"

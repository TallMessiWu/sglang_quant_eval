#!/bin/bash
set -euo pipefail

# 在线 W4A8：必须使用原始 BF16/FP16 checkpoint，不可使用离线 packed 权重。
if [ -z "${MODEL_PATH:-}" ]; then
    echo "错误：请设置 MODEL_PATH 为原始 BF16/FP16 权重目录。" >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误：原始权重目录不存在：$MODEL_PATH" >&2
    exit 1
fi
if [ -f "$MODEL_PATH/quant_model_description.json" ]; then
    echo "错误：$MODEL_PATH 是 ModelSlim checkpoint，请使用 offline 脚本，避免重复量化/打包。" >&2
    exit 1
fi

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch "$@" || exit $?

# A5 prefill/decode 使用 FIA，避免默认 ATB 路径。
export ASCEND_USE_FIA=1

EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi

exec sglang serve \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT:-6969}" \
    --quantization mxfp_w4a8 \
    --device npu \
    --tp "${TP_SIZE:-1}" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code \
    "${EXTRA_ARGS_ARR[@]}"

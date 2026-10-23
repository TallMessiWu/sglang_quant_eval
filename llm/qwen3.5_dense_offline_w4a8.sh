#!/bin/bash
set -euo pipefail

# 离线 ModelSlim：已打包的 W4A8 checkpoint，不再在线量化。
MODEL_PATH="${MODEL_PATH:-/mnt/share/weight/Qwen3.6-27B_mxfp-w4a8-0821}"
if [ ! -f "$MODEL_PATH/quant_model_description.json" ]; then
    echo "错误：离线 ModelSlim 权重缺少 $MODEL_PATH/quant_model_description.json" >&2
    exit 1
fi

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

# A5 prefill/decode 使用 FIA，避免默认 ATB 路径。
export ASCEND_USE_FIA=1
# NPU GDN/SSM state 默认 BF16，与权重量化格式无关；允许显式覆盖。
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

# 与既有脚本一致，EXTRA_ARGS 按空白分词，例如 --disable-cuda-graph。
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi

# quant_model_description.json 自动选择 ModelSlim；不传 --quantization。
exec sglang serve \
    --model-path "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port "${VLLM_PORT:-6969}" \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code \
    "${EXTRA_ARGS_ARR[@]}"

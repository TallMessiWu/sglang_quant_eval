#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

export ASCEND_USE_FIA=1
# NPU GDN/SSM state 默认 BF16，与权重量化格式无关；允许显式覆盖。
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

# Offline W4A4 MXFP4: msmodelslim checkpoint exported from
# qwen_mxfp_quant/qwen3_5_35b_moe_w4a4_mxfp4.yaml (mixed precision — experts
# W4A4 MXFP4, attention / shared MLP / vision tower W8A8 MXFP8). The scheme is
# auto-detected from quant_model_description.json, so no --quantization flag.
sglang serve \
    --model-path /mnt/weight/Qwen3.5-35B-A3B-mxfp-w4a4 \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

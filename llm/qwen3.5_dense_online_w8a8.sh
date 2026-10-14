#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch "$@" || exit $?

export ASCEND_USE_FIA=1

sglang serve \
    --model-path /mnt/share/weight/Qwen3.6-27B \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --quantization mxfp8 \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

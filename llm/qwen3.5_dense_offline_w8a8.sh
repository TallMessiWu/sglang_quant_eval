#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

export ASCEND_USE_FIA=1

sglang serve \
    --model-path /mnt/weight/Qwen3.5-27B-mxw8a8 \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

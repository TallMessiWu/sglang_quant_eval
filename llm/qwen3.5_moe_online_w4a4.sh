#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch "$@" || exit $?

export ASCEND_USE_FIA=1

# Online W4A4 MXFP4: BF16 weights, experts quantised to single-level MXFP4 at
# load time (non-expert Linear stays dual-level MXFP4). Requires Ascend 950 (A5).
sglang serve \
    --model-path /mnt/share/weight/Qwen3.5-35B-A3B \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --quantization mxfp4 \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

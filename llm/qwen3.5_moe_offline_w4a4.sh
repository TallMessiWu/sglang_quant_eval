#!/bin/bash

if [ $# -eq 0 ]; then
    unset ASCEND_RT_VISIBLE_DEVICES
else
    OLD_IFS="$IFS"
    IFS=","; export ASCEND_RT_VISIBLE_DEVICES="$*"; IFS="$OLD_IFS"
fi

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
if [ -x "$SCRIPT_DIR/npu-cleaner.sh" ]; then
    "$SCRIPT_DIR/npu-cleaner.sh" "$@"
    sleep 1
fi

export ASCEND_USE_FIA=1

# Offline W4A4 MXFP4: msmodelslim checkpoint exported from
# qwen_mxfp_quant/qwen3_5_35b_moe_w4a4_mxfp4.yaml (mixed precision — experts
# W4A4 MXFP4, attention / shared MLP / vision tower W8A8 MXFP8). The scheme is
# auto-detected from quant_model_description.json, so no --quantization flag.
sglang serve \
    --model-path /mnt/weight/Qwen3.5-35B-A3B-mxfp-w4a4 \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code

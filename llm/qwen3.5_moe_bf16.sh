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

sglang serve \
    --model-path /mnt/share/weight/Qwen3.5-35B-A3B \
    --host 127.0.0.1 \
    --port $VLLM_PORT \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code

#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/../llm/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

echo "离线量化mxfp4"
sglang generate --model-path /home/weights/Wan2.2-T2V-A14B-Diffusers-MXFP4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus "$NPU_TP_SIZE" --num-frames 81 --num-inference-steps 40  --warmup
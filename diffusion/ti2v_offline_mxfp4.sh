#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/../llm/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

sglang generate --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers-mxfp4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus "$NPU_TP_SIZE" --num-frames 81 --num-inference-steps 40  --warmup

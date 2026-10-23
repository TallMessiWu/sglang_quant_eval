#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/../llm/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.4
SGLANG_CACHE_DIT_MC=4
SGLANG_CACHE_DIT_TAYLORSEER=true
SGLANG_CACHE_DIT_TS_ORDER=2
SGLANG_CACHE_DIT_ENABLED=true
sglang generate --model-path /home/weights/Wan2.2-TI2V-5B-Diffusers \
--quantization mxfp4_npu \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus "$NPU_TP_SIZE" --num-frames 81 --num-inference-steps 40  --warmup

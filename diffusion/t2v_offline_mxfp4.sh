#!/bin/bash
echo "离线量化mxfp4"
SGLANG_CACHE_DIT_FN=2
SGLANG_CACHE_DIT_BN=1
SGLANG_CACHE_DIT_WARMUP=4
SGLANG_CACHE_DIT_RDT=0.4
SGLANG_CACHE_DIT_MC=4
SGLANG_CACHE_DIT_TAYLORSEER=true
SGLANG_CACHE_DIT_TS_ORDER=2
SGLANG_CACHE_DIT_ENABLED=true
sglang generate --model-path /home/weights/Wan2.2-T2V-A14B-Diffusers-MXFP4 \
--prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
--height 704 --width 1280 --num-gpus 1 --num-frames 81 --num-inference-steps 40  --warmup
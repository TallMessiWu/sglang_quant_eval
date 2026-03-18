#!/bin/bash
# Wan2.2 TI2V W8A8 INT8 量化推理 (sglang generate CLI)

MODEL_PATH=${MODEL_PATH:-"/home/weights/Wan2.2-TI2V-5B-Diffusers"}
QUANT_MODEL_PATH=${QUANT_MODEL_PATH:-"/home/weights/Wan2.2-TI2V-5B-w8a8c8-self-attn-bf16-rot"}
IMAGE_PATH=${IMAGE_PATH:-"gyro.jpg"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs_w8a8"}
NUM_GPUS=${NUM_GPUS:-1}

export USE_NZ=${USE_NZ:-0}

PROMPT="杰作，最高画质，8K，超高细节，官方原画，荒木飞吕彦画风，JOJO的奇妙冒险画风，\
单人男性，杰洛·齐贝林，JOJO的奇妙冒险第七部飙马野郎，帅气男性，银色长发，\
紫色眼眸，绿色嘴唇，标志性宽边牛仔帽，帽子带有紫色铁球装饰，双手扶着帽檐，\
紫蓝色骑马制服，胸前银色蜻蜓胸针，帽子和衣服上覆盖积雪与霜冻，户外雨夹雪场景，\
下落的雨滴与动态雨丝，模糊的绿色自然背景，柔和电影级打光，冷色调，厚涂质感，\
自然的微动态，缓慢眨眼，呼吸带来的胸腔轻微起伏，头发和衣服被风轻轻吹动，\
雨滴动态下落，镜头缓慢轻微推近，动作丝滑，画面稳定无抖动，24帧"

sglang generate \
    --model-path "$MODEL_PATH" \
    --num-gpus "$NUM_GPUS" \
    --output-path "$OUTPUT_DIR" \
    --image-path "$IMAGE_PATH" \
    --num-frames 81 \
    --height 704 \
    --width 1280 \
    --num-inference-steps 50 \
    --guidance-scale 5.0 \
    --fps 24 \
    --seed 42 \
    --prompt "$PROMPT" \
    --transformer-path "$QUANT_MODEL_PATH"

python -m sglang.launch_server \
    --model-path /home/weights/Qwen3-8B-mxfp8 \
    --quantization modelslim \
    --device npu \
    --tp 1

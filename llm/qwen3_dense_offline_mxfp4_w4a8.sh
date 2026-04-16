sglang serve \
    --model-path /home/weights/Qwen3-8B-w4a8-full \
    --host 0.0.0.0 \
    --port 6969 \
    --quantization modelslim \
    --device npu \
    --tp 1 \
    --trust-remote-code

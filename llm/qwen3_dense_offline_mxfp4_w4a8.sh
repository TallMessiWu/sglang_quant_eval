sglang serve \
    --model-path /home/weights/qwen3-8B-dense-w4a8-040202 \
    --host 0.0.0.0 \
    --port 6969 \
    --quantization modelslim \
    --device npu \
    --tp 1 \
    --trust-remote-code

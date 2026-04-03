sglang serve \
    --model-path /home/weights/Qwen3-8B \
    --host 0.0.0.0 \
    --port 6969 \
    --quantization mxfp8 \
    --device npu \
    --tp 1 \
    --trust-remote-code

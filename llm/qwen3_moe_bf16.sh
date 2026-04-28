sglang serve \
    --model-path /home/weights/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 6969 \
    --device npu \
    --tp 8 \
    --trust-remote-code

sglang serve \
    --model-path /home/weights/qwen3-8B-dense-w4a4-041302 \
    --host 0.0.0.0 \
    --port 6969 \
    --quantization modelslim \
    --device npu \
    --tp 1 \
    --trust-remote-code

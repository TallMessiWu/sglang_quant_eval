sglang serve \
    --model-path /home/weights/Qwen3-30B-A3B \
    --host 0.0.0.0 \
    --port 6969 \
    --quantization mxfp8 \
    --device npu \
    --tp 1 \
    --trust-remote-code \
    --disable-cuda-graph

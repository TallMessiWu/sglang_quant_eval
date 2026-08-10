#!/usr/bin/env bash

set -euo pipefail

host=${SERVER_HOST:-127.0.0.1}
port=${SERVER_PORT:-${VLLM_PORT:-30000}}
base_url="http://${host}:${port}"
num_prefill_prompts=${NUM_PREFILL_PROMPTS:-4}
prefill_input_len=${PREFILL_INPUT_LEN:-4096}
decode_output_len=${DECODE_OUTPUT_LEN:-2048}
first_token_timeout=${FIRST_TOKEN_TIMEOUT:-120}

debug_dir=$(mktemp -d /tmp/a5-fia-mixed-debug.XXXXXX)
decode_stream="${debug_dir}/decode-stream.out"
decode_pid=""

cleanup() {
    if [[ -n "$decode_pid" ]] && kill -0 "$decode_pid" 2>/dev/null; then
        kill "$decode_pid" 2>/dev/null || true
        wait "$decode_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "Starting one short-prefill, long-decode request..."
curl --fail --silent --show-error --no-buffer \
    --request POST \
    --header 'Content-Type: application/json' \
    --data-binary "{\"text\":\"Hello from the mixed FIA debug workload.\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":${decode_output_len},\"ignore_eos\":true},\"stream\":true}" \
    "${base_url}/generate" >"$decode_stream" &
decode_pid=$!

deadline=$((SECONDS + first_token_timeout))
while [[ ! -s "$decode_stream" ]]; do
    if ! kill -0 "$decode_pid" 2>/dev/null; then
        wait "$decode_pid"
        echo "Decode request exited before producing its first streamed token." >&2
        exit 1
    fi
    if ((SECONDS >= deadline)); then
        echo "Timed out waiting for the decode request's first streamed token." >&2
        exit 1
    fi
    sleep 0.1
done

echo "Decode is active; injecting long prefill requests now."
python3 -m sglang.benchmark.serving \
    --backend sglang \
    --base-url "$base_url" \
    --dataset-name random \
    --num-prompts "$num_prefill_prompts" \
    --random-input-len "$prefill_input_len" \
    --random-output-len 1 \
    --random-range-ratio 1 \
    --request-rate inf \
    --max-concurrency "$num_prefill_prompts" \
    --tokenize-prompt \
    --seed 0 \
    --disable-tqdm

echo "Mixed workload completed. Inspect the server log with:"
echo "  grep -E 'Ascend FIA mixed split|Ascend FIA mixed forward' <server.log>"

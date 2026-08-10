#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  a5_fia_mixed_split_profile.sh serve  <off|on> [npu_devices]
  a5_fia_mixed_split_profile.sh profile <off|on>

Examples:
  ./llm/a5_fia_mixed_split_profile.sh serve off 0
  ./llm/a5_fia_mixed_split_profile.sh profile off
  ./llm/a5_fia_mixed_split_profile.sh serve on 0
  ./llm/a5_fia_mixed_split_profile.sh profile on

Environment overrides:
  MODEL_PATH, SERVER_HOST, SERVER_PORT, TP_SIZE, CONTEXT_LENGTH
  CHUNKED_PREFILL_SIZE
  PROFILE_ROOT, NUM_PROMPTS, INPUT_LEN, OUTPUT_LEN, REQUEST_RATE
  MAX_CONCURRENCY, PROFILE_NUM_STEPS
EOF
}

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

action=$1
mode=$2

case "$mode" in
    off)
        split_enabled=0
        ;;
    on)
        split_enabled=1
        ;;
    *)
        echo "Invalid mode: $mode (expected off or on)" >&2
        exit 2
        ;;
esac

model_path=${MODEL_PATH:-/mnt/share/weights/Qwen3.5-27B}
host=${SERVER_HOST:-127.0.0.1}
port=${SERVER_PORT:-${VLLM_PORT:-30000}}
tp_size=${TP_SIZE:-1}
context_length=${CONTEXT_LENGTH:-5000}
chunked_prefill_size=${CHUNKED_PREFILL_SIZE:-1024}
profile_root=${PROFILE_ROOT:-/tmp/fia-profile}
profile_dir="${profile_root}/${mode}"

case "$action" in
    serve)
        if [[ $# -ge 3 ]]; then
            export ASCEND_RT_VISIBLE_DEVICES=$3
        fi

        export ASCEND_USE_FIA=1
        export SGLANG_NPU_FIA_MIXED_SPLIT=$split_enabled
        export SGLANG_TORCH_PROFILER_DIR=$profile_dir
        mkdir -p "$profile_dir"

        echo "Starting server with FIA mixed split: $mode"
        echo "Profiler output directory: $profile_dir"
        exec sglang serve \
            --model-path "$model_path" \
            --host "$host" \
            --port "$port" \
            --device npu \
            --tp "$tp_size" \
            --context-length "$context_length" \
            --enable-mixed-chunk \
            --chunked-prefill-size "$chunked_prefill_size" \
            --reasoning-parser qwen3 \
            --trust-remote-code
        ;;
    profile)
        num_prompts=${NUM_PROMPTS:-12}
        input_len=${INPUT_LEN:-4096}
        output_len=${OUTPUT_LEN:-64}
        request_rate=${REQUEST_RATE:-8}
        max_concurrency=${MAX_CONCURRENCY:-8}
        profile_num_steps=${PROFILE_NUM_STEPS:-8}
        base_url="http://${host}:${port}"
        trace_dir="${profile_dir}/$(date +%Y%m%d-%H%M%S)"

        mkdir -p "$trace_dir"
        echo "Profiling FIA mixed split mode: $mode"
        echo "Trace directory: $trace_dir"
        echo "If no mixed marker appears, rerun with REQUEST_RATE=16."

        printf -v profile_payload \
            '{"activities":["CPU","GPU"],"num_steps":%s,"profile_by_stage":false,"with_stack":false,"record_shapes":false,"output_dir":"%s","profile_prefix":"%s"}' \
            "$profile_num_steps" "$trace_dir" "$mode"

        curl --fail --silent --show-error \
            --request POST \
            --header 'Content-Type: application/json' \
            --data-binary "$profile_payload" \
            "${base_url}/start_profile"
        echo

        stop_profile() {
            curl --silent --show-error \
                --request POST \
                "${base_url}/stop_profile" >/dev/null 2>&1 || true
        }
        trap stop_profile EXIT

        python3 -m sglang.benchmark.serving \
            --backend sglang \
            --base-url "$base_url" \
            --dataset-name random \
            --num-prompts "$num_prompts" \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 1 \
            --request-rate "$request_rate" \
            --max-concurrency "$max_concurrency" \
            --tokenize-prompt \
            --seed 0 \
            --warmup-requests 0 \
            --output-file "${trace_dir}/benchmark.jsonl"

        trap - EXIT
        stop_profile
        ;;
    *)
        echo "Invalid action: $action (expected serve or profile)" >&2
        usage
        exit 2
        ;;
esac

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
  MAX_CONCURRENCY, WARMUP_REQUESTS, PROFILE_NUM_STEPS
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
        num_prompts=${NUM_PROMPTS:-128}
        input_len=${INPUT_LEN:-4096}
        output_len=${OUTPUT_LEN:-512}
        request_rate=${REQUEST_RATE:-4}
        max_concurrency=${MAX_CONCURRENCY:-32}
        warmup_requests=${WARMUP_REQUESTS:-16}
        profile_num_steps=${PROFILE_NUM_STEPS:-20}

        mkdir -p "$profile_dir"
        echo "Profiling FIA mixed split mode: $mode"
        echo "If no mixed marker appears, rerun with REQUEST_RATE=8, then 16."
        exec python3 -m sglang.benchmark.serving \
            --backend sglang \
            --base-url "http://${host}:${port}" \
            --dataset-name random \
            --num-prompts "$num_prompts" \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 1 \
            --request-rate "$request_rate" \
            --max-concurrency "$max_concurrency" \
            --tokenize-prompt \
            --seed 0 \
            --warmup-requests "$warmup_requests" \
            --profile \
            --profile-by-stage \
            --profile-stages prefill \
            --profile-num-steps "$profile_num_steps" \
            --profile-activities CPU GPU \
            --profile-prefix "$mode" \
            --profile-output-dir "$profile_dir" \
            --output-file "${profile_dir}/benchmark.jsonl"
        ;;
    *)
        echo "Invalid action: $action (expected serve or profile)" >&2
        usage
        exit 2
        ;;
esac

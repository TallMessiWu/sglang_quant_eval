#!/usr/bin/env bash
# ShareGPT 服务性能测试；只选择客户端 tokenizer，不启动或清理模型服务。
set -euo pipefail

usage() {
    cat <<'EOF'
用法: ./llm/bench_serving.sh

↑/↓ 选择 BF16 模型对应的 tokenizer，Enter 开始，Esc/Q 退出。
选项读取同目录 *_bf16.sh 中独立一行的 --model-path /绝对路径。
请先启动对应模型的服务；选择菜单不会切换服务端模型。

环境变量（均可覆盖）:
  VLLM_PORT        服务端口，默认 6969
  SERVER_HOST      服务地址，默认 127.0.0.1
  DATASET_PATH     默认 /mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
  MAX_CONCURRENCY  并发数，默认 32
  NUM_PROMPTS      请求数，默认 500
  FLUSH_CACHE      设为 1 时在压测前清空服务 prefix cache，默认 0
  OUTPUT_DETAILS   设为 1 保存原生逐请求明细并打印短摘要，默认 0
  TRACE_STALLS     设为 1 记录收流时间轴和客户端停顿，要求 OUTPUT_DETAILS=1
  BENCH_ROOT       结果目录，默认 当前工作目录/artifacts/bench-serving
  NO_COLOR         非空时不强调模型名或淡化路径

每次运行使用独立子目录，保存 bench.log / results.jsonl 及临时文件、缓存。
EOF
}

if (( $# )); then
    if [[ $# == 1 && ( $1 == -h || $1 == --help ) ]]; then
        usage
        exit 0
    fi
    usage >&2
    exit 2
fi

cache_args=()
case ${FLUSH_CACHE:-0} in
    0) ;;
    1) cache_args+=(--flush-cache) ;;
    *) printf 'FLUSH_CACHE 只支持 0 或 1。\n' >&2; exit 2 ;;
esac
detail_args=()
case ${OUTPUT_DETAILS:-0} in
    0) ;;
    1) detail_args+=(--output-details) ;;
    *) printf 'OUTPUT_DETAILS 只支持 0 或 1。\n' >&2; exit 2 ;;
esac
case ${TRACE_STALLS:-0} in
    0) ;;
    1)
        if [[ ${OUTPUT_DETAILS:-0} != 1 ]]; then
            printf 'TRACE_STALLS=1 要求 OUTPUT_DETAILS=1。\n' >&2
            exit 2
        fi
        detail_args+=(--trace-stalls)
        ;;
    *) printf 'TRACE_STALLS 只支持 0 或 1。\n' >&2; exit 2 ;;
esac

script_dir=$(dirname "$(readlink -f "$0")")
model_paths=()
shopt -s nullglob
for launcher in "$script_dir"/*_bf16.sh; do
    # 只读取固定路径；绝不 source 启动脚本，避免启动服务或清理 NPU。
    model_path=$(awk '$1 == "--model-path" { print $2 }' "$launcher")
    if [[ $model_path != /* || $model_path == *$'\n'* || $model_path == *'$'* || $model_path == *'`'* ]]; then
        printf '无法读取固定模型路径，请检查 %s 的 --model-path。\n' "$launcher" >&2
        exit 2
    fi
    model_paths+=("$model_path")
done
if (( ${#model_paths[@]} == 0 )); then
    printf '未找到 BF16 启动脚本: %s/*_bf16.sh\n' "$script_dir" >&2
    exit 2
fi

if ! { exec {tty_fd}<>/dev/tty; } 2>/dev/null; then
    printf '方向键菜单需要交互终端，请直接在终端运行本脚本。\n' >&2
    exit 2
fi
tty_state=$(stty -g <&"$tty_fd")
cleanup_menu() {
    stty "$tty_state" <&"$tty_fd" || true
    printf '\033[0m\033[?25h\033[?1049l' >&"$tty_fd"
}
trap cleanup_menu EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
stty -echo -icanon min 1 time 0 <&"$tty_fd"
printf '\033[?1049h\033[?25l' >&"$tty_fd"

bold='' dim='' reset=''
if [[ -z ${NO_COLOR:-} && ${TERM:-dumb} != dumb ]]; then
    bold=$'\033[1m'
    dim=$'\033[2m'
    reset=$'\033[0m'
fi
selected=0
confirmed=0
while true; do
    printf '\033[H\033[2J选择压测 tokenizer\n\n' >&"$tty_fd"
    for i in "${!model_paths[@]}"; do
        marker=' '
        style=''
        if (( i == selected )); then marker='>'; style=$bold; fi
        printf ' %s %s%s%s %s(%s)%s\n' \
            "$marker" "$style" "${model_paths[i]##*/}" "$reset" \
            "$dim" "${model_paths[i]}" "$reset" >&"$tty_fd"
    done
    printf '\n↑/↓ 选择 · Enter 开始 · Esc/Q 退出\n仅选择分词器，不会切换服务端模型。\n' >&"$tty_fd"
    key=''
    IFS= read -rsn1 -u "$tty_fd" key || break
    case "$key" in
        '') confirmed=1; break ;;
        q|Q) break ;;
        $'\033')
            sequence=''
            if ! IFS= read -rsn2 -t 0.3 -u "$tty_fd" sequence; then break; fi
            case "$sequence" in
                '[A'|'OA') selected=$(( (selected + ${#model_paths[@]} - 1) % ${#model_paths[@]} )) ;;
                '[B'|'OB') selected=$(( (selected + 1) % ${#model_paths[@]} )) ;;
            esac
            ;;
    esac
done
cleanup_menu
trap - EXIT INT TERM
exec {tty_fd}>&-
if (( ! confirmed )); then
    printf '已取消测试。\n'
    exit 0
fi

tokenizer=${model_paths[selected]}
host=${SERVER_HOST:-127.0.0.1}
port=${VLLM_PORT:-6969}
bench_root=${BENCH_ROOT:-$PWD/artifacts/bench-serving}
mkdir -p "$bench_root"
run_dir=$(mktemp -d "$bench_root/$(date +%Y%m%d-%H%M%S)-XXXXXX")
export TMPDIR="$run_dir/tmp"
export XDG_CACHE_HOME="$run_dir/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$TMPDIR" "$HF_HUB_CACHE"
printf 'Tokenizer: %s (%s)\n服务: http://%s:%s\n结果: %s\n' "${tokenizer##*/}" "$tokenizer" "$host" "$port" "$run_dir"

python3 -m sglang.bench_serving \
    --backend sglang \
    --host "$host" \
    --port "$port" \
    --dataset-name sharegpt \
    --dataset-path "${DATASET_PATH:-/mnt/share/dataset/ShareGPT_V3_unfiltered_cleaned_split.json}" \
    --tokenizer "$tokenizer" \
    --max-concurrency "${MAX_CONCURRENCY:-32}" \
    --num-prompts "${NUM_PROMPTS:-500}" \
    "${cache_args[@]}" \
    "${detail_args[@]}" \
    --output-file "$run_dir/results.jsonl" \
    2>&1 | tee "$run_dir/bench.log"

if [[ ${OUTPUT_DETAILS:-0} == 1 ]]; then
    python3 "$script_dir/diagnostics/summarize_serving_tails.py" "$run_dir/results.jsonl" \
        | tee "$run_dir/summary.txt"
fi

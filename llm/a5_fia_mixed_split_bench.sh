#!/usr/bin/env bash
#
# A5 混合 chunked-prefill FIA 拆分：端到端性能测试
#
# 对着 a5_fia_mixed_split_serve.sh 起的服务打流量。off/on 两组要分别重启
# 服务再各跑一次，最后用 compare 对比。
#
# 默认负载是照着 vllm-ascend#11948 测出收益的那个 batch 形态设计的：
# prefill 段 11274 token / 1 个请求，decode 段 36 token / 9 个请求。收益的
# 大头（566us / 全部 811us）来自 prefill 段本身变规整，而不是 decode 段。
#
# 要复现这种形态，三个参数缺一不可：
#   * 服务端 --chunked-prefill-size 要大。它是 mixed batch 里 prefill 段的
#     长度上限，卡在 1024 的话，输入调多长都没用。
#   * OUTPUT_LEN 要长，请求在 decode 阶段停留得久，才堆得起多个 decode 请求。
#   * MAX_CONCURRENCY 要大，否则同时在跑的请求不够多。
#
# 判断负载有没有造对，跑完用 inspect 看服务日志里 mixed batch 的实际形态，
# 不要靠猜。decode 请求只有 1~2 个时，拆分多出来的 kernel launch、
# empty_like 和两次 copy_ 会吃掉全部收益，测出来必然是负的。

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  a5_fia_mixed_split_bench.sh run <off|on> [--profile]
  a5_fia_mixed_split_bench.sh compare
  a5_fia_mixed_split_bench.sh inspect <服务日志>

示例:
  ./llm/a5_fia_mixed_split_bench.sh run off
  ./llm/a5_fia_mixed_split_bench.sh run on
  ./llm/a5_fia_mixed_split_bench.sh compare

  # 先确认负载造对了：看 mixed batch 的实际形态
  ./llm/a5_fia_mixed_split_bench.sh inspect on.log

  # 只想抓一段 trace 看 ascend.fia_mixed.* marker
  ./llm/a5_fia_mixed_split_bench.sh run on --profile

环境变量覆盖:
  SERVER_HOST      服务地址        (默认 127.0.0.1)
  SERVER_PORT      服务端口        (默认 $VLLM_PORT，再默认 30000)
  DATASET_PATH     ShareGPT 数据集 (默认 /home/hajimi/benchmark/ShareGPT_V3_unfiltered_cleaned_split.json)
  NUM_PROMPTS      请求总数        (默认 96)
  INPUT_LEN        输入长度        (默认 8192，要大于服务端分块大小)
  OUTPUT_LEN       输出长度        (默认 1024，越长 decode 请求堆得越多)
  MAX_CONCURRENCY  并发上限        (默认 32)
  REQUEST_RATE     发送速率        (默认 inf，由并发上限控速)
  PROFILE_NUM_STEPS  --profile 时抓几步 (默认 8)
  BENCH_ROOT       结果根目录      (默认 <repo>/llm/fia_bench)

注意 INPUT_LEN + OUTPUT_LEN 必须小于服务端的 CONTEXT_LENGTH(默认 10240)。
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

script_dir=$(dirname "$(readlink -f "$0")")
repo_root=$(dirname "$script_dir")

host=${SERVER_HOST:-127.0.0.1}
port=${SERVER_PORT:-${VLLM_PORT:-30000}}
base_url="http://${host}:${port}"
bench_root=${BENCH_ROOT:-${repo_root}/llm/fia_bench}

action=$1
shift

case "$action" in
    run)
        if [[ $# -lt 1 ]]; then
            echo "❌ run 需要指定 off 或 on" >&2
            usage
            exit 2
        fi
        mode=$1
        shift
        case "$mode" in
            off|on) ;;
            *)
                echo "❌ 无效模式: $mode (应为 off 或 on)" >&2
                exit 2
                ;;
        esac

        with_profile=0
        if [[ $# -gt 0 ]]; then
            case "$1" in
                --profile) with_profile=1 ;;
                *)
                    echo "❌ 无效参数: $1" >&2
                    exit 2
                    ;;
            esac
        fi

        dataset_path=${DATASET_PATH:-/home/hajimi/benchmark/ShareGPT_V3_unfiltered_cleaned_split.json}
        num_prompts=${NUM_PROMPTS:-96}
        input_len=${INPUT_LEN:-8192}
        output_len=${OUTPUT_LEN:-1024}
        max_concurrency=${MAX_CONCURRENCY:-32}
        request_rate=${REQUEST_RATE:-inf}
        profile_num_steps=${PROFILE_NUM_STEPS:-8}

        run_dir="${bench_root}/${mode}/$(date +%Y%m%d-%H%M%S)"
        mkdir -p "$run_dir"

        echo "📊 FIA mixed split = ${mode}"
        echo "    结果目录: $run_dir"
        echo "    负载: ${num_prompts} 请求, 输入 ${input_len}, 输出 ${output_len}, 并发 ${max_concurrency}, 速率 ${request_rate}"

        stop_profile() {
            curl --silent --show-error \
                --request POST \
                "${base_url}/stop_profile" >/dev/null 2>&1 || true
        }

        if [[ $with_profile -eq 1 ]]; then
            echo "    抓 trace: 前 ${profile_num_steps} 步 -> $run_dir"
            printf -v profile_payload \
                '{"activities":["CPU","GPU"],"num_steps":%s,"profile_by_stage":false,"with_stack":false,"record_shapes":false,"output_dir":"%s","profile_prefix":"%s"}' \
                "$profile_num_steps" "$run_dir" "$mode"

            curl --fail --silent --show-error \
                --request POST \
                --header 'Content-Type: application/json' \
                --data-binary "$profile_payload" \
                "${base_url}/start_profile"
            echo
            trap stop_profile EXIT
        fi

        # --warmup-requests 1: 让第一个请求先把权重和 kernel 都跑热，
        # 否则首个请求的编译/tiling 开销会算进 TTFT。
        python3 -m sglang.benchmark.serving \
            --backend sglang \
            --base-url "$base_url" \
            --dataset-name random \
            --dataset-path "$dataset_path" \
            --num-prompts "$num_prompts" \
            --random-input-len "$input_len" \
            --random-output-len "$output_len" \
            --random-range-ratio 1 \
            --request-rate "$request_rate" \
            --max-concurrency "$max_concurrency" \
            --tokenize-prompt \
            --seed 0 \
            --warmup-requests 1 \
            --output-file "${run_dir}/benchmark.jsonl"

        echo
        echo "✅ 完成: ${run_dir}/benchmark.jsonl"
        echo "   两组都跑完后执行: ./llm/a5_fia_mixed_split_bench.sh compare"
        ;;

    compare)
        off_file=$(ls -1d "${bench_root}"/off/*/benchmark.jsonl 2>/dev/null | tail -1 || true)
        on_file=$(ls -1d "${bench_root}"/on/*/benchmark.jsonl 2>/dev/null | tail -1 || true)

        if [[ -z "$off_file" || -z "$on_file" ]]; then
            echo "❌ 找不到两组结果，off 和 on 各跑一次再来 compare" >&2
            echo "   off: ${off_file:-<无>}" >&2
            echo "   on : ${on_file:-<无>}" >&2
            exit 1
        fi

        echo "off: $off_file"
        echo "on : $on_file"
        echo

        python3 - "$off_file" "$on_file" <<'PY'
import json
import sys

# (字段, 展示名, 越大越好)
METRICS = [
    ("output_throughput", "Output tok/s", True),
    ("total_token_throughput", "Total tok/s", True),
    ("request_throughput", "Req/s", True),
    ("median_ttft_ms", "TTFT p50 ms", False),
    ("p99_ttft_ms", "TTFT p99 ms", False),
    ("median_tpot_ms", "TPOT p50 ms", False),
    ("p99_tpot_ms", "TPOT p99 ms", False),
    ("median_itl_ms", "ITL p50 ms", False),
    ("p99_itl_ms", "ITL p99 ms", False),
    ("median_e2e_latency_ms", "E2E p50 ms", False),
]


def load(path):
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        sys.exit(f"{path} 里没有结果")
    return rows[-1]


off, on = load(sys.argv[1]), load(sys.argv[2])

print(f"{'指标':<16}{'OFF':>14}{'ON':>14}{'变化':>12}")
print("-" * 56)
for key, label, higher_is_better in METRICS:
    if key not in off or key not in on:
        continue
    a, b = off[key], on[key]
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a == 0:
        continue
    delta = (b - a) / a * 100.0
    if not higher_is_better:
        delta = -delta
    print(f"{label:<16}{a:>14.2f}{b:>14.2f}{delta:>11.2f}%")

print()
print("正数 = on 更好。注意这里比的是端到端，混合 batch 只占其中一部分步数，")
print("所以整体涨幅一定小于算子级 microbenchmark 的涨幅。")
PY
        ;;

    inspect)
        if [[ $# -lt 1 ]]; then
            echo "❌ inspect 需要一个服务日志路径" >&2
            usage
            exit 2
        fi
        log_file=$1
        if [[ ! -f "$log_file" ]]; then
            echo "❌ 找不到日志: $log_file" >&2
            exit 1
        fi

        python3 - "$log_file" <<'PY'
import re
import statistics
import sys

# SGLang 调度日志里，带 #running-req > 0 的 Prefill batch 就是 mixed batch:
# 那一步既有新请求的 prefill 分块，又有正在 decode 的请求。
NEW_TOKEN = re.compile(r"#new-token:\s*(\d+)")
RUNNING_REQ = re.compile(r"#running-req:\s*(\d+)")

prefill_steps = 0
mixed = []  # (prefill_tokens, decode_reqs)

with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if "Prefill batch" not in line:
            continue
        tokens = NEW_TOKEN.search(line)
        running = RUNNING_REQ.search(line)
        if not tokens or not running:
            continue
        prefill_steps += 1
        decode_reqs = int(running.group(1))
        if decode_reqs > 0:
            mixed.append((int(tokens.group(1)), decode_reqs))

if not prefill_steps:
    sys.exit(
        "日志里没有可解析的 Prefill batch 行。确认这是服务端日志，"
        "而不是压测端输出。"
    )

print(f"Prefill batch 总步数 : {prefill_steps}")
print(f"其中 mixed batch     : {len(mixed)}  ({len(mixed) / prefill_steps:.1%})")

if not mixed:
    print()
    print("一个 mixed batch 都没有，拆分完全没被触发。检查服务端有没有带")
    print("--enable-mixed-chunk，以及 INPUT_LEN 是否大于 --chunked-prefill-size。")
    sys.exit(0)

tokens = [t for t, _ in mixed]
reqs = [r for _, r in mixed]

print()
print(f"{'':<18}{'p50':>8}{'max':>8}{'ref':>8}   ref = vllm-ascend#11948")
print(
    f"{'prefill tokens':<18}{statistics.median(tokens):>8.0f}"
    f"{max(tokens):>8}{11274:>8}"
)
print(
    f"{'decode reqs':<18}{statistics.median(reqs):>8.0f}"
    f"{max(reqs):>8}{9:>8}"
)
print()

median_tokens = statistics.median(tokens)
median_reqs = statistics.median(reqs)
problems = []
if median_tokens < 2048:
    problems.append(
        f"prefill 段只有 {median_tokens:.0f} token —— 调大服务端的 "
        "--chunked-prefill-size，它是这一段的长度上限"
    )
if median_reqs < 4:
    problems.append(
        f"decode 请求只有 {median_reqs:.0f} 个 —— 调大 OUTPUT_LEN 和 "
        "MAX_CONCURRENCY，让请求在 decode 阶段堆起来"
    )
if len(mixed) / prefill_steps < 0.2:
    problems.append(
        f"mixed batch 只占 {len(mixed) / prefill_steps:.1%} 的 prefill 步 —— "
        "端到端收益会被稀释到测不出来"
    )

if problems:
    print("负载没造对:")
    for item in problems:
        print(f"  - {item}")
else:
    print("负载形态合理，端到端差异可以采信。")
PY
        ;;

    *)
        echo "❌ 无效动作: $action (应为 run / compare / inspect)" >&2
        usage
        exit 2
        ;;
esac

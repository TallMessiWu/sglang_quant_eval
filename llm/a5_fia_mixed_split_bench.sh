#!/usr/bin/env bash
#
# A5 混合 chunked-prefill FIA 拆分：端到端性能测试
#
# 对着 a5_fia_mixed_split_serve.sh 起的服务打流量。off/on 两组要分别重启
# 服务再各跑一次，最后用 compare 对比。
#
# 算子级已经实测过（a5_fia_mixed_split_op_bench.py，Qwen3.5-27B TP=1）：
# prefill 段 3968 token 配 23 个 decode 请求时拆分快 16.3%，prefill 段从
# 1024 扫到 8192 全程为正（+29.5% 递减到 +11.1%）。所以算子本身没问题。
#
# 端到端能吃到多少，约等于:
#     算子涨幅 x mixed 步占总步数的比例 x attention 占单步的比重
# 拆分只作用于 mixed 步，纯 decode 步完全不走这条路径。实测 OUTPUT_LEN=1024
# 时 mixed 只占总步数 5%，16% 稀释完不到 1%，比单次运行的抖动还小。
#
# 于是 OUTPUT_LEN 是把双刃剑：
#   * 长 -> 请求在 decode 阶段停留久，mixed batch 里的 decode 请求堆得多；
#   * 长 -> 纯 decode 步随之线性增长，把 mixed 步的占比压下去。
# 默认取 256 是这两者的折中。服务端 --chunked-prefill-size 则要大，它是
# mixed batch 里 prefill 段的长度上限，卡在 1024 的话输入调多长都没用。
#
# 跑完先用 inspect 确认 mixed 占总步数的比例，再看 compare，不要靠猜。

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  a5_fia_mixed_split_bench.sh run <off|on> [--profile]
  a5_fia_mixed_split_bench.sh compare
  a5_fia_mixed_split_bench.sh verify [<目录A> <目录B>]
  a5_fia_mixed_split_bench.sh inspect <服务日志>

示例:
  ./llm/a5_fia_mixed_split_bench.sh run off
  ./llm/a5_fia_mixed_split_bench.sh run on
  ./llm/a5_fia_mixed_split_bench.sh compare

  # 精度：off/on 生成文本逐字节比对
  ./llm/a5_fia_mixed_split_bench.sh verify

  # 底噪基线：同为 off 的两次运行互比，量硬件本身的不确定性
  ./llm/a5_fia_mixed_split_bench.sh verify \n      llm/fia_bench/off/20260812-100000 llm/fia_bench/off/20260812-101500

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
  OUTPUT_LEN       输出长度        (默认 256，太长会让纯 decode 步淹没 mixed 步)
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
        # decode 步数随 OUTPUT_LEN 线性增长，而拆分只作用于 mixed 步。
        # 1024 时实测 mixed 只占总步数 5%，算子那 16% 稀释后不足 1%，读不出来。
        output_len=${OUTPUT_LEN:-256}
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
            --output-details \
            --output-file "${run_dir}/benchmark.jsonl"

        echo
        echo "✅ 完成: ${run_dir}/benchmark.jsonl"
        echo "   两组都跑完后执行: ./llm/a5_fia_mixed_split_bench.sh compare"
        ;;

    compare)
        if ! compgen -G "${bench_root}/off/*/benchmark.jsonl" >/dev/null \
            || ! compgen -G "${bench_root}/on/*/benchmark.jsonl" >/dev/null; then
            echo "❌ 找不到两组结果，off 和 on 各跑一次再来 compare" >&2
            exit 1
        fi

        python3 - "${bench_root}/off" "${bench_root}/on" <<'PY'
import glob
import json
import os
import statistics
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


def load_runs(directory):
    runs = []
    for path in sorted(glob.glob(os.path.join(directory, "*", "benchmark.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if rows:
            runs.append(rows[-1])
    if not runs:
        sys.exit(f"{directory} 下没有结果")
    return runs


def series(runs, key):
    values = [
        run[key]
        for run in runs
        if isinstance(run.get(key), (int, float))
    ]
    return values


off_runs, on_runs = load_runs(sys.argv[1]), load_runs(sys.argv[2])
print(f"off: {len(off_runs)} 次运行    on: {len(on_runs)} 次运行")
if len(off_runs) < 2 or len(on_runs) < 2:
    print("⚠️  每组只有一次运行，无法判断方差。这个特性的端到端差异预期在 1~2%,")
    print("   比单次运行的抖动还小 —— 每组至少跑 3 次再看结论。")
print()

print(f"{'metric':<16}{'OFF p50':>12}{'ON p50':>12}{'delta':>9}{'OFF spread':>12}{'ON spread':>11}")
print("-" * 72)
for key, label, higher_is_better in METRICS:
    a_values, b_values = series(off_runs, key), series(on_runs, key)
    if not a_values or not b_values:
        continue
    a, b = statistics.median(a_values), statistics.median(b_values)
    if a == 0:
        continue
    delta = (b - a) / a * 100.0
    if not higher_is_better:
        delta = -delta
    # 极差相对中位数的比例：它比 delta 大就说明这个差异读不出来
    a_spread = (max(a_values) - min(a_values)) / a * 100.0 if a else 0.0
    b_spread = (max(b_values) - min(b_values)) / b * 100.0 if b else 0.0
    print(
        f"{label:<16}{a:>12.2f}{b:>12.2f}{delta:>8.2f}%"
        f"{a_spread:>11.2f}%{b_spread:>10.2f}%"
    )

print()
print("正数 = on 更好。delta 小于任一侧的 spread 时，这个差异是噪声，不是信号。")
print("拆分只作用于 mixed batch，先用 inspect 确认 mixed 占总步数的比例 ——")
print("端到端涨幅的上限约等于「算子涨幅 x mixed 步占比 x attention 占单步的比重」。")
PY
        ;;

    verify)
        # 精度验证：off/on 用同一个 seed、同样的长度、temperature=0，逐条比对
        # 生成文本。这是唯一能覆盖 SGLang 真实实现的多 decode 请求路径的手段
        # —— op bench 的对拍是自己复现了一遍拆分逻辑，走不到
        # _forward_fia_mixed_split 里 block_table / seq_lens 的多请求切片。
        # 默认比 off 与 on。也可以显式传两个目录，用来先跑一次同模式的基线
        # 对照（off 的两次运行互比）—— bf16 归约顺序在 NPU 上不保证稳定，
        # 个别 token 的 argmax 可能翻转，不先量出这个底噪就无法判断 off/on
        # 的差异是拆分算错了还是硬件抖动。
        dir_a=${1:-${bench_root}/off}
        dir_b=${2:-${bench_root}/on}

        python3 - "$dir_a" "$dir_b" <<'PY'
import glob
import json
import os
import sys


def latest_texts(directory):
    # 既接受模式目录(取其中最新一次运行)，也接受具体某一次运行的目录
    direct = os.path.join(directory, "benchmark.jsonl")
    if os.path.isfile(direct):
        path = direct
    else:
        paths = sorted(glob.glob(os.path.join(directory, "*", "benchmark.jsonl")))
        if not paths:
            sys.exit(f"{directory} 下没有结果")
        path = paths[-1]
    with open(path, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        sys.exit(f"{path} 里没有结果")
    texts = rows[-1].get("generated_texts")
    if texts is None:
        sys.exit(
            f"{path} 里没有 generated_texts。这次运行是在加 --output-details "
            "之前跑的，off 和 on 都要重跑一次。"
        )
    return path, texts


off_path, off_texts = latest_texts(sys.argv[1])
on_path, on_texts = latest_texts(sys.argv[2])
print(f"off: {off_path}  ({len(off_texts)} 条)")
print(f"on : {on_path}  ({len(on_texts)} 条)")
print()

if len(off_texts) != len(on_texts):
    sys.exit(
        f"❌ 请求数不一致 ({len(off_texts)} vs {len(on_texts)})，"
        "两组必须用同一套负载参数"
    )

mismatches = [
    (index, a, b)
    for index, (a, b) in enumerate(zip(off_texts, on_texts))
    if a != b
]

if not mismatches:
    print(f"✅ {len(off_texts)}/{len(off_texts)} 条输出逐字节一致")
    sys.exit(0)

print(f"❌ {len(mismatches)}/{len(off_texts)} 条输出不一致")
for index, a, b in mismatches[:3]:
    # 定位第一个分叉的字符，贪心解码下这里就是数值差异首次改变 argmax 的位置
    diverge = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    print()
    print(f"--- 请求 {index}，第 {diverge} 个字符处分叉 ---")
    print(f"  off: ...{a[max(0, diverge - 40):diverge + 40]!r}")
    print(f"  on : ...{b[max(0, diverge - 40):diverge + 40]!r}")
sys.exit(1)
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
decode_steps = 0
mixed = []  # (prefill_tokens, decode_reqs)

with open(sys.argv[1], encoding="utf-8", errors="replace") as handle:
    for line in handle:
        if "Decode batch" in line:
            decode_steps += 1
            continue
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

total_steps = prefill_steps + decode_steps
# 端到端收益的上限就卡在这个占比上：拆分只作用于 mixed 步，纯 decode 步
# 完全不走这条路径。mixed 占总步数 5%、算子快 16%，整体最多也就 0.8%,
# 单次运行的噪声比这个还大。
mixed_share = len(mixed) / total_steps if total_steps else 0.0

print(f"Prefill batch 步数   : {prefill_steps}  (其中 mixed {len(mixed)})")
print(f"Decode batch 步数    : {decode_steps}")
print(f"mixed 占总步数       : {mixed_share:.1%}   <- 端到端收益的天花板由它决定")

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
if mixed_share < 0.15:
    problems.append(
        f"mixed 只占总步数 {mixed_share:.1%} —— 纯 decode 步太多，收益会被"
        "稀释到噪声以下。调小 OUTPUT_LEN(decode 步随它线性减少)，"
        "或调大 INPUT_LEN(mixed 步随它增加)"
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
        echo "❌ 无效动作: $action (应为 run / compare / verify / inspect)" >&2
        usage
        exit 2
        ;;
esac

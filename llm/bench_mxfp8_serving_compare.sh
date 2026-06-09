#!/usr/bin/env bash
# Qwen3-8B Dense @ Ascend A5: BF16 vs MXFP8 端到端 serving 对比。
#
# 目的:把 PR #22352 的收益拆成 prefill(TTFT)和 decode(TPOT)两个信号。
#   - 算子级 micro-bench(bench_mxfp8_dense_op.py)已证实:prefill GEMM ~1.8x,
#     decode 小矩阵被 npu_quant_matmul ~0.085ms 设备侧固定地板吃掉 → decode 是 wash。
#   - 这里用真实 serving 验证该结论:mxfp8 应在 prefill-heavy 上压低 TTFT,
#     在 decode-heavy 上 TPOT 与 bf16 接近持平。
#
# 自动化流程:对 bf16 / mxfp8 各 → 后台起 server → 等就绪 → 两种 workload bench → 关 server。
#
# 用法:
#   bash llm/bench_mxfp8_serving_compare.sh
# 环境变量覆盖:
#   MODEL=/home/weights/Qwen3-8B PORT=6979 TP=1 NUM_PROMPTS=64 \
#   bash llm/bench_mxfp8_serving_compare.sh

set -uo pipefail

MODEL="${MODEL:-/home/weights/Qwen3-8B}"
PORT="${PORT:-6979}"
TP="${TP:-1}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
LOGDIR="${LOGDIR:-./bench_logs}"
mkdir -p "$LOGDIR"

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">>> stopping server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null
    # 给子进程(NPU runtime)时间释放显存
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_ready() {
  echo -n ">>> waiting for server on :$PORT "
  for _ in $(seq 1 300); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo " SERVER DIED"
      return 1
    fi
    # /health_generate 会真正跑一次前向,确保权重加载 + 图捕获完成
    if curl -sf "http://localhost:$PORT/health_generate" >/dev/null 2>&1; then
      echo " ready"
      return 0
    fi
    echo -n "."
    sleep 2
  done
  echo " TIMEOUT"
  return 1
}

bench() {
  local tag="$1" wl="$2" in_len="$3" out_len="$4"
  echo ">>> [$tag] bench $wl (in=$in_len out=$out_len, n=$NUM_PROMPTS)"
  python -m sglang.bench_serving --backend sglang \
      --host localhost --port "$PORT" \
      --dataset-name random \
      --random-input-len "$in_len" --random-output-len "$out_len" \
      --num-prompts "$NUM_PROMPTS" \
      2>&1 | tee "$LOGDIR/bench_${tag}_${wl}.log"
}

run_one() {
  local tag="$1"; shift
  local quant_args=("$@")

  echo "============================================================"
  echo ">>> launching server [$tag] : ${quant_args[*]:-（bf16 baseline）}"
  echo "============================================================"
  # shellcheck disable=SC2086
  sglang serve \
      --model-path "$MODEL" \
      --host 0.0.0.0 \
      --port "$PORT" \
      --device npu \
      --tp "$TP" \
      --trust-remote-code \
      "${quant_args[@]}" \
      > "$LOGDIR/server_$tag.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_ready; then
    echo "!!! server [$tag] 启动失败,见 $LOGDIR/server_$tag.log"
    cleanup
    return 1
  fi

  bench "$tag" prefill 2048 16    # prefill-heavy → 看 TTFT
  bench "$tag" decode  128  512   # decode-heavy  → 看 TPOT

  cleanup
  sleep 5
}

run_one bf16
run_one mxfp8 --quantization mxfp8

echo
echo "############################################################"
echo "# SUMMARY  model=$MODEL  tp=$TP  num_prompts=$NUM_PROMPTS"
echo "############################################################"
for wl in prefill decode; do
  echo "==================== workload: $wl ===================="
  for tag in bf16 mxfp8; do
    f="$LOGDIR/bench_${tag}_${wl}.log"
    [[ -f "$f" ]] || { echo "[$tag] (无结果)"; continue; }
    echo "---- $tag ----"
    grep -E "Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Output token throughput|Request throughput" "$f" \
      || echo "  (未找到指标,完整输出见 $f)"
  done
done
echo
echo "判读:"
echo "  prefill workload → 比 Mean/Median TTFT,mxfp8 应明显低(算子级 ~1.8x 的体现)。"
echo "  decode  workload → 比 Mean/Median TPOT,mxfp8 应与 bf16 接近持平(小矩阵地板所致)。"
echo "  若 prefill TTFT 也没降 → 回头查 scale dtype / A5 fast-path 是否真走对。"

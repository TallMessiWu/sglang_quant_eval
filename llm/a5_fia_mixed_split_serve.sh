#!/usr/bin/env bash
#
# A5 混合 chunked-prefill FIA 拆分：起服务
#
# 用 a5_fia_mixed_split_bench.sh 打流量。off/on 两组必须分别重启本脚本，
# 因为开关是启动期读取的环境变量。
#
# 拆分只在调度器真的形成 mixed batch 时才生效，所以这里必须带
# --enable-mixed-chunk，且 --chunked-prefill-size 要小于请求的输入长度，
# 否则 prefill 一步做完，永远不会和 decode 混在同一个 batch 里。

set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  a5_fia_mixed_split_serve.sh <off|on> [NPU 卡号...]

示例:
  ./llm/a5_fia_mixed_split_serve.sh off 0
  ./llm/a5_fia_mixed_split_serve.sh on 0

环境变量覆盖:
  MODEL_PATH            模型权重      (默认 /home/weights/Qwen3.5-27B)
  SERVER_HOST           监听地址      (默认 127.0.0.1)
  SERVER_PORT           监听端口      (默认 $VLLM_PORT，再默认 30000)
  TP_SIZE               张量并行      (默认 1)
  CONTEXT_LENGTH        最大上下文    (默认 5000)
  CHUNKED_PREFILL_SIZE  prefill 分块  (默认 1024)
  BENCH_ROOT            结果/trace 根 (默认 <repo>/llm/fia_bench)
EOF
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

mode=$1
shift

case "$mode" in
    off) split_enabled=0 ;;
    on)  split_enabled=1 ;;
    *)
        echo "❌ 无效模式: $mode (应为 off 或 on)" >&2
        usage
        exit 2
        ;;
esac

script_dir=$(dirname "$(readlink -f "$0")")
repo_root=$(dirname "$script_dir")

model_path=${MODEL_PATH:-/home/weights/Qwen3.5-27B}
host=${SERVER_HOST:-127.0.0.1}
port=${SERVER_PORT:-${VLLM_PORT:-30000}}
tp_size=${TP_SIZE:-1}
context_length=${CONTEXT_LENGTH:-5000}
chunked_prefill_size=${CHUNKED_PREFILL_SIZE:-1024}
bench_root=${BENCH_ROOT:-${repo_root}/llm/fia_bench}
profile_dir="${bench_root}/${mode}"

# ================= 1. 前置清理 NPU =================
if [[ $# -eq 0 ]]; then
    echo "⚠️  未传入 NPU 卡号，将 unset ASCEND_RT_VISIBLE_DEVICES"
    unset ASCEND_RT_VISIBLE_DEVICES || true
else
    cleaner_path="$script_dir/npu-cleaner.sh"
    if [[ -x "$cleaner_path" ]]; then
        echo "🧹 [前置任务] 准备释放 NPU ($*) 的资源..."
        "$cleaner_path" "$@"
        sleep 1
        echo "✅ [前置任务] 清理完毕！"
    else
        echo "⚠️  未找到可执行的清理脚本 $cleaner_path，跳过清理直接启动。"
    fi

    old_ifs="$IFS"
    IFS=","
    export ASCEND_RT_VISIBLE_DEVICES="$*"
    IFS="$old_ifs"
fi

# ================= 2. 环境变量 =================
# A5(Ascend 950) 走 FIA 融合注意力算子 npu_fused_infer_attention_score；
# 否则默认走 ATB SelfAttentionOperation，在 A5 上 warmup 会崩。
export ASCEND_USE_FIA=1

# 混合 batch 的 FIA 拆分开关。代码里默认就是开的，这里显式写死是为了让
# off 组也能明确关掉，两组之间只差这一个变量。
export SGLANG_NPU_FIA_MIXED_SPLIT=$split_enabled

# /start_profile 需要这个目录已经存在。
export SGLANG_TORCH_PROFILER_DIR=$profile_dir
mkdir -p "$profile_dir"

echo "🚀 [启动任务] FIA mixed split = ${mode} (SGLANG_NPU_FIA_MIXED_SPLIT=${split_enabled})"
echo "    设备可见性: ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-<unset>}"
echo "    profiler 目录: $profile_dir"
echo "    分块大小: $chunked_prefill_size (输入长度必须大于它才会产生 mixed batch)"

# ================= 3. 起服务 =================
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

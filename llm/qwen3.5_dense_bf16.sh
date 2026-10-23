#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

# A5(Ascend 950) 走 FIA 融合注意力算子 npu_fused_infer_attention_score；
# 否则默认走 ATB SelfAttentionOperation(_npu_flash_attention_qlens)，在 A5 上 CreateOperation 失败、warmup 请求崩溃。
# prefill+decode 一起绕开 ATB（decode 默认走 _npu_paged_attention，同源）。
export ASCEND_USE_FIA=1
# NPU GDN/SSM state 默认 BF16，与权重量化格式无关；允许显式覆盖。
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

# ================= 4. MTP（NEXTN 投机解码）总开关 =================
# MTP=1 开启，默认关闭。draft 层在 Qwen3.5 checkpoint 内，不需要额外的 draft 权重路径。
# topk=1 的链式 NEXTN 里 num-draft-tokens = num-steps + 1，要改一起改。
# num-draft-tokens 同时是 GDN 算子一次 verify 的 token 数，不能超过 kernel 的 MAX_MTP=8。
MTP_ARGS=()
if [ "${MTP:-0}" = "1" ]; then
    MTP_ARGS=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
    # MTP 与普通推理共用上方的 BF16 SSM state 默认值，不依赖 MTP 才触发 dtype 修正。
    echo "🔮 [MTP] NEXTN 已开启：num-steps=3, eagle-topk=1, num-draft-tokens=4"
fi

# 额外参数透传，例如 EXTRA_ARGS="--disable-cuda-graph"
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
    echo "➕ [EXTRA] 追加启动参数：${EXTRA_ARGS}"
fi

# ========== 下方是原有的模型启动命令 ==========
sglang serve \
    --model-path /mnt/share/weights/Qwen3.5-27B \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code \
    "${MTP_ARGS[@]}" "${EXTRA_ARGS_ARR[@]}"

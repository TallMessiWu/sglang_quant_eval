#!/bin/bash

# ================= 1. 确定目标卡号 =================
if [ $# -eq 0 ]; then
    echo "⚠️ 未传入 NPU 卡号，将默认unset ASCEND_RT_VISIBLE_DEVICES"
    unset ASCEND_RT_VISIBLE_DEVICES
fi

# ================= 2. [新增] 联动清理脚本 =================
# 获取当前启动脚本所在的绝对路径目录
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
CLEANER_PATH="$SCRIPT_DIR/npu-cleaner.sh"

# 检查清理脚本是否存在且具备执行权限 (-x)
if [ -x "$CLEANER_PATH" ]; then
    echo "🧹 [前置任务] 准备释放 NPU ($@) 的资源..."
    # 将当前的参数无缝透传给清理脚本
    "$CLEANER_PATH" "$@"

    # 稍微停顿 1 秒，确保 NPU 驱动层已完全回收 HBM 显存
    sleep 1
    echo "✅ [前置任务] 清理完毕！"
else
    echo "⚠️ 警告：未找到可执行的清理脚本 $CLEANER_PATH，将跳过清理直接启动。"
fi

# ================= 3. 参数转换与环境变量挂载 =================
OLD_IFS="$IFS"
IFS=","
export ASCEND_RT_VISIBLE_DEVICES="$*"
IFS="$OLD_IFS"

echo "🚀 [启动任务] 当前设备可见性: ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"

# A5(Ascend 950) 走 FIA 融合注意力算子 npu_fused_infer_attention_score；
# 否则默认走 ATB SelfAttentionOperation(_npu_flash_attention_qlens)，在 A5 上 CreateOperation 失败、warmup 请求崩溃。
# prefill+decode 一起绕开 ATB（decode 默认走 _npu_paged_attention，同源）。
export ASCEND_USE_FIA=1

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
    # verify 用的 torch.ops.npu.recurrent_gated_delta_rule 只有 bf16 一份实例化
    # （kernel 里是 RGDR<bfloat16_t, bfloat16_t>，host 的 UB 预算也按每元素 2 字节算，
    # 而且没有任何按 dtype 的分发），而 MambaPool 的 ssm_dtype 默认是 float32。
    # 位宽不匹配不会报错：算子按 2 字节步长读写 4 字节的 state，每隔一个元素落在 fp32 的
    # 高半字（正好是它的 bf16 截断），数值看着正常但 state 是错位的，表现为输出流畅但复读。
    export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"
    echo "🔮 [MTP] NEXTN 已开启：num-steps=3, eagle-topk=1, num-draft-tokens=4"
    echo "🔮 [MTP] SGLANG_MAMBA_SSM_DTYPE=$SGLANG_MAMBA_SSM_DTYPE（算子只支持 bf16 state）"
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
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code \
    "${MTP_ARGS[@]}" "${EXTRA_ARGS_ARR[@]}"

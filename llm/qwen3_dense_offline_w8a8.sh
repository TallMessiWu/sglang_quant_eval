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

# ========== 下方是原有的模型启动命令 ==========
sglang serve \
    --model-path /mnt/weight/Qwen3-32B-w8a8 \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code

#!/bin/bash

# ================= 1. 确定目标卡号 =================
if [ $# -eq 0 ]; then
    echo "⚠️ 未传入 NPU 卡号，将默认unset ASCEND_RT_VISIBLE_DEVICES"
    unset ASCEND_RT_VISIBLE_DEVICES
fi

# ================= 2. 联动清理脚本 =================
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
CLEANER_PATH="$SCRIPT_DIR/npu-cleaner.sh"

if [ -x "$CLEANER_PATH" ]; then
    echo "🧹 [前置任务] 准备释放 NPU ($@) 的资源..."
    "$CLEANER_PATH" "$@"
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

# ========== 下方是原有的模型启动命令 ==========
sglang serve \
    --model-path /home/weights/Qwen3-30B-A3B-w8a8_mxfp8-0120-full \
    --host 127.0.0.1 \
    --port $VLLM_PORT \
    --quantization modelslim \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length 5000 \
    --trust-remote-code

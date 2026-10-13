#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch "$@" || exit $?

# A5(Ascend 950) 走 FIA 融合注意力算子 npu_fused_infer_attention_score；
# 否则默认走 ATB SelfAttentionOperation(_npu_flash_attention_qlens)，在 A5 上 CreateOperation 失败、warmup 请求崩溃。
# prefill+decode 一起绕开 ATB（decode 默认走 _npu_paged_attention，同源）。
export ASCEND_USE_FIA=1

# ========== 下方是原有的模型启动命令 ==========
sglang serve \
    --model-path /mnt/share/weight/Qwen3-32B-mxw4a4 \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --device npu \
    --tp 1 \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
# 显式做 A/B 时检查已安装的 SGLang，避免旧版本静默忽略开关。
if [[ -n "${SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT+x}${SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY+x}" ]]; then
    python3 - <<'PY' || exit $?
import os

from sglang.srt.environ import envs

for name in (
    "SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT",
    "SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY",
):
    if name not in os.environ:
        continue
    option = getattr(envs, name, None)
    if option is None:
        raise SystemExit(f"当前 SGLang 不支持 {name}，请先选择对应验证分支并确认安装路径。")
    print(f"{name}={option.get()} (set before server startup)")
PY
fi
npu_prepare_launch 1 "$@" || exit $?

export ASCEND_USE_FIA=1
# NPU GDN/SSM state 默认 BF16，与权重量化格式无关；允许显式覆盖。
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

sglang serve \
    --model-path /mnt/share/weight/Qwen3.6-35B-A3B \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --quantization mxfp_w4a8 \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code

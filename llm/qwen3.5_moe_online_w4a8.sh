#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
# 显式做路由 A/B 时检查已安装的 SGLang，避免旧版本静默忽略开关。
if [[ -n "${SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT+x}" ]]; then
    python3 - <<'PY' || exit $?
from sglang.srt.environ import envs

option = getattr(envs, "SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT", None)
if option is None:
    raise SystemExit("当前 SGLang 不支持路由 A/B 开关，请先更新所选 W4A8 分支并确认安装路径。")
print(f"W4A8 TP routing quantization fusion={option.get()} (resolved at server startup)")
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

#!/bin/bash

# 用法: $0 [-f|--force] [NPU_ID ... | all]
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
# 显式做 A/B 时检查已安装的 SGLang，避免旧版本静默忽略开关。
if [[ -n "${SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT+x}${SGLANG_NPU_W4A8_MOE_UNFUSED_ROUTING_MIN_TOKENS+x}${SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY+x}${SGLANG_NPU_MAMBA_CLEAR_TRACE+x}" ]]; then
    python3 - <<'PY' || exit $?
import os

from sglang.srt.environ import envs

for name in (
    "SGLANG_NPU_W4A8_MOE_FUSE_ROUTING_QUANT",
    "SGLANG_NPU_W4A8_MOE_UNFUSED_ROUTING_MIN_TOKENS",
    "SGLANG_NPU_MAMBA_CLEAR_USE_INDEX_COPY",
    "SGLANG_NPU_MAMBA_CLEAR_TRACE",
):
    if name not in os.environ:
        continue
    option = getattr(envs, name, None)
    if option is None:
        raise SystemExit(f"当前 SGLang 不支持 {name}，请先选择对应验证分支并确认安装路径。")
    if name == "SGLANG_NPU_W4A8_MOE_UNFUSED_ROUTING_MIN_TOKENS":
        try:
            value = int(os.environ[name])
        except ValueError:
            raise SystemExit(f"{name} 必须是非负整数。")
        if value < 0:
            raise SystemExit(f"{name} 必须是非负整数。")
    print(f"{name}={option.get()} (set before server startup)")
PY
fi
npu_prepare_launch 1 "$@" || exit $?

export ASCEND_USE_FIA=1
# NPU GDN/SSM state 默认 BF16，与权重量化格式无关；允许显式覆盖。
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"

# 与 Dense 启动器一致，EXTRA_ARGS 按空白分词，不执行 shell 表达式。
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi

sglang serve \
    --model-path /mnt/share/weight/Qwen3.6-35B-A3B \
    --host 127.0.0.1 \
    --port ${VLLM_PORT:-6969} \
    --quantization mxfp_w4a8 \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code \
    "${EXTRA_ARGS_ARR[@]}"

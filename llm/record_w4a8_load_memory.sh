#!/bin/bash

# 用法: bash ./record_w4a8_load_memory.sh LABEL [NPU_ID ... | all]
# 用户自行选择/安装 SGLang 版本；复用 W4A4 采集器，不改普通启动脚本。
set -o pipefail
if [[ $# -eq 0 || "$1" == -h || "$1" == --help ]]; then
    echo "用法: $0 LABEL [NPU_ID ... | all]"
    echo "示例: $0 w4a8_before 6"
    echo "结果: 当前目录 artifacts/load-memory/LABEL-时间-唯一后缀/"
    exit 0
fi
label=$1
shift
if [[ ! "$label" =~ ^[a-zA-Z][a-zA-Z0-9_.-]*$ ]]; then
    echo "错误：LABEL 只能包含字母、数字、点、下划线和横线，且以字母开头（例如 w4a8_before/w4a8_after）。" >&2
    exit 2
fi
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
source "$SCRIPT_DIR/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

root="$PWD/artifacts/load-memory"
mkdir -p "$root" || exit $?
run_dir=$(mktemp -d "$root/$label-$(date +%Y%m%d-%H%M%S)-XXXXXX") || exit $?
export NPU_LOAD_MEMORY_DIR="$run_dir"
export NPU_LOAD_MEMORY_LABEL="$label"
export ASCEND_USE_FIA=1
export SGLANG_MAMBA_SSM_DTYPE="${SGLANG_MAMBA_SSM_DTYPE:-bfloat16}"
echo "加载显存报告与服务日志: $run_dir"

# 与普通 W4A8 启动入口相同；EXTRA_ARGS 按空白分词，不执行 shell 表达式。
EXTRA_ARGS_ARR=()
if [ -n "${EXTRA_ARGS:-}" ]; then
    read -r -a EXTRA_ARGS_ARR <<< "$EXTRA_ARGS"
fi
python3 "$SCRIPT_DIR/diagnostics/npu_load_memory.py" serve \
    --model-path /mnt/share/weight/Qwen3.6-35B-A3B \
    --host 127.0.0.1 \
    --port "${VLLM_PORT:-6969}" \
    --quantization mxfp_w4a8 \
    --device npu \
    --tp "$NPU_TP_SIZE" \
    --reasoning-parser qwen3 \
    --context-length "${CONTEXT_LENGTH:-131072}" \
    --trust-remote-code \
    "${EXTRA_ARGS_ARR[@]}" 2>&1 | tee "$run_dir/server.log"

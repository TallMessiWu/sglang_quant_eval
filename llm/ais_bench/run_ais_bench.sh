#!/usr/bin/env bash
set -euo pipefail

# AIS Bench 精度评测公共入口；在 llm/ 中调用：
#   bash ais_bench/run_ais_bench.sh <dataset> [AIS Bench 附加参数...]
# gsm8k.sh / gpqa.sh / mmmu.sh 都走这里。
#
# 服务：VLLM_IP=localhost、VLLM_PORT=6969；VLLM_URL 非空时优先。
# MODEL_NAME 不给则由服务的 /v1/models 探测。
# AIS_MODEL_CFG 默认 vllm_api_general_chat。
#
# AIS_SAMPLING_PRESET 默认 thinking，可选：
#   thinking                 通用任务思考：T=1.0, top_p=0.95, presence_penalty=1.5
#   thinking-coding          精确编码思考：T=0.6, top_p=0.95, presence_penalty=0.0
#   non-thinking             通用非思考：  T=0.7, top_p=0.8,  presence_penalty=1.5
#   non-thinking-reasoning   推理任务非思考：与 thinking 的采样参数相同
#   qwen35-397b-thinking      Qwen3.5 397B 思考：与 thinking-coding 相同
#   template                 保留模板采样参数（用于其他模型/框架）
# 除 template 外均为 top_k=20, min_p=0.0, repetition_penalty=1.0。
# 397B 非思考用 non-thinking；预设不负责开关服务端 thinking。
#
# 单项覆盖（优先于预设/数据集默认）：
#   AIS_MAX_OUT_LEN           GPQA/MMMU 默认 32768，GSM8K 默认 4096，其余保留模板
#   AIS_TEMPERATURE / AIS_TOP_P / AIS_TOP_K / AIS_MIN_P
#   AIS_PRESENCE_PENALTY / AIS_REPETITION_PENALTY
#
# 新旧 AIS Bench 统一生成配置，避免新版 CLI 路径遗漏采样参数。
# 值在生成器中解析为字面量，不在 mmengine lazy-import 配置里读环境变量。
# 每次运行在当前目录 .ais_bench_configs/ 下创建独立目录；并发评测不互相覆盖，
# 不修改 site-packages，保留模板的后处理和其他参数。
# 数据集之后的参数仍原样透传，例如 --work-dir、--batch-size、--debug；
# AIS Bench 自身支持的 CLI 覆盖仍可能覆盖生成配置，以最终运行配置为准。

if [[ $# -lt 1 ]]; then
    echo "用法: $0 <dataset> [ais_bench 附加参数...]" >&2
    exit 2
fi

dataset="$1"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v ais_bench >/dev/null 2>&1; then
    echo "RED: 找不到 ais_bench，确认它在 PATH 里" >&2
    exit 127
fi

# 优先用 AIS Bench 自己的解释器；/usr/bin/env python3 等 shebang 走 PATH 兜底。
ais_python="$(head -1 "$(command -v ais_bench)" | sed 's|^#!||' | awk '{print $1}')"
if [[ ! -x "$ais_python" ]] || ! "$ais_python" -c "import ais_bench" >/dev/null 2>&1; then
    ais_python=python3
fi

template="${AIS_MODEL_CFG:-vllm_api_general_chat}"
template="${template%.py}"
mkdir -p "$PWD/.ais_bench_configs"
cfg_dir="$(mktemp -d "$PWD/.ais_bench_configs/run.XXXXXXXX")"
cfg_name="sglang_eval"
gen_args=(--out-dir "$cfg_dir" --name "$cfg_name" --template "$template"
          --dataset "$dataset" --sampling-preset "${AIS_SAMPLING_PRESET:-thinking}")

if [[ -n "${VLLM_URL:-}" ]]; then
    endpoint="$VLLM_URL"
    gen_args+=(--url "$VLLM_URL")
else
    endpoint="${VLLM_IP:-localhost}:${VLLM_PORT:-6969}"
    gen_args+=(--host-ip "${VLLM_IP:-localhost}" --host-port "${VLLM_PORT:-6969}")
fi
if [[ -n "${MODEL_NAME:-}" ]]; then
    gen_args+=(--model-name "$MODEL_NAME")
fi
for field in MAX_OUT_LEN TEMPERATURE TOP_P TOP_K MIN_P PRESENCE_PENALTY REPETITION_PENALTY; do
    var="AIS_$field"
    if [[ -n "${!var:-}" ]]; then
        flag="${field,,}"
        gen_args+=("--${flag//_/-}=${!var}")
    fi
done

generated="$("$ais_python" "$script_dir/gen_ais_bench_model_cfg.py" "${gen_args[@]}")"
echo "[ais_bench] dataset=$dataset endpoint=$endpoint config=$generated" >&2
exec ais_bench --datasets "$dataset" --dump-eval-details \
    --models "$cfg_name" --config-dir "$cfg_dir" "$@"

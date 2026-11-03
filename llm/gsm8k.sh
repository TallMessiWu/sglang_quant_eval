#!/usr/bin/env bash
set -euo pipefail

# GSM8K 精度评测，跑在已经起好的服务上（本仓是 SGLang，OpenAI 兼容接口）。
#
# 用环境变量指定服务，不用复制 ais_bench 的 config 文件（为什么不能在 config 里
# 读环境变量、这些开关又是怎么生效的，见 ais_bench/run_ais_bench.sh 顶部注释）：
#
#   VLLM_IP         服务 IP 或主机名，默认 localhost
#   VLLM_PORT       服务端口，默认 6969
#   VLLM_URL        完整 URL，给了它就忽略 VLLM_IP / VLLM_PORT
#   MODEL_NAME      模型名，不给则由服务的 /v1/models 自行探测
#   AIS_MODEL_CFG   ais_bench 自带的模型配置模板，默认 vllm_api_general_chat
#   AIS_MAX_OUT_LEN 默认 4096，可覆盖
#   AIS_SAMPLING_PRESET 默认 thinking（与 GPQA 相同，仅输出长度不同）
#   AIS_TEMPERATURE / AIS_TOP_P / AIS_TOP_K / AIS_MIN_P /
#   AIS_PRESENCE_PENALTY / AIS_REPETITION_PENALTY 可逐项覆盖预设。
#   全部预设见 ais_bench/run_ais_bench.sh；只调采样，不开关 thinking。
#
# 脚本名之后的参数原样透传给 ais_bench，例如 --work-dir、--batch-size、--debug。
#
#   ./gsm8k.sh
#   VLLM_PORT=7969 ./gsm8k.sh                        # 同机的另一个服务
#   VLLM_IP=10.0.0.5 VLLM_PORT=8000 ./gsm8k.sh       # 别的机器上的服务
#   VLLM_URL=http://gw.example/prefix/ ./gsm8k.sh    # 带路径的网关
#   ./gsm8k.sh --batch-size 16
#
#   AIS_MAX_OUT_LEN=8192 bash ./gsm8k.sh
#   AIS_SAMPLING_PRESET=non-thinking-reasoning bash ./gsm8k.sh
#
# 新旧版均在当前目录 .ais_bench_configs/ 下生成独立配置，不修改已安装的模板。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$script_dir/ais_bench/run_ais_bench.sh" gsm8k_gen_0_shot_cot_chat_prompt "$@"

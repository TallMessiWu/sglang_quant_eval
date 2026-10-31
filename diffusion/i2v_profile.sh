#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        '用法: MODEL_PATH=/path/to/model bash ./i2v_profile.sh bf16|mxfp8 [-f|--force] [NPU_ID ... | all]' \
        'BF16 必须指定原始 Wan2.2-I2V-A14B-Diffusers 权重；MXFP8 默认使用现成 MXFP8-G2。' \
        'PROFILE_STEPS=5: 请求采集前段短窗口；设为 -1 覆盖整个去噪过程（仍遵循原生 profiler warmup）。' \
        '两组均使用 gyro.jpg、1024x720、81 帧、40 步、seed=42 和相同 CPU offload 设置。' \
        '输出: 当前目录/outputs_i2v_profile/<模式>-<唯一运行目录>/{run.log,traces/,video.mp4}' \
        '请先手动安装 #45550 分支；本脚本不切换分支、不修改 PYTHONPATH。' \
        '短窗口不覆盖所有高/低噪声专家；profile 耗时不能当作无 profiler 的性能结果。'
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
    bf16|mxfp8) mode=$1; shift ;;
    *) usage >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
profile_steps=${PROFILE_STEPS:-5}
if [[ ! "$profile_steps" =~ ^(-1|[1-9]|[1-3][0-9]|40)$ ]]; then
    echo '错误：PROFILE_STEPS 必须为 1..40 或 -1。' >&2
    exit 2
fi
if [[ "$mode" == bf16 ]]; then
    model_path=${MODEL_PATH:?BF16 模式需要 MODEL_PATH 指向原始 I2V A14B 权重}
    export SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION=0
else
    model_path=${MODEL_PATH:-/mnt/share/weight/Wan2.2-I2V-A14B-Diffusers-MXFP8-G2}
    export SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION=1
fi
if [[ ! -d "$model_path" ]]; then
    echo "错误：本地模型目录不存在：$model_path" >&2
    exit 2
fi
if [[ "$mode" == bf16 ]]; then
    for component in '' /transformer /transformer_2; do
        if [[ -f "$model_path$component/quant_model_description.json" ]]; then
            echo "错误：BF16 对照不能使用 ModelSlim 量化 checkpoint：$model_path" >&2
            exit 2
        fi
    done
fi

output_root="$PWD/outputs_i2v_profile"
mkdir -p "$output_root"
output_dir=$(mktemp -d "$output_root/${mode}-$(date +%Y%m%d-%H%M%S)-XXXXXX")
export SGLANG_DIFFUSION_TORCH_PROFILER_DIR="$output_dir/traces"
# 与普通 I2V 入口一致，将临时文件与缓存留在执行目录内；两组共享下载缓存。
export TMPDIR="$output_dir/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$output_root/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export SGLANG_DIFFUSION_CACHE_ROOT="$XDG_CACHE_HOME/sgl_diffusion"
export ASCEND_WORK_PATH="$output_dir/ascend"
export ASCEND_CACHE_PATH="$XDG_CACHE_HOME/ascend"
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
# 不让调用方遗留的 Cache-DiT 开关跳过待采集的 DiT 计算。
export SGLANG_CACHE_DIT_ENABLED=0
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$SGLANG_DIFFUSION_TORCH_PROFILER_DIR" "$ASCEND_WORK_PATH" "$ASCEND_CACHE_PATH"

source "$script_dir/../llm/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

# 不传 attention backend 或 quantization：离线 scheme 自动检测，FA 由 #45550 自动选择。
# 保留原 I2V 对齐尺寸；两组都固定 offload，避免自动显存策略改变对照条件。
command=(sglang generate
    --model-path "$model_path"
    --image-path "$script_dir/gyro.jpg"
    --prompt 'A man slowly blinks, his hair and clothes move gently in the wind, raindrops fall, subtle natural motion, stable camera.'
    --height 720 --width 1024 --num-gpus "$NPU_TP_SIZE"
    --num-frames 81 --num-inference-steps 40
    --guidance-scale 5.0 --fps 24 --seed 42
    --dit-cpu-offload true --text-encoder-cpu-offload true
    --image-encoder-cpu-offload true --vae-cpu-offload false
    --warmup-mode request --warmup-steps 2
    --profile --num-profiled-timesteps "$profile_steps"
    --output-path "$output_dir" --output-file-name video.mp4
)
{
    echo "[I2V profile] mode=$mode; model=$model_path; profile_steps=$profile_steps"
    echo "[I2V profile] devices=$ASCEND_RT_VISIBLE_DEVICES; MXFP8_attention=$SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION"
    echo "[I2V profile] traces=$SGLANG_DIFFUSION_TORCH_PROFILER_DIR; executable=$(command -v sglang)"
    printf '[I2V profile] command: '; printf '%q ' "${command[@]}"; printf '\n'
    "${command[@]}"
} 2>&1 | tee "$output_dir/run.log"

if [[ "$mode" == mxfp8 ]] && ! grep -Fq 'Using MXFP8 quantized Ascend Flash Attention.' "$output_dir/run.log"; then
    echo "错误：未看到运行时 MXFP8 Attention 生效日志；不能将 fallback 当作 #45550 采集成功。检查 $output_dir/run.log" >&2
    exit 1
fi
trace=$(find "$SGLANG_DIFFUSION_TORCH_PROFILER_DIR" -type f -name trace_view.json -size +0c -print -quit)
if [[ -z "$trace" ]]; then
    echo "错误：没有找到非空 NPU trace_view.json；检查 $output_dir/run.log 和 traces/ 下的 profiler 输出。" >&2
    exit 1
fi
echo "采集结束：$output_dir"
echo "NPU 时间线：$trace（多卡时保留整个 traces/ 目录）"
echo '注意：短窗口和开启 profiler 的耗时不构成全模型性能或精度结论。'

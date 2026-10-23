#!/usr/bin/env bash
set -euo pipefail

# 用法: $0 [-f|--force] [NPU_ID ... | all]
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# 两步 MXFP8 Attention 路径验证，单独保存，避免覆盖原来的 40 步视频和混用日志。
output_dir="$PWD/outputs_i2v_mxfp8_fa_check"

# Keep generated video, logs, temporary files and caches in the current directory.
export TMPDIR="$output_dir/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$output_dir/cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$TORCH_HOME/extensions"
export TORCHINDUCTOR_CACHE_DIR="$TORCH_HOME/inductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export SGLANG_DIFFUSION_CACHE_ROOT="$XDG_CACHE_HOME/sgl_diffusion"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec > >(tee -a "$output_dir/run.log") 2>&1

source "$script_dir/../llm/common/npu-launch.sh" || exit $?
npu_prepare_launch 1 "$@" || exit $?

export SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION="${SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION:-1}"
# 不指定 backend：验证自动选择。当前 gyro.jpg 按 1024x720 的目标面积调整为
# 704x1024，81 帧对应 59136 tokens，可被 64 整除；换图片后需重新核对。
# 请求尺寸不在模型推荐列表，会有警告；2 步仅验证路径，不用于画质评测。
echo "[MXFP8 FA check] SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION=$SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION; steps=2"
echo "[MXFP8 FA check] 检查运行时 MXFP8 专用日志；仅选到 fa 后端不算生效。"

sglang generate \
  --model-path /mnt/share/weight/Wan2.2-I2V-A14B-Diffusers-MXFP8-G2 \
  --image-path "$script_dir/gyro.jpg" \
  --prompt "A man slowly blinks, his hair and clothes move gently in the wind, raindrops fall, subtle natural motion, stable camera." \
  --height 720 --width 1024 --num-gpus "$NPU_TP_SIZE" \
  --num-frames 81 --num-inference-steps 2 \
  --guidance-scale 5.0 --fps 24 --seed 42 \
  --output-path "$output_dir" --output-file-name wan22_i2v_mxfp8.mp4

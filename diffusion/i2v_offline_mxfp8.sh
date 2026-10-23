#!/usr/bin/env bash
set -euo pipefail

# 用法: $0 [-f|--force] [NPU_ID ... | all]
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output_dir="$PWD/outputs_i2v_mxfp8"

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

sglang generate \
  --model-path /mnt/share/weight/Wan2.2-I2V-A14B-Diffusers-MXFP8-G2 \
  --image-path "$script_dir/gyro.jpg" \
  --prompt "A man slowly blinks, his hair and clothes move gently in the wind, raindrops fall, subtle natural motion, stable camera." \
  --height 720 --width 1280 --num-gpus "$NPU_TP_SIZE" \
  --num-frames 81 --num-inference-steps 40 \
  --guidance-scale 5.0 --fps 24 --seed 42 \
  --output-path "$output_dir" --output-file-name wan22_i2v_mxfp8.mp4

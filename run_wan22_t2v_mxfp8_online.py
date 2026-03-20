"""
Wan2.2 T2V (Text-to-Video) Online MXFP8 量化推理脚本
使用 SGLang DiffGenerator API + 在线 MXFP8 量化

无需预量化权重 — 从原始 FP16/BF16 模型在加载时自动量化为 MXFP8，
利用 torch_npu.npu_dynamic_mx_quant + npu_quant_matmul 进行推理。

要求:
- Ascend NPU (Atlas 800I A2/A3)
- CANN >= 8.0.RC3 (支持 npu_dynamic_mx_quant)
- torch_npu 已安装
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch_npu.npu.set_compile_mode(jit_compile=False)
if bool(os.environ.get("USE_NZ", 0)):
    torch.npu.config.allow_internal_format = True
else:
    torch.npu.config.allow_internal_format = False


def cleanup_scheduler_processes():
    """强制清理任何遗留的 scheduler 工作进程（解决 graceful shutdown 延迟问题）"""
    try:
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *sglang*diffusion*"],
            capture_output=True,
            timeout=3
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 T2V Online MXFP8 量化视频生成")
    parser.add_argument("--model-path", type=str, default="/home/weights/Wan2.2-T2V-5B-Diffusers",
                        help="原始 FP16/BF16 模型路径（无需预量化）")
    parser.add_argument("--prompt", type=str, default=(
            "一只小猫在花园里追蝴蝶，阳光明媚，花朵盛开，"
            "镜头缓慢跟随小猫移动，画面清晰稳定，自然光照，"
            "高清画质，24帧"
        ), help="提示词")
    parser.add_argument("--num-gpus", type=int, default=2,
                        help="使用的 NPU 数量")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="生成帧数 (必须满足 (n-1) %% 4 == 0，如 81, 121)")
    parser.add_argument("--height", type=int, default=480,
                        help="视频高度")
    parser.add_argument("--width", type=int, default=832,
                        help="视频宽度")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="引导强度")
    parser.add_argument("--fps", type=int, default=24,
                        help="输出视频帧率")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str, default="./outputs_t2v_mxfp8_online",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    print(f"模型路径: {args.model_path}")
    print(f"量化方式: Online MXFP8")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"帧数: {args.num_frames}, FPS: {args.fps}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"NPU 数量: {args.num_gpus}")
    print(f"种子: {args.seed}")
    print()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    print("正在加载模型（Online MXFP8 量化）...")
    gen = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        output_path=args.output_dir,
        quantization="mxfp8",
    )

    print("正在生成视频...")
    gen.generate(
        sampling_params_kwargs={
            "prompt": args.prompt,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
        }
    )

    gen.shutdown()
    cleanup_scheduler_processes()

    print("完成!")


if __name__ == "__main__":
    main()

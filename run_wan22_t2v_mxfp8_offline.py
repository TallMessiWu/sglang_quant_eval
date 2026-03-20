"""
Wan2.2 T2V (Text-to-Video) Offline MXFP8 (msmodelslim pre-quantized) 推理脚本
使用 SGLang DiffGenerator API

使用 repack_wan22_ti2v_mxfp8.py 预处理后的模型目录，
SGLang 自动检测 quant_model_description.json 并启用 modelslim 量化。

要求:
- Ascend NPU (Atlas 800I A2/A3)
- CANN >= 8.0.RC3
- torch_npu 已安装
- 模型已通过 repack 脚本预处理
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
    parser = argparse.ArgumentParser(description="Wan2.2 T2V Offline MXFP8 量化视频生成")
    parser.add_argument("--model-path", type=str,
                        default="/home/weights/Wan2.2-T2V-5B-Diffusers-MXFP8",
                        help="Repacked MXFP8 模型路径（repack 脚本的输出）")
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
    parser.add_argument("--output-dir", type=str, default="./outputs_t2v_mxfp8_offline",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    print(f"模型路径: {args.model_path}")
    print(f"量化方式: Offline MXFP8 (msmodelslim pre-quantized)")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"帧数: {args.num_frames}, FPS: {args.fps}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"NPU 数量: {args.num_gpus}")
    print(f"种子: {args.seed}")
    print()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    # 无需指定 quantization — SGLang 自动从 quant_model_description.json 检测 modelslim
    print("正在加载模型（Offline MXFP8 量化）...")
    gen = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        output_path=args.output_dir,
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

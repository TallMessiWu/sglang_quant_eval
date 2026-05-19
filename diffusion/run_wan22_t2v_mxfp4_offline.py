"""
Wan2.2 T2V (Text-to-Video) MXFP4 离线预量化加载推理脚本
使用 SGLang DiffGenerator API + 策略 B (离线 MXFP4 预量化加载)

从 msmodelslim 导出的预量化 MXFP4 权重直接加载推理，
利用 torch_npu.npu_dynamic_dual_level_mx_quant + npu_dual_level_quant_matmul 进行双级量化推理。

要求:
- Ascend NPU (Atlas 800I A2/A3)
- CANN >= 8.0.RC3 (支持 npu_dynamic_dual_level_mx_quant, npu_dual_level_quant_matmul)
- torch_npu 已安装
- msmodelslim 预量化的 Wan2.2 模型权重
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
torch.npu.set_device(int(os.environ.get("ASCEND_DEVICE_ID", 0)))


def cleanup_scheduler_processes():
    """强制清理任何遗留的 scheduler 工作进程（解决 graceful shutdown 延迟问题）"""
    try:
        # 查找并杀死所有 sglang-diffusionWorker 进程
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *sglang*diffusion*"],
            capture_output=True,
            timeout=3
        )
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 T2V MXFP4 离线预量化加载视频生成")
    parser.add_argument("--model-path", type=str,
                        default="/home/weights/Wan2.2-T2V-A14B-Diffusers-MXFP4",
                        help="msmodelslim 预量化的 MXFP4 模型路径（含 quant_model_description.json）")
    parser.add_argument("--prompt", type=str, default=(
            "Two anthropomorphic cats in comfy boxing gear and bright gloves "
            "fight intensely on a spotlighted stage."
        ), help="提示词")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="使用的 NPU 数量")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="生成帧数 (必须满足 (n-1) %% 4 == 0，如 81, 121)")
    parser.add_argument("--height", type=int, default=704,
                        help="视频高度")
    parser.add_argument("--width", type=int, default=1280,
                        help="视频宽度")
    parser.add_argument("--num-inference-steps", type=int, default=40,
                        help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="引导强度")
    parser.add_argument("--fps", type=int, default=24,
                        help="输出视频帧率")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str, default="./outputs_t2v_mxfp4_offline",
                        help="输出目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    # 检查是否存在 quant_model_description.json（标志预量化模型）
    # repack 后 description 在 transformer/ 子目录中
    quant_desc_candidates = [
        os.path.join(args.model_path, "transformer", "quant_model_description.json"),
        os.path.join(args.model_path, "quant_model_description.json"),
    ]
    if not any(os.path.exists(p) for p in quant_desc_candidates):
        print(f"警告: 未找到 quant_model_description.json，模型可能不是预量化版本")
        print(f"    已检查: {quant_desc_candidates}")

    print(f"模型路径: {args.model_path}")
    print(f"量化方式: Offline MXFP4 (策略 B，预量化权重加载)")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"帧数: {args.num_frames}, FPS: {args.fps}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"NPU 数量: {args.num_gpus}")
    print(f"种子: {args.seed}")
    print()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    # 初始化生成器
    # quantization="modelslim" 使系统自动从 quant_model_description.json 检测量化类型
    # 若检测到 W4A4_MXFP4，则使用 ModelSlimMXFP4Scheme 加载预量化权重
    print("正在加载模型（MXFP4 离线预量化权重）...")
    gen = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        output_path=args.output_dir,
        quantization="modelslim",
    )

    # 生成视频
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

    # 强制清理遗留的 scheduler 进程（修复 graceful shutdown 延迟问题）
    cleanup_scheduler_processes()

    print("完成!")


if __name__ == "__main__":
    main()

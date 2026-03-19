"""
Wan2.2 TI2V (Text-Image-to-Video) Online MXFP8 量化推理脚本 (带 Profiling)
使用 SGLang DiffGenerator API + 路径 B (在线 MXFP8 量化)

无需预量化权重 — 从原始 FP16/BF16 模型在加载时自动量化为 MXFP8，
利用 torch_npu.npu_dynamic_mx_quant + npu_quant_matmul 进行推理。

Profiling 功能:
- 使用 torch_npu.profiler 捕获 NPU 算子级别的执行统计
- 过滤并统计 npu_dynamic_mx_quant / npu_quant_matmul 调用次数以确认 MXFP8 生效
- 导出 Chrome Trace 文件供 chrome://tracing 可视化分析
- 使用 --profile 开启（默认关闭，因为 profiling 有额外开销）

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
        # 查找并杀死所有 sglang-diffusionWorker 进程
        result = subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq *sglang*diffusion*"],
            capture_output=True,
            timeout=3
        )
    except Exception:
        pass


def _print_mxfp8_stats(key_averages):
    """从 profiler 结果中过滤并打印 MXFP8 相关算子统计。"""
    # 关键算子名称（Ascend CANN kernel 命名可能含下划线或驼峰，做 lower-case 模糊匹配）
    MXFP8_KEYWORDS = ["dynamic_mx_quant", "quant_matmul", "npu_quant", "mxfp8"]

    mxfp8_events = [
        e for e in key_averages
        if any(kw in e.key.lower() for kw in MXFP8_KEYWORDS)
    ]

    print("\n" + "=" * 70)
    print("MXFP8 量化算子统计")
    print("=" * 70)

    if not mxfp8_events:
        print("[警告] 未检测到任何 MXFP8 相关算子！")
        print("  可能原因：")
        print("  1. quantization='mxfp8' 参数未生效，实际仍在跑 FP16/BF16")
        print("  2. Worker 进程的 NPU ops 未被当前进程的 profiler 捕获")
        print("  3. Kernel 名称与预期不符（请检查 Chrome Trace 文件中的算子名）")
    else:
        print(f"共检测到 {len(mxfp8_events)} 种 MXFP8 相关算子：\n")
        print(f"  {'算子名':<45} {'调用次数':>8} {'NPU时间(ms)':>14}")
        print(f"  {'-'*45} {'-'*8} {'-'*14}")
        for e in sorted(mxfp8_events, key=lambda x: x.self_device_time_total, reverse=True):
            npu_ms = e.self_device_time_total / 1000  # us -> ms
            print(f"  {e.key:<45} {e.count:>8} {npu_ms:>14.3f}")
        print(f"\n  [OK] MXFP8 量化已确认生效（检测到 npu_dynamic_mx_quant / npu_quant_matmul 调用）")
    print("=" * 70 + "\n")


def _run_with_profiling(gen, sampling_params, args):
    """使用 torch_npu.profiler 包裹推理，捕获 NPU 算子统计。"""
    profile_dir = os.path.join(args.output_dir, "profiling")
    os.makedirs(profile_dir, exist_ok=True)
    trace_path = os.path.join(profile_dir, "mxfp8_trace.json")

    print(f"[Profiling] 开启 NPU profiling（前 {args.profile_steps} 步）...")
    print(f"[Profiling] Chrome Trace 将保存至: {trace_path}")
    print("正在生成视频（Profiling 模式）...")

    # 注意：若推理在子进程中执行，profiler 在主进程中仅能捕获主进程提交的 NPU ops。
    # torch_npu.profiler 在 CANN 层面通过设备级时间线捕获，通常可覆盖跨进程的 NPU 调度。
    activities = [
        torch_npu.profiler.ProfilerActivity.CPU,
        torch_npu.profiler.ProfilerActivity.NPU,
    ]

    # schedule: wait=0, warmup=1, active=profile_steps, repeat=1
    # warmup=1 跳过第一步的冷启动开销，active=profile_steps 只收集指定步数
    schedule = torch_npu.profiler.schedule(
        wait=0,
        warmup=1,
        active=max(1, args.profile_steps),
        repeat=1,
    )

    with torch_npu.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        ),
    ) as prof:
        gen.generate(sampling_params_kwargs=sampling_params)
        prof.step()  # 确保最后一批数据被刷入

    # 导出 Chrome Trace
    prof.export_chrome_trace(trace_path)
    print(f"[Profiling] Chrome Trace 已保存: {trace_path}")

    # 打印 Top-30 NPU 算子（按 NPU 时间排序）
    print("\n[Profiling] Top-30 NPU 算子（按 NPU 执行时间排序）:")
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=30))

    # 打印 MXFP8 专项统计
    _print_mxfp8_stats(prof.key_averages())


def main():
    parser = argparse.ArgumentParser(description="Wan2.2 TI2V Online MXFP8 量化视频生成")
    parser.add_argument("--model-path", type=str, default="/home/weights/Wan2.2-TI2V-5B-Diffusers",
                        help="原始 FP16/BF16 模型路径（无需预量化）")
    parser.add_argument("--image-path", type=str, default="gyro.jpg",
                        help="输入图片路径")
    parser.add_argument("--prompt", type=str, default=(
            "杰作，最高画质，8K，超高细节，官方原画，荒木飞吕彦画风，JOJO的奇妙冒险画风，"
            "单人男性，杰洛·齐贝林，JOJO的奇妙冒险第七部飙马野郎，帅气男性，银色长发，"
            "紫色眼眸，绿色嘴唇，标志性宽边牛仔帽，帽子带有紫色铁球装饰，双手扶着帽檐，"
            "紫蓝色骑马制服，胸前银色蜻蜓胸针，帽子和衣服上覆盖积雪与霜冻，户外雨夹雪场景，"
            "下落的雨滴与动态雨丝，模糊的绿色自然背景，柔和电影级打光，冷色调，厚涂质感，"
            "自然的微动态，缓慢眨眼，呼吸带来的胸腔轻微起伏，头发和衣服被风轻轻吹动，"
            "雨滴动态下落，镜头缓慢轻微推近，动作丝滑，画面稳定无抖动，24帧"
        ), help="提示词")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="使用的 NPU 数量")
    parser.add_argument("--num-frames", type=int, default=81,
                        help="生成帧数 (必须满足 (n-1) %% 4 == 0，如 81, 121)")
    parser.add_argument("--height", type=int, default=704,
                        help="视频高度")
    parser.add_argument("--width", type=int, default=1280,
                        help="视频宽度")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="推理步数")
    parser.add_argument("--guidance-scale", type=float, default=5.0,
                        help="引导强度")
    parser.add_argument("--fps", type=int, default=24,
                        help="输出视频帧率")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--output-dir", type=str, default="./outputs_mxfp8_online",
                        help="输出目录")
    parser.add_argument("--profile-steps", type=int, default=3,
                        help="只对前 N 个推理步骤做 profiling（-1 表示全部步骤）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 校验输入
    image_path = str(Path(args.image_path).resolve())
    if not os.path.exists(image_path):
        print(f"错误: 输入图片不存在: {image_path}")
        return

    if not os.path.isdir(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return

    print(f"模型路径: {args.model_path}")
    print(f"量化方式: Online MXFP8 (路径 B)")
    print(f"输入图片: {image_path}")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"帧数: {args.num_frames}, FPS: {args.fps}")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"NPU 数量: {args.num_gpus}")
    print(f"种子: {args.seed}")
    print()

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

    # 初始化生成器
    # quantization="mxfp8" 使 transformer 的所有 Linear 层自动在线量化为 MXFP8
    print("正在加载模型（Online MXFP8 量化）...")
    gen = DiffGenerator.from_pretrained(
        model_path=args.model_path,
        num_gpus=args.num_gpus,
        output_path=args.output_dir,
        quantization="mxfp8",
    )

    sampling_params = {
        "prompt": args.prompt,
        "image_path": image_path,
        "num_frames": args.num_frames,
        "fps": args.fps,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
    }

    # 生成视频（始终开启 profiling）
    _run_with_profiling(gen, sampling_params, args)

    gen.shutdown()

    # 强制清理遗留的 scheduler 进程（修复 graceful shutdown 延迟问题）
    cleanup_scheduler_processes()

    print("完成!")


if __name__ == "__main__":
    main()

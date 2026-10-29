# Diffusion 启动入口

在 `diffusion/` 运行，使用用户已安装的目标 SGLang 环境；脚本不 fetch、切分支、改 PYTHONPATH 或自动打补丁。

```bash
bash ./ti2v_bf16.sh 0
bash ./ti2v_online_mxfp8.sh 0
bash ./i2v_offline_mxfp8.sh 0
```

| 入口 | 默认模型 / 用途 |
| --- | --- |
| `t2v_bf16.sh`、`t2v_online_mxfp4.sh` | 原始 Wan2.2-T2V-A14B-Diffusers，BF16 / 在线 NPU FP4 |
| `t2v_offline_mxfp4.sh` | T2V 预量化 MXFP4 checkpoint |
| `ti2v_bf16.sh`、`ti2v_online_mxfp4.sh`、`ti2v_online_mxfp8.sh` | 原始 Wan2.2-TI2V-5B-Diffusers |
| `ti2v_offline_mxfp4.sh`、`ti2v_offline_mxfp8.sh`、`ti2v_int8.sh` | 对应预量化 TI2V checkpoint |
| `i2v_offline_mxfp8.sh` | 现成 Wan2.2-I2V-A14B-Diffusers-MXFP8-G2，自动 MXFP8 FA 验证 |

路径、prompt、尺寸和步数直接写在各脚本中，修改前核对实际权重。在线必须用原始 BF16/FP16 权重；离线以描述自动检测 scheme。TI2V BF16 与 I2V 入口传入 `gyro.jpg`；其余现有 TI2V 入口未传图片，是该模型的文本生成模式，不能与图片输入结果直接比较。

所有入口复用 `../llm/common/npu-launch.sh`：显式卡号数量传给 `--num-gpus`，不硬设 LLM TP；`all` 使用全部卡，不传默认 1 卡且不清理。占用时 Enter 清理、P 保留、R 换空闲卡、Esc 取消；详见 [设备规则](../llm/README.md)。不要脱离主仓单独复制 shell 文件。

I2V 脚本将视频、日志和缓存写入执行目录的 `outputs_i2v_mxfp8_fa_check/`，当前请求 1024×720、81 帧、40 步。默认开启 `SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION=1`，不强制后端；需要看到运行时 `Using MXFP8 quantized Ascend Flash Attention.`，fallback 生成成功不算 MXFP8 FA 验收。换图片后检查缩放后的 token 对齐。

普通脚本的 `--warmup` 和实际输出以命令为准；运行成功、视频观感、数值与性能不是同一结论。原 G2 已报告场景及新 head 未测范围见 [验收计划](../docs/mxfp-refactoring-plan.md)。本轮将 TI2V BF16 改为原生 `sglang generate` 并删除重复 Python 包装器；只做 CPU/mock 参数回归，尚无此入口迁移后的 NPU 生成结果。

旧的未导出 Cache-DiT shell 赋值已移除，不自动启用缓存；需要时由用户显式配置环境。旧 PR 草稿和 Python `DiffGenerator` 包装器已删除，历史从 Git 取回，不作为部署前置。`diffusion_quant_descriptions/` 只是格式样例，不能代替 checkpoint 自己的完整描述。

# sgl-kernel-npu 构建与 target wheel

本文记录当前开发 fork、上游同步方式，以及 910/950 target wheel 的构建和验证边界。实时分支与 PR 见 [branches.md](branches.md)。Agent 执行此类任务时使用 [`$sgl-kernel-npu-dev`](../.agents/skills/sgl-kernel-npu-dev/SKILL.md) skill。

## 仓库与远程

```text
origin   https://github.com/TallMessiWu/sgl-kernel-npu.git
upstream https://github.com/sgl-project/sgl-kernel-npu.git
```

主仓把 `sgl-kernel-npu/` 作为子模块跟踪，`.gitmodules` URL 指向开发 fork；官方更新从子模块内的 `upstream` 获取。

首次初始化：

```bash
git submodule update --init --recursive sgl-kernel-npu
git -C sgl-kernel-npu remote add upstream https://github.com/sgl-project/sgl-kernel-npu.git
```

如果 `upstream` 已存在，不要重复添加；先用 `git -C sgl-kernel-npu remote -v` 核对。

## 当前 Gemma RMSNorm 架构

[sgl-kernel-npu #638](https://github.com/sgl-project/sgl-kernel-npu/pull/638) 与 [SGLang #32745](https://github.com/sgl-project/sglang/pull/32745) 配对：

- SGLang 始终调用 `sgl_kernel_npu.norm.gemma_rmsnorm.npu_gemma_rms_norm`。
- 910 wheel（A2/A3）只安装 native Gemma ACLNN provider。
- 950 wheel（A5）只安装基于 `npu_rms_norm(input, 1 + weight, eps)` 的 ACLNN provider。
- provider 在 setuptools `build_py` 的 staging 目录选择；最终 `build/lib`、wheel、`site-packages` 只含目标实现，不做运行时 SoC 分支。
- ordinary 与 residual API 保持稳定；residual `add_gemma_rms_norm` 使用 `npu_add_rms_norm(..., 1 + weight, eps)`。

逻辑 wheel target 与底层编译 target 分离：950 wheel 的 Python provider 是 950 版本，但主 C++ kernel bundle 仍可能使用 `Ascend910_9382` 兼容 CMake target。

## 构建

先看当前分支帮助，避免沿用旧命令：

```bash
cd sgl-kernel-npu
bash build.sh -h
```

只构建 `sgl_kernel_npu` wheel：

```bash
# 在有 NPU 的机器上自动检测 A2/A3/A5
bash build.sh -a kernels

# 显式目标
bash build.sh -a kernels 910    # A2/A3 native Gemma provider
bash build.sh -a kernels 950    # A5 ACLNN Gemma provider
```

支持的常用别名：

| 逻辑平台 | 常用输入 | wheel provider |
| --- | --- | --- |
| A2 | `910B` / `Ascend910B1` | 910 native |
| A3 | `910` / `910C` / `Ascend910_9382` | 910 native |
| A5 | `950` / `Ascend950` / `Ascend950PR_*` / `Ascend950DT_*` | 950 ACLNN |

不传目标时使用 `npu-smi` 检测；无设备环境回落到 `Ascend910_9382`。为 A5 发布或验收时建议显式传 `950`，并核对构建日志中的 `Wheel SOC_VERSION: Ascend950`。

产物位于 `output/`：

```bash
ls output/sgl_kernel_npu-*.whl
pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
```

## 验证

验证必须覆盖最终 staged/installed artifact，不能只 import 源码树：

```bash
# PR #638 的 provider/staging 单测
python -m pytest \
  tests/python/sgl_kernel_npu/test_build_targets.py \
  tests/python/sgl_kernel_npu/test_gemma_rmsnorm_provider.py -q

# 安装后稳定 API
python -c "from sgl_kernel_npu.norm.gemma_rmsnorm import npu_gemma_rms_norm; print(npu_gemma_rms_norm)"
```

构建验收至少检查：

1. 910 和 950 的 `build/lib/sgl_kernel_npu/norm/gemma_rmsnorm.py` 分别只包含目标 provider。
2. wheel 内不残留相反 provider 的私有模板。
3. 安装后的 `site-packages` 与 wheel 内容一致。
4. 950 跑 ordinary/residual 数值测试与 Qwen3.5 e2e；910B/910C 跑 native provider smoke。

静态/CPU 测试不能替代真实 NPU 验证。使用错误 target 的 wheel 不受支持。

## 环境问题

旧版文档记录过 CANN beta 头文件缺失和 `-Wframe-larger-than` 被 `-Werror` 提升等环境问题。这些不是默认源码修改步骤；只有当前构建日志再次出现相同错误时，才按实际 CANN 安装定位。不要预先修改编译参数或系统头文件。

常用定位：

```bash
find /usr/local/Ascend -name '<missing-header>.h' 2>/dev/null
git diff --check
```

若只需要 kernel wheel，优先 `-a kernels`，避免把 DeepEP、attentions 或 memory-saver 的独立环境问题混入当前验证。

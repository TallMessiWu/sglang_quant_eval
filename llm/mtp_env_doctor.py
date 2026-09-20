#!/usr/bin/env python3
"""一次性打印 Ascend GDN/MTP 这条链上所有跨仓配对的实际状态。

这条链有好几处 SGLang 与 sgl-kernel-npu 必须成对的契约，任何一处错配都不会明着报错，
只会让输出变差。逐个猜代价太高，这里一次把它们全量出来：

  - 跑的是哪个 SGLang checkout、哪个分支、HEAD 是什么
  - ascend_gdn_backend.py / memory_pool.py 上还挂着哪些调试补丁
  - wheel 里哪些算子注册上了
  - SSM state layout：pool 有没有转置、decode triton 是 V-major 还是 K-major
  - gating 输出是几维、SGLang 侧有没有补维
  - move_intermediate_cache 的 h_block_size（950 上必须是 1）
  - MambaPool 的 ssm_dtype 实际取值

用法（NPU 机器）：
    python3 llm/mtp_env_doctor.py
"""

from __future__ import annotations

import importlib
import inspect
import os
import pathlib
import re
import subprocess
import sys

OK, BAD, WARN, INFO = "[ok]", "[!!]", "[? ]", "[--]"


def line(tag: str, label: str, value: str) -> None:
    print(f"  {tag} {label:<34}{value}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def git(repo: pathlib.Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def src_of(mod_name: str) -> tuple[pathlib.Path | None, str]:
    try:
        mod = importlib.import_module(mod_name)
    except Exception as exc:  # noqa: BLE001
        return None, f"import 失败: {type(exc).__name__}: {exc}"
    f = getattr(mod, "__file__", None)
    return (pathlib.Path(f) if f else None), ""


def main() -> int:
    section("SGLang")
    sg_path, err = src_of("sglang")
    if sg_path is None:
        line(BAD, "import sglang", err)
        return 2
    root = sg_path.parent.parent.parent  # .../python/sglang/__init__.py -> repo root
    line(INFO, "路径", str(sg_path.parent))
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD")
    head = git(root, "rev-parse", "--short=10", "HEAD")
    dirty = git(root, "status", "--short")
    if branch:
        line(INFO, "分支 / HEAD", f"{branch} @ {head}")
        line(WARN if dirty else OK, "工作区", f"{len(dirty.splitlines())} 处未提交改动")
    else:
        line(WARN, "分支 / HEAD", f"{root} 不是 git 仓库（wheel 安装？）")

    section("挂在源码上的调试补丁")
    gdn_path, err = src_of("sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend")
    pool_path, _ = src_of("sglang.srt.mem_cache.memory_pool")
    marks = [
        (gdn_path, "V-major state experiment", "patch_gdn_verify_state_major", BAD),
        (gdn_path, "torch fallbacks injected", "patch_gdn_verify_torch", BAD),
        (gdn_path, "gdn probe injected", "patch_gdn_verify_probe", BAD),
        (gdn_path, "gating shape shim", "patch_gdn_gating_shape", OK),
        (pool_path, "old-wheel SSM layout shim", "patch_npu_spec_state_transpose", BAD),
    ]
    any_patch = False
    for path, marker, name, tag_when_on in marks:
        if path is None or not path.exists():
            continue
        on = marker in path.read_text()
        if on:
            any_patch = True
            line(tag_when_on, name, "已打上" + ("" if tag_when_on == OK else "  <- 配新 wheel 时应当 --restore"))
    if not any_patch:
        line(OK, "无调试补丁", "干净")

    section("sgl-kernel-npu wheel")
    k_path, err = src_of("sgl_kernel_npu")
    if k_path is None:
        line(BAD, "import sgl_kernel_npu", err)
        return 2
    kroot = k_path.parent
    line(INFO, "路径", str(kroot))
    try:
        import torch  # noqa: PLC0415
        import torch_npu  # noqa: F401,PLC0415

        for op in ("recurrent_gated_delta_rule", "chunk_gated_delta_rule", "causal_conv1d"):
            has = hasattr(torch.ops.npu, op)
            line(OK if has else WARN, f"torch.ops.npu.{op}", "已注册" if has else "未注册")
    except Exception as exc:  # noqa: BLE001
        line(WARN, "算子探测", f"{type(exc).__name__}: {exc}")

    section("SSM state layout（三方必须一致）")
    pool_txt = pool_path.read_text() if pool_path and pool_path.exists() else ""
    has_transpose = "temporal_state = temporal_state.transpose(-1, -2)" in pool_txt
    line(
        INFO,
        "pool 是否转置",
        "是 -> 逻辑 (HV,K,V)，配 #747 之前的 wheel" if has_transpose
        else "否 -> 逻辑=物理 (HV,V,K)，配含 #747 的 wheel",
    )

    dec = kroot / "fla" / "fused_sigmoid_gating_recurrent.py"
    dec_txt = dec.read_text() if dec.exists() else ""
    if "tl.zeros([BV, BK]" in dec_txt:
        dec_major = "V-major（含 #747）"
    elif "tl.zeros([BK, BV]" in dec_txt:
        dec_major = "K-major（#747 之前）"
    else:
        dec_major = "识别不出"
    line(INFO, "decode triton 布局", dec_major)

    consistent = has_transpose == ("K-major" in dec_major)
    line(
        OK if consistent else BAD,
        "pool 与 decode 是否配套",
        "一致" if consistent else "错配 -> prefill 与 verify 差一个转置，输出会崩",
    )

    section("gating 输出维度")
    gat = kroot / "fla" / "fused_gdn_gating.py"
    gat_txt = gat.read_text() if gat.exists() else ""
    two_d = bool(re.search(r"g = torch\.empty\(batch, num_heads", gat_txt))
    line(INFO, "wheel 侧 fused_gdn_gating_npu", "[T, H] 二维（含 #747）" if two_d else "[1, T, H] 三维")
    shim = "gating shape shim" in (gdn_path.read_text() if gdn_path and gdn_path.exists() else "")
    fixed = "if g.dim() == 2" in (gdn_path.read_text() if gdn_path and gdn_path.exists() else "")
    line(
        OK if (not two_d or fixed) else BAD,
        "SGLang 侧是否补维",
        ("有（补丁）" if shim else "有（分支自带）") if fixed
        else "没有" + ("  <- prefill 会 AssertionError" if two_d else "（旧 wheel 不需要）"),
    )

    section("其他 950 前提")
    mst = kroot / "mamba" / "mamba_state_update_triton.py"
    if mst.exists():
        vals = re.findall(r"h_block_size=(\d+)", mst.read_text())
        first = vals[0] if vals else "?"
        line(
            OK if first == "1" else BAD,
            "h_block_size",
            f"{first}" + ("" if first == "1" else "  <- 950 上会 UB overflow，需要 #651"),
        )

    env = os.environ.get("SGLANG_MAMBA_SSM_DTYPE")
    line(INFO, "SGLANG_MAMBA_SSM_DTYPE", env or "未设置（由 MambaPool 决定）")
    try:
        from sglang.srt.configs.mamba_utils import mamba2_state_dtype  # noqa: PLC0415

        line(INFO, "mamba2_state_dtype().temporal", str(mamba2_state_dtype(None).temporal))
    except Exception as exc:  # noqa: BLE001
        line(WARN, "mamba2_state_dtype", f"{type(exc).__name__}: {exc}")

    src = inspect.getsource(importlib.import_module("sglang.srt.mem_cache.memory_pool"))
    # 这条 warning 在源码里是跨行拼接的，别拿整句去匹配
    forced = "ssm_dtype = torch.bfloat16" in src
    line(
        OK if forced else WARN,
        "投机路径强制 bf16 state",
        "有（#40419）" if forced else "没有 -> 默认 float32 会静默出坏结果",
    )

    print("\n说明：[!!] 是需要处理的，[? ] 存疑，[ok] 正常，[--] 只是信息。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""把 sgl_kernel_npu 里两个 mamba state Triton kernel 换成纯 torch 实现（可回滚）。

背景：在 Ascend 950 上 tests/python/sgl_kernel_npu/test_mamba_state_update.py 的
test_conv_state_rollback 失败（结构性错误，不是精度问题），而 conv_state_rollback 与
move_intermediate_cache 都在 MTP verify 之后提交 / 回滚 state，写坏就会让后续解码复读。

做法：往已安装的 sgl_kernel_npu/mamba/mamba_state_update_triton.py 末尾追加同名函数定义，
后定义的会覆盖前面的，所有 import 方都会拿到 torch 版本。改的是 site-packages 里的纯 Python
文件，不需要重新编译 wheel；重装 wheel 后失效。

用法：
    python3 llm/patch_mamba_triton_fallback.py --self-test   # 只在 CPU 上验证 torch 实现的语义
    python3 llm/patch_mamba_triton_fallback.py --apply       # 打补丁（自动备份）
    python3 llm/patch_mamba_triton_fallback.py --restore     # 还原
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- torch fallback injected by patch_mamba_triton_fallback.py ---"

FALLBACK = '''

# --- torch fallback injected by patch_mamba_triton_fallback.py ---
# 同名重定义会覆盖上面的 Triton 版本；语义与 tests/python/sgl_kernel_npu/
# test_mamba_state_update.py 里的参考实现一致。
def move_intermediate_cache(  # noqa: F811
    ssm_states,
    intermediate_state_cache,
    dst_indices_tensor,
    src_indices_tensor,
    last_steps_tensor,
    h_block_size=1,
):
    """dst[:, dst_idx] = src[:, src_idx, step]，step < 0 的请求跳过。"""
    dst_list = dst_indices_tensor.tolist()
    src_list = src_indices_tensor.tolist()
    step_list = last_steps_tensor.tolist()
    for dst_idx, src_idx, step in zip(dst_list, src_list, step_list):
        if step < 0:
            continue
        ssm_states[:, int(dst_idx)] = intermediate_state_cache[
            :, int(src_idx), int(step)
        ]
    return ssm_states


def conv_state_rollback(  # noqa: F811
    conv_states,
    state_indices,
    step_indices,
    draft_token_num,
):
    """把 conv window 右移 (draft_token_num - 1 - step) 格，丢掉被拒绝的 token。"""
    idx_list = state_indices.tolist()
    step_list = step_indices.tolist()
    for idx, step in zip(idx_list, step_list):
        step = int(step)
        if step < 0:
            continue
        shift = (draft_token_num - 1) - step
        if shift <= 0:
            continue
        req = conv_states[:, int(idx)]
        req[:, shift:, :] = req[:, :-shift, :].clone()
    return conv_states
'''


def module_path() -> Path:
    import sgl_kernel_npu  # noqa: F401

    from sgl_kernel_npu.mamba import mamba_state_update_triton as mod

    return Path(mod.__file__)


def self_test() -> int:
    """在 CPU 上用随机数据对比 torch 实现与测试文件里的参考实现。"""
    import torch

    ns: dict = {}
    exec(FALLBACK, ns)  # noqa: S102  只执行本文件里的常量字符串
    move_fallback = ns["move_intermediate_cache"]
    rollback_fallback = ns["conv_state_rollback"]

    torch.manual_seed(0)
    failures = 0

    # conv_state_rollback：参考实现取自 test_mamba_state_update.py::conv_state_rollback_ref
    for num_layers, pool, dims, draft, reqs in [(4, 8, 64, 3, 3), (2, 4, 32, 4, 4)]:
        window = 3 + draft - 1
        base = torch.randn(num_layers, pool, window, dims)
        indices = torch.randint(0, pool, (reqs,), dtype=torch.int32)
        steps = torch.randint(-1, draft, (reqs,), dtype=torch.int32)

        ref = base.clone()
        for idx, step in zip(indices.tolist(), steps.tolist()):
            shift = (draft - 1) - step
            if step >= 0 and shift > 0:
                req = ref[:, idx, :, :]
                req[:, shift:, :] = req[:, :-shift, :].clone()

        got = rollback_fallback(base.clone(), indices, steps, draft)
        ok = torch.equal(ref, got)
        failures += 0 if ok else 1
        print(f"conv_state_rollback  layers={num_layers} draft={draft} reqs={reqs}: "
              f"{'一致' if ok else '不一致'}")

    # move_intermediate_cache：语义取自该函数的 docstring
    for num_layers, pool, draft, heads, dv, dk, reqs in [(3, 6, 4, 4, 16, 16, 3)]:
        ssm = torch.randn(num_layers, pool, heads, dv, dk)
        inter = torch.randn(num_layers, reqs, draft, heads, dv, dk)
        dst = torch.randint(0, pool, (reqs,), dtype=torch.int64)
        src = torch.arange(reqs, dtype=torch.int64)
        steps = torch.randint(-1, draft, (reqs,), dtype=torch.int64)

        ref = ssm.clone()
        for d, s, st in zip(dst.tolist(), src.tolist(), steps.tolist()):
            if st >= 0:
                ref[:, d] = inter[:, s, st]

        got = move_fallback(ssm.clone(), inter, dst, src, steps)
        ok = torch.equal(ref, got)
        failures += 0 if ok else 1
        print(f"move_intermediate_cache draft={draft} reqs={reqs}: "
              f"{'一致' if ok else '不一致'}")

    print("self-test 通过" if not failures else f"self-test 失败 {failures} 项")
    return 1 if failures else 0


def check_kernel() -> int:
    """在 NPU 上按文档语义逐位校验这两个 kernel（纯数据搬运，应当完全相等）。

    注意 test_mamba_state_update.py 的 conv_state_rollback_ref 用的是重叠内存上的
    copy_，在 CPU 上并不是右移，所以那个测试失败不足以判定 kernel 有错；这里直接和
    “真右移”“按 step 取切片”比。
    """
    import torch

    try:
        import torch_npu  # noqa: F401
        import sgl_kernel_npu  # noqa: F401
        from sgl_kernel_npu.mamba.mamba_state_update_triton import (
            conv_state_rollback as kernel_rollback,
        )
        from sgl_kernel_npu.mamba.mamba_state_update_triton import (
            move_intermediate_cache as kernel_move,
        )
    except ImportError as exc:
        print(f"--check-kernel 需要 NPU 环境（torch_npu + sgl_kernel_npu）：{exc}", file=sys.stderr)
        return 2

    if MARKER in module_path().read_text():
        print("当前模块已经打过 torch 补丁，--check-kernel 请先 --restore", file=sys.stderr)
        return 2

    torch.manual_seed(0)
    failures = 0

    for num_layers, pool, dims, draft, reqs in [(4, 8, 2048, 4, 4), (2, 6, 1024, 3, 3)]:
        window = 3 + draft - 1
        base = torch.randn(num_layers, pool, window, dims, dtype=torch.float32)
        base = base.to(torch.bfloat16)
        # 每个请求一个不同的槽，避免同槽多次回滚导致期望值有歧义
        indices = torch.randperm(pool)[:reqs].to(torch.int32)
        steps = torch.arange(reqs, dtype=torch.int32) % (draft + 1) - 1

        want = base.clone()
        for idx, step in zip(indices.tolist(), steps.tolist()):
            shift = (draft - 1) - step
            if step >= 0 and shift > 0:
                req = want[:, idx]
                req[:, shift:, :] = req[:, :-shift, :].clone()

        got = kernel_rollback(base.npu().clone(), indices.npu(), steps.npu(), draft)
        ok = torch.equal(got.cpu(), want)
        failures += 0 if ok else 1
        bad = int((got.cpu() != want).sum())
        print(
            f"conv_state_rollback  layers={num_layers} dims={dims} draft={draft} "
            f"steps={steps.tolist()}: {'逐位一致' if ok else f'不一致，{bad} 个元素不同'}"
        )

    for num_layers, pool, draft, heads, dv, dk, reqs in [(3, 8, 4, 8, 128, 128, 4)]:
        ssm = torch.randn(num_layers, pool, heads, dv, dk).to(torch.bfloat16)
        inter = torch.randn(num_layers, reqs, draft, heads, dv, dk).to(torch.bfloat16)
        dst = torch.randperm(pool)[:reqs].to(torch.int64)
        src = torch.arange(reqs, dtype=torch.int64)
        steps = torch.arange(reqs, dtype=torch.int64) % (draft + 1) - 1

        want = ssm.clone()
        for d, s, st in zip(dst.tolist(), src.tolist(), steps.tolist()):
            if st >= 0:
                want[:, d] = inter[:, s, st]

        got = kernel_move(ssm.npu().clone(), inter.npu(), dst.npu(), src.npu(), steps.npu())
        ok = torch.equal(got.cpu(), want)
        failures += 0 if ok else 1
        bad = int((got.cpu() != want).sum())
        print(
            f"move_intermediate_cache draft={draft} steps={steps.tolist()}: "
            f"{'逐位一致' if ok else f'不一致，{bad} 个元素不同'}"
        )

    print("两个 kernel 都符合文档语义" if not failures else f"有 {failures} 项不符合文档语义")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="打补丁")
    g.add_argument("--restore", action="store_true", help="从备份还原")
    g.add_argument("--self-test", action="store_true", help="只在 CPU 上验证语义")
    g.add_argument("--check-kernel", action="store_true", help="在 NPU 上逐位校验原 kernel")
    args = p.parse_args()

    if args.self_test:
        return self_test()
    if args.check_kernel:
        return check_kernel()

    path = module_path()
    backup = path.with_suffix(path.suffix + ".orig")

    if args.restore:
        if not backup.exists():
            print(f"没有备份文件 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print(f"{path} 已经打过补丁了")
        return 0
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    path.write_text(text + FALLBACK)
    print(f"已打补丁 {path}（重启服务生效，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

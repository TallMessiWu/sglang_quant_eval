#!/usr/bin/env python3
"""把 NPU 的 GDN prefill 从 triton 切到 #747 新增的 AscendC chunk 算子（可回滚）。

为什么要切：#747 之后 pool、decode triton、verify 算子三方都是 V-major
（state 形状 (nv, dv, dk)），只有 SGLang 仍在调的那条 triton prefill
（sgl_kernel_npu.fla.chunk）返回 **K-major** 的 (nv, dk, dv)。实测：

    triton   final_state (1, 4, 64, 32)  -> (nv, dk, dv)
    AscendC  final_state (1, 4, 32, 64)  -> (nv, dv, dk)
    attention 输出两者一致（差 3.6e-02，bf16 累积量级）

模型的 dk == dv == 128，形状相同所以 `ssm_states[cache_indices] = last_recurrent_state`
不会报错，只是把转置后的内容写进 pool，verify 读到的历史就是错的。

两边的接口差异由这个补丁适配：
  - AscendC 算子不做 L2 norm，要调用方先归一化（triton 是 use_qk_l2norm_in_kernel=True）
  - 它吃的是 [T, N, D]，不带 batch 维
  - cu_seqlens（累积）换成 actual_seq_lengths（每段长度）
  - 只返回 (out, final_state)，没有 per-chunk 的 h；mamba track 那条会因此跳过
    （`if h is not None`），只影响跨 page 的状态跟踪

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_gdn_prefill_ascendc.py --apply
    python3 llm/patch_gdn_prefill_ascendc.py --restore
    python3 llm/patch_gdn_prefill_ascendc.py --show
改完要重启服务。算子没注册时会自动回退到 triton，并打一条 warning。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- AscendC prefill switch (patch_gdn_prefill_ascendc.py) ---"

OLD = """        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **inplace_update_args,
"""

NEW = '''        # --- AscendC prefill switch (patch_gdn_prefill_ascendc.py) ---
        # #747 flipped the pool, the decode kernel and the verify operator to
        # (nv, dv, dk); the triton path here still returns (nv, dk, dv), so the
        # state written back to the pool is transposed -- silently, because
        # dk == dv == 128 makes the shapes identical. Use the operator #747
        # added, which is native to the unified layout.
        if is_npu():
            import torch as _torch

            _op = getattr(_torch.ops.npu, "chunk_gated_delta_rule", None)
            if _op is None:
                import logging as _logging

                _logging.getLogger(__name__).warning(
                    "chunk_gated_delta_rule is not registered; falling back to the "
                    "triton prefill, whose state layout does not match the pool."
                )
            else:
                from sgl_kernel_npu.fla.l2norm import l2norm_fwd as _l2norm

                _t, _nk, _dk = q.shape[-3], q.shape[-2], q.shape[-1]
                _nv, _dv = v.shape[-2], v.shape[-1]
                # q/k/v arrive as strided views of mixed_qkv, so one reshape each
                # is the copy the operator needs; l2norm_fwd returns contiguous.
                _q = _l2norm(q.reshape(-1, _dk)).view(_t, _nk, _dk)
                _k = _l2norm(k.reshape(-1, _dk)).view(_t, _nk, _dk)
                _lens = _torch.diff(query_start_loc).to(_torch.int32)
                # The operator's chunk grid is sum_b ceil(len_b / 64), the same
                # grid _init_track_ssm_indices builds for GDN, and it writes each
                # chunk's entering state -- so chunk_state is exactly the `h` the
                # mamba page tracking reads. Sizing it costs one host sync.
                _chunks = int(((_lens + 63) // 64).sum())
                _h = _torch.empty(
                    _chunks,
                    _nv,
                    _dv,
                    _dk,
                    dtype=recurrent_state.dtype,
                    device=recurrent_state.device,
                )
                _out, _state = _op(
                    _q,
                    _k,
                    v.reshape(_t, _nv, _dv),
                    beta=beta.reshape(_t, _nv),
                    initial_state=recurrent_state,
                    actual_seq_lengths=_lens,
                    scale=_dk**-0.5,
                    g=g.reshape(_t, _nv).to(_torch.float32),
                    chunk_state=_h,
                )
                return _out.unsqueeze(0), _state, _h.unsqueeze(0)
        # --- end switch ---

        return chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=True,
            **recurrent_state_indices_args,
            **inplace_update_args,
'''


def target_path() -> Path:
    from sglang.srt.layers.attention.linear.kernels import (  # noqa: PLC0415
        gdn_triton as mod,
    )

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="切到 AscendC 算子")
    g.add_argument("--restore", action="store_true", help="从备份还原回 triton")
    g.add_argument("--show", action="store_true", help="打印目标文件与当前状态")
    p.add_argument("--file", help="直接指定 gdn_triton.py")
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}；可用 --file 指定路径", file=sys.stderr)
            return 2

    backup = path.with_suffix(path.suffix + ".prefill-orig")

    if args.show:
        text = path.read_text()
        print(path)
        print(f"  已切到 AscendC: {MARKER in text}")
        print(f"  备份存在: {backup.exists()}")
        return 0

    if args.restore:
        if not backup.exists():
            print(f"没有备份文件 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}")
        return 0

    text = path.read_text()
    if MARKER in text:
        print(f"{path} 已经切过了")
        return 0
    if text.count(OLD) != 1:
        print(
            f"{path} 里没有找到唯一的 chunk_gated_delta_rule 调用点"
            f"（找到 {text.count(OLD)} 处）",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已切到 AscendC prefill 算子 {path}（重启服务生效，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

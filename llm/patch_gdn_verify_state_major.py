#!/usr/bin/env python3
"""实验补丁：给 NEXTN verify 的 GDN 算子喂 V-major 的 SSM state（可回滚）。

背景：`torch.ops.npu.recurrent_gated_delta_rule` 的契约是 state 形如
(N, nv, dv, dk)，也就是**行按 v 走、列按 k 走**（host 里 `dv = size(2)`、
`dk = size(3)`，kernel 里 `delta` 的长度是 V 的分块）。但 SGLang 的 GDN state 池
与 prefill/decode 两个 FLA kernel 都是 K-major（decode 的 triton kernel 把
`b_h` 当成 `[K, V]`、存在 `k*V + v`），传进算子时只是 `view(-1, nv, head_k_dim,
head_v_dim)`。dk == dv == 128 时两者互为转置且不会报错，于是每次 verify 读写的都是
转置后的 state：输出仍然流畅（全注意力层还在），但线性注意力的记忆是错的，解码几十步后
塌成复读。

这个补丁在调用算子前把本批次用到的 state 转成 V-major，调用后再转回去写入
intermediate cache，用来验证上面的判断。它会带来明显的拷贝开销，只用于定位问题，不是最终修法。

用法（在 NPU 机器上，SGLang 以源码方式安装）：
    python3 llm/patch_gdn_verify_state_major.py --apply     # 打补丁（自动备份）
    python3 llm/patch_gdn_verify_state_major.py --restore    # 还原
    python3 llm/patch_gdn_verify_state_major.py --show       # 只打印目标文件路径

补丁生效需要重启服务；建议同时关闭 NPU graph（补丁里有额外的张量分配）：
    MTP=1 EXTRA_ARGS="--disable-cuda-graph" llm/qwen3.5_dense_bf16.sh 0
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- V-major state experiment (patch_gdn_verify_state_major.py) ---"

OLD = """        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )
"""

NEW = """        # --- V-major state experiment (patch_gdn_verify_state_major.py) ---
        # The op reads/writes state as (N, nv, dv, dk); the pool is K-major.
        # Gather only this batch's slots, transpose, and write the result back.
        _cache_idx = cache_indices.to(torch.int64)
        _rs_vmajor = recurrent_state[_cache_idx].transpose(-1, -2).contiguous()
        _local_cache_indices = torch.arange(
            batch_size, dtype=cache_indices.dtype, device=cache_indices.device
        )
        _ints_vmajor = None
        if intermediate_state is not None:
            _used = batch_size * seq_len
            _ints_vmajor = (
                intermediate_state[:_used].transpose(-1, -2).contiguous()
            )

        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            _rs_vmajor,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=_ints_vmajor,
            cache_indices=_local_cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )

        if _ints_vmajor is not None:
            intermediate_state[: batch_size * seq_len].copy_(
                _ints_vmajor.transpose(-1, -2)
            )
        else:
            recurrent_state[_cache_idx] = _rs_vmajor.transpose(-1, -2)
        # --- end V-major state experiment ---
"""


def target_path() -> Path:
    from sglang.srt.hardware_backend.npu.attention import (  # noqa: PLC0415
        ascend_gdn_backend as mod,
    )

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--apply", action="store_true", help="打补丁")
    g.add_argument("--restore", action="store_true", help="从备份还原")
    g.add_argument("--show", action="store_true", help="打印目标文件路径")
    p.add_argument("--file", help="直接指定 ascend_gdn_backend.py（跳过 import 探测）")
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}；可用 --file 指定路径", file=sys.stderr)
            return 2

    if args.show:
        print(path)
        return 0

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
    if text.count(OLD) != 1:
        print(
            f"{path} 里没有找到唯一的调用点（找到 {text.count(OLD)} 处），"
            "该文件可能已被改动，请人工确认",
            file=sys.stderr,
        )
        return 1
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    path.write_text(text.replace(OLD, NEW, 1))
    print(f"已打补丁 {path}（重启服务生效，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

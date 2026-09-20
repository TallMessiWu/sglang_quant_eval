#!/usr/bin/env python3
"""探针补丁：在真机 verify 里把 recurrent_gated_delta_rule 和 torch 参考实现当场对比。

背景：patch_gdn_verify_torch.py --conv 后仍复读，--gdn 后反而截断+乱码，说明 torch 兜底
和算子在真机上并不等价 —— 但兜底通过了算子自己那套 CPU 参考实现的自检。所以真机输入里
一定有个假设不成立，只能量出来。

做法：不替换算子。前 N 次 verify 调用里：
    1. 存下 recurrent_state / intermediate_state
    2. 正常调算子，拿走它的输出和写进去的 state
    3. 把 state 还原，用同样的输入跑 torch 参考实现
    4. 打印两者的差异（含“按最后两维转置后是否更接近”，用来判定 layout）
    5. 把算子的结果写回去，返回算子的输出 —— 这一轮的行为和不打补丁完全一致
同时打印输入的真实特征（g 是否在对数域、beta 是否已经过 sigmoid、q/k 是否已归一化、
ssm_state_indices / cache_indices / num_accepted_tokens 的实际取值、state 是否连续等）。

用法（NPU 机器）：
    python3 llm/patch_gdn_verify_probe.py --apply
    GDN_PROBE_CALLS=4 MTP=1 EXTRA_ARGS="--disable-cuda-graph" llm/qwen3.5_dense_bf16.sh 0
    # 发一个请求，然后到服务端日志里看 [GDN-PROBE] 开头的行
    python3 llm/patch_gdn_verify_probe.py --restore

GDN_PROBE_CALLS 控制探测多少次调用（默认 4，即第一次 verify 的前 4 个 GDN 层）。
探针会跑两遍数学，只适合调试，不要留在正常服务里。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_gdn_verify_torch import GDN_OLD, HELPERS  # noqa: E402  复用同一份 torch 参考实现

MARKER = "# --- gdn probe injected by patch_gdn_verify_probe.py ---"

PROBE = '''

# --- gdn probe injected by patch_gdn_verify_probe.py ---
import os as _dbg_os

_DBG_PROBE = {"calls": 0, "limit": int(_dbg_os.environ.get("GDN_PROBE_CALLS", "4"))}


def _dbg_span(name, t):
    if t is None:
        return f"{name}=None"
    f = t.detach().to(torch.float32)
    return (
        f"{name}[{tuple(t.shape)},{t.dtype},contig={t.is_contiguous()}"
        f",stride={tuple(t.stride())}]"
        f" min={f.min().item():.4g} max={f.max().item():.4g}"
        f" nan={int(torch.isnan(f).sum())} inf={int(torch.isinf(f).sum())}"
    )


def _dbg_diff(name, got, want):
    g = got.detach().to(torch.float32)
    w = want.detach().to(torch.float32)
    d = (g - w).abs()
    rel = d.max().item() / max(w.abs().max().item(), 1e-9)
    print(
        f"[GDN-PROBE]   {name}: 最大绝对差={d.max().item():.4g}"
        f" 平均绝对差={d.mean().item():.4g} 相对={rel:.4g}",
        flush=True,
    )
    return d.max().item()


def _dbg_probe_gdn(
    mix_qkv,
    recurrent_state,
    beta,
    scale,
    actual_seq_lengths,
    ssm_state_indices,
    nk,
    nv,
    dk,
    dv,
    intermediate_state,
    cache_indices,
    num_accept_tokens,
    g,
    batch_size,
    seq_len,
):
    call = _DBG_PROBE["calls"]
    probe = call < _DBG_PROBE["limit"]

    # 只取这一批真正用到的行，整池 clone 会是几百 MB
    idx_pool = cache_indices.reshape(-1).to(torch.int64)
    idx_inter = ssm_state_indices.reshape(-1).to(torch.int64)
    saved_inter = (
        intermediate_state[idx_inter].detach().clone()
        if (probe and intermediate_state is not None)
        else None
    )
    saved_pool = recurrent_state[idx_pool].detach().clone() if probe else None

    out = torch.ops.npu.recurrent_gated_delta_rule(
        mix_qkv,
        recurrent_state,
        beta=beta,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
        nk=nk,
        nv=nv,
        intermediate_state=intermediate_state,
        cache_indices=cache_indices,
        num_accepted_tokens=num_accept_tokens,
        g=g,
    )
    if not probe:
        return out

    _DBG_PROBE["calls"] = call + 1
    op_out = out.detach().clone()
    op_inter = (
        intermediate_state[idx_inter].detach().clone()
        if intermediate_state is not None
        else None
    )
    pool_touched = not torch.equal(recurrent_state[idx_pool], saved_pool)

    print(f"[GDN-PROBE] 第 {call} 次 verify 调用", flush=True)
    print(
        f"[GDN-PROBE]   bs={batch_size} seq_len={seq_len} nk={nk} nv={nv}"
        f" dk={dk} dv={dv} scale={scale:.6g}",
        flush=True,
    )
    print(f"[GDN-PROBE]   {_dbg_span('mix_qkv', mix_qkv)}", flush=True)
    print(f"[GDN-PROBE]   {_dbg_span('g', g)}  （对数域时应全为负）", flush=True)
    print(
        f"[GDN-PROBE]   {_dbg_span('beta', beta)}  （落在 [0,1] 说明外面已经 sigmoid 过）",
        flush=True,
    )
    print(f"[GDN-PROBE]   {_dbg_span('recurrent_state', recurrent_state)}", flush=True)
    if intermediate_state is not None:
        print(
            f"[GDN-PROBE]   {_dbg_span('intermediate_state', intermediate_state)}",
            flush=True,
        )
    print(
        f"[GDN-PROBE]   cache_indices={cache_indices.reshape(-1)[:8].tolist()}"
        f" num_accepted={num_accept_tokens.reshape(-1)[:8].tolist()}"
        f" actual_seq={actual_seq_lengths.reshape(-1)[:8].tolist()}",
        flush=True,
    )
    print(
        f"[GDN-PROBE]   ssm_state_indices={ssm_state_indices.reshape(-1)[:16].tolist()}"
        f" 主 pool 被算子改写={pool_touched}",
        flush=True,
    )

    # q/k 是否已经归一化：算子内部会再做一次 L2 norm（幂等），这里只是确认输入形态
    _mix = mix_qkv.reshape(batch_size * seq_len, -1).to(torch.float32)
    _q = _mix[:, : nk * dk].reshape(-1, nk, dk)
    _k = _mix[:, nk * dk : 2 * nk * dk].reshape(-1, nk, dk)
    print(
        f"[GDN-PROBE]   ||q|| 均值={_q.norm(dim=-1).mean().item():.4g}"
        f" ||k|| 均值={_k.norm(dim=-1).mean().item():.4g}（≈1 说明外面已归一化）",
        flush=True,
    )

    # 用同一份输入跑 torch 参考实现
    if intermediate_state is not None:
        intermediate_state[idx_inter] = saved_inter
    ref_out = _dbg_torch_gdn_verify(
        mix_qkv,
        recurrent_state,
        beta,
        scale,
        ssm_state_indices,
        nk,
        nv,
        dk,
        dv,
        intermediate_state,
        cache_indices,
        g,
        batch_size,
        seq_len,
    )
    ref_inter = (
        intermediate_state[idx_inter].detach().clone()
        if intermediate_state is not None
        else None
    )

    _dbg_diff("attention 输出 算子 vs torch", op_out, ref_out)
    if op_inter is not None:
        a = op_inter.to(torch.float32)
        b = ref_inter.to(torch.float32)
        straight = _dbg_diff("state 算子 vs torch", a, b)
        flipped = _dbg_diff("state 算子 vs torch 转置", a, b.transpose(-1, -2))
        if flipped < straight * 0.5:
            print(
                "[GDN-PROBE]   ⚠ 转置后明显更接近：算子和参考实现对 state 后两维的约定不一致",
                flush=True,
            )
        # slot0 会被“处理完 token0 的 state”覆盖，所以它应当不等于主 pool 里的初值
        pool_rows = saved_pool.to(torch.float32)
        first = a.view(batch_size, seq_len, nv, dk, dv)[:, 0]
        print(
            f"[GDN-PROBE]   算子写的 slot0 与主 pool 初值的差="
            f"{(first - pool_rows).abs().max().item():.4g}（应当不为 0）",
            flush=True,
        )
        intermediate_state[idx_inter] = op_inter

    return out
'''

PROBE_CALL = """        attn_core_out = _dbg_probe_gdn(
            mix_qkv,
            recurrent_state,
            beta,
            scale,
            actual_seq_lengths,
            ssm_state_indices,
            num_heads,
            num_value_heads,
            head_k_dim,
            head_v_dim,
            intermediate_state,
            cache_indices,
            num_accept_tokens,
            g,
            batch_size,
            seq_len,
        )
"""


def target_path() -> Path:
    from sglang.srt.hardware_backend.npu.attention import (  # noqa: PLC0415
        ascend_gdn_backend as mod,
    )

    return Path(mod.__file__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--apply", action="store_true", help="打探针")
    grp.add_argument("--restore", action="store_true", help="从备份还原")
    grp.add_argument("--show", action="store_true", help="打印目标文件与当前状态")
    p.add_argument("--file", help="直接指定 ascend_gdn_backend.py")
    args = p.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        try:
            path = target_path()
        except ImportError as exc:
            print(f"无法 import sglang：{exc}；可用 --file 指定路径", file=sys.stderr)
            return 2

    backup = path.with_suffix(path.suffix + ".orig")

    if args.show:
        text = path.read_text()
        print(path)
        print(f"  探针已注入: {MARKER in text}")
        print(f"  备份存在: {backup.exists()}")
        return 0

    if args.restore:
        if not backup.exists():
            print(f"没有备份文件 {backup}", file=sys.stderr)
            return 1
        shutil.copy2(backup, path)
        print(f"已还原 {path}")
        return 0

    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"已备份到 {backup}")
    text = backup.read_text()  # 总是从干净的备份出发，可反复切换各个补丁

    if text.count(GDN_OLD) != 1:
        print(
            f"没有找到唯一的 recurrent_gated_delta_rule 调用点（{text.count(GDN_OLD)} 处）",
            file=sys.stderr,
        )
        return 1
    text = text.replace(GDN_OLD, PROBE_CALL, 1)
    path.write_text(text + HELPERS + PROBE)
    print(f"已注入探针 {path}（重启服务生效，--restore 可还原）")
    print("日志里看 [GDN-PROBE] 开头的行；GDN_PROBE_CALLS 控制探测次数（默认 4）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

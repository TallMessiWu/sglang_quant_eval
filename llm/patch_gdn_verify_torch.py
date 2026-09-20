#!/usr/bin/env python3
"""把 MTP verify 路径上的两个 AscendC 算子分别换成纯 torch 实现，用来二分复读问题。

verify（target_verify）里 GDN 层只用到两个自定义算子：
    1. torch.ops.npu.causal_conv1d(run_mode=1, num_accepted_tokens=...)  —— conv 部分
    2. torch.ops.npu.recurrent_gated_delta_rule(...)                     —— 线性注意力部分
decode / prefill 走的是另外的分支，所以关掉 MTP 一切正常并不能给这两条路径背书。

本补丁把它们逐个替换成等价的 torch 实现（慢，但语义明确），于是：
    --conv 后不复读  → conv 算子（verify/spec 分支）有问题
    --gdn  后不复读  → recurrent_gated_delta_rule 有问题
    --both 后仍复读  → 两个算子都没问题，问题在 Python 侧的 state 提交/回退或别的层

torch 替换保持和算子完全相同的 layout 约定（state 的 axis-2/axis-3 原样不动），
所以它是 layout 中立的：只验证 kernel 实现，不改变 state 语义。

用法（NPU 机器，SGLang 源码安装）：
    python3 llm/patch_gdn_verify_torch.py --self-test    # 先在 CPU 上验证兜底实现本身
    python3 llm/patch_gdn_verify_torch.py --conv         # 只换 conv
    python3 llm/patch_gdn_verify_torch.py --gdn          # 只换 GDN
    python3 llm/patch_gdn_verify_torch.py --both
    python3 llm/patch_gdn_verify_torch.py --restore
每次改完都要重启服务。建议配 EXTRA_ARGS="--disable-cuda-graph"，替换实现里有额外的临时张量。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MARKER = "# --- torch fallbacks injected by patch_gdn_verify_torch.py ---"

HELPERS = '''

# --- torch fallbacks injected by patch_gdn_verify_torch.py ---
def _dbg_torch_causal_conv1d_verify(
    x, weight_t, conv_states, bias, cache_indices, batch_size, seq_len
):
    """等价于 causal_conv1d(run_mode=1, num_accepted_tokens=seq_len)。

    full     = cat([conv_states[slot], x_req])
    y[t]     = silu(sum_j weight_t[j] * full[-(width-1+seq)+t+j])
    新 state = full[-state_len:]
    """
    width = weight_t.shape[0]
    state_len = conv_states.shape[1]
    dim = x.shape[-1]

    idx = cache_indices.to(torch.int64)
    idx_safe = idx.clamp(min=0)
    valid = (idx >= 0).view(-1, 1, 1)

    hist = conv_states[idx_safe].to(torch.float32)
    tokens = x.view(batch_size, seq_len, dim).to(torch.float32)
    full = torch.cat([hist, tokens], dim=1)
    comp = full[:, -(width - 1 + seq_len) :, :]

    w = weight_t.to(torch.float32)
    acc = comp[:, 0:seq_len, :] * w[0]
    for j in range(1, width):
        acc = acc + comp[:, j : j + seq_len, :] * w[j]
    if bias is not None:
        acc = acc + bias.to(torch.float32)
    acc = acc * torch.sigmoid(acc)  # silu

    new_state = full[:, -state_len:, :].to(conv_states.dtype)
    conv_states[idx_safe] = torch.where(valid, new_state, conv_states[idx_safe])
    return acc.to(x.dtype).reshape(-1, dim)


def _dbg_torch_gdn_verify(
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
):
    """等价于 recurrent_gated_delta_rule（num_accepted_tokens 全 1 的 verify 形式）。

    按算子的物理约定做（axis-2 配 v、axis-3 配 k），intermediate_state 是连续的，
    所以直接写进去就和算子写的内存排布一致；主 pool 则要按 is_contiguous() 还原。
    intermediate_state[ssm_state_indices[b, t]] = 处理完 token t 的 state。
    """
    idx = cache_indices.to(torch.int64)
    # 算子的 host 没有对 recurrent_state 调 .contiguous()，kernel 按物理内存读成
    # (nv, dv, dk)。而 SGLang 在 NPU + 投机解码下把 pool 换成 transpose(-1, -2) 的
    # 视图（memory_pool.py: `if _is_npu: temporal_state = temporal_state.transpose(-1, -2)`），
    # 逻辑视图是 (nv, dk, dv)。所以要先还原成物理排布，才是算子看到的那份。
    phys = (
        recurrent_state
        if recurrent_state.is_contiguous()
        else recurrent_state.transpose(-1, -2)
    )
    state = phys[idx].to(torch.float32)  # [bs, nv, dv, dk]

    mix = mix_qkv.view(batch_size, seq_len, -1).to(torch.float32)
    q, k, v = torch.split(mix, [nk * dk, nk * dk, nv * dv], dim=-1)
    q = q.view(batch_size, seq_len, nk, dk)
    k = k.view(batch_size, seq_len, nk, dk)
    v = v.view(batch_size, seq_len, nv, dv)
    q = torch.nn.functional.normalize(q, p=2, dim=-1) * scale
    k = torch.nn.functional.normalize(k, p=2, dim=-1)
    if nv != nk:
        q = q.repeat_interleave(nv // nk, dim=2)
        k = k.repeat_interleave(nv // nk, dim=2)

    g_exp = g.view(batch_size, seq_len, nv).to(torch.float32).exp()
    beta_s = beta.view(batch_size, seq_len, nv).to(torch.float32).sigmoid()

    flat_indices = ssm_state_indices.view(batch_size, seq_len).to(torch.int64)
    out = torch.empty(
        batch_size, seq_len, nv, dv, dtype=torch.float32, device=mix_qkv.device
    )
    for t in range(seq_len):
        k_t = k[:, t].unsqueeze(-2)  # [bs, nv, 1, dk]
        state = state * g_exp[:, t].view(batch_size, nv, 1, 1)
        pred = (state * k_t).sum(dim=-1)  # [bs, nv, dv]
        delta = (v[:, t] - pred) * beta_s[:, t].unsqueeze(-1)
        state = state + delta.unsqueeze(-1) * k_t
        out[:, t] = (state * q[:, t].unsqueeze(-2)).sum(dim=-1)
        if intermediate_state is not None:
            intermediate_state[flat_indices[:, t]] = state.to(
                intermediate_state.dtype
            )
    if intermediate_state is None:
        phys[idx] = state.to(recurrent_state.dtype)
    return out.to(mix_qkv.dtype)
'''

CONV_OLD = """            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
"""

CONV_NEW = """            # patch_gdn_verify_torch.py: conv fallback
            mixed_qkv = _dbg_torch_causal_conv1d_verify(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states,
                layer.bias,
                cache_indices,
                batch_size,
                draft_token_num,
            )
"""

GDN_OLD = """        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
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

GDN_NEW = """        # patch_gdn_verify_torch.py: recurrent_gated_delta_rule fallback
        attn_core_out = _dbg_torch_gdn_verify(
            mix_qkv,
            recurrent_state,
            beta,
            scale,
            ssm_state_indices,
            num_heads,
            num_value_heads,
            head_k_dim,
            head_v_dim,
            intermediate_state,
            cache_indices,
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


def self_test() -> int:
    """在 CPU 上验证两个兜底实现的语义（conv 对上游 unfold 参考，GDN 对逐元素循环参考）。"""
    import torch
    import torch.nn.functional as F

    ns: dict = {"torch": torch}
    exec(HELPERS, ns)  # noqa: S102  只执行本文件里的常量字符串
    conv_fb = ns["_dbg_torch_causal_conv1d_verify"]
    gdn_fb = ns["_dbg_torch_gdn_verify"]

    torch.manual_seed(0)
    failures = 0

    # --- conv：对上游 native_causal_conv1d_update_mtp 的 unfold 写法 ---
    for bs, seq, dim, width in [(3, 4, 64, 4), (2, 2, 32, 3)]:
        pool = bs + 2
        state_len = width - 1 + seq - 1
        x = torch.randn(bs * seq, dim, dtype=torch.float32)
        weight_t = torch.randn(width, dim)
        bias = torch.randn(dim)
        conv_states = torch.randn(pool, state_len, dim)
        idx = torch.randperm(pool)[:bs].to(torch.int32)

        hist = conv_states[idx.to(torch.int64)].transpose(1, 2).contiguous()
        full = torch.cat([hist, x.view(bs, seq, dim).transpose(1, 2)], dim=-1)
        comp = full[:, :, -(width - 1 + seq) :]
        windows = comp.unfold(-1, width, 1)
        want = F.silu(
            (windows * weight_t.t()[None, :, None, :]).sum(-1) + bias[None, :, None]
        )
        want_y = want.transpose(1, 2).reshape(-1, dim)
        want_states = conv_states.clone()
        want_states[idx.to(torch.int64)] = full[:, :, -state_len:].transpose(1, 2)

        got_states = conv_states.clone()
        got_y = conv_fb(x, weight_t, got_states, bias, idx, bs, seq)
        ok_y = torch.allclose(got_y, want_y, rtol=1e-5, atol=1e-5)
        ok_s = torch.allclose(got_states, want_states, rtol=1e-6, atol=1e-6)
        failures += 0 if (ok_y and ok_s) else 1
        print(
            f"conv fallback bs={bs} seq={seq} width={width}: "
            f"输出{'一致' if ok_y else '不一致'}，state{'一致' if ok_s else '不一致'}"
        )

    # --- GDN：对逐 head/逐 token 的循环参考（与 recurrent_gated_delta_rule_check.py 同源）---
    for bs, seq, nk, nv, dk, dv in [(2, 4, 2, 4, 16, 16), (3, 2, 4, 4, 32, 32)]:
        pool = bs + 2
        mix = torch.randn(bs, seq, nk * dk * 2 + nv * dv)
        recurrent_state = torch.randn(pool, nv, dv, dk)
        intermediate = torch.randn(bs * seq, nv, dv, dk)
        idx = torch.randperm(pool)[:bs].to(torch.int32)
        ssm_idx = torch.arange(bs * seq, dtype=torch.int32)
        beta = torch.randn(bs, seq, nv)
        g = -torch.rand(bs, seq, nv)
        scale = dk**-0.5

        # 参考实现：逐 batch / 逐 head / 逐 token
        q, k, v = torch.split(
            mix.reshape(bs * seq, -1), [nk * dk, nk * dk, nv * dv], dim=-1
        )
        q = F.normalize(q.view(bs * seq, nk, dk), p=2, dim=-1) * scale
        k = F.normalize(k.view(bs * seq, nk, dk), p=2, dim=-1)
        v = v.view(bs * seq, nv, dv)
        g_e = g.reshape(bs * seq, nv).exp()
        b_s = beta.reshape(bs * seq, nv).sigmoid()
        want_inter = intermediate.clone()
        want_out = torch.empty(bs, seq, nv, dv)
        for b in range(bs):
            for h in range(nv):
                cur = recurrent_state[int(idx[b])][h].clone()
                qk_h = h // (nv // nk)
                for t in range(seq):
                    slot = b * seq + t
                    cur = cur * g_e[slot][h]
                    x_pred = (cur * k[slot][qk_h].unsqueeze(-2)).sum(-1)
                    y = (v[slot][h] - x_pred) * b_s[slot][h]
                    cur = cur + y[:, None] * k[slot][qk_h][None, :]
                    want_inter[int(ssm_idx[slot])][h] = cur
                    want_out[b][t][h] = (cur * q[slot][qk_h].unsqueeze(-2)).sum(-1)

        got_inter = intermediate.clone()
        got_out = gdn_fb(
            mix, recurrent_state.clone(), beta, scale, ssm_idx, nk, nv, dk, dv,
            got_inter, idx, g, bs, seq,
        )
        ok_o = torch.allclose(got_out, want_out, rtol=1e-4, atol=1e-4)
        ok_s = torch.allclose(got_inter, want_inter, rtol=1e-4, atol=1e-4)
        failures += 0 if (ok_o and ok_s) else 1
        print(
            f"gdn fallback bs={bs} seq={seq} nk={nk} nv={nv}: "
            f"输出{'一致' if ok_o else '不一致'}，state{'一致' if ok_s else '不一致'}"
        )

        # 生产形态：pool 是 transpose(-1, -2) 的非连续视图。算子读的是物理内存，
        # 所以物理内容相同的视图必须给出和连续版本完全相同的结果。
        strided_pool = recurrent_state.transpose(-1, -2)
        assert not strided_pool.is_contiguous()
        got_inter_s = intermediate.clone()
        got_out_s = gdn_fb(
            mix, strided_pool, beta, scale, ssm_idx, nk, nv, dk, dv,
            got_inter_s, idx, g, bs, seq,
        )
        ok_t = torch.allclose(got_out_s, want_out, rtol=1e-4, atol=1e-4) and torch.allclose(
            got_inter_s, want_inter, rtol=1e-4, atol=1e-4
        )
        failures += 0 if ok_t else 1
        print(
            f"gdn fallback（pool 为非连续转置视图，生产形态）: "
            f"{'一致' if ok_t else '不一致'}"
        )

    print("self-test 通过" if not failures else f"self-test 失败 {failures} 项")
    return 1 if failures else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--conv", action="store_true", help="用 torch 替换 verify 的 conv 算子")
    p.add_argument("--gdn", action="store_true", help="用 torch 替换 recurrent_gated_delta_rule")
    p.add_argument("--both", action="store_true", help="两个都换")
    p.add_argument("--restore", action="store_true", help="从备份还原")
    p.add_argument("--show", action="store_true", help="打印目标文件路径与当前状态")
    p.add_argument("--self-test", action="store_true", help="只在 CPU 上验证兜底实现")
    p.add_argument("--file", help="直接指定 ascend_gdn_backend.py")
    args = p.parse_args()

    if args.self_test:
        return self_test()

    want_conv = args.conv or args.both
    want_gdn = args.gdn or args.both
    if not (want_conv or want_gdn or args.restore or args.show):
        p.error("至少要给一个 --conv / --gdn / --both / --restore / --show / --self-test")

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
        print(f"  已注入兜底函数: {MARKER in text}")
        print(f"  conv 已替换: {'conv fallback' in text}")
        print(f"  gdn 已替换: {'recurrent_gated_delta_rule fallback' in text}")
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
    # 每次都从干净的备份出发，这样 --conv / --gdn 可以反复切换
    text = backup.read_text()

    if want_conv:
        if text.count(CONV_OLD) != 1:
            print(
                f"没有找到唯一的 verify conv 调用点（{text.count(CONV_OLD)} 处），"
                "文件可能已被改动",
                file=sys.stderr,
            )
            return 1
        text = text.replace(CONV_OLD, CONV_NEW, 1)
    if want_gdn:
        if text.count(GDN_OLD) != 1:
            print(
                f"没有找到唯一的 recurrent_gated_delta_rule 调用点"
                f"（{text.count(GDN_OLD)} 处），文件可能已被改动",
                file=sys.stderr,
            )
            return 1
        text = text.replace(GDN_OLD, GDN_NEW, 1)

    path.write_text(text + HELPERS)
    replaced = " + ".join(x for x, on in (("conv", want_conv), ("gdn", want_gdn)) if on)
    print(f"已替换 {replaced}：{path}（重启服务生效，--restore 可还原）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""探针:验证 sglang 在 NPU 上用的 causal_conv1d 数值是否正确。

在 A5 上跑,直接调用模型实际使用的两个函数:
  - causal_conv1d_fn_npu       (prefill / extend,图片长序列走这条)
  - causal_conv1d_update_npu   (decode 单 token)
分别与 CPU fp32 golden reference 对拍,报告 cos 相似度 / 最大绝对误差,
并与 bf16 舍入容差比较,判断是"只是 bf16 舍入"还是"kernel 算错了"。

用法(在 A5 机器上,已装 torch_npu + sgl_kernel_npu 的环境):
    python probe_causal_conv1d.py
    python probe_causal_conv1d.py --dim 5120 --width 4 --seqlen 3000 --dtype bf16

注意:dim/width 用的是"代表性默认值",真要对齐 Qwen3.5-27B,请把
      --dim 设成 conv_dim = linear_key_head_dim*linear_num_key_heads*2
                        + linear_value_head_dim*linear_num_value_heads
      --width 设成 linear_conv_kernel_dim(通常 4),都在模型 config.json 里。
      不过验证 kernel 正确性用默认值即可,数值 bug 与具体 dim 无关。
"""

import argparse

import torch
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401

    DEVICE = "npu"
except ImportError:
    raise SystemExit("此探针必须在装了 torch_npu 的 NPU 机器上运行")

from sgl_kernel_npu.mamba.causal_conv1d import (
    causal_conv1d_fn_npu,
    causal_conv1d_update_npu,
)


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _report(tag: str, out_npu: torch.Tensor, out_golden: torch.Tensor, dt: torch.dtype):
    a = out_npu.float().flatten()
    b = out_golden.float().flatten()
    cos = F.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    # 与"纯 bf16/fp16 舍入"能造成的误差量级做对照:把 golden(fp32)round 到低精度再还原
    ref_round = out_golden.to(dt).float().flatten()
    round_err = (b - ref_round).abs().max().item()
    verdict = "OK(仅舍入量级)" if max_abs <= max(round_err * 4, 1e-3) else ">>> 可疑:超出舍入容差 <<<"
    print(
        f"[{tag}] cos={cos:.6f}  max_abs_err={max_abs:.3e}  "
        f"(bf16舍入基线={round_err:.3e})  -> {verdict}"
    )


def golden_conv1d_varlen(x, weight, bias, query_start_loc, conv_states, cache_indices,
                         has_initial_state, activation):
    """CPU fp32 参考:逐序列做 causal depthwise conv1d。
    x: (dim, cu_seq_len)   weight: (dim, width)   conv_states: (n, dim, width-1)
    返回 out: (dim, cu_seq_len)
    """
    x = x.float().cpu()
    weight = weight.float().cpu()
    bias = bias.float().cpu() if bias is not None else None
    conv_states = conv_states.float().cpu()
    qsl = query_start_loc.cpu().tolist()
    dim, width = weight.shape
    out = torch.zeros_like(x)
    for i in range(len(qsl) - 1):
        s, e = qsl[i], qsl[i + 1]
        seg = x[:, s:e]  # (dim, L)
        if has_initial_state[i]:
            init = conv_states[cache_indices[i]]  # (dim, width-1)
        else:
            init = torch.zeros(dim, width - 1)
        padded = torch.cat([init, seg], dim=-1)  # (dim, width-1+L)
        # depthwise conv,权重按时间对齐(causal)
        seg_out = F.conv1d(
            padded.unsqueeze(0), weight.unsqueeze(1), bias, padding=0, groups=dim
        ).squeeze(0)  # (dim, L)
        if activation in ("silu", "swish"):
            seg_out = F.silu(seg_out)
        out[:, s:e] = seg_out
    return out


def probe_prefill(dim, width, seqlen, dt):
    print(f"\n=== Prefill: causal_conv1d_fn_npu  (dim={dim}, width={width}, seqlen={seqlen}, dtype={dt}) ===")
    torch.manual_seed(0)
    # 单序列 varlen 布局:x (dim, cu_seq_len)
    x = torch.randn(dim, seqlen, dtype=dt, device=DEVICE)
    weight = torch.randn(dim, width, dtype=dt, device=DEVICE) * 0.2
    bias = torch.randn(dim, dtype=dt, device=DEVICE) * 0.1
    query_start_loc = torch.tensor([0, seqlen], dtype=torch.int32, device=DEVICE)
    cache_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    has_initial_state = torch.tensor([False], dtype=torch.bool, device=DEVICE)
    conv_states = torch.zeros(1, dim, width - 1, dtype=dt, device=DEVICE)

    out = causal_conv1d_fn_npu(
        x.clone(),
        weight,
        bias,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        conv_states=conv_states,
        activation="silu",
    )
    golden = golden_conv1d_varlen(
        x, weight, bias, query_start_loc, conv_states,
        cache_indices.cpu().tolist(), has_initial_state.cpu().tolist(), "silu",
    )
    _report("prefill", out[:, :seqlen], golden[:, :seqlen], dt)


def probe_decode(dim, width, dt):
    print(f"\n=== Decode: causal_conv1d_update_npu  (dim={dim}, width={width}, dtype={dt}) ===")
    torch.manual_seed(1)
    state_len = width - 1
    x = torch.randn(1, dim, dtype=dt, device=DEVICE)  # (batch, dim) 单 token
    weight = torch.randn(dim, width, dtype=dt, device=DEVICE) * 0.2
    bias = torch.randn(dim, dtype=dt, device=DEVICE) * 0.1
    conv_state = torch.randn(1, dim, state_len, dtype=dt, device=DEVICE)
    conv_state_before = conv_state.clone()
    conv_state_indices = torch.tensor([0], dtype=torch.int32, device=DEVICE)

    out = causal_conv1d_update_npu(
        x.clone(),
        conv_state,
        weight,
        bias=bias,
        activation="silu",
        conv_state_indices=conv_state_indices,
    )

    # golden: window = concat(state, x_token)[-width:] ; out = sum(window * weight) + bias ; silu
    xf = x.float().cpu()
    wf = weight.float().cpu()
    bf = bias.float().cpu()
    sf = conv_state_before[0].float().cpu()  # (dim, state_len)
    window = torch.cat([sf, xf[0].unsqueeze(-1)], dim=-1)  # (dim, width)
    golden = (window * wf).sum(-1) + bf  # (dim,)
    golden = F.silu(golden)
    _report("decode-out", out[0], golden, dt)

    # 顺便检查 conv_state 是否被正确滚动更新
    new_state_golden = torch.cat([sf, xf[0].unsqueeze(-1)], dim=-1)[:, -state_len:]
    _report("decode-state", conv_state[0], new_state_golden, dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=5120)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--seqlen", type=int, default=3000, help="模拟图片那种长 prefill")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dt = _dtype(args.dtype)

    print(f"torch_npu soc: {torch.npu.get_soc_version()}")
    probe_prefill(args.dim, args.width, args.seqlen, dt)
    probe_decode(args.dim, args.width, dt)
    print(
        "\n判读:两项都 'OK(仅舍入量级)' => conv1d 数值正确,元凶不在 conv1d,"
        "\n      往 gated_delta_rule(TritonGDNKernel)或 vision tower 继续查。"
    )


if __name__ == "__main__":
    main()

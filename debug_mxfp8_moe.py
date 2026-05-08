#!/usr/bin/env python3
"""Debug script: compare BF16 reference vs MXFP8 quant for Qwen3 MoE FusedMoE layer.

Run on the NPU server:
    python debug_mxfp8_moe.py 2>&1 | tee debug_out.txt

Then share debug_out.txt back.
"""

import os
import sys

# Force import order so everything resolves
import torch
import torch_npu

print(f"torch: {torch.__version__}")
print(f"torch_npu: {torch_npu.__version__}")
print(f"Device count: {torch.npu.device_count()}")
print(f"Current device: {torch.npu.current_device()}")
print(f"Device name: {torch.npu.get_device_name()}")
print()

# ──────────────────────────────────────────────
# Qwen3-30B-A3B MoE dimensions (tp=1)
# ──────────────────────────────────────────────
NUM_EXPERTS = 60          # total experts in Qwen3-30B-A3B
HIDDEN = 2560             # hidden_size
INTERMEDIATE = 1536       # moe_intermediate_size (per expert)
NUM_TOKENS = 8
TOP_K = 4
BLOCK_SIZE = 32
QUANT_DTYPE = torch_npu.float8_e4m3fn
F8E8M0 = getattr(torch_npu, "float8_e8m0fnu", None)
print(f"FLOAT8_E8M0FNU available: {F8E8M0 is not None}")
print()

device = f"npu:{torch.npu.current_device()}"
rng = torch.Generator(device="cpu").manual_seed(42)


def make_bf16_weight(shape):
    """Create random BF16 weights with reasonable scale."""
    w = torch.randn(shape, generator=rng, dtype=torch.bfloat16, device="cpu")
    # Scale to realistic magnitude (~0.5 std)
    w = w * 0.1
    return w


# ──────────────────────────────────────────────
# Create input and weights
# ──────────────────────────────────────────────
# A few experts only for quick test
E = 4
hidden_states = torch.randn(NUM_TOKENS, HIDDEN, generator=rng, dtype=torch.bfloat16, device=device)

# w13: gate+up projection, shape [E, 2*I, H]
w13_bf16 = make_bf16_weight((E, 2 * INTERMEDIATE, HIDDEN)).to(device)
# w2: down projection, shape [E, H, I]
w2_bf16 = make_bf16_weight((E, HIDDEN, INTERMEDIATE)).to(device)

print("=== Weight shapes (before any transform) ===")
print(f"w13_bf16: {tuple(w13_bf16.shape)}  dtype={w13_bf16.dtype}  "
      f"is_contiguous={w13_bf16.is_contiguous()}  strides={w13_bf16.stride()}")
print(f"w2_bf16:  {tuple(w2_bf16.shape)}  dtype={w2_bf16.dtype}  "
      f"is_contiguous={w2_bf16.is_contiguous()}  strides={w2_bf16.stride()}")


# ──────────────────────────────────────────────
# Reference: BF16 forward (unquantized)
# Uses same routing + grouped_matmul pattern as NPU native
# ──────────────────────────────────────────────
def ref_forward(hidden, w13, w2, topk_ids, topk_weights, top_k):
    """BF16 reference forward via npu_grouped_matmul (no quant)."""
    num_tokens = hidden.shape[0]
    num_experts = w13.shape[0]

    # Route
    sorted_hidden, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden, topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # gmm1 (gate_up_proj): weights are [E, N, K], need [E, K, N] for matmul
    gate = torch.ops.npu.npu_grouped_matmul(
        x=[sorted_hidden],
        weight=[w13.transpose(1, 2)],  # [E, K, N]
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=torch.bfloat16,
    )[0]

    # swiglu
    gate = torch.ops.npu.npu_swiglu(gate)

    # gmm2 (down_proj)
    out = torch.ops.npu.npu_grouped_matmul(
        x=[gate],
        weight=[w2.transpose(1, 2)],  # [E, K, N]
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=torch.bfloat16,
    )[0]

    # Finalize routing
    final = torch.ops.npu.npu_moe_finalize_routing(
        out, skip1=None, skip2=None, bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )
    return final


# ──────────────────────────────────────────────
# MXFP8 forward (matching NPUMXFP8FusedMoEMethod)
# ──────────────────────────────────────────────
def mxfp8_forward(hidden, w13_bf16, w2_bf16, topk_ids, topk_weights, top_k,
                  use_x_dtype_kwargs=True, use_contiguous=False):
    """MXFP8 quantized forward matching NPUMXFP8FusedMoEMethod + npu_fused_experts_mxfp8."""

    num_experts = w13_bf16.shape[0]
    num_tokens = hidden.shape[0]

    # ── Quantise weights online ──
    qw13_list, s13_list = [], []
    for e in range(num_experts):
        w_e = w13_bf16[e]  # [2I, H]
        if not w_e.is_npu:
            w_e = w_e.to(device)
        qw_e, s_e = torch_npu.npu_dynamic_mx_quant(w_e, dst_type=QUANT_DTYPE)
        qw13_list.append(qw_e.view(torch.uint8))
        s13_list.append(s_e)

    qw2_list, s2_list = [], []
    for e in range(num_experts):
        w_e = w2_bf16[e]  # [H, I]
        if not w_e.is_npu:
            w_e = w_e.to(device)
        qw_e, s_e = torch_npu.npu_dynamic_mx_quant(w_e, dst_type=QUANT_DTYPE)
        qw2_list.append(qw_e.view(torch.uint8))
        s2_list.append(s_e)

    qw13 = torch.stack(qw13_list).view(QUANT_DTYPE)  # [E, N, K]
    s13 = torch.stack(s13_list)                      # [E, N, K//64, 2] uint8
    qw2 = torch.stack(qw2_list).view(QUANT_DTYPE)    # [E, N, K]
    s2 = torch.stack(s2_list)                        # [E, N, K//64, 2] uint8

    print(f"\n  After quant + stack:")
    print(f"    qw13: {tuple(qw13.shape)}  dtype={qw13.dtype}  strides={qw13.stride()}  is_contiguous={qw13.is_contiguous()}")
    print(f"    s13:  {tuple(s13.shape)}   dtype={s13.dtype}  strides={s13.stride()}  is_contiguous={s13.is_contiguous()}")
    print(f"    qw2:  {tuple(qw2.shape)}  dtype={qw2.dtype}  strides={qw2.stride()}  is_contiguous={qw2.is_contiguous()}")
    print(f"    s2:   {tuple(s2.shape)}   dtype={s2.dtype}  strides={s2.stride()}  is_contiguous={s2.is_contiguous()}")

    # ── Transform layout: transpose(1, 2) ± contiguous ──
    if use_contiguous:
        # WRONG (allegedly): contiguous after transpose
        t_w13 = qw13.transpose(1, 2).contiguous()
        t_s13 = s13.transpose(1, 2).contiguous()
        t_w2 = qw2.transpose(1, 2).contiguous()
        t_s2 = s2.transpose(1, 2).contiguous()
        layout_name = "transpose+contiguous"
    else:
        # RIGHT (allegedly): strided view only
        t_w13 = qw13.transpose(1, 2)            # [E, K, N] strided
        t_s13 = s13.transpose(1, 2)             # [E, K//64, N, 2] strided
        t_w2 = qw2.transpose(1, 2)              # [E, K, N] strided
        t_s2 = s2.transpose(1, 2)               # [E, K//64, N, 2] strided
        layout_name = "transpose-only (strided)"

    print(f"\n  After {layout_name}:")
    print(f"    t_w13: {tuple(t_w13.shape)}  strides={t_w13.stride()}  is_contiguous={t_w13.is_contiguous()}")
    print(f"    t_s13: {tuple(t_s13.shape)}   strides={t_s13.stride()}  is_contiguous={t_s13.is_contiguous()}")
    print(f"    t_w2:  {tuple(t_w2.shape)}  strides={t_w2.stride()}  is_contiguous={t_w2.is_contiguous()}")
    print(f"    t_s2:  {tuple(t_s2.shape)}   strides={t_s2.stride()}  is_contiguous={t_s2.is_contiguous()}")

    # Verify: "n_dim of weight and n_dim of scale should be equal"
    # w13: [E, K, N] → N = w13.shape[2] = 2*I
    # s13: [E, K//64, N, 2] → N = s13.shape[2] = 2*I
    assert t_w13.shape[2] == t_s13.shape[2], \
        f"n-dim mismatch: w13_N={t_w13.shape[2]} != s13_N={t_s13.shape[2]}"
    assert t_w2.shape[2] == t_s2.shape[2], \
        f"n-dim mismatch: w2_N={t_w2.shape[2]} != s2_N={t_s2.shape[2]}"
    print(f"  ✓ n-dim check passed (weight.N == scale.N == {t_w13.shape[2]} / {t_w2.shape[2]})")

    # ── Route ──
    sorted_hidden, expanded_row_idx, expert_tokens, _ = (
        torch.ops.npu.npu_moe_init_routing_v2(
            hidden, topk_ids,
            active_num=num_tokens * top_k,
            expert_num=num_experts,
            expert_tokens_num_type=1,
            expert_tokens_num_flag=True,
            active_expert_range=[0, num_experts],
            quant_mode=-1,
        )
    )
    expert_tokens = expert_tokens.to(torch.int64)

    # ── gmm1: gate_up_proj ──
    qx1, pertoken_scale1 = torch_npu.npu_dynamic_mx_quant(sorted_hidden, dst_type=QUANT_DTYPE)
    print(f"\n  gmm1: qx shape={tuple(qx1.shape)} dtype={qx1.dtype}")
    print(f"  gmm1: pertoken_scale shape={tuple(pertoken_scale1.shape)} dtype={pertoken_scale1.dtype}")

    # Build kwargs
    gmm_kwargs = dict(
        x=[qx1],
        weight=[t_w13],
        scale=[t_s13],
        per_token_scale=[pertoken_scale1],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=torch.bfloat16,
    )
    if use_x_dtype_kwargs:
        gmm_kwargs.update(dict(
            x_dtype=QUANT_DTYPE,
            weight_dtype=QUANT_DTYPE,
            scale_dtype=F8E8M0,
            per_token_scale_dtype=F8E8M0,
        ))
    print(f"  gmm1 kwargs: { {k: v for k, v in gmm_kwargs.items() if k != 'x' and k != 'weight' and k != 'scale' and k != 'per_token_scale' and k != 'group_list'} }")

    gate = torch.ops.npu.npu_grouped_matmul(**gmm_kwargs)[0]

    # ── swiglu ──
    gate = torch.ops.npu.npu_swiglu(gate)

    # ── gmm2: down_proj ──
    qx2, pertoken_scale2 = torch_npu.npu_dynamic_mx_quant(gate, dst_type=QUANT_DTYPE)

    gmm2_kwargs = dict(
        x=[qx2],
        weight=[t_w2],
        scale=[t_s2],
        per_token_scale=[pertoken_scale2],
        split_item=2,
        group_list_type=1,
        group_type=0,
        group_list=expert_tokens,
        output_dtype=torch.bfloat16,
    )
    if use_x_dtype_kwargs:
        gmm2_kwargs.update(dict(
            x_dtype=QUANT_DTYPE,
            weight_dtype=QUANT_DTYPE,
            scale_dtype=F8E8M0,
            per_token_scale_dtype=F8E8M0,
        ))

    out = torch.ops.npu.npu_grouped_matmul(**gmm2_kwargs)[0]

    # ── Finalize ──
    final = torch.ops.npu.npu_moe_finalize_routing(
        out, skip1=None, skip2=None, bias=None,
        scales=topk_weights,
        expanded_src_to_dst_row=expanded_row_idx,
        export_for_source_row=topk_ids,
        drop_pad_mode=2,
    )
    return final


# ──────────────────────────────────────────────
# Test: compare all variants
# ──────────────────────────────────────────────
print("=" * 72)
print("Running tests...")
print("=" * 72)

# Generate fixed routing (so all variants see same dispatch)
topk_ids = torch.randint(0, E, (NUM_TOKENS, TOP_K), device=device, dtype=torch.int32)
topk_weights = torch.ones(NUM_TOKENS, TOP_K, device=device, dtype=torch.bfloat16) / TOP_K

print(f"\nInput stats: mean={hidden_states.mean().item():.4f} std={hidden_states.std().item():.4f}")

# ── Reference ──
print("\n" + "-" * 60)
print("[Ref] BF16 baseline forward")
print("-" * 60)
ref_out = ref_forward(hidden_states, w13_bf16, w2_bf16, topk_ids, topk_weights, TOP_K)
print(f"  Output shape: {tuple(ref_out.shape)}")
print(f"  Output stats: mean={ref_out.mean().item():.6f} std={ref_out.std().item():.6f}")
print(f"  Output[:3,:5]:\n{ref_out[:3, :5]}")

# ── Variant A: strided + no x_dtype/weight_dtype ──
print("\n" + "-" * 60)
print("[A] MXFP8: strided (no contiguous), WITHOUT x_dtype/weight_dtype")
print("-" * 60)
out_a = mxfp8_forward(hidden_states, w13_bf16, w2_bf16, topk_ids, topk_weights, TOP_K,
                       use_x_dtype_kwargs=False, use_contiguous=False)
print(f"  Output shape: {tuple(out_a.shape)}")
print(f"  Output stats: mean={out_a.mean().item():.6f} std={out_a.std().item():.6f}")
print(f"  Output[:3,:5]:\n{out_a[:3, :5]}")
cos_a = torch.nn.functional.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                               out_a.flatten().unsqueeze(0))
print(f"  Cosine similarity vs ref: {cos_a.item():.6f}")

# ── Variant B: strided + WITH x_dtype/weight_dtype ──
print("\n" + "-" * 60)
print("[B] MXFP8: strided (no contiguous), WITH x_dtype/weight_dtype")
print("-" * 60)
out_b = mxfp8_forward(hidden_states, w13_bf16, w2_bf16, topk_ids, topk_weights, TOP_K,
                       use_x_dtype_kwargs=True, use_contiguous=False)
print(f"  Output shape: {tuple(out_b.shape)}")
print(f"  Output stats: mean={out_b.mean().item():.6f} std={out_b.std().item():.6f}")
print(f"  Output[:3,:5]:\n{out_b[:3, :5]}")
cos_b = torch.nn.functional.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                               out_b.flatten().unsqueeze(0))
print(f"  Cosine similarity vs ref: {cos_b.item():.6f}")

# ── Variant C: contiguous + WITH x_dtype/weight_dtype ──
print("\n" + "-" * 60)
print("[C] MXFP8: contiguous (after transpose), WITH x_dtype/weight_dtype")
print("-" * 60)
out_c = mxfp8_forward(hidden_states, w13_bf16, w2_bf16, topk_ids, topk_weights, TOP_K,
                       use_x_dtype_kwargs=True, use_contiguous=True)
print(f"  Output shape: {tuple(out_c.shape)}")
print(f"  Output stats: mean={out_c.mean().item():.6f} std={out_c.std().item():.6f}")
print(f"  Output[:3,:5]:\n{out_c[:3, :5]}")
cos_c = torch.nn.functional.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                               out_c.flatten().unsqueeze(0))
print(f"  Cosine similarity vs ref: {cos_c.item():.6f}")

# ── Variant D: contiguous + WITHOUT x_dtype/weight_dtype ──
print("\n" + "-" * 60)
print("[D] MXFP8: contiguous (after transpose), WITHOUT x_dtype/weight_dtype")
print("-" * 60)
out_d = mxfp8_forward(hidden_states, w13_bf16, w2_bf16, topk_ids, topk_weights, TOP_K,
                       use_x_dtype_kwargs=False, use_contiguous=True)
print(f"  Output shape: {tuple(out_d.shape)}")
print(f"  Output stats: mean={out_d.mean().item():.6f} std={out_d.std().item():.6f}")
print(f"  Output[:3,:5]:\n{out_d[:3, :5]}")
cos_d = torch.nn.functional.cosine_similarity(ref_out.flatten().unsqueeze(0),
                                               out_d.flatten().unsqueeze(0))
print(f"  Cosine similarity vs ref: {cos_d.item():.6f}")

# ── Summary ──
print("\n" + "=" * 72)
print("SUMMARY: cosine similarity vs BF16 reference")
print("=" * 72)
print(f"  [A] strided, no x_dtype:        {cos_a.item():.6f}")
print(f"  [B] strided, WITH x_dtype:      {cos_b.item():.6f}")
print(f"  [C] contiguous, WITH x_dtype:   {cos_c.item():.6f}")
print(f"  [D] contiguous, no x_dtype:     {cos_d.item():.6f}")
print()
# ──────────────────────────────────────────────
# Test E: vllm-ascend's fused kernel path (npu_grouped_matmul_swiglu_quant_v2)
# ──────────────────────────────────────────────
print("\n" + "-" * 60)
print("[E] vllm-ascend fused: npu_grouped_matmul_swiglu_quant_v2 + gmm2")
print("-" * 60)

# Quantise weights same as before
qw13_list, s13_list = [], []
for e in range(E):
    w_e = w13_bf16[e]
    qw_e, s_e = torch_npu.npu_dynamic_mx_quant(w_e, dst_type=QUANT_DTYPE)
    qw13_list.append(qw_e.view(torch.uint8))
    s13_list.append(s_e)
qw2_list, s2_list = [], []
for e in range(E):
    w_e = w2_bf16[e]
    qw_e, s_e = torch_npu.npu_dynamic_mx_quant(w_e, dst_type=QUANT_DTYPE)
    qw2_list.append(qw_e.view(torch.uint8))
    s2_list.append(s_e)
t_w13 = torch.stack(qw13_list).view(QUANT_DTYPE).transpose(1, 2)  # [E, H, 2I] strided
t_s13 = torch.stack(s13_list).transpose(1, 2)                     # [E, H//64, 2I, 2] strided
t_w2 = torch.stack(qw2_list).view(QUANT_DTYPE).transpose(1, 2)    # [E, I, H] strided
t_s2 = torch.stack(s2_list).transpose(1, 2)                       # [E, I//64, H, 2] strided

# Routing
sorted_hidden, expanded_row_idx, expert_tokens, _ = (
    torch.ops.npu.npu_moe_init_routing_v2(
        hidden_states, topk_ids,
        active_num=NUM_TOKENS * TOP_K,
        expert_num=E,
        expert_tokens_num_type=1,
        expert_tokens_num_flag=True,
        active_expert_range=[0, E],
        quant_mode=-1,
    )
)
expert_tokens = expert_tokens.to(torch.int64)

# Activation quant
qx, pertoken_scale = torch_npu.npu_dynamic_mx_quant(sorted_hidden, dst_type=QUANT_DTYPE)

# vllm-ascend uses cumsum group_list for the fused kernel
group_list_cumsum = torch.cat([expert_tokens[:1], torch.diff(expert_tokens, dim=0)])

# Check if npu_grouped_matmul_swiglu_quant_v2 exists
has_fused_kernel = hasattr(torch.ops.npu, 'npu_grouped_matmul_swiglu_quant_v2')
print(f"  npu_grouped_matmul_swiglu_quant_v2 available: {has_fused_kernel}")

if has_fused_kernel:
    try:
        # gmm1 + swiglu + quant fused (vllm-ascend's MXFP path)
        gate, out_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=qx,
            weight=[t_w13],
            group_list=group_list_cumsum,
            weight_scale=[t_s13],
            x_scale=pertoken_scale,
            dequant_mode=2,
            quant_mode=2,
            dequant_dtype=torch.float32,
            quant_dtype=QUANT_DTYPE,
            x_dtype=QUANT_DTYPE,
            weight_dtype=QUANT_DTYPE,
            weight_scale_dtype=F8E8M0,
            x_scale_dtype=F8E8M0,
        )
        print(f"  after fused: shape={tuple(gate.shape)} dtype={gate.dtype} mean={gate.mean().item():.4f}")

        # gmm2
        out_gmm2 = torch.ops.npu.npu_grouped_matmul(
            x=[gate],
            weight=[t_w2],
            scale=[t_s2],
            per_token_scale=[out_scale],
            x_dtype=QUANT_DTYPE,
            weight_dtype=QUANT_DTYPE,
            scale_dtype=F8E8M0,
            per_token_scale_dtype=F8E8M0,
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=torch.bfloat16,
        )[0]

        out_e = torch.ops.npu.npu_moe_finalize_routing(
            out_gmm2, skip1=None, skip2=None, bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
            drop_pad_mode=2,
        )
        cos_e = torch.nn.functional.cosine_similarity(
            ref_out.flatten().unsqueeze(0), out_e.flatten().unsqueeze(0))
        print(f"  Output stats: mean={out_e.mean().item():.6f} std={out_e.std().item():.6f}")
        print(f"  Cosine similarity vs ref: {cos_e.item():.6f}")
    except Exception as e:
        print(f"  Fused kernel failed: {e}")
        cos_e = torch.tensor(0.0)
else:
    cos_e = torch.tensor(0.0)

# ── Summary ──
print("\n" + "=" * 72)
print("SUMMARY: cosine similarity vs BF16 reference")
print("=" * 72)
print(f"  [A] strided, no x_dtype:        {cos_a.item():.6f}")
print(f"  [B] strided, WITH x_dtype:      {cos_b.item():.6f}")
print(f"  [C] contiguous, WITH x_dtype:   {cos_c.item():.6f}")
print(f"  [D] contiguous, no x_dtype:     {cos_d.item():.6f}")
if has_fused_kernel:
    print(f"  [E] vllm fused kernel path:    {cos_e.item():.6f}")
print()

if max(cos_a.item(), cos_b.item(), cos_c.item(), cos_d.item(), cos_e.item()) < 0.9:
    print("⚠  All variants show low similarity — problem is deeper than layout/dtype kwargs.")
    print("   Possible causes: nopoe kernel not installed, wrong moe routing params,")
    print("   or npu_dynamic_mx_quant returning unexpected scale format.")
    print()
    print("   Let's also test simple matmuls:")
    # Test A: npu_quant_matmul (dense linear path — known working)
    M, K, N = 16, HIDDEN, 2 * INTERMEDIATE
    x_simple = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    w_simple = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    ref_simple = x_simple @ w_simple.T
    qx_s, ps_s = torch_npu.npu_dynamic_mx_quant(x_simple, dst_type=QUANT_DTYPE)
    qw_s, ws_s = torch_npu.npu_dynamic_mx_quant(w_simple, dst_type=QUANT_DTYPE)
    out_s = torch_npu.npu_quant_matmul(
        qx_s, qw_s.T.contiguous(), ws_s.transpose(0, 1).contiguous(),
        scale_dtype=F8E8M0, pertoken_scale=ps_s,
        pertoken_scale_dtype=F8E8M0, output_dtype=torch.bfloat16,
        group_sizes=[1, 1, BLOCK_SIZE],
    )
    cos_s = torch.nn.functional.cosine_similarity(
        ref_simple.flatten().unsqueeze(0), out_s.flatten().unsqueeze(0))
    print(f"   [QMatmul] npu_quant_matmul cos_sim: {cos_s.item():.6f}")

    # Test B: npu_grouped_matmul with single expert (bypass routing)
    # This is the kernel MoE path actually uses
    gw_s = qw_s.transpose(0, 1)  # [K, N] → for grouped_matmul weight=[E, K, N], E=1
    gws_s = ws_s.transpose(0, 1)  # [K//64, N, 2]
    # Wrap as single-element lists mimicking [E=1, ...]
    out_g = torch.ops.npu.npu_grouped_matmul(
        x=[qx_s],
        weight=[gw_s.unsqueeze(0)],
        scale=[gws_s.unsqueeze(0)],
        per_token_scale=[ps_s],
        x_dtype=QUANT_DTYPE,
        weight_dtype=QUANT_DTYPE,
        scale_dtype=F8E8M0,
        per_token_scale_dtype=F8E8M0,
        split_item=2,
        group_list_type=0,  # single expert
        group_type=0,
        group_list=torch.tensor([M], dtype=torch.int64, device=device),
        output_dtype=torch.bfloat16,
    )[0]
    cos_g = torch.nn.functional.cosine_similarity(
        ref_simple.flatten().unsqueeze(0), out_g.flatten().unsqueeze(0))
    print(f"   [GMatmul] npu_grouped_matmul cos_sim: {cos_g.item():.6f}")

    if cos_s.item() > 0.99 and cos_g.item() < 0.9:
        print("   ✓ npu_quant_matmul works, npu_grouped_matmul fails — problem is in")
        print("     grouped_matmul MXFP path specifically (layout params).")
    elif cos_s.item() > 0.99 and cos_g.item() > 0.99:
        print("   ✓ Both matmul paths work — problem is MoE routing + matmul combo.")
    elif cos_s.item() < 0.9:
        print("   ⚠  Even simple npu_quant_matmul fails — fundamental issue (CANN/kernel).")
elif cos_b.item() > 0.99 and cos_a.item() < 0.9:
    print("✓ x_dtype/weight_dtype kwargs are the KEY fix.")
elif cos_a.item() > 0.99:
    print("✓ Even without x_dtype/weight_dtype, strided layout works. Problem elsewhere.")
elif cos_c.item() > 0.99:
    print("✓ contiguous works with x_dtype/weight_dtype. Problem elsewhere.")
else:
    print("Mixed results — see individual scores above.")

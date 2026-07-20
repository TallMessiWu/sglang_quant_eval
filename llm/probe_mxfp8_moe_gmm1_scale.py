"""A5 probe: does the MXFP8 MoE gmm1 output scale need the 2D->3D normalize?

Production (`NPUMXFP8MoEMethod.apply_fused_gmm1_swiglu` ->
`AscendRunnerCore.run` -> `NPUMXFP8MoEMethod.apply`) feeds the scale returned by
`npu_grouped_matmul_swiglu_quant_v2` straight into gmm2's `per_token_scale`.
The Jul-9 probe that validated the kernel chain (cos~0.997) instead ran it
through `_normalize_scale` (flat 2D [N, K/32] -> pair-split 3D [N, K/64, 2])
first. If the op returns a 2D scale, production reads the block scales with the
wrong layout -- no error, just garbage tokens.

This probe mirrors the production call sequence byte for byte (torch.ops.npu.*,
same dtype sources, weights kept as strided transpose views, activation quant
fused into routing via quant_mode=3) and runs gmm2 twice: once with the raw
gmm1 scale (what production does today) and once with the normalized one. Both
are scored against a bf16 per-expert reference over the same permuted tokens.

Run on the A5 box:  python llm/probe_mxfp8_moe_gmm1_scale.py
"""

import torch
import torch.nn.functional as F
import torch_npu

# Dtype sources must match production exactly (see AGENTS.md known pitfalls):
# e4m3 comes from torch, e8m0 MUST come from torch_npu (int 293) -- the grouped
# matmuls reject torch's dtype object for the *_scale_dtype arguments.
E4M3 = torch.float8_e4m3fn
E8M0 = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))

DEVICE = f"npu:{torch.npu.current_device()}"
DTYPE = torch.bfloat16

# Qwen3-30B-A3B-ish shapes, shrunk. K must stay divisible by 64 so the block
# count K/32 is even (pair-split needs it).
H = 2048  # hidden_size
I = 768  # moe_intermediate_size
E = 4  # experts
NUM_TOKENS = 8
TOP_K = 2

torch.manual_seed(0)


def _stat(name, t):
    if not isinstance(t, torch.Tensor):
        print(f"    {name}: {t}")
        return
    print(
        f"    {name}: shape={tuple(t.shape)} ndim={t.ndim} dtype={t.dtype} "
        f"contig={t.is_contiguous()}"
    )


def _cos(ref, out):
    return F.cosine_similarity(ref.float().flatten(), out.float().flatten(), dim=0).item()


def _normalize_scale(scale):
    """Same as sglang's `_normalize_mxfp_scale`: 2D [N, M] -> 3D [N, M//2, 2]."""
    if scale is None or scale.ndim != 2:
        return scale
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


def _swiglu_ref(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


def _build_weights():
    """Online MXFP8 weight quant, kept as strided transpose views (production)."""
    w13_bf = torch.randn(E, 2 * I, H, device=DEVICE, dtype=DTYPE) * 0.05
    w2_bf = torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.05
    qw13, s13 = torch.ops.npu.npu_dynamic_mx_quant(w13_bf, dst_type=E4M3)
    qw2, s2 = torch.ops.npu.npu_dynamic_mx_quant(w2_bf, dst_type=E4M3)
    return (
        w13_bf,
        w2_bf,
        qw13.transpose(1, 2),
        s13.transpose(1, 2),
        qw2.transpose(1, 2),
        s2.transpose(1, 2),
    )


def _init_routing(hidden, topk_ids, quant_mode):
    return torch.ops.npu.npu_moe_init_routing_v2(
        hidden,
        topk_ids,
        active_num=NUM_TOKENS * TOP_K,
        expert_num=E,
        expert_tokens_num_type=1,  # COUNT
        expert_tokens_num_flag=True,
        active_expert_range=[0, E],
        quant_mode=quant_mode,
    )


def _gmm2(g1_out, per_token_scale, expert_tokens, w2, w2_scale):
    return torch.ops.npu.npu_grouped_matmul(
        x=[g1_out],
        weight=[w2],
        scale=[w2_scale],
        per_token_scale=[per_token_scale],
        scale_dtype=E8M0,
        per_token_scale_dtype=E8M0,
        x_dtype=None,
        weight_dtype=None,
        split_item=2,
        group_list_type=1,  # COUNT
        group_type=0,
        group_list=expert_tokens,
        output_dtype=DTYPE,
    )[0]


def probe_dense_linear():
    """Does the DENSE MXFP8 matmul care where its e8m0 dtype comes from?

    ``NPUMXFP8LinearMethod.apply`` (upstream-merged, used by every attention
    linear -- including in a MoE model) passes ``scale_dtype`` /
    ``pertoken_scale_dtype`` as ``getattr(torch, "float8_e8m0fnu")``, i.e. the
    torch dtype object, not torch_npu's int enum. The grouped matmuls reject
    that loudly; if npu_quant_matmul instead *accepts* it and reads the block
    scales as something else, every attention projection quietly returns
    garbage -- which looks exactly like garbled generation with a numerically
    perfect MoE path.
    """
    print("=" * 100)
    print("DENSE npu_quant_matmul: e8m0 from torch vs torch_npu")
    M, K, N = 128, H, 2 * I
    x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
    w = torch.randn(N, K, device=DEVICE, dtype=DTYPE) * 0.05  # [out, in]
    ref = x @ w.t()

    qx, x_scale = torch.ops.npu.npu_dynamic_mx_quant(x, dst_type=E4M3)
    qw, w_scale = torch.ops.npu.npu_dynamic_mx_quant(w, dst_type=E4M3)
    # Strided transpose views, exactly like process_weights_after_loading.
    qw_t, w_scale_t = qw.transpose(0, 1), w_scale.transpose(0, 1)

    torch_e8m0 = getattr(torch, "float8_e8m0fnu", None)
    npu_e8m0 = getattr(torch_npu, "float8_e8m0fnu", None)
    for label, dtype in (("torch.float8_e8m0fnu  (production today)", torch_e8m0),
                         ("torch_npu.float8_e8m0fnu (int enum)     ", npu_e8m0)):
        if dtype is None:
            print(f"    {label}: dtype unavailable, skipped")
            continue
        try:
            y = torch.ops.npu.npu_quant_matmul(
                qx,
                qw_t,
                w_scale_t,
                scale_dtype=dtype,
                pertoken_scale=x_scale,
                pertoken_scale_dtype=dtype,
                bias=None,
                output_dtype=DTYPE,
                group_sizes=[1, 1, 32],
            )
            cos = _cos(ref, y)
            verdict = "PASS" if cos > 0.97 else "FAIL (garbage -> dense path is the bug)"
            print(f"    {label}: cos={cos:.5f}  => {verdict}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {label}: RAISED {type(exc).__name__}: {exc}")


def main():
    print("=" * 100)
    print("ENV")
    print(f"    torch     = {torch.__version__}")
    print(f"    torch_npu = {getattr(torch_npu, '__version__', '?')}")
    print(f"    E4M3 = {E4M3}    E8M0 = {E8M0}  (e8m0 must be the torch_npu int enum)")

    w13_bf, w2_bf, w13, w13_scale, w2, w2_scale = _build_weights()
    print("=" * 100)
    print("WEIGHTS (strided transpose views, no .contiguous())")
    _stat("w13      ", w13)
    _stat("w13_scale", w13_scale)
    _stat("w2       ", w2)
    _stat("w2_scale ", w2_scale)

    hidden = torch.randn(NUM_TOKENS, H, device=DEVICE, dtype=DTYPE)
    topk_ids = torch.randint(0, E, (NUM_TOKENS, TOP_K), device=DEVICE, dtype=torch.int32)

    # bf16 reference over the same permutation (quant_mode=-1 keeps bf16 states).
    ref_states, _, ref_counts, _ = _init_routing(hidden, topk_ids, quant_mode=-1)
    ref_counts = ref_counts.to(torch.int64)
    parts, off = [], 0
    for e in range(E):
        n = int(ref_counts[e])
        xe = ref_states[off : off + n]
        parts.append(_swiglu_ref(xe @ w13_bf[e].transpose(0, 1)) @ w2_bf[e].transpose(0, 1))
        off += n
    y_ref = torch.cat(parts, 0)

    # Production path: activation quant fused into routing (quant_mode=3).
    qx, _, expert_tokens, x_scale_raw = _init_routing(hidden, topk_ids, quant_mode=3)
    expert_tokens = expert_tokens.to(torch.int64)
    print("=" * 100)
    print("ROUTING (quant_mode=3)")
    _stat("qx           ", qx)
    _stat("x_scale (raw)", x_scale_raw)
    x_scale = _normalize_scale(x_scale_raw)
    _stat("x_scale (norm)", x_scale)
    print(f"    routing scale needed normalize (was 2D)? {x_scale_raw.ndim == 2}")

    g1_out, g1_scale_raw = torch.ops.npu.npu_grouped_matmul_swiglu_quant_v2(
        x=qx,
        weight=[w13],
        group_list=expert_tokens.cumsum(0),
        weight_scale=[w13_scale],
        x_scale=x_scale,
        dequant_mode=2,
        quant_mode=2,
        dequant_dtype=torch.float32,
        quant_dtype=E4M3,
        x_dtype=None,
        weight_dtype=None,
        weight_scale_dtype=E8M0,
        x_scale_dtype=E8M0,
    )
    print("=" * 100)
    print("GMM1  <<< THE QUESTION")
    _stat("g1_out       ", g1_out)
    _stat("g1_scale(raw)", g1_scale_raw)
    g1_scale_norm = _normalize_scale(g1_scale_raw)
    _stat("g1_scale(norm)", g1_scale_norm)
    print(f"    gmm1 scale is 2D (production would feed gmm2 the wrong layout)? "
          f"{g1_scale_raw.ndim == 2}")

    print("=" * 100)
    print("GMM2  A/B")
    for label, scale in (("RAW  (production today)", g1_scale_raw),
                         ("NORM (Jul-9 probe)     ", g1_scale_norm)):
        try:
            y = _gmm2(g1_out, scale, expert_tokens, w2, w2_scale)
            cos = _cos(y_ref, y)
            verdict = "PASS" if cos > 0.97 else "FAIL (garbage -> this is the bug)"
            print(f"    {label}: cos={cos:.5f}  => {verdict}")
        except Exception as exc:  # noqa: BLE001
            print(f"    {label}: RAISED {type(exc).__name__}: {exc}")

    print("=" * 100)
    print("If RAW fails and NORM passes -> apply_fused_gmm1_swiglu must normalize")
    print("its returned scale before it reaches gmm2. If both pass, the garbling")
    print("is elsewhere (routing / finalize / weight layout).")

    probe_dense_linear()


if __name__ == "__main__":
    main()

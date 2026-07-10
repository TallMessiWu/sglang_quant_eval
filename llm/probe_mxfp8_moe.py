"""A5 MXFP8 MoE op-level probe — pin down the grouped-matmul kernel contract.

Why this exists
---------------
The previous online MXFP8 MoE branch produced garbage and churned 11 commits
flip-flopping on transpose/contiguous, scale layout, group_sizes and
x_dtype/weight_dtype — because the contract of the two grouped-matmul kernels
was never *measured* on real torch_npu. This script measures it, so the
re-implementation is written against facts, not guesses.

It answers, on THIS box's torch_npu, with hard prints and a bf16-reference
numeric check:

  Q1. What EXACT shape/ndim does `npu_dynamic_mx_quant(x, dst=e4m3)` return for
      the scale?  2D `[N, K/32]`  or  3D pair-split `[N, K/64, 2]` ?
      (dense linear code assumes 3D; vllm-ascend maybe_normalize assumes 2D —
       only one is right for this torch_npu version.)
  Q2. Does the dense MXFP8 matmul (`npu_quant_matmul`, group_sizes=[1,1,32])
      reproduce bf16 within tolerance here? (sanity that e8m0 works at all.)
  Q3. gmm1 `npu_grouped_matmul_swiglu_quant_v2` (MX): exact call that runs,
      cumulative group_list, single-elem weight list, e8m0 scale dtypes,
      x_dtype/weight_dtype=None. Output + out_scale shapes.
  Q4. gmm2 `npu_grouped_matmul` (MX): original count group_list + explicit
      group_list_type, scale=[w2_scale], e8m0 dtypes, NO group_sizes.
  Q5. Does the strided (non-contiguous) transpose-view weight layout get
      accepted by the grouped kernels, or must we .contiguous()?
  Q6. GROUND TRUTH: full gmm1->swiglu->gmm2 vs a pure-bf16 per-expert
      reference (top_k=1 permuted tokens). cosine + max-rel-err. This is the
      arbiter of "is the kernel path numerically correct".

Run on the A5 box:
    python llm/probe_mxfp8_moe.py

Everything is wrapped so one failing section still lets the rest print.
Signatures mirror vllm-ascend A5DeviceAdaptor (device/device_op.py) exactly.
"""

import torch
import torch.nn.functional as F
import torch_npu

# ---------------------------------------------------------------------------
# constants / dtypes
# ---------------------------------------------------------------------------
MXFP8_BLOCK_SIZE = 32
E4M3 = torch.float8_e4m3fn
E8M0 = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))

DEVICE = f"npu:{torch.npu.current_device()}"
DTYPE = torch.bfloat16

# Qwen3-30B-A3B-ish shapes (hidden=2048, moe_intermediate=768). Kept tiny+valid:
# K must be divisible by 64 so K/32 (block count) is even (pair-split needs it).
H = 2048          # hidden_size            2048/64 = 32  ok
I = 768           # moe_intermediate       768/64  = 12  ok
TWO_I = 2 * I     # gate_up out            1536
E = 4             # experts (tiny)
torch.manual_seed(0)


def _line(c="-"):
    print(c * 100)


def _stat(name, t):
    if t is None:
        print(f"    {name}: None")
        return
    print(f"    {name}: shape={tuple(t.shape)} ndim={t.ndim} dtype={t.dtype} "
          f"contig={t.is_contiguous()}")


def _cos_relerr(ref, out):
    ref = ref.float().flatten()
    out = out.float().flatten()
    cos = F.cosine_similarity(ref, out, dim=0).item()
    denom = ref.abs().clamp_min(1e-6)
    max_rel = ((out - ref).abs() / denom).max().item()
    mean_rel = ((out - ref).abs() / denom).mean().item()
    return cos, max_rel, mean_rel


def _swiglu_ref(gate_up):
    """SGLang/vLLM convention: w13 = concat(gate, up); silu(gate) * up."""
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


# ---------------------------------------------------------------------------
# Q0 — capability probe
# ---------------------------------------------------------------------------
def probe_capabilities():
    _line("=")
    print("Q0  CAPABILITIES")
    print(f"    torch      = {torch.__version__}")
    print(f"    torch_npu  = {getattr(torch_npu, '__version__', '?')}")
    print(f"    device     = {DEVICE}   compute dtype = {DTYPE}")
    print(f"    float8_e4m3fn = {E4M3}")
    print(f"    float8_e8m0fnu(E8M0) = {E8M0}")
    ops = [
        "npu_dynamic_mx_quant",
        "npu_quant_matmul",
        "npu_grouped_matmul",
        "npu_grouped_matmul_swiglu_quant_v2",
        "npu_moe_init_routing_v2",
        "npu_moe_finalize_routing",
        "npu_swiglu",
    ]
    for op in ops:
        print(f"    torch_npu.{op:<38} = {'OK' if hasattr(torch_npu, op) else 'MISSING'}")


# ---------------------------------------------------------------------------
# Q1 — the load-bearing question: npu_dynamic_mx_quant scale ndim/shape
# ---------------------------------------------------------------------------
def probe_scale_layout():
    _line("=")
    print("Q1  npu_dynamic_mx_quant SCALE LAYOUT  (2D [N,K/32] vs 3D [N,K/64,2] ?)")
    for tag, N, K in [("weight-like [2I,H]", TWO_I, H),
                      ("act-like    [M,H] ", 128, H),
                      ("down-w      [H,I] ", H, I)]:
        try:
            x = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
            qx, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=E4M3)
            print(f"  [{tag}] N={N} K={K}  (K/32={K//32}, K/64={K//64})")
            _stat("qx   ", qx)
            _stat("scale", scale)
            if scale.ndim == 2:
                print(f"    => 2D. K/32={K//32} == scale.shape[-1]={scale.shape[-1]}? "
                      f"{K//32 == scale.shape[-1]}  (needs maybe_normalize -> 3D)")
            elif scale.ndim == 3:
                print(f"    => 3D pair-split already. matches [N,K/64,2]="
                      f"[{N},{K//64},2]? {tuple(scale.shape) == (N, K//64, 2)}")
        except Exception as e:  # noqa: BLE001
            print(f"  [{tag}] FAILED: {type(e).__name__}: {e}")
    # 3D input (per-expert stacked) — does the kernel accept [E,N,K] directly?
    try:
        x = torch.randn(E, TWO_I, H, device=DEVICE, dtype=DTYPE)
        qx, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=E4M3)
        print("  [3D input [E,2I,H]] kernel accepted 3D directly:")
        _stat("qx   ", qx)
        _stat("scale", scale)
    except Exception as e:  # noqa: BLE001
        print(f"  [3D input [E,2I,H]] rejected (=> must loop per-expert): "
              f"{type(e).__name__}: {e}")


def _normalize_scale(scale):
    """vllm-ascend maybe_normalize_mxfp_scale_layout: 2D [.,K] -> 3D [.,K/2,2]."""
    if scale is None or scale.ndim != 2:
        return scale
    if scale.shape[-1] % 2 != 0:
        raise ValueError(f"odd MXFP scale last dim: {tuple(scale.shape)}")
    return scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)


# ---------------------------------------------------------------------------
# Q2 — dense MXFP8 matmul sanity (this path is known-good in prod)
# ---------------------------------------------------------------------------
def probe_dense_matmul():
    _line("=")
    print("Q2  DENSE npu_quant_matmul MXFP8 vs bf16  (sanity of e8m0/group_sizes)")
    M, K, N = 128, H, TWO_I
    try:
        x = torch.randn(M, K, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)  # [out,in]
        ref = torch.matmul(x, w.t())

        qw, w_scale = torch_npu.npu_dynamic_mx_quant(w, dst_type=E4M3)
        qw_t = qw.transpose(0, 1)              # [in,out] strided
        w_scale_t = w_scale.transpose(0, 1)    # strided
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=E4M3)
        out = torch_npu.npu_quant_matmul(
            qx, qw_t, w_scale_t,
            scale_dtype=E8M0,
            pertoken_scale=x_scale,
            pertoken_scale_dtype=E8M0,
            bias=None,
            output_dtype=DTYPE,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )
        cos, mx, mn = _cos_relerr(ref, out)
        print(f"    cos={cos:.5f}  max_rel={mx:.3f}  mean_rel={mn:.4f}  "
              f"{'PASS' if cos > 0.99 else 'CHECK'}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# helpers to build MoE weights online (mirror process_weights_after_loading)
# ---------------------------------------------------------------------------
def _mx_quant_expertwise(w3d):
    """w3d [E,N,K] bf16 -> (qw [E,N,K] e4m3, scale [E,N,K/64,2] uint8-ish).

    fp8 can't torch.stack on NPU -> stack via uint8 view. Adapts to whatever
    ndim npu_dynamic_mx_quant returns per expert (2D or 3D).
    """
    e = w3d.shape[0]
    qw_list, s_list = [], []
    for i in range(e):
        qw_i, s_i = torch_npu.npu_dynamic_mx_quant(w3d[i], dst_type=E4M3)
        if s_i.ndim == 2:            # [N, K/32] -> [N, K/64, 2]
            s_i = s_i.reshape(s_i.shape[0], s_i.shape[1] // 2, 2)
        qw_list.append(qw_i.view(torch.uint8))
        s_list.append(s_i)
    qw = torch.stack(qw_list, 0).view(E4M3)     # [E,N,K]
    scale = torch.stack(s_list, 0)              # [E,N,K/64,2]
    return qw, scale


def _build_moe_weights(contiguous):
    """Return w13/w2 fp8 + scales in the vllm-ascend transposed layout.

    w13: [E,2I,H] -> transpose(1,2) -> [E,H,2I]      scale [E,2I,H/64,2]->[E,H/64,2I,2]
    w2 : [E,H,I ] -> transpose(1,2) -> [E,I,H ]      scale [E,H,I/64,2 ]->[E,I/64,H,2 ]
    """
    w13_bf = torch.randn(E, TWO_I, H, device=DEVICE, dtype=DTYPE) * 0.05
    w2_bf = torch.randn(E, H, I, device=DEVICE, dtype=DTYPE) * 0.05

    qw13, s13 = _mx_quant_expertwise(w13_bf)      # [E,2I,H], [E,2I,H/64,2]
    qw2, s2 = _mx_quant_expertwise(w2_bf)         # [E,H,I],  [E,H,I/64,2]

    w13 = qw13.transpose(1, 2)                    # [E,H,2I]
    w13_scale = s13.transpose(1, 2)               # [E,H/64,2I,2]
    w2 = qw2.transpose(1, 2)                       # [E,I,H]
    w2_scale = s2.transpose(1, 2)                  # [E,I/64,H,2]
    if contiguous:
        w13, w13_scale = w13.contiguous(), w13_scale.contiguous()
        w2, w2_scale = w2.contiguous(), w2_scale.contiguous()
    return (w13_bf, w2_bf, w13, w13_scale, w2, w2_scale)


# ---------------------------------------------------------------------------
# Q3-Q6 — gmm1 + gmm2, strided vs contiguous, numeric ground truth
# ---------------------------------------------------------------------------
def probe_moe_forward(contiguous):
    tag = "CONTIGUOUS" if contiguous else "STRIDED-VIEW"
    _line("=")
    print(f"Q3-Q6  MoE gmm1->swiglu->gmm2  [{tag} weights]  vs bf16 per-expert ref")
    try:
        w13_bf, w2_bf, w13, w13_scale, w2, w2_scale = _build_moe_weights(contiguous)
        _stat("w13(e4m3)", w13)
        _stat("w13_scale", w13_scale)
        _stat("w2 (e4m3)", w2)
        _stat("w2_scale ", w2_scale)

        # top_k=1: build tokens grouped by expert (already permuted / sorted).
        counts = [3, 5, 2, 6]                      # tokens per expert, sum=16
        assert len(counts) == E
        x_sorted = torch.randn(sum(counts), H, device=DEVICE, dtype=DTYPE)
        group_count = torch.tensor(counts, device=DEVICE, dtype=torch.int64)
        group_cumsum = group_count.cumsum(0)       # [3,8,10,16]

        # ---- bf16 per-expert reference ----
        ref_parts = []
        off = 0
        for e in range(E):
            n = counts[e]
            xe = x_sorted[off:off + n]
            gate_up = xe @ w13_bf[e].transpose(0, 1)   # [n,2I]
            h = _swiglu_ref(gate_up)                   # [n,I]
            ye = h @ w2_bf[e].transpose(0, 1)          # [n,H]
            ref_parts.append(ye)
            off += n
        y_ref = torch.cat(ref_parts, 0)

        # ---- MXFP8 path ----
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(x_sorted, dst_type=E4M3)
        x_scale = _normalize_scale(x_scale)
        _stat("qx       ", qx)
        _stat("x_scale  ", x_scale)

        # gmm1: cumulative group_list
        g1_out, g1_scale = torch_npu.npu_grouped_matmul_swiglu_quant_v2(
            x=qx,
            weight=[w13],
            group_list=group_cumsum,
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
        print("  gmm1 OK")
        _stat("g1_out   ", g1_out)
        _stat("g1_scale ", g1_scale)
        g1_scale = _normalize_scale(g1_scale)

        # gmm2: original COUNT group_list + explicit group_list_type=1
        y_mx = torch_npu.npu_grouped_matmul(
            x=[g1_out],
            weight=[w2],
            scale=[w2_scale],
            bias=None,
            per_token_scale=[g1_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_count,
            output_dtype=DTYPE,
            scale_dtype=E8M0,
            per_token_scale_dtype=E8M0,
            x_dtype=None,
            weight_dtype=None,
        )[0]
        print("  gmm2 OK")
        _stat("y_mx     ", y_mx)
        _stat("y_ref    ", y_ref)

        cos, mx, mn = _cos_relerr(y_ref, y_mx)
        verdict = "PASS (kernel path numerically correct)" if cos > 0.97 else \
                  "FAIL (garbage — layout/contract still wrong)"
        print(f"  >>> cos={cos:.5f}  max_rel={mx:.3f}  mean_rel={mn:.4f}  => {verdict}")
    except Exception as e:  # noqa: BLE001
        import traceback
        print(f"  FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Q7 — npu_moe_init_routing_v2 signature/return (for the real apply())
# ---------------------------------------------------------------------------
def probe_init_routing():
    _line("=")
    print("Q7  npu_moe_init_routing_v2 return arity/shapes (top_k routing)")
    try:
        num_tokens, top_k = 8, 2
        x = torch.randn(num_tokens, H, device=DEVICE, dtype=DTYPE)
        topk_ids = torch.randint(0, E, (num_tokens, top_k), device=DEVICE, dtype=torch.int32)
        ret = torch_npu.npu_moe_init_routing_v2(
            x, topk_ids,
            scale=None,
            active_num=num_tokens * top_k,
            expert_num=E,
            expert_tokens_num_type=1,      # COUNT
            expert_tokens_num_flag=True,
            active_expert_range=[0, E],
            quant_mode=-1,
            x_dtype=None,
        )
        print(f"    returned {len(ret)} tensors:")
        for idx, t in enumerate(ret):
            _stat(f"ret[{idx}]", t if isinstance(t, torch.Tensor) else None)
            if not isinstance(t, torch.Tensor):
                print(f"    ret[{idx}] = {t!r}")
    except Exception as e:  # noqa: BLE001
        print(f"    FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    probe_capabilities()
    probe_scale_layout()
    probe_dense_matmul()
    probe_init_routing()
    probe_moe_forward(contiguous=False)   # STRIDED view first (what prod wants)
    probe_moe_forward(contiguous=True)    # fallback if strided rejected
    _line("=")
    print("DONE. Read Q1 (scale ndim), Q6 cos per layout. The layout whose Q6")
    print("prints PASS is the one process_weights_after_loading must produce.")

"""A5 MXFP8 MoE OFFLINE checkpoint probe — pin down why offline is garbage.

Online MXFP8 MoE works; offline (msmodelslim W8A8_MXFP8 checkpoint) is garbage.
Static review shows the offline code path is structurally identical to online
and the scale pairing is contiguous (probe Q8), so the only remaining suspect
is the VALUE domain: do msmodelslim's stored fp8 weight + (exp+127) uint8 scale
bytes actually mean, under the kernel's e8m0 semantics, the same numbers the
kernel expects?

This loads ONE real expert projection from the offline checkpoint and answers,
decisively:

  A. Does the LOADED offline weight+scale, laid out exactly as
     process_weights_after_loading does it, reproduce a bf16 dequant reference
     through npu_quant_matmul?  (cos high => per-proj repr+layout CORRECT for
     the kernel => bug is in fusion/routing/assembly, not the scale convention.
     cos low => the offline weight/scale/layout is wrong for the kernel.)

  B. If we dequantise the offline weight and RE-quantise it with
     npu_dynamic_mx_quant (the online/kernel-native path), do we get back the
     SAME fp8 bytes and the SAME scale bytes?  (match => msmodelslim convention
     == kernel convention. mismatch => found the convention bug, and the diff
     shows the fix.)

Run on the A5 box, pointing at the msmodelslim offline checkpoint dir:
    python llm/probe_mxfp8_moe_offline.py /path/to/Qwen3-30B-A3B-W8A8_MXFP8

(the dir with *.safetensors + quant_model_description.json)
"""

import glob
import os
import sys

import torch
import torch.nn.functional as F
import torch_npu
from safetensors import safe_open

MXFP8_BLOCK_SIZE = 32
E4M3 = torch.float8_e4m3fn
E8M0 = getattr(torch_npu, "float8_e8m0fnu", getattr(torch, "float8_e8m0fnu", None))
DEVICE = f"npu:{torch.npu.current_device()}"
DTYPE = torch.bfloat16

# layer 0, expert 0 — adjust here if your checkpoint names differ.
GATE_W = "model.layers.0.mlp.experts.0.gate_proj.weight"
GATE_S = "model.layers.0.mlp.experts.0.gate_proj.weight_scale"
DOWN_W = "model.layers.0.mlp.experts.0.down_proj.weight"
DOWN_S = "model.layers.0.mlp.experts.0.down_proj.weight_scale"


def _stat(name, t):
    print(f"    {name}: shape={tuple(t.shape)} ndim={t.ndim} dtype={t.dtype}")


def _load_tensors(ckpt_dir, names):
    """Find and load the given tensor names from any *.safetensors shard."""
    want = set(names)
    found = {}
    files = sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no *.safetensors under {ckpt_dir}")
    for f in files:
        with safe_open(f, framework="pt") as fh:
            keys = set(fh.keys())
            for n in list(want):
                if n in keys:
                    found[n] = fh.get_tensor(n)
                    want.discard(n)
        if not want:
            break
    if want:
        raise KeyError(f"missing tensors in checkpoint: {sorted(want)}")
    return found


def _dequant_ref(w_fp8, scale_u8, block_axis):
    """Dequantise offline weight -> bf16 the way the kernel's e8m0 does:
    real = fp8 * 2^(scale_byte - 127), one scale per 32-wide block along block_axis.
    """
    w = w_fp8.float()
    exp = scale_u8.float() - 127.0                       # e8m0 byte -> exponent
    factor = torch.pow(torch.tensor(2.0, device=w.device), exp)
    # expand block factor back to per-element along block_axis
    factor = torch.repeat_interleave(factor, MXFP8_BLOCK_SIZE, dim=block_axis)
    return (w * factor).to(DTYPE)


def _cos(ref, out):
    a, b = ref.float().flatten(), out.float().flatten()
    return F.cosine_similarity(a, b, dim=0).item()


def probe_offline_proj(tag, w_fp8, scale_u8):
    """w_fp8 [out,in] fp8, scale_u8 [out, in//32] uint8. Block over `in` (dim 1)."""
    print("=" * 96)
    print(f"{tag}: offline weight+scale repr / layout / convention")
    w_fp8 = w_fp8.to(DEVICE)
    scale_u8 = scale_u8.to(DEVICE)
    _stat("w_fp8   ", w_fp8)
    _stat("scale_u8", scale_u8)
    out_dim, in_dim = w_fp8.shape
    if scale_u8.shape != (out_dim, in_dim // MXFP8_BLOCK_SIZE):
        print(f"    !! unexpected scale shape; expected {(out_dim, in_dim//32)}")

    # bf16 dequant reference (kernel e8m0 semantics)
    w_deq = _dequant_ref(w_fp8, scale_u8, block_axis=1)

    # ---- Test A: run LOADED offline weight+scale through the kernel ----
    # Mirror process_weights_after_loading offline branch (dense analogue):
    #   scale [out, in//32] -> [out, in//64, 2] -> transpose -> [in//64, out, 2]
    #   weight [out, in] -> transpose -> [in, out]  (strided)
    n, kb = scale_u8.shape
    scale_kernel = scale_u8.reshape(n, kb // 2, 2).transpose(0, 1)   # [in//64, out, 2]
    w_kernel = w_fp8.transpose(0, 1)                                  # [in, out] strided
    x = torch.randn(64, in_dim, device=DEVICE, dtype=DTYPE)
    ref = torch.matmul(x, w_deq.transpose(0, 1))
    try:
        qx, x_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=E4M3)
        out = torch_npu.npu_quant_matmul(
            qx, w_kernel, scale_kernel,
            scale_dtype=E8M0,
            pertoken_scale=x_scale, pertoken_scale_dtype=E8M0,
            bias=None, output_dtype=DTYPE,
            group_sizes=[1, 1, MXFP8_BLOCK_SIZE],
        )
        cosA = _cos(ref, out)
        print(f"  [A] kernel(loaded offline w+scale) vs bf16-dequant matmul: "
              f"cos={cosA:.5f}  {'PASS repr+layout OK' if cosA > 0.97 else 'FAIL repr/layout WRONG'}")
    except Exception as e:  # noqa: BLE001
        print(f"  [A] FAILED: {type(e).__name__}: {e}")

    # ---- Test B: re-quantise the dequant with npu (kernel-native) ----
    try:
        qw_on, s_on = torch_npu.npu_dynamic_mx_quant(w_deq, dst_type=E4M3)  # [out,in], [out,in//64,2]
        s_on_flat = s_on.reshape(n, kb)                                     # back to [out, in//32]
        # compare bytes
        w_match = (qw_on.view(torch.uint8) == w_fp8.view(torch.uint8)).float().mean().item()
        s_match = (s_on_flat == scale_u8).float().mean().item()
        s_diff = (s_on_flat.float() - scale_u8.float())
        print(f"  [B] re-quantised vs loaded:  weight-byte match={w_match:.4f}  "
              f"scale-byte match={s_match:.4f}")
        print(f"      scale diff (npu - offline): min={s_diff.min().item():.0f} "
              f"max={s_diff.max().item():.0f} mean={s_diff.mean().item():.3f}")
        if s_match > 0.99:
            print("      => msmodelslim scale convention == kernel e8m0. Scale is fine.")
        else:
            uniq = torch.unique(s_diff)
            print(f"      => convention MISMATCH. unique diffs: {uniq[:8].tolist()}"
                  f"{' ...' if uniq.numel() > 8 else ''}")
            print("      (a single constant diff => uniform bias fix; varied => deeper.)")
    except Exception as e:  # noqa: BLE001
        print(f"  [B] FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("ERROR: pass the offline checkpoint dir as argv[1].")
        sys.exit(1)
    ckpt = sys.argv[1]
    print(f"device={DEVICE}  E8M0={E8M0}  ckpt={ckpt}")
    tensors = _load_tensors(ckpt, [GATE_W, GATE_S, DOWN_W, DOWN_S])
    probe_offline_proj("GATE_PROJ (block over in=H)", tensors[GATE_W], tensors[GATE_S])
    probe_offline_proj("DOWN_PROJ (block over in=I)", tensors[DOWN_W], tensors[DOWN_S])
    print("=" * 96)
    print("Read: [A] PASS + [B] match => offline weight/scale/layout all correct for the")
    print("kernel; the garbage is in fusion/routing/assembly, not the per-proj repr.")
    print("[A] FAIL or [B] mismatch => the offline weight/scale itself is wrong (convention).")

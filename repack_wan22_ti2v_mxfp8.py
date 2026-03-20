"""Repack msmodelslim MXFP8 quantized weights for Wan2.2 TI2V into HF Diffusers format.

Merges:
- Original HF model weights (for unquantized layers, VAE, text_encoder, etc.)
- msmodelslim quantized weights (MXFP8 layers: weight + weight_scale)

Output: a self-contained model directory loadable by SGLang with --quantization modelslim.

Usage:
    python repack_wan22_ti2v_mxfp8.py \
        --original-model-path /home/weights/Wan2.2-TI2V-5B-Diffusers \
        --quant-weights-path /path/to/msmodelslim/output \
        --output-path /home/weights/Wan2.2-TI2V-5B-Diffusers-MXFP8
"""

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors.torch import load_file, save_file

# Official (msmodelslim) -> HF Diffusers key mapping
# Sourced from sglang/python/sglang/multimodal_gen/tools/wan_repack.py
TRANSFORMER_KEYS_RENAME_DICT = {
    "time_embedding.0": "condition_embedder.time_embedder.linear_1",
    "time_embedding.2": "condition_embedder.time_embedder.linear_2",
    "text_embedding.0": "condition_embedder.text_embedder.linear_1",
    "text_embedding.2": "condition_embedder.text_embedder.linear_2",
    "time_projection.1": "condition_embedder.time_proj",
    "head.modulation": "scale_shift_table",
    "head.head": "proj_out",
    "modulation": "scale_shift_table",
    "ffn.0": "ffn.net.0.proj",
    "ffn.2": "ffn.net.2",
    # Hack to swap the layer names
    # The original model calls the norms in following order: norm1, norm3, norm2
    # We convert it to: norm1, norm2, norm3
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # For the I2V model
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    # for the FLF2V model
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Attention component mappings
    "self_attn.q": "attn1.to_q",
    "self_attn.k": "attn1.to_k",
    "self_attn.v": "attn1.to_v",
    "self_attn.o": "attn1.to_out.0",
    "self_attn.norm_q": "attn1.norm_q",
    "self_attn.norm_k": "attn1.norm_k",
    "cross_attn.q": "attn2.to_q",
    "cross_attn.k": "attn2.to_k",
    "cross_attn.v": "attn2.to_v",
    "cross_attn.o": "attn2.to_out.0",
    "cross_attn.norm_q": "attn2.norm_q",
    "cross_attn.norm_k": "attn2.norm_k",
    "attn2.to_k_img": "attn2.add_k_proj",
    "attn2.to_v_img": "attn2.add_v_proj",
    "attn2.norm_k_img": "attn2.norm_added_k",
}


def rename_key(key: str) -> str:
    """Rename a key from Official (msmodelslim) format to HF Diffusers format."""
    new_key = key
    for old, new in TRANSFORMER_KEYS_RENAME_DICT.items():
        new_key = new_key.replace(old, new)
    return new_key


def load_original_transformer_weights(original_model_path: Path) -> dict:
    """Load all transformer weights from the original HF model."""
    transformer_dir = original_model_path / "transformer"
    state_dict = {}
    for sf_file in sorted(transformer_dir.glob("diffusion_pytorch_model*.safetensors")):
        print(f"  Loading original weights: {sf_file.name}")
        state_dict.update(load_file(str(sf_file)))
    if not state_dict:
        raise FileNotFoundError(
            f"No diffusion_pytorch_model*.safetensors found in {transformer_dir}"
        )
    return state_dict


def load_quant_weights_and_config(quant_weights_path: Path):
    """Load msmodelslim quantized weights and quant description."""
    # Find quantized safetensors
    quant_sf_candidates = sorted(quant_weights_path.glob("quant_model_weight*.safetensors"))
    if not quant_sf_candidates:
        raise FileNotFoundError(
            f"No quant_model_weight*.safetensors found in {quant_weights_path}"
        )
    quant_sf_file = quant_sf_candidates[0]
    print(f"  Loading quantized weights: {quant_sf_file.name}")
    quant_state_dict = load_file(str(quant_sf_file))

    # Find quant description JSON
    quant_desc_candidates = sorted(quant_weights_path.glob("quant_model_description*.json"))
    if not quant_desc_candidates:
        raise FileNotFoundError(
            f"No quant_model_description*.json found in {quant_weights_path}"
        )
    quant_desc_file = quant_desc_candidates[0]
    print(f"  Loading quant description: {quant_desc_file.name}")
    with open(quant_desc_file) as f:
        quant_description = json.load(f)

    return quant_state_dict, quant_description


def repack(original_model_path: Path, quant_weights_path: Path, output_path: Path):
    print("Step 1: Loading original transformer weights...")
    original_sd = load_original_transformer_weights(original_model_path)

    print("Step 2: Loading msmodelslim quantized weights and description...")
    quant_sd, quant_desc = load_quant_weights_and_config(quant_weights_path)

    print("Step 3: Renaming quantized keys (Official -> HF Diffusers)...")
    renamed_quant_sd = {}
    for key, tensor in quant_sd.items():
        new_key = rename_key(key)
        renamed_quant_sd[new_key] = tensor
        if new_key != key:
            print(f"    {key} -> {new_key}")

    renamed_quant_desc = {}
    for key, value in quant_desc.items():
        new_key = rename_key(key)
        renamed_quant_desc[new_key] = value

    print("Step 4: Merging weights...")
    # Start with original, override with quantized
    merged_sd = dict(original_sd)
    overridden = 0
    new_keys = 0
    for key, tensor in renamed_quant_sd.items():
        if key in merged_sd:
            overridden += 1
        else:
            new_keys += 1
        merged_sd[key] = tensor
    print(f"    Overridden: {overridden}, New (weight_scale etc.): {new_keys}")
    print(f"    Total keys in merged: {len(merged_sd)}")

    print("Step 5: Saving repacked model...")
    output_transformer_dir = output_path / "transformer"
    output_transformer_dir.mkdir(parents=True, exist_ok=True)

    # Save merged safetensors
    output_sf = output_transformer_dir / "diffusion_pytorch_model.safetensors"
    save_file(merged_sd, str(output_sf))
    print(f"    Saved: {output_sf}")

    # Save remapped quant description
    output_desc = output_transformer_dir / "quant_model_description.json"
    with open(output_desc, "w") as f:
        json.dump(renamed_quant_desc, f, indent=2)
    print(f"    Saved: {output_desc}")

    # Copy transformer config.json
    src_config = original_model_path / "transformer" / "config.json"
    if src_config.is_file():
        shutil.copy2(str(src_config), str(output_transformer_dir / "config.json"))
        print(f"    Copied: config.json")

    print("Step 6: Symlinking other components from original model...")
    # Symlink VAE, text_encoder, and other top-level components
    for item in original_model_path.iterdir():
        if item.name == "transformer":
            continue
        dst = output_path / item.name
        if dst.exists():
            continue
        if item.is_dir():
            os.symlink(str(item.resolve()), str(dst), target_is_directory=True)
            print(f"    Symlinked dir: {item.name}")
        elif item.is_file():
            os.symlink(str(item.resolve()), str(dst))
            print(f"    Symlinked file: {item.name}")

    print(f"\nRepack complete! Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Repack msmodelslim MXFP8 weights for Wan2.2 TI2V into HF format"
    )
    parser.add_argument(
        "--original-model-path",
        type=str,
        required=True,
        help="Path to original HF model (e.g., /home/weights/Wan2.2-TI2V-5B-Diffusers)",
    )
    parser.add_argument(
        "--quant-weights-path",
        type=str,
        required=True,
        help="Path to msmodelslim output directory with quantized weights",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output directory for repacked model",
    )
    args = parser.parse_args()

    repack(
        original_model_path=Path(args.original_model_path),
        quant_weights_path=Path(args.quant_weights_path),
        output_path=Path(args.output_path),
    )


if __name__ == "__main__":
    main()

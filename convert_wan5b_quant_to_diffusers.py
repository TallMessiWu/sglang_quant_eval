import argparse
import json
import pathlib
from typing import Any, Dict

from safetensors.torch import load_file, save_file

# Key rename mapping to align with the original Wan architecture
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
    # Fix the stacking order of normalization layers
    "norm2": "norm__placeholder",
    "norm3": "norm2",
    "norm__placeholder": "norm3",
    # Visual feature embeddings specific to image-to-video (I2V/TI2V)
    "img_emb.proj.0": "condition_embedder.image_embedder.norm1",
    "img_emb.proj.1": "condition_embedder.image_embedder.ff.net.0.proj",
    "img_emb.proj.3": "condition_embedder.image_embedder.ff.net.2",
    "img_emb.proj.4": "condition_embedder.image_embedder.norm2",
    "img_emb.emb_pos": "condition_embedder.image_embedder.pos_embed",
    # Attention mechanism mappings
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

def update_dict_(dict_obj: Dict[str, Any], old_key: str, new_key: str) -> None:
    dict_obj[new_key] = dict_obj.pop(old_key)

def convert_quantized_transformer(quant_dir: str, output_dir: str):
    quant_path = pathlib.Path(quant_dir)
    out_path = pathlib.Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Auto-match quantization files (supports any quantization suffix)
    model_files = list(quant_path.glob("quant_model_weight*.safetensors"))
    json_files = list(quant_path.glob("quant_model_description*.json"))

    if not model_files:
        raise FileNotFoundError(f"No file matching quant_model_weight*.safetensors found in {quant_path}")
    if len(model_files) > 1:
        raise FileNotFoundError(f"Multiple weight files found in {quant_path}, please check: {model_files}")
    if not json_files:
        raise FileNotFoundError(f"No file matching quant_model_description*.json found in {quant_path}")
    if len(json_files) > 1:
        raise FileNotFoundError(f"Multiple config files found in {quant_path}, please check: {json_files}")

    model_file = model_files[0]
    json_file = json_files[0]

    print(f"[*] Loading quantized weights: {model_file.name}...")
    state_dict = load_file(model_file)

    print(f"[*] Loading quantization config: {json_file.name}...")
    with open(json_file, "r") as f:
        quant_config = json.load(f)

    # 2. Rename keys to Diffusers format
    print("[*] Converting model structure (mapping to diffusers naming convention)...")
    for key in list(state_dict.keys()):
        new_key = key[:]
        for replace_key, rename_key in TRANSFORMER_KEYS_RENAME_DICT.items():
            new_key = new_key.replace(replace_key, rename_key)

        # Only update the dict if the key changed
        if new_key != key:
            update_dict_(state_dict, key, new_key)
            # The quant JSON may not contain all model keys (typically only quantized Linear layers)
            if key in quant_config:
                update_dict_(quant_config, key, new_key)

    # 3. Save using Diffusers standard naming convention
    out_model_path = out_path / "diffusion_pytorch_model.safetensors"
    out_json_path = out_path / "quant_model_description.json"

    print(f"[*] Saving Diffusers-format weights to: {out_model_path}")
    save_file(state_dict, out_model_path)

    print(f"[*] Saving config to: {out_json_path}")
    with open(out_json_path, "w") as f:
        json.dump(quant_config, f, indent=2)

    print("\n[+] Conversion complete! You can now replace the contents of your diffusers transformer directory with the output directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Wan2.2 quantized weights to Diffusers format")
    parser.add_argument("--quant-path", type=str, required=True, help="Directory containing quantized weights")
    parser.add_argument("--output-path", type=str, required=True, help="Output directory (typically diffusers_backup/transformer)")
    args = parser.parse_args()

    convert_quantized_transformer(args.quant_path, args.output_path)
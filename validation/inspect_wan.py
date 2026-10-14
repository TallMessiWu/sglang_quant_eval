"""Read Wan checkpoint metadata and safetensors headers without loading weights.

Usage: python3 validation/inspect_wan.py /mnt/share/weight/Wan2.2*
No torch, torch_npu, or safetensors installation is required.
"""

import argparse
import collections
import hashlib
import json
import struct
from pathlib import Path


def safetensors_header(path):
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("Missing safetensors header length")
        length = struct.unpack("<Q", length_bytes)[0]
        if length > 100 * 1024 * 1024:
            raise ValueError("Header exceeds 100 MiB; file may be a Git LFS pointer")
        header = json.loads(stream.read(length))
    return {key: value for key, value in header.items() if key != "__metadata__"}


def inspect(directory):
    result = {
        "path": str(directory),
        "resolved_path": str(directory.resolve()),
        "exists": directory.is_dir(),
        "components": [],
        "scope": "Metadata/header inspection only; no weight conversion or NPU validation",
    }
    if not directory.is_dir():
        return result
    result["top_level_entries"] = sorted(
        p.name for p in directory.iterdir() if not p.name.startswith(".")
    )
    # Include one extra directory level for /Wan2.2/<variant>/<noise_model>.
    folders = [directory]
    for _ in range(2):
        folders += [
            child
            for parent in list(folders)
            for child in sorted(parent.iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and child not in folders
        ]
        folders = list(dict.fromkeys(folders))
    for folder in folders:
        descriptions = sorted(folder.glob("quant_model_description*.json"))
        configs = [
            folder / name
            for name in ("model_index.json", "config.json")
            if (folder / name).is_file()
        ]
        weights = sorted(folder.glob("*.safetensors"))
        if not (configs or descriptions or weights):
            continue
        row = {
            "directory": str(folder.relative_to(directory)),
            "configs": {},
            "quantization": [],
            "weight_files": [p.name for p in weights],
        }
        quantized_keys = set()
        for path in configs:
            try:
                data = json.loads(path.read_text())
                row["configs"][path.name] = data
            except (OSError, ValueError) as exc:
                row["configs"][path.name] = {"error": str(exc)}
        for path in descriptions:
            try:
                data = json.loads(path.read_text())
                counts = collections.Counter(
                    value
                    for key, value in data.items()
                    if key.endswith(".weight") and isinstance(value, str)
                )
                quantized_keys.update(
                    key
                    for key, value in data.items()
                    if key.endswith(".weight") and value != "FLOAT"
                )
                row["quantization"].append(
                    {
                        "file": path.name,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "weight_scheme_counts": dict(counts),
                        "examples_by_scheme": {
                            scheme: [
                                key for key, value in data.items() if value == scheme
                            ][:3]
                            for scheme in counts
                        },
                        "non_weight_metadata_examples": dict(
                            [
                                (key, value)
                                for key, value in data.items()
                                if not key.endswith(
                                    (
                                        ".weight",
                                        ".bias",
                                        ".weight_scale",
                                        ".weight_offset",
                                    )
                                )
                            ][:20]
                        ),
                    }
                )
            except (OSError, ValueError) as exc:
                row["quantization"].append({"file": path.name, "error": str(exc)})
        row["headers"] = []
        all_tensors = {}
        for path in weights:
            try:
                header = safetensors_header(path)
                duplicates = sorted(all_tensors.keys() & header.keys())
                all_tensors.update(header)
                examples = [
                    key
                    for key in header
                    if "weight_scale" in key or key.endswith(".weight")
                ][:8]
                auxiliary = [
                    key
                    for key in header
                    if any(
                        token in key.lower()
                        for token in (
                            "rot",
                            "smooth",
                            "mul_scale",
                            "hadamard",
                            "scale_bias",
                            "offset",
                        )
                    )
                ]
                row["headers"].append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "dtype_counts": dict(
                            collections.Counter(v["dtype"] for v in header.values())
                        ),
                        "duplicate_keys_in_other_shards": duplicates[:10],
                        "auxiliary_tensor_count": len(auxiliary),
                        "auxiliary_examples": {
                            key: {
                                "dtype": header[key]["dtype"],
                                "shape": header[key]["shape"],
                            }
                            for key in auxiliary[:12]
                        },
                        "examples": {
                            k: {
                                "dtype": header[k]["dtype"],
                                "shape": header[k]["shape"],
                            }
                            for k in examples
                        },
                    }
                )
            except (OSError, ValueError, KeyError) as exc:
                row["headers"].append({"file": path.name, "error": str(exc)})
        row["quantized_weight_examples"] = {}
        for key in sorted(quantized_keys)[:8]:
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            row["quantized_weight_examples"][key] = {
                name: {
                    "dtype": all_tensors[name]["dtype"],
                    "shape": all_tensors[name]["shape"],
                }
                if name in all_tensors
                else "not found in this directory"
                for name in (key, scale_key)
            }
        row["quantized_keys_absent_from_headers"] = sorted(
            quantized_keys - all_tensors.keys()
        )[:20]
        result["components"].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            [inspect(path) for path in args.directories], indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()

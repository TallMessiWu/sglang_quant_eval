#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_PROMPTS = [
    "Give one concise reason tensor parallelism is useful.",
    "Explain mixture-of-experts routing in two sentences.",
    "Write a short Python function that adds two integers.",
]
ADAPTER_NAMES = ("adapter_a", "adapter_b")
PER_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
SHARED_OUTER_RE = re.compile(r"\.experts\.(?!\d+\.)")


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def _payload(prompts: list[str], mapping: list[str | None] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "text": prompts,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 32,
            "seed": 0,
        },
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 1,
    }
    if mapping is not None:
        payload["lora_path"] = mapping
    return payload


def _cases(case: str, replays: int) -> list[tuple[str, list[str | None] | None]]:
    if case == "base":
        return [(f"base-{index}", None) for index in range(replays)]

    fixed = [
        ("base-only", [None, None, None]),
        ("single-a", ["adapter_a", "adapter_a", "adapter_a"]),
        ("mixed", [None, "adapter_a", "adapter_b"]),
        ("base-only-replay", [None, None, None]),
        ("reverse-mixed", ["adapter_b", None, "adapter_a"]),
    ]
    fixed.extend(
        (f"mixed-replay-{index}", [None, "adapter_a", "adapter_b"])
        for index in range(replays)
    )
    return fixed


def capture(args: argparse.Namespace) -> None:
    prompts = DEFAULT_PROMPTS
    records = []
    endpoint = args.base_url.rstrip("/") + "/generate"
    for label, mapping in _cases(args.case, args.replays):
        print(f"request={label} mapping={mapping}", flush=True)
        response = _post_json(endpoint, _payload(prompts, mapping), args.timeout)
        if not isinstance(response, list) or len(response) != len(prompts):
            raise RuntimeError(
                f"Expected a {len(prompts)}-row batch response, got {type(response)}"
            )
        records.append({"label": label, "mapping": mapping, "response": response})

    artifact = {
        "schema_version": 1,
        "case": args.case,
        "base_url": args.base_url,
        "prompts": prompts,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote={args.output}")


def _load_artifact(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not data.get("records"):
        raise ValueError(f"Not an Ascend MoE-LoRA validation artifact: {path}")
    return data


def _snapshot(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta_info", {})
    return {
        "text": row.get("text"),
        "output_ids": row.get("output_ids"),
        "output_token_logprobs": meta.get("output_token_logprobs"),
    }


def _token_ids(row: dict[str, Any]) -> list[int]:
    output_ids = row.get("output_ids")
    if output_ids is not None:
        return output_ids
    logprobs = row.get("meta_info", {}).get("output_token_logprobs") or []
    return [item[1] for item in logprobs]


def _selected_logprobs(row: dict[str, Any]) -> list[float]:
    values = row.get("meta_info", {}).get("output_token_logprobs") or []
    return [float(item[0]) for item in values]


def _baseline_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    first = artifact["records"][0]["response"]
    expected = [_snapshot(row) for row in first]
    for record in artifact["records"][1:]:
        actual = [_snapshot(row) for row in record["response"]]
        if actual != expected:
            raise AssertionError(
                f"Pure baseline is not repeatable at record {record['label']}"
            )
    return first


def _iter_rows(artifact: dict[str, Any]):
    for record in artifact["records"]:
        mapping = record.get("mapping")
        if mapping is None:
            mapping = [None] * len(record["response"])
        for prompt_index, (adapter, row) in enumerate(
            zip(mapping, record["response"], strict=True)
        ):
            yield record["label"], prompt_index, adapter, row


def _reference_rows(artifact: dict[str, Any]) -> dict[tuple[int, str], dict[str, Any]]:
    rows: dict[tuple[int, str], dict[str, Any]] = {}
    for _, prompt_index, adapter, row in _iter_rows(artifact):
        if adapter is not None:
            rows.setdefault((prompt_index, adapter), row)
    return rows


def _max_logprob_diff(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_ids, right_ids = _token_ids(left), _token_ids(right)
    if left_ids != right_ids:
        return math.inf
    left_values, right_values = _selected_logprobs(left), _selected_logprobs(right)
    if len(left_values) != len(right_values):
        return math.inf
    return max(
        (abs(a - b) for a, b in zip(left_values, right_values, strict=True)),
        default=0.0,
    )


def compare(args: argparse.Namespace) -> None:
    baseline = _load_artifact(args.baseline)
    candidate = _load_artifact(args.candidate)
    if baseline["prompts"] != candidate["prompts"]:
        raise AssertionError("Baseline and candidate prompts differ")

    baseline_rows = _baseline_rows(baseline)
    base_checks = 0
    adapter_changed = {name: False for name in ADAPTER_NAMES}
    repeated_mixed: list[list[dict[str, Any]]] = []

    for record in candidate["records"]:
        if record["label"] == "mixed" or record["label"].startswith("mixed-replay-"):
            repeated_mixed.append([_snapshot(row) for row in record["response"]])

    if repeated_mixed and any(rows != repeated_mixed[0] for rows in repeated_mixed[1:]):
        raise AssertionError("Fixed-shape mixed mapping changed across graph replays")

    for label, prompt_index, adapter, row in _iter_rows(candidate):
        pure = baseline_rows[prompt_index]
        if adapter is None:
            if _snapshot(row) != _snapshot(pure):
                raise AssertionError(
                    f"Base-only row differs from pure MoE baseline: {label}[{prompt_index}]"
                )
            base_checks += 1
        elif adapter in adapter_changed:
            if _snapshot(row) != _snapshot(pure):
                adapter_changed[adapter] = True

    if base_checks == 0:
        raise AssertionError("Candidate artifact contains no base-only row")
    unchanged = [name for name, changed in adapter_changed.items() if not changed]
    if unchanged:
        raise AssertionError(
            "No observable token/logprob delta for adapters: " + ", ".join(unchanged)
        )

    if args.adapter_reference is not None:
        reference = _reference_rows(_load_artifact(args.adapter_reference))
        comparisons = 0
        for label, prompt_index, adapter, row in _iter_rows(candidate):
            if adapter is None:
                continue
            key = (prompt_index, adapter)
            if key not in reference:
                continue
            expected = reference[key]
            if _token_ids(row) != _token_ids(expected):
                raise AssertionError(
                    f"Adapter token IDs differ from reference: {label}[{prompt_index}]"
                )
            difference = _max_logprob_diff(row, expected)
            if difference > args.logprob_atol:
                raise AssertionError(
                    f"Adapter logprob diff {difference:.6f} exceeds "
                    f"{args.logprob_atol}: {label}[{prompt_index}]"
                )
            comparisons += 1
        if comparisons == 0:
            raise AssertionError("No adapter rows matched the supplied reference")
        print(f"adapter_reference_checks={comparisons}")

    print(f"base_only_exact_checks={base_checks}")
    print(f"adapter_delta={adapter_changed}")
    print(f"fixed_mapping_replays={len(repeated_mixed)}")
    print("PASS")


def _adapter_keys(path: Path) -> list[str]:
    index_path = path / "adapter_model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        return sorted(index["weight_map"])

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No adapter safetensors found in {path}")
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError(
            "Install safetensors to inspect a non-sharded adapter"
        ) from error

    keys: list[str] = []
    for filename in files:
        with safe_open(filename, framework="pt", device="cpu") as handle:
            keys.extend(handle.keys())
    return sorted(keys)


def preflight_adapter(args: argparse.Namespace) -> None:
    config_path = args.adapter / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rank = int(config.get("r", 0))
    if rank not in (8, 16, 32, 64):
        raise AssertionError(f"BGMV rank must be 8, 16, 32, or 64; got {rank}")

    keys = _adapter_keys(args.adapter)
    shared_outer = [key for key in keys if SHARED_OUTER_RE.search(key)]
    shared_expert = [key for key in keys if ".shared_expert" in key]
    per_expert = [key for key in keys if PER_EXPERT_RE.search(key)]
    expert_ids = sorted(
        {int(match.group(1)) for key in per_expert if (match := PER_EXPERT_RE.search(key))}
    )
    required_parts = ("gate_proj", "up_proj", "down_proj")
    missing_parts = [part for part in required_parts if not any(f".{part}." in key for key in per_expert)]

    if shared_outer:
        raise AssertionError(
            "Shared-outer expert tensors are outside the MVP; example: "
            + shared_outer[0]
        )
    if shared_expert:
        raise AssertionError(
            "Shared-expert tensors are outside the MVP; example: " + shared_expert[0]
        )
    if not per_expert:
        raise AssertionError("No '.experts.<id>.' per-expert LoRA tensors found")
    if missing_parts:
        raise AssertionError("Missing expert LoRA parts: " + ", ".join(missing_parts))

    print(f"adapter={args.adapter}")
    print(f"rank={rank}")
    print(f"per_expert_tensor_count={len(per_expert)}")
    print(f"expert_ids={expert_ids}")
    print("PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ascend MoE-LoRA MVP validator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture_parser = subparsers.add_parser("capture", help="Capture a fixed batch")
    capture_parser.add_argument("--case", choices=("base", "mixed"), required=True)
    capture_parser.add_argument(
        "--base-url", default=f"http://127.0.0.1:{os.environ.get('VLLM_PORT', '6969')}"
    )
    capture_parser.add_argument("--output", type=Path, required=True)
    capture_parser.add_argument("--replays", type=int, default=3)
    capture_parser.add_argument("--timeout", type=float, default=600.0)
    capture_parser.set_defaults(func=capture)

    compare_parser = subparsers.add_parser("compare", help="Compare captures")
    compare_parser.add_argument("--baseline", type=Path, required=True)
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--adapter-reference", type=Path)
    compare_parser.add_argument("--logprob-atol", type=float, default=0.1)
    compare_parser.set_defaults(func=compare)

    preflight_parser = subparsers.add_parser(
        "preflight-adapter", help="Reject adapter formats outside the MVP"
    )
    preflight_parser.add_argument("adapter", type=Path)
    preflight_parser.set_defaults(func=preflight_adapter)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "replays") and args.replays < 1:
        raise SystemExit("--replays must be positive")
    try:
        args.func(args)
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

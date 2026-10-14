"""G2 NPU evidence runner. Run with the service's Python and environment.

No mock operators, checkpoint edits, process killing, or precision fallback.
The linear comparison loads the original baseline Scheme from its own checkout;
its dependencies come from the selected target environment. Wan runs each whole
checkout in a separate invocation. A successful probe is not full G2 acceptance.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASELINE = "62ff466ac0c2cdd5521acf7d4c673fa7630eb083"
SCHEME_PATH = (
    "python/sglang/multimodal_gen/runtime/layers/quantization/modelslim_mxfp8_scheme.py"
)
ENV_KEYS = (
    "TMPDIR",
    "XDG_CACHE_HOME",
    "HF_HOME",
    "HF_HUB_CACHE",
    "TORCH_HOME",
    "TORCH_EXTENSIONS_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "SGLANG_DIFFUSION_CACHE_ROOT",
    "PYTHONPATH",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "LD_LIBRARY_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_VISIBLE_DEVICES",
    "ASCEND_DEVICE_ID",
    "ASCEND_USE_FIA",
    "SGLANG_MAMBA_SSM_DTYPE",
    "SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION",
    "SGLANG_DIFFUSION_ATTENTION_BACKEND",
    "SGLANG_CACHE_DIT_ENABLED",
    "SGLANG_CACHE_DIT_FN",
    "SGLANG_CACHE_DIT_BN",
    "SGLANG_CACHE_DIT_WARMUP",
    "SGLANG_CACHE_DIT_RDT",
    "SGLANG_CACHE_DIT_MC",
    "SGLANG_CACHE_DIT_TAYLORSEER",
    "SGLANG_CACHE_DIT_TS_ORDER",
    "HCCL_CONNECT_TIMEOUT",
    "HCCL_EXEC_TIMEOUT",
    "OMP_NUM_THREADS",
    "USE_NZ",
)


def command(argv):
    result = subprocess.run(
        argv, text=True, capture_output=True, timeout=60, check=False
    )
    if result.returncode:
        raise RuntimeError(f"{argv}: {result.stderr.strip()}")
    return result.stdout.strip()


def checkout(path, expected):
    path = Path(path).resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("--expected-head must be a full 40-character SHA")
    head = command(["git", "-C", str(path), "rev-parse", "HEAD"])
    dirty = command(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=normal"]
    )
    if head != expected or dirty:
        raise ValueError(
            f"Wrong/dirty source: {path}, HEAD={head}, expected={expected}, diff={dirty!r}"
        )
    return {"path": str(path), "head": head, "working_tree_changes": dirty}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(report, output):
    path = output / "result.json"
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n")
    temporary.replace(path)


def initialize(args, report):
    report["source"] = checkout(args.source, args.expected_head)
    python_path = str(Path(report["source"]["path"]) / "python")
    sys.path.insert(0, python_path)
    # Scheduler children must import the same checkout.
    os.environ["PYTHONPATH"] = python_path + (
        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
    )
    report["environment"] = {key: os.environ.get(key) for key in ENV_KEYS}
    report["packages"] = {}
    for name in (
        "torch",
        "torch-npu",
        "sgl-kernel-npu",
        "sglang",
        "transformers",
        "diffusers",
        "triton",
    ):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = "not installed as a distribution"
    report["ascend_versions"] = {}
    toolkit = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    version_paths = [
        Path("/usr/local/Ascend/driver/version.info"),
        toolkit / "version.cfg",
    ]
    version_paths.extend(toolkit.glob("*/ascend_toolkit_install.info"))
    for path in version_paths:
        report["ascend_versions"][str(path)] = (
            path.read_text() if path.is_file() else "unavailable"
        )
    try:
        report["npu_smi"] = command(["npu-smi", "info"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        report["npu_smi"] = str(exc)

    import torch
    import torch_npu  # Required here: this executable is an NPU-only probe.

    if not torch.npu.is_available():
        raise RuntimeError("NPU unavailable; CPU/mock execution is not accepted")
    torch.npu.set_device(args.device)
    report["device"] = {
        "index": args.device,
        "name": torch.npu.get_device_name(args.device),
        "count": torch.npu.device_count(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
    }
    import sglang

    actual = Path(sglang.__file__).resolve()
    expected = Path(python_path) / "sglang" / "__init__.py"
    report["imported_sglang"] = str(actual)
    if actual != expected.resolve():
        raise RuntimeError(f"Imported {actual}, expected {expected}")
    # Match the import order of the registered contract tests.
    importlib.import_module("sglang.srt.layers.quantization")

    report["parallelism"] = {
        "tp": 1,
        "ep": 1,
        "communication_backend": "none (local Linear probe)",
    }
    if args.action == "wan":
        report["parallelism"] = {
            "num_gpus": args.num_gpus,
            "resolved": "recorded after generator startup",
        }

    return torch


def tensor_info(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def dequantize(torch, quantized, scales):
    values = quantized.detach().cpu().float()
    blocks = (
        scales.detach()
        .cpu()
        .contiguous()
        .view(torch.uint8)
        .reshape(values.shape[0], -1)
    )
    blocks = blocks[:, : (values.shape[-1] + 31) // 32]
    if bool((blocks == 255).any()):
        raise ValueError("UE8M0 contains NaN encoding 255")
    factors = torch.exp2(blocks.float() - 127).repeat_interleave(32, dim=-1)
    return values * factors[:, : values.shape[-1]]


def compare(torch, actual, expected, rtol, atol):
    if rtol < 0 or atol < 0:
        raise ValueError("Comparison tolerances must be nonnegative")
    actual = actual.detach().cpu().float()
    expected = expected.detach().cpu().float()
    if actual.shape != expected.shape:
        raise AssertionError(f"Output shape differs: {actual.shape}/{expected.shape}")
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = (actual - expected).abs()
    passed = finite and bool((difference <= atol + rtol * expected.abs()).all())
    return {
        "passed": passed,
        "finite": finite,
        "max_abs": difference.max().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "rtol": rtol,
        "atol": atol,
    }


def make_layer(torch, scheme, qw, scales, bias):
    n, k = qw.shape
    layer = torch.nn.Module()
    scheme.create_weights(layer, k, [n], k, n, torch.bfloat16)
    layer.to(qw.device)
    layer.weight.data.copy_(qw)
    layer.weight_scale.data.copy_(scales)
    layer.bias = torch.nn.Parameter(bias.clone(), requires_grad=False)
    return layer


def linear(args, report, output, torch):
    from sglang.multimodal_gen.runtime.layers.quantization.modelslim_mxfp8_scheme import (
        ModelSlimMXFP8Scheme,
    )

    baseline_cls = None
    if not args.boundary:
        if not args.baseline_source:
            raise ValueError("--baseline-source is required for old/new comparison")
        report["baseline"] = checkout(args.baseline_source, BASELINE)
        path = Path(report["baseline"]["path"]) / SCHEME_PATH
        spec = importlib.util.spec_from_file_location("g22_original_scheme", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline_cls = module.ModelSlimMXFP8Scheme
        report["baseline_scheme_sha256"] = digest(path)
    report["scope"] = (
        "Synthetic raw checkpoint -> real post-load/apply; single NPU, no TP/EP or model loading. "
        + (
            "Boundary compared to FP32 dequantized MXFP reference, not baseline equivalence."
            if args.boundary
            else "Original baseline Scheme with target-environment dependencies."
        )
    )
    report["cases"] = []
    torch.manual_seed(args.seed)
    for k in args.k or ([4304] if args.boundary else [64, 128, 512]):
        if k <= 0 or (not args.boundary and k % 64):
            raise ValueError("Baseline comparison requires positive K divisible by 64")
        report["active_case"] = {"k": k, "stage": "weight_quantization"}
        save(report, output)
        raw = torch.randn(args.n, k, dtype=torch.float32) * 0.05
        inputs = torch.randn(6, k, dtype=torch.float32) * 0.2
        bias_cpu = torch.randn(args.n, dtype=torch.float32) * 0.1
        qw, ws = torch.ops.npu.npu_dynamic_mx_quant(
            raw.to(device=f"npu:{args.device}", dtype=torch.bfloat16),
            dst_type=torch.float8_e4m3fn,
        )
        scales = (
            ws.contiguous()
            .view(torch.uint8)
            .reshape(args.n, -1)[:, : (k + 31) // 32]
            .contiguous()
        )
        # Save raw checkpoint bytes and inputs before either implementation runs.
        torch.save(
            {
                "raw_weight": raw,
                "weight_bytes": qw.cpu().view(torch.uint8),
                "scale_bytes": scales.cpu(),
                "input_fp32": inputs,
                "bias_fp32": bias_cpu,
            },
            output / f"fixture_k{k}.pt",
        )
        bias = bias_cpu.to(device=qw.device, dtype=torch.float16)
        new = ModelSlimMXFP8Scheme()
        new_layer = make_layer(torch, new, qw, scales, bias)
        report["active_case"] = {"k": k, "stage": "new_post_load"}
        save(report, output)
        new.process_weights_after_loading(new_layer)
        layouts = {
            "weight": tensor_info(new_layer.weight),
            "scale": tensor_info(new_layer.weight_scale_inv),
        }
        old = old_layer = None
        if baseline_cls:
            old = baseline_cls()
            old_layer = make_layer(torch, old, qw, scales, bias)
            report["active_case"] = {"k": k, "stage": "baseline_post_load"}
            save(report, output)
            old.process_weights_after_loading(old_layer)
            for left, right in (
                (old_layer.weight.T, new_layer.weight),
                (old_layer.weight_scale.transpose(0, 1), new_layer.weight_scale_inv),
            ):
                if left.stride() != right.stride() or not torch.equal(
                    left.cpu().view(torch.uint8), right.cpu().view(torch.uint8)
                ):
                    raise AssertionError(
                        "Prepared bytes or strides differ before matmul"
                    )
        # Bias mutation exercises the real adapter's post-load cache handling.
        new_layer.bias.data.add_(0.25)
        if old_layer is not None:
            old_layer.bias.data.add_(0.25)
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            base = inputs.to(device=qw.device, dtype=dtype)
            shapes = (
                base,
                base.reshape(2, 3, k),
                base.reshape(2, 1, 3, k),
                base.reshape(2, 3, k).transpose(0, 1),
            )
            for x in shapes:
                for kind in ("none", "fp32", "updated_layer_fp16"):
                    bnew = (
                        None
                        if kind == "none"
                        else (
                            bias_cpu.to(qw.device) if kind == "fp32" else new_layer.bias
                        )
                    )
                    bold = (
                        old_layer.bias
                        if old_layer is not None and kind == "updated_layer_fp16"
                        else bnew
                    )
                    case = {
                        "k": k,
                        "input": tensor_info(x),
                        "bias": kind,
                        "layout": layouts,
                        "stage": "new_apply",
                    }
                    report["active_case"] = case
                    save(report, output)
                    actual = new.apply_weights(new_layer, x, bnew)
                    torch.npu.synchronize()
                    expected_dtype = dtype if dtype != torch.float32 else torch.bfloat16
                    if actual.dtype != expected_dtype or actual.shape != (
                        *x.shape[:-1],
                        args.n,
                    ):
                        raise AssertionError(
                            f"Invalid output contract: {tensor_info(actual)}"
                        )
                    if old is not None:
                        case["stage"] = "baseline_apply"
                        save(report, output)
                        expected = old.apply_weights(old_layer, x, bold)
                        torch.npu.synchronize()
                        if (
                            expected.dtype != actual.dtype
                            or expected.shape != actual.shape
                        ):
                            raise AssertionError(
                                "Old/new output shape or dtype differs"
                            )
                        metrics = compare(torch, actual, expected, args.rtol, args.atol)
                    else:
                        qx, sx = torch.ops.npu.npu_dynamic_mx_quant(
                            x.to(expected_dtype).reshape(-1, k),
                            dst_type=torch.float8_e4m3fn,
                        )
                        expected = (
                            dequantize(torch, qx, sx) @ dequantize(torch, qw, ws).T
                        )
                        if bnew is not None:
                            expected += bnew.detach().cpu().float()
                        metrics = compare(
                            torch,
                            actual,
                            expected.reshape(actual.shape),
                            args.reference_rtol,
                            args.reference_atol,
                        )
                    case["stage"] = "compared"
                    case["output"] = tensor_info(actual)
                    case["comparison"] = metrics
                    report["cases"].append(case)
                    print(json.dumps(case), flush=True)
                    save(report, output)
                    if not metrics["passed"]:
                        torch.save(
                            {"actual": actual.cpu(), "expected": expected.cpu()},
                            output / "failure_tensors.pt",
                        )
                        raise AssertionError(
                            "Numerical comparison failed; thresholds are not widened"
                        )


def wan(args, report, output, torch):
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    model = Path(args.model_path).resolve(strict=True)
    image = Path(args.image_path).resolve(strict=True)
    if not model.is_dir() or not image.is_file():
        raise ValueError("Model must be a directory and image must be a file")
    descriptions = sorted(model.glob("quant_model_description.json")) + sorted(
        model.glob("*/quant_model_description.json")
    )
    if args.mode == "offline" and not descriptions:
        raise ValueError("Offline run needs a ModelSlim quant_model_description.json")
    if args.mode == "offline":
        formats = {
            value
            for path in descriptions
            for key, value in json.loads(path.read_text()).items()
            if key.endswith(".weight") and isinstance(value, str) and value != "FLOAT"
        }
        if formats != {"W8A8_MXFP8"}:
            raise ValueError(
                f"G2.2 needs MXFP8 Linear weights; found {sorted(formats)}. Repacking does not change quantization."
            )
    if args.mode == "online" and descriptions:
        raise ValueError(
            "Online run needs original weights, not a ModelSlim checkpoint"
        )
    if args.num_frames <= 0 or (args.num_frames - 1) % 4:
        raise ValueError("Use num_frames=4n+1")
    if not args.smoke and args.repeats < 3:
        raise ValueError("Use at least three repeats and num_frames=4n+1")
    report["model"] = {"path": str(model), "configs": {}}
    for name in ("config.json", "model_index.json", "quant_model_description.json"):
        for path in sorted(model.glob(name)) + sorted(model.glob(f"*/{name}")):
            report["model"]["configs"][str(path.relative_to(model))] = {
                "sha256": digest(path),
                "contents": json.loads(path.read_text()),
            }
    report["image"] = {"path": str(image), "sha256": digest(image)}
    report["scope"] = (
        "Wan load + one image-to-video request; no numerical or performance comparison"
        if args.smoke
        else "Wan load + one full-shape warmup request + repeated image-to-video requests; media quality requires review"
    )
    kwargs = {
        "model_path": str(model),
        "num_gpus": args.num_gpus,
        "output_path": str(output),
        "warmup_mode": "off",
    }
    if args.mode == "online":
        kwargs["quantization"] = "mxfp8"
    report["server_kwargs"] = kwargs
    sampling = {
        "prompt": args.prompt,
        "image_path": str(image),
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "fps": args.fps,
        "guidance_scale": args.guidance_scale,
        "save_output": True,
    }
    report["sampling"] = sampling
    report["timings"] = []
    report["stage"] = "load"
    save(report, output)
    start = time.perf_counter()
    gen = DiffGenerator.from_pretrained(**kwargs)
    report["load_seconds"] = time.perf_counter() - start
    try:
        report["resolved_server_args"] = vars(gen.server_args)
        save(report, output)
        stages = (
            ["smoke"]
            if args.smoke
            else ["warmup"]
            + [f"stable_{index}" for index in range(1, args.repeats + 1)]
        )
        for stage in stages:
            report["stage"] = stage
            save(report, output)
            start = time.perf_counter()
            result = gen.generate(
                sampling_params_kwargs={**sampling, "output_file_name": f"{stage}.mp4"}
            )
            duration = time.perf_counter() - start
            results = result if isinstance(result, list) else [result]
            if not results:
                raise RuntimeError(f"{stage}: generator returned an empty list")
            artifacts = []
            for item in results:
                if item is None or item.metrics.get("warmup_failed"):
                    raise RuntimeError(
                        f"{stage}: generator returned no result or failed warmup"
                    )
                path = Path(item.output_file_path or "")
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"{stage}: no nonempty output media: {path}")
                artifact = {
                    "path": str(path.resolve()),
                    "sha256": digest(path),
                    "metrics": item.metrics,
                    "generation_seconds": item.generation_time,
                }
                if args.save_samples and stage == "stable_1":
                    import numpy as np

                    if item.samples is None:
                        raise RuntimeError(
                            "No decoded samples returned for numerical comparison"
                        )
                    sample = item.samples
                    if isinstance(sample, torch.Tensor):
                        sample = sample.detach().cpu().float().numpy()
                    sample = np.asarray(sample)
                    if not np.isfinite(sample).all():
                        raise RuntimeError("Nonfinite decoded samples")
                    sample_path = output / f"stable_1_sample_{len(artifacts)}.npy"
                    np.save(sample_path, sample, allow_pickle=False)
                    artifact["samples"] = {
                        "path": str(sample_path),
                        "shape": list(sample.shape),
                        "dtype": str(sample.dtype),
                        "sha256": digest(sample_path),
                    }
                artifacts.append(artifact)
            row = {"stage": stage, "wall_seconds": duration, "artifacts": artifacts}
            report["timings"].append(row)
            print(json.dumps(row, default=str), flush=True)
            save(report, output)
        if not args.smoke:
            report["stable_median_seconds"] = statistics.median(
                row["wall_seconds"] for row in report["timings"][1:]
            )
    finally:
        gen.shutdown()


def capabilities(args, report, output, torch):
    from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
        NPUMXFP8LinearMethod,
    )
    from sglang.srt.hardware_backend.npu.quantization.mxfp_capabilities import (
        MXFP_CAPABILITIES,
        W8A8,
        get_mxfp_capability,
        require_mxfp_operators,
        validate_mxfp_configuration,
    )

    report["scope"] = (
        "Actual operator registration; isolated validation; synthetic W8A8 Linear post-load/apply. No MoE runner, TP/EP, or checkpoint/model e2e."
    )
    report["inventory"] = []
    for cap in MXFP_CAPABILITIES:
        row = {
            "key": cap.key,
            "implementation": cap.implementation,
            "fused_ops": list(cap.fused_ops),
            "fallback": cap.fallback,
            "blocker": cap.blocker,
        }
        if cap.fallback or cap.blocker:
            row["preflight"] = "not tested: existing fallback or blocked path"
            report["inventory"].append(row)
            continue
        try:
            require_mxfp_operators(cap)
            row["preflight"] = "passed (operator presence only)"
        except ValueError as exc:
            row["preflight"] = str(exc)
        if cap.fused_ops:
            try:
                require_mxfp_operators(cap, fused=True)
                row["fused_preflight"] = "passed (operator presence only)"
            except ValueError as exc:
                row["fused_preflight"] = str(exc)
        report["inventory"].append(row)
    save(report, output)
    report["checks"] = []

    def reject(label, function, message):
        try:
            function()
        except ValueError as exc:
            if message not in str(exc):
                raise AssertionError(f"{label}: unexpected reason: {exc}") from exc
            report["checks"].append(
                {"label": label, "result": "rejected", "reason": str(exc)}
            )
            save(report, output)
        else:
            raise AssertionError(f"{label}: invalid input accepted")

    moe = get_mxfp_capability("srt", "moe", W8A8, "online")
    reject(
        "MoE ascend_fuseep configuration",
        lambda: validate_mxfp_configuration(moe, a2a_backend="ascend_fuseep"),
        "different INT8",
    )
    reject(
        "Injected missing-op list (isolated metadata check)",
        lambda: require_mxfp_operators(moe, ()),
        "operators not registered",
    )
    for runtime in ("srt", "diffusion"):
        module_name = (
            "sglang.srt.layers.quantization.modelslim.modelslim"
            if runtime == "srt"
            else "sglang.multimodal_gen.runtime.layers.quantization.modelslim"
        )
        module = importlib.import_module(module_name)
        config = module.ModelSlimConfig({"linear.weight": W8A8})
        method = module.ModelSlimLinearMethod(config)

        def construct(k=64, config=config, runtime=runtime, method=method):
            layer = torch.nn.Module()
            selector = (
                config.get_linear_scheme if runtime == "srt" else config.get_scheme
            )
            layer.scheme = selector(layer, "linear")
            method.create_weights(layer, k, [128], k, 128, torch.bfloat16)
            layer.to(f"npu:{args.device}")
            return layer

        reject(
            f"{runtime} actual create_weights local K=32",
            lambda: construct(32),
            "local input K=32",
        )
        for bad in ("weight_dtype", "scale_dtype", "scale_count"):
            layer = construct()
            if bad == "weight_dtype":
                layer.weight = torch.nn.Parameter(
                    torch.zeros(
                        128, 64, device=f"npu:{args.device}", dtype=torch.bfloat16
                    ),
                    requires_grad=False,
                )
            elif bad == "scale_dtype":
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale.float(), requires_grad=False
                )
            else:
                layer.weight_scale = torch.nn.Parameter(
                    layer.weight_scale[:, :1], requires_grad=False
                )
            reject(
                f"{runtime} actual post-load {bad}",
                lambda method=method, layer=layer: method.process_weights_after_loading(
                    layer
                ),
                "expected ceil(K/32)"
                if bad == "scale_count"
                else "checkpoint requires",
            )
        weight = (
            torch.randn(128, 64, dtype=torch.bfloat16).to(f"npu:{args.device}") * 0.05
        )
        qw, ws = torch.ops.npu.npu_dynamic_mx_quant(
            weight, dst_type=torch.float8_e4m3fn
        )
        layer = construct()
        layer.weight.data.copy_(qw)
        layer.weight_scale.data.copy_(ws.view(torch.uint8).reshape(128, 2))
        method.process_weights_after_loading(layer)
        x = torch.randn(6, 64, dtype=torch.bfloat16).to(weight.device) * 0.2
        actual = method.apply(layer, x)
        qx, xs = torch.ops.npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        expected = dequantize(torch, qx, xs) @ dequantize(torch, qw, ws).T
        metrics = compare(
            torch, actual, expected, args.reference_rtol, args.reference_atol
        )
        report["checks"].append(
            {"label": f"{runtime} ModelSlim W8A8 real post-load/apply", **metrics}
        )
        if not metrics["passed"]:
            raise AssertionError(f"{runtime}: valid ModelSlim linear failed")
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight.clone(), requires_grad=False)
    online = NPUMXFP8LinearMethod()
    online.process_weights_after_loading(layer)
    metrics = compare(
        torch,
        online.apply(layer, x),
        expected,
        args.reference_rtol,
        args.reference_atol,
    )
    report["checks"].append(
        {"label": "srt online W8A8 real post-load/apply", **metrics}
    )
    if not metrics["passed"]:
        raise AssertionError("Valid online W8A8 linear failed")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for name in ("linear", "wan", "capabilities"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--source", required=True, help="Actual SGLang checkout")
        sub.add_argument("--expected-head", required=True)
        sub.add_argument("--output-dir", required=True, help="Fresh evidence directory")
        sub.add_argument("--device", type=int, default=0)
        sub.add_argument("--seed", type=int, default=42)
        sub.add_argument("--reference-rtol", type=float, default=0.02)
        sub.add_argument("--reference-atol", type=float, default=0.02)
        if name == "linear":
            sub.add_argument("--baseline-source")
            sub.add_argument(
                "--boundary",
                action="store_true",
                help="New implementation versus dequantized reference, in a separate process",
            )
            sub.add_argument("--k", type=int, action="append")
            sub.add_argument("--n", type=int, default=128)
            sub.add_argument("--rtol", type=float, default=0.0)
            sub.add_argument("--atol", type=float, default=0.0)
        if name == "wan":
            sub.add_argument(
                "--smoke",
                action="store_true",
                help="Generate once, without a separate warmup or repeated timing runs",
            )
            sub.add_argument("--model-path", required=True)
            sub.add_argument("--image-path", required=True)
            sub.add_argument("--mode", choices=("offline", "online"), default="offline")
            sub.add_argument(
                "--prompt",
                default="A person slowly turns their head, natural motion, static camera.",
            )
            sub.add_argument("--num-gpus", type=int, default=1)
            sub.add_argument("--height", type=int, default=704)
            sub.add_argument("--width", type=int, default=1280)
            sub.add_argument("--num-frames", type=int, default=81)
            sub.add_argument("--num-inference-steps", type=int, default=40)
            sub.add_argument("--guidance-scale", type=float, default=5.0)
            sub.add_argument("--fps", type=int, default=24)
            sub.add_argument("--repeats", type=int, default=3)
            sub.add_argument(
                "--save-samples",
                action="store_true",
                help="Save first stable decoded sample as .npy for baseline comparison (may be large)",
            )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "running",
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "full_g2_acceptance": False,
        "python": sys.executable,
        "python_version": sys.version,
        "arguments": vars(args),
        "original_environment": {key: os.environ.get(key) for key in ENV_KEYS},
    }
    save(report, output)
    try:
        torch = initialize(args, report)
        torch.manual_seed(args.seed)
        save(report, output)
        {"linear": linear, "wan": wan, "capabilities": capabilities}[args.action](
            args, report, output, torch
        )
        report["status"] = "probe_passed"
        report["full_g2_acceptance"] = False
        return 0
    except Exception:  # noqa: BLE001 -- persist any probe failure and its traceback
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, flush=True)
        return 1
    finally:
        save(report, output)
        print(f"Evidence: {output / 'result.json'}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

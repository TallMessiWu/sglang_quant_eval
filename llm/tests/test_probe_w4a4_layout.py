"""CPU import-order contracts for the standalone W4A4 probe, not NPU tests."""

import builtins
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "diagnostics/probe_w4a4_layout.py"


class DeviceSelected(Exception):
    """Stop after imports, before executing any hardware probe operations."""


class TestProbeW4A4Imports(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("probe_w4a4_layout", SCRIPT)
        self.probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.probe)

    def test_quantization_registry_precedes_hardware_helpers(self):
        imports = []
        real_import = builtins.__import__
        registry = "sglang.srt.layers.quantization"
        hardware = "sglang.srt.hardware_backend.npu.quantization"

        def set_device(device):
            self.assertEqual(device, 6)
            raise DeviceSelected

        def cold_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append(name)
            if name == "torch":
                return SimpleNamespace(npu=SimpleNamespace(set_device=set_device))
            if name == "sglang.srt.utils":
                return SimpleNamespace(is_npu=lambda: True)
            if name in (registry, "torch_npu"):
                return SimpleNamespace()
            if name == hardware:
                # Mirrors the unquant -> partially initialized moe_methods edge.
                if registry not in imports:
                    raise ImportError(
                        "cannot import name 'NPUUnquantMoEMethod' from partially "
                        "initialized module 'moe_methods'"
                    )
                return SimpleNamespace(moe_methods=SimpleNamespace())
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=cold_import):
            with self.assertRaises(DeviceSelected):
                self.probe.run(SimpleNamespace(device=6), {})
        self.assertLess(imports.index(registry), imports.index(hardware))

    def test_non_npu_fails_before_importing_npu_modules(self):
        imports = []
        real_import = builtins.__import__

        def non_npu_import(name, globals=None, locals=None, fromlist=(), level=0):
            imports.append(name)
            if name == "torch":
                return SimpleNamespace()
            if name == "sglang.srt.utils":
                return SimpleNamespace(is_npu=lambda: False)
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=non_npu_import):
            with self.assertRaisesRegex(RuntimeError, "Ascend NPU environment"):
                self.probe.run(SimpleNamespace(device=6), {})
        self.assertNotIn("torch_npu", imports)
        self.assertNotIn("sglang.srt.layers.quantization", imports)
        self.assertNotIn("sglang.srt.hardware_backend.npu.quantization", imports)


if __name__ == "__main__":
    unittest.main()

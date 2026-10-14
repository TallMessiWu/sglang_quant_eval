"""CPU checks for evidence tools only; these do not validate any NPU operator."""

import json
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from g2_npu import checkout, compare, dequantize
from inspect_wan import inspect, safetensors_header


class EvidenceToolTests(unittest.TestCase):
    def test_numerical_comparison_rejects_nonfinite_or_changed_outputs(self):
        values = torch.tensor([0.0, 1.0])
        self.assertTrue(compare(torch, values, values, 0, 0)["passed"])
        self.assertFalse(compare(torch, values + 0.1, values, 0, 0)["passed"])
        for special in (float("nan"), float("inf")):
            bad = torch.tensor([special, 1.0])
            self.assertFalse(compare(torch, bad, bad, 0, 0)["passed"])
        with self.assertRaises(AssertionError):
            compare(torch, values, values.unsqueeze(0), 0, 0)

    def test_ue8m0_reference_preserves_partial_blocks(self):
        weight = torch.ones(1, 33).to(torch.float8_e4m3fn)
        scale = torch.tensor([[127, 128]], dtype=torch.uint8)
        expected = torch.ones(1, 33)
        expected[0, -1] = 2
        torch.testing.assert_close(dequantize(torch, weight, scale), expected)
        scale[0, 1] = 255
        with self.assertRaisesRegex(ValueError, "NaN encoding"):
            dequantize(torch, weight, scale)

    def test_source_pin_rejects_wrong_head_and_untracked_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", temporary, *args], text=True
                ).strip()

            git("init", "-q")
            (path / "source.py").write_text("pass\n")
            git("add", "source.py")
            git(
                "-c",
                "user.name=Probe Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "Fixture",
            )
            head = git("rev-parse", "HEAD")
            self.assertEqual(checkout(path, head)["head"], head)
            with self.assertRaisesRegex(ValueError, "Wrong/dirty"):
                checkout(path, "0" * 40)
            (path / "injected.py").write_text("pass\n")
            with self.assertRaisesRegex(ValueError, "Wrong/dirty"):
                checkout(path, head)

    def test_model_inventory_reads_metadata_without_tensor_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            component = root / "high_noise_model"
            component.mkdir()
            (component / "quant_model_description.json").write_text(
                json.dumps({"blocks.0.ffn.0.weight": "W8A8_MXFP8"})
            )
            metadata = {
                "blocks.0.ffn.0.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [128, 64],
                    "data_offsets": [0, 8192],
                }
            }
            encoded = json.dumps(metadata).encode()
            # Header only: reading tensor contents would fail on this fixture.
            (component / "quant_model_weight.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded
            )
            result = inspect(root)["components"][0]
            self.assertEqual(
                result["quantization"][0]["weight_scheme_counts"], {"W8A8_MXFP8": 1}
            )
            self.assertEqual(result["headers"][0]["dtype_counts"], {"F8_E4M3": 1})

    def test_inventory_detects_lfs_pointer_instead_of_claiming_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "weight.safetensors"
            path.write_text("version https://git-lfs.github.com/spec/v1\n")
            with self.assertRaisesRegex(ValueError, "LFS pointer"):
                safetensors_header(path)


if __name__ == "__main__":
    unittest.main()

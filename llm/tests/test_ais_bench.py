"""AIS Bench config/shell regressions with a fake package/CLI; no model requests."""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
TEMPLATE = '''\
from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="vllm-api-general-chat",
        path="",
        model="",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=8192,
        batch_size=200,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=0.01,
            top_p=0.5,
            top_k=1,
            ignore_eos=False,
            seed=42,
            chat_template_kwargs=dict(enable_thinking=False),
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
'''
DEFAULT_SAMPLING = dict(
    temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
    presence_penalty=1.5, repetition_penalty=1.0,
)


class AISBenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.llm = self.root / "scripts with spaces"
        self.llm.mkdir()
        shutil.copytree(LLM_DIR / "ais_bench", self.llm / "ais_bench")
        for name in ("gpqa", "gsm8k", "mmmu"):
            shutil.copy2(LLM_DIR / f"{name}.sh", self.llm / f"{name}.sh")
        package = self.root / "packages/ais_bench"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        self.template = package / "benchmark/configs/models/vllm_api/vllm_api_general_chat.py"
        self.template.parent.mkdir(parents=True)
        self.template.write_text(TEMPLATE)
        bin_dir = self.root / "bin"
        bin_dir.mkdir()
        cli = bin_dir / "ais_bench"
        cli.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "if '--help' in sys.argv:\n"
            "    print('--host-ip' if os.environ.get('TEST_MODERN') == '1' else '--models')\n"
            "else:\n"
            "    pathlib.Path(os.environ['TEST_ARGV']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        cli.chmod(0o755)
        (bin_dir / "python3").symlink_to(sys.executable)
        self.argv_file = self.root / "argv.json"
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("AIS_", "VLLM_", "TEST_"))
            and key not in ("MODEL_NAME", "BASH_ENV", "PYTHONPATH")
        }
        self.env.update(
            PATH=f"{bin_dir}:{self.env['PATH']}",
            PYTHONPATH=str(package.parent),
            PYTHONDONTWRITEBYTECODE="1",
            TEST_ARGV=str(self.argv_file),
        )

    def run_script(self, name="gpqa", overrides=None, args=(), check=True):
        result = subprocess.run(
            ["bash", str(self.llm / f"{name}.sh"), *args],
            cwd=self.llm, env=self.env | (overrides or {}),
            capture_output=True, text=True, timeout=10,
        )
        if check:
            self.assertEqual(result.returncode, 0, result.stderr)
        return result

    def generated(self, argv_file=None):
        args = json.loads((argv_file or self.argv_file).read_text())
        config = (
            Path(args[args.index("--config-dir") + 1]) / "models"
            / f"{args[args.index('--models') + 1]}.py"
        )
        self.assertTrue(config.is_relative_to(self.llm / ".ais_bench_configs"))
        return self.load_model(config), args, config

    @staticmethod
    def load_model(path):
        tree = ast.parse(path.read_text())
        # Stub the two imported symbols, but evaluate the actual generated models expression.
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = dict(VLLMCustomAPIChat="chat-class", extract_non_reasoning_content="postprocessor")
        exec(compile(tree, str(path), "exec"), namespace)
        return namespace["models"][0]

    def test_dataset_defaults_and_preserved_template_fields_on_old_and_new_cli(self):
        for modern in ("0", "1"):
            for name, dataset, length in (
                ("gpqa", "gpqa_gen_0_shot_cot_chat_prompt", 32768),
                ("gsm8k", "gsm8k_gen_0_shot_cot_chat_prompt", 4096),
                ("mmmu", "mmmu_gen", 8192),
            ):
                with self.subTest(modern=modern, dataset=dataset):
                    self.run_script(name, {"TEST_MODERN": modern})
                    model, args, _ = self.generated()
                    self.assertEqual(args[:3], ["--datasets", dataset, "--dump-eval-details"])
                    self.assertEqual(model["max_out_len"], length)
                    self.assertEqual(model["host_ip"], "localhost")
                    self.assertEqual(model["host_port"], 6969)
                    self.assertEqual(model["generation_kwargs"], DEFAULT_SAMPLING | {
                        "ignore_eos": False, "seed": 42,
                        "chat_template_kwargs": {"enable_thinking": False},
                    })
                    self.assertEqual(model["type"], "chat-class")
                    self.assertEqual(model["pred_postprocessor"], {"type": "postprocessor"})
                    self.assertEqual(model["batch_size"], 200)
                    self.assertFalse(model["stream"])
        self.assertEqual(self.template.read_text(), TEMPLATE)

    def test_all_sampling_presets(self):
        for preset, temperature, top_p, presence_penalty in (
            ("thinking", 1.0, 0.95, 1.5),
            ("thinking-coding", 0.6, 0.95, 0.0),
            ("non-thinking", 0.7, 0.8, 1.5),
            ("non-thinking-reasoning", 1.0, 0.95, 1.5),
            ("qwen35-397b-thinking", 0.6, 0.95, 0.0),
        ):
            with self.subTest(preset=preset):
                self.run_script(overrides={"AIS_SAMPLING_PRESET": preset})
                model, _, _ = self.generated()
                expected = DEFAULT_SAMPLING | dict(
                    temperature=temperature, top_p=top_p, presence_penalty=presence_penalty,
                )
                self.assertEqual({key: model["generation_kwargs"][key] for key in expected}, expected)
                self.assertEqual(model["max_out_len"], 32768)

    def test_individual_overrides_take_precedence_including_zero_and_negative(self):
        self.run_script("gsm8k", {
            "AIS_SAMPLING_PRESET": "qwen35-397b-thinking",
            "AIS_MAX_OUT_LEN": "1234", "AIS_TEMPERATURE": "0",
            "AIS_TOP_P": "0.9", "AIS_TOP_K": "-1", "AIS_MIN_P": "0.1",
            "AIS_PRESENCE_PENALTY": "-0.5", "AIS_REPETITION_PENALTY": "1.2",
        })
        model, _, _ = self.generated()
        self.assertEqual(model["max_out_len"], 1234)
        expected = dict(temperature=0.0, top_p=0.9, top_k=-1, min_p=0.1,
                        presence_penalty=-0.5, repetition_penalty=1.2)
        self.assertEqual({key: model["generation_kwargs"][key] for key in expected}, expected)

    def test_template_preset_preserves_sampling_with_optional_single_override(self):
        original = self.load_model(self.template)["generation_kwargs"]
        for overrides in ({}, {"AIS_TOP_K": "40"}):
            with self.subTest(overrides=overrides):
                self.run_script(overrides={"AIS_SAMPLING_PRESET": "template"} | overrides)
                model, _, _ = self.generated()
                expected = original | ({"top_k": 40} if overrides else {})
                self.assertEqual(model["generation_kwargs"], expected)
                self.assertEqual(model["max_out_len"], 32768)

    def test_url_model_and_cli_passthrough(self):
        url = "http://gw.example/prefix/?a=1&b=2"
        model_name = "model 'quoted' name"
        result = self.run_script(overrides={
            "VLLM_URL": url, "VLLM_IP": "ignored", "VLLM_PORT": "bad-ignored",
            "MODEL_NAME": model_name,
        }, args=("--batch-size", "16", "--work-dir", "results with spaces", "--debug"))
        model, args, _ = self.generated()
        self.assertEqual(model["url"], url)
        self.assertEqual(model["model"], model_name)
        self.assertEqual(model["host_port"], 8080)
        self.assertEqual(args[-5:], ["--batch-size", "16", "--work-dir", "results with spaces", "--debug"])
        self.assertIn("max_out_len=32768", result.stderr)
        self.assertIn("config=", result.stderr)

    def test_host_overrides_clear_template_url(self):
        self.template.write_text(TEMPLATE.replace('url=""', 'url="http://stale.example/"'))
        self.run_script(overrides={"VLLM_IP": "10.0.0.5", "VLLM_PORT": "7979"})
        model, _, _ = self.generated()
        self.assertEqual((model["host_ip"], model["host_port"], model["url"]), ("10.0.0.5", 7979, ""))

    def test_inline_dict_literal_custom_template_and_missing_generation_kwargs(self):
        custom = self.template.with_name("custom.py")
        for text in (
            "models = [{'max_out_len': 777, 'generation_kwargs': {'ignore_eos': False}}]\n",
            "models = [dict(max_out_len=777)]\n",
        ):
            with self.subTest(text=text):
                custom.write_text(text)
                self.run_script("mmmu", {"AIS_MODEL_CFG": "custom.py"})
                model, _, _ = self.generated()
                self.assertEqual(model["max_out_len"], 777)
                self.assertEqual(model["generation_kwargs"]["min_p"], 0.0)
                self.assertNotIn("chat_template_kwargs", model["generation_kwargs"])

    def test_concurrent_runs_on_same_endpoint_use_independent_configs(self):
        jobs = (
            ("gpqa", "thinking", 32768, 1.0),
            ("gsm8k", "thinking", 4096, 1.0),
            ("gpqa", "non-thinking", 32768, 0.7),
        )
        argv_files = [self.root / f"argv-{i}.json" for i in range(len(jobs))]
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(self.run_script, name, {
                "AIS_SAMPLING_PRESET": preset, "TEST_ARGV": str(argv_file),
            }) for (name, preset, _, _), argv_file in zip(jobs, argv_files)]
            for future in futures:
                future.result()
        paths = []
        for (_, _, length, temperature), argv_file in zip(jobs, argv_files):
            model, _, path = self.generated(argv_file)
            paths.append(path)
            self.assertEqual(model["max_out_len"], length)
            self.assertEqual(model["generation_kwargs"]["temperature"], temperature)
        self.assertEqual(len(set(paths)), 3)

    def test_invalid_inputs_fail_before_benchmark(self):
        for overrides in (
            {"AIS_SAMPLING_PRESET": "typo"}, {"AIS_MAX_OUT_LEN": "0"},
            {"AIS_MAX_OUT_LEN": "-1"}, {"AIS_MAX_OUT_LEN": "nope"},
            {"AIS_TEMPERATURE": "nan"}, {"AIS_TOP_P": "inf"},
            {"AIS_TOP_K": "1.5"}, {"AIS_MODEL_CFG": "missing"},
        ):
            with self.subTest(overrides=overrides):
                result = self.run_script(overrides=overrides, check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.argv_file.exists())

    def test_unsupported_template_fails_instead_of_patching_wrong_fields(self):
        for text in (
            "models = [dict(), dict()]\n",
            "models = [dict(generation_kwargs=defaults)]\n",
            "models = [dict(generation_kwargs=dict(**defaults))]\n",
        ):
            with self.subTest(text=text):
                self.template.write_text(text)
                result = self.run_script(check=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("无法改写", result.stderr)
                self.assertFalse(self.argv_file.exists())

    def test_env_python_shebang_fallback(self):
        cli = self.root / "bin/ais_bench"
        text = cli.read_text()
        cli.write_text("#!/usr/bin/env python3\n" + text.split("\n", 1)[1])
        self.run_script()
        model, _, _ = self.generated()
        self.assertEqual(model["max_out_len"], 32768)


if __name__ == "__main__":
    unittest.main()

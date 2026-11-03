#!/usr/bin/env python3
"""从本机 AIS Bench 模板生成带服务地址、输出长度和采样参数的独立配置。

配置中的值全部是 Python 字面量，不在 mmengine lazy-import 配置里读取环境变量。
保留模板的 imports、模型类型、后处理和未覆盖参数，不修改 site-packages。

在 llm/ 中也可单独生成（--name 相同时会覆盖，运行脚本会分配独立目录）：
    python3 ais_bench/gen_ais_bench_model_cfg.py --out-dir ./.ais_bench_configs \\
        --name sglang_localhost_6969 --host-ip localhost --host-port 6969 \\
        --dataset gpqa_gen_0_shot_cot_chat_prompt --sampling-preset thinking
"""

import argparse
import ast
import importlib.util
import math
import os
import pathlib
import sys


# 预设只选择采样参数，不开关 chat_template_kwargs.enable_thinking。
COMMON_SAMPLING = dict(top_k=20, min_p=0.0, repetition_penalty=1.0)
SAMPLING_PRESETS = {
    "thinking": dict(temperature=1.0, top_p=0.95, presence_penalty=1.5),
    "thinking-coding": dict(temperature=0.6, top_p=0.95, presence_penalty=0.0),
    "non-thinking": dict(temperature=0.7, top_p=0.8, presence_penalty=1.5),
    "non-thinking-reasoning": dict(temperature=1.0, top_p=0.95, presence_penalty=1.5),
    "qwen35-397b-thinking": dict(temperature=0.6, top_p=0.95, presence_penalty=0.0),
    "template": {},
}
SAMPLING_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
)
DATASET_MAX_OUT_LEN = {
    "gpqa_gen_0_shot_cot_chat_prompt": 32768,
    "gsm8k_gen_0_shot_cot_chat_prompt": 4096,
}


def find_template(template: str) -> pathlib.Path:
    spec = importlib.util.find_spec("ais_bench")
    if spec is None or not spec.origin:
        sys.exit("RED: 没找到 ais_bench 包，确认它装在当前 python 环境里")
    root = pathlib.Path(spec.origin).parent
    name = template[:-3] if template.endswith(".py") else template
    matches = sorted(root.glob(f"benchmark/configs/models/**/{name}.py"))
    if not matches:
        sys.exit(f"RED: 在 {root}/benchmark/configs/models 下没找到模板 {name}.py")
    return matches[0]


def dict_items(node: ast.expr) -> dict[str, ast.expr]:
    """只处理静态 dict(...) / {...}，不执行模板，也不猜测动态配置。"""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
        and all(kw.arg is not None for kw in node.keywords)
    ):
        return {kw.arg: kw.value for kw in node.keywords}
    if isinstance(node, ast.Dict) and all(
        isinstance(key, ast.Constant) and isinstance(key.value, str)
        for key in node.keys
    ):
        return {key.value: value for key, value in zip(node.keys, node.values)}
    raise ValueError("模型及 generation_kwargs 必须是静态 dict(...) 或 {...}，不支持展开或引用")


def update_dict(node: ast.expr, updates: dict[str, ast.expr]) -> ast.expr:
    items = dict_items(node)
    items.update(updates)
    if isinstance(node, ast.Dict):
        return ast.Dict(
            keys=[ast.Constant(key) for key in items], values=list(items.values())
        )
    return ast.Call(
        func=ast.Name(id="dict", ctx=ast.Load()), args=[],
        keywords=[ast.keyword(arg=key, value=value) for key, value in items.items()],
    )


def render_config(text: str, fields: dict, sampling: dict) -> str:
    tree = ast.parse(text)
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "models" for t in node.targets)
    ]
    if len(assignments) != 1:
        raise ValueError("模板必须包含唯一的 models = [...] 赋值")
    models = assignments[0].value
    if not isinstance(models, ast.List) or len(models.elts) != 1:
        raise ValueError("模板 models 必须恰好包含一个模型")
    model = models.elts[0]
    updates = {key: ast.Constant(value) for key, value in fields.items()}
    if sampling:
        generation = dict_items(model).get("generation_kwargs", ast.Dict(keys=[], values=[]))
        updates["generation_kwargs"] = update_dict(
            generation, {key: ast.Constant(value) for key, value in sampling.items()}
        )
    models.elts[0] = update_dict(model, updates)
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", required=True, help="config 目录，文件写到它的 models/ 下")
    parser.add_argument("--name", required=True, help="生成的配置名，不带 .py")
    parser.add_argument("--template", default="vllm_api_general_chat", help="AIS Bench 自带的模板名")
    parser.add_argument("--host-ip")
    parser.add_argument("--host-port", type=int)
    parser.add_argument("--url")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset", help="GPQA 默认 32768，GSM8K 默认 4096，其余保留模板长度")
    parser.add_argument("--sampling-preset", choices=SAMPLING_PRESETS, default="thinking")
    parser.add_argument("--max-out-len", type=int)
    for field in SAMPLING_FIELDS:
        parser.add_argument(f"--{field.replace('_', '-')}", type=int if field == "top_k" else float)
    args = parser.parse_args()

    fields = {}
    if args.url:
        # url 非空时 AIS Bench 会忽略 host_ip/host_port。
        fields["url"] = args.url
    else:
        if args.host_ip is not None:
            fields["host_ip"] = args.host_ip
        if args.host_port is not None:
            fields["host_port"] = args.host_port
        if args.host_ip is not None or args.host_port is not None:
            fields["url"] = ""
    if args.model_name:
        fields["model"] = args.model_name
    max_out_len = args.max_out_len
    if max_out_len is None:
        max_out_len = DATASET_MAX_OUT_LEN.get(args.dataset)
    if max_out_len is not None:
        if max_out_len <= 0:
            parser.error("--max-out-len 必须大于 0")
        fields["max_out_len"] = max_out_len

    sampling = {}
    if args.sampling_preset != "template":
        sampling.update(COMMON_SAMPLING)
        sampling.update(SAMPLING_PRESETS[args.sampling_preset])
    for field in SAMPLING_FIELDS:
        value = getattr(args, field)
        if value is not None:
            if not math.isfinite(value):
                parser.error(f"--{field.replace('_', '-')} 必须是有限数值")
            sampling[field] = value

    src = find_template(args.template)
    try:
        text = render_config(src.read_text(encoding="utf-8"), fields, sampling)
    except (SyntaxError, ValueError) as exc:
        parser.error(f"模板 {src} 无法改写: {exc}")
    header = (
        f"# 由 {pathlib.Path(__file__).name} 生成，勿手改；改配置请重跑评测脚本。\n"
        f"# 模板来自 {src}\n"
        f"# dataset={args.dataset!r} sampling_preset={args.sampling_preset!r}\n"
    )
    out_dir = pathlib.Path(args.out_dir) / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.name}.py"
    tmp = out_dir / f".{args.name}.{os.getpid()}.tmp"
    tmp.write_text(header + text, encoding="utf-8", newline="\n")
    tmp.replace(out)
    print(
        f"[ais_bench] max_out_len={max_out_len if max_out_len is not None else 'template'} "
        f"sampling_preset={args.sampling_preset} generation_overrides={sampling}",
        file=sys.stderr,
    )
    print(out)


if __name__ == "__main__":
    main()

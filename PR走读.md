# PR #20922 代码走读：Ascend NPU MXFP8 量化支持

PR 包含 **3 个部分**：LLM 侧 online MXFP8、Diffusion 侧 online MXFP8、测试。

---

## 1. LLM 侧：`--quantization mxfp8` on NPU

### 入口分发 — `fp8.py:225-230`

在 `Fp8Config.get_quant_method()` 里加了 6 行：

```python
if _is_npu and self.use_mxfp8:
    from ...mxfp8_method_npu import NPUMXFP8LinearMethod
    return NPUMXFP8LinearMethod(self)
```

用户传 `--quantization mxfp8` 时，`Fp8Config` 设 `use_mxfp8=True`。原来这在 CUDA 上走 `Fp8LinearMethod`（Triton/CUTLASS kernel），现在如果 `is_npu()` 为 True，改走 NPU 专用的 `NPUMXFP8LinearMethod`。

### 核心实现 — `mxfp8_method_npu.py`

实现 `LinearMethodBase` 接口的 3 个方法：

#### `create_weights()`（L35-88）— 注册参数

根据 `is_checkpoint_fp8_serialized` 分两条路：

- **offline 模式**（已有 FP8 checkpoint）：
  - weight 用 `float8_e4m3fn`
  - 同时注册 `weight_scale_inv`（`uint8`，存储 `float8_e8m0fnu`）
  - scale shape 为 `[out, ceil(in/32)]`

- **online 模式**（FP16/BF16 权重）：
  - weight 保持原始 dtype
  - `weight_scale_inv = None`（权重加载完再量化）

#### `process_weights_after_loading()`（L90-120）— 加载后处理

- **offline**：
  - 检查 weight 是否为 `float8_e4m3fn`
  - 如果不是（比如 checkpoint 存的 int8）则调 `npu_dtype_cast` 转换

- **online**：
  - 调 `npu_dynamic_mx_quant(weight)` 把 FP16/BF16 权重量化为 MXFP8
  - 返回 `(qw, w_scale)`，写入 `layer.weight` 和 `layer.weight_scale_inv`

#### `apply()`（L122-152）— 每次 forward

```
输入 x (FP16/BF16)
  → npu_dynamic_mx_quant(x) → (qx, input_scale)   # 动态量化激活，block_size=32
  → npu_quant_matmul(
        qx, weight^T, weight_scale^T,
        scale_dtype=float8_e8m0fnu,
        pertoken_scale=input_scale,
        group_sizes=[1, 1, 32]                      # MXFP8 block_size
    )
  → output (FP16/BF16)
```

**注意**：`bias` 必须转 `float32` 才能传给 `npu_quant_matmul`（NPU API 限制）。

---

## 2. Diffusion 侧：Wan2.2 MXFP8

Diffusion 子系统（`multimodal_gen`）与 LLM serving（`srt`）是完全独立的两套代码，量化体系不共享，所以需要单独实现。

### 量化配置 — `mxfp8_npu.py`

- **`MXFP8Config`**：注册为 `"mxfp8"` 的 `QuantizationConfig`，`get_quant_method()` 对 `LinearBase` 返回 `NPUMXFP8DiffusionLinearMethod`

- **`NPUMXFP8DiffusionLinearMethod`**：结构与 LLM 侧相同，但**只支持 online 模式**。

与 LLM 侧的关键差异在 `apply()`（L141-164）：Diffusion 的输入可能是 3D+ 张量（如 `[B, T, C]`），而 `npu_dynamic_mx_quant` 需要 2D 输入，所以先 flatten：

```python
x_2d = x.reshape(-1, x.shape[-1])          # [B*T, C]
qx, input_scale = npu_dynamic_mx_quant(x_2d)
output = npu_quant_matmul(qx, ...)
output = output.reshape(input_shape[:-1] + [output.shape[-1]])  # 还原
```

### 注册 — `quantization/__init__.py`

```python
_CUSTOMIZED_METHOD_TO_QUANT_CONFIG = {
    "modelslim": ModelSlimConfig,
    "fp8": Fp8Config,
    "mxfp8": MXFP8Config,  # 新增
}
```

### Transformer Loader — `transformer_loader.py:87-93`

`_resolve_quant_config()` 里增加了最高优先级的判断：

如果用户显式传了 `--quantization`，直接用 `quant_cls.from_config({})` 创建配置，跳过 checkpoint metadata 的自动检测。

### ServerArgs — `server_args.py:172`

新增字段：

```python
quantization: str | None = None
```

让用户可以通过 `--quantization mxfp8` 显式指定。

### Wan2.2 模型 — `mova_video_dit.py`

这个文件本身没有改动，`quant_config` 是原有的参数，通过 `WanModel.__init__()` 逐层传入每个 `ColumnParallelLinear` / `RowParallelLinear`，这些 Linear 层调用 `quant_config.get_quant_method()` 时自动走到 `NPUMXFP8DiffusionLinearMethod`。

---

## 3. 测试 — `test_ascend_mxfp8_quantization.py`

用 `Qwen2.5-0.5B-Instruct` 启动服务，带 `--quantization mxfp8 --device npu --attention-backend ascend`：

- **test_gsm8k**：200 道数学题，accuracy ≥ 25%、throughput ≥ 500 tokens/s
- **test_throughput**：生成 256 tokens，CI 环境要求 ≥ 20 tokens/s

---

## 数据流总结

```
用户: --quantization mxfp8

LLM 路径 (srt):
  Fp8Config(use_mxfp8=True)
    → get_quant_method() → is_npu → NPUMXFP8LinearMethod
    → create_weights(): 注册 weight + weight_scale_inv
    → process_weights_after_loading(): npu_dynamic_mx_quant(weight) [online]
    → apply(x): npu_dynamic_mx_quant(x) + npu_quant_matmul()

Diffusion 路径 (multimodal_gen):
  MXFP8Config()
    → get_quant_method() → NPUMXFP8DiffusionLinearMethod
    → process_weights_after_loading(): npu_dynamic_mx_quant(weight)
    → apply(x): flatten → npu_dynamic_mx_quant(x) + npu_quant_matmul() → reshape
```

---

## 关键 NPU API

| API | 用途 |
|-----|------|
| `torch_npu.npu_dynamic_mx_quant(x, dst_type=float8_e4m3fn)` | MXFP8 动态量化，block_size=32，返回 `(qx, scale)` |
| `torch_npu.npu_quant_matmul(..., group_sizes=[1,1,32])` | MXFP8 量化矩阵乘法 |
| `torch_npu.npu_dtype_cast(x, float8_e4m3fn)` | 类型转换（int8 → float8_e4m3fn） |
| `torch_npu.float8_e4m3fn` | FP8 数据类型（量化值） |
| `torch_npu.float8_e8m0fnu` | FP8 scale 类型（指数型） |

---

## 改动统计

| 文件 | 改动 |
|------|------|
| `mxfp8_method_npu.py` | **新增** (152 行) — LLM MXFP8 |
| `fp8.py` | +6 行 — NPU 分发 |
| `mxfp8_npu.py` | **新增** (167 行) — Diffusion MXFP8 |
| `quantization/__init__.py` | +1 行 — 注册 MXFP8Config |
| `transformer_loader.py` | +6 行 — quantization 优先级处理 |
| `mova_video_dit.py` | 无改动 — 使用现有 quant_config 参数 |
| `server_args.py` | +1 行 — 新增 quantization 字段 |
| `test_ascend_mxfp8_quantization.py` | **新增** (103 行) — 测试 |

**总计**：+261 行新增，影响 8 个文件。

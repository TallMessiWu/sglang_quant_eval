---
name: mxfp4-impl-ref
description: MXFP4 在线/离线量化完整实现参考——API 签名、shape 要求、已知 gotchas、与 MXFP8 差异
---

# MXFP4 实现参考

当需要实现或调试 MXFP4 量化时，参考以下内容。

## 核心 API（MXFP4 双级量化）

```python
# 量化（权重或激活）→ 返回 (quantized, l0_scale, l1_scale)
y, l0_scale, l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x, smooth_scale=None)
# x: FLOAT16/BFLOAT16, ND
# y: FLOAT4_E2M1, ND — shape 与 x 相同
# l0_scale: FLOAT32, shape[-1] = ceil(x.shape[-1] / 512)
# l1_scale: FLOAT8_E8M0, shape = x.shape + (ceil(x.shape[-1]/32+1)/2, 2)

# 矩阵乘（双级 scale）
out = torch_npu.npu_dual_level_quant_matmul(
    x1, x2,
    x1_level0_scale, x2_level0_scale,
    x1_level1_scale, x2_level1_scale,
    bias=None, output_dtype=torch.bfloat16
)
# x1: FLOAT4_E2M1, ND
# x2: FLOAT4_E2M1, **FRACTAL_NZ 格式**（必须！）
# x1_level0_scale: FLOAT32, ND
# x2_level0_scale: FLOAT32, shape = [x2.shape[-1]/512, x2.shape[-2]]（转置！）
# x1_level1_scale: FLOAT8_E8M0, ND
# x2_level1_scale: FLOAT8_E8M0, ND

# NZ 格式转换（用于权重）
weight_nz = torch_npu.npu_format_cast(weight.view(torch.int8), 29)
# 注意：只接受 2 个位置参数，customize_dtype 是 keyword-only
```

## ⚠️ 已知 Gotchas（调试过）

### 1. `x2 only supports NZ format`
`npu_dual_level_quant_matmul` 的 x2（权重）**必须**是 FRACTAL_NZ 格式（format=29）。

```python
# 量化后必须转换
qw = torch_npu.npu_format_cast(qw.view(torch.int8), 29)
```

### 2. `Check input x2Level0Scale shape failed, expected [A, B], but got [B, A]`
`npu_dynamic_dual_level_mx_quant` 对权重 `[out, in]` 返回的 `l0_scale` shape 是 `[out, in/512]`，
但 matmul 要求 `x2_level0_scale` 为 `[in/512, out]`（转置）。

```python
w_dual_scale = w_dual_scale.squeeze(-1).transpose(0, 1).contiguous()
```

### 3. `npu_format_cast` 第三个位置参数报错
签名是 `(Tensor input, int acl_format, *, int? customize_dtype=None)`，
`customize_dtype` 是 keyword-only，不能作为位置参数传：
```python
# 错误：torch_npu.npu_format_cast(x, 29, torch.int8)
# 正确：
torch_npu.npu_format_cast(x, 29)
```

## 在线量化完整实现模式（`process_weights_after_loading`）

```python
weight_fp = layer.weight.data
if weight_fp.dtype not in (torch.float16, torch.bfloat16):
    weight_fp = weight_fp.to(torch.bfloat16)
if not weight_fp.is_npu:
    weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

qw, w_dual_scale, w_scale = torch_npu.npu_dynamic_dual_level_mx_quant(weight_fp, smooth_scale=None)

# 必须的后处理（见 gotchas）
qw = torch_npu.npu_format_cast(qw.view(torch.int8), 29)
w_dual_scale = w_dual_scale.squeeze(-1).transpose(0, 1).contiguous()

layer.weight = Parameter(qw, requires_grad=False)
layer.weight_dual_scale = Parameter(w_dual_scale, requires_grad=False)
layer.weight_scale = Parameter(w_scale, requires_grad=False)
```

## 在线量化完整实现模式（`apply`）

```python
original_dtype = x.dtype
input_shape = x.shape
x_2d = x.reshape(-1, x.shape[-1])  # 3D→2D

qx, act_l0_scale, act_l1_scale = torch_npu.npu_dynamic_dual_level_mx_quant(x_2d, smooth_scale=None)

output = torch_npu.npu_dual_level_quant_matmul(
    qx, layer.weight,
    act_l0_scale, layer.weight_dual_scale,
    act_l1_scale, layer.weight_scale,
    bias=bias.to(torch.float32) if bias is not None else None,
    output_dtype=original_dtype,
)

output_shape = list(input_shape[:-1]) + [output.shape[-1]]
output = output.reshape(output_shape)
```

## MXFP8 vs MXFP4 对比

| | MXFP8 | MXFP4 (DualScale) |
|--|-------|-------------------|
| 量化 API | `npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)` | `npu_dynamic_dual_level_mx_quant(x)` |
| 矩阵乘 API | `npu_quant_matmul(..., group_sizes=[1,1,32])` | `npu_dual_level_quant_matmul(...)` |
| Scale 数量 | 1 级 | 2 级（l0 + l1） |
| 权重 dtype | `float8_e4m3fn` | `float4_e2m1fn_x2` |
| 权重转置 | 是（T） | 否 |
| 权重格式 | NZ（format=29） | NZ（format=29），必须 `.view(int8)` 后转 |
| dual_scale 转置 | 不需要 | 需要 `squeeze(-1).transpose(0,1).contiguous()` |

## msmodelslim 导出的 MXFP4 权重格式（离线加载用）

| 字段 | dtype | shape | 说明 |
|------|-------|-------|------|
| `weight` | `uint8`（两个 FP4 nibble/byte） | `[out, in/2]` | checkpoint 已由 `pack_fp4_to_uint8` 打包；推理前直接 transpose，禁止再次 `npu_dtype_cast` |
| `weight_scale` | `uint8` | `[out, in/32]` | e8m0 偏移 +127，还原：`.to(int16) - 127` |
| `weight_dual_scale` | `bfloat16` | `[in/512, out]` | L0 scale，已经是转置后的 shape |
| `bias`（可选） | `float32` | `[out]` | |

## 参考文件

- 实现：`sglang/.../quantization/mxfp4_npu.py`（在线）、`modelslim_mxfp4_scheme.py`（离线）
- 参考：`MindIE-SD/mindiesd/quantization/layer.py` → `W4A4MXFP4DualQuantLinear`
- API 文档：`docs/npu-api/` 下 `DualLevelQuantBatchMatmul.md`、`DynamicDualLevelMxQuant.md`

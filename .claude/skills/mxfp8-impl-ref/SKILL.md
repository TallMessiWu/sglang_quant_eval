---
name: mxfp8-impl-ref
description: MXFP8 在线/离线量化完整实现参考——API 签名、shape 要求、在线与离线对比、与 MXFP4 差异
---

# MXFP8 实现参考

当需要实现或调试 MXFP8 量化时，参考以下内容。

## 核心 API（MXFP8 单级量化）

```python
# 量化（权重或激活）→ 返回 (quantized, scale)
qx, scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch_npu.float8_e4m3fn)
# x: FLOAT16/BFLOAT16, 2D（需先 reshape）
# qx: float8_e4m3fn
# scale: float8_e8m0fnu, shape = [tokens, ceil(hidden/32)]

# 矩阵乘（单级 scale，block_size=32）
output = torch_npu.npu_quant_matmul(
    qx,
    weight.transpose(0, 1),       # weight 需转置
    weight_scale.transpose(0, 1), # weight_scale 需转置
    scale_dtype=torch_npu.float8_e8m0fnu,
    pertoken_scale=input_scale,
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    bias=bias.to(torch.float32) if bias is not None else None,
    output_dtype=original_dtype,
    group_sizes=[1, 1, 32],       # block_size=32
)

# dtype 常量
torch_npu.float8_e4m3fn    # FP8 权重/激活类型
torch_npu.float8_e8m0fnu   # MXFP8 scale 类型（e8m0）
```

## 在线量化实现模式（`mxfp8_npu.py`）

```python
# process_weights_after_loading
weight_fp = layer.weight.data
if weight_fp.dtype not in (torch.float16, torch.bfloat16):
    weight_fp = weight_fp.to(torch.bfloat16)
if not weight_fp.is_npu:
    weight_fp = weight_fp.to(f"npu:{torch.npu.current_device()}")

qw, w_scale = torch_npu.npu_dynamic_mx_quant(weight_fp, dst_type=torch_npu.float8_e4m3fn)
layer.weight = Parameter(qw, requires_grad=False)
layer.weight_scale_inv = Parameter(w_scale, requires_grad=False)

# apply（推理）
input_shape = x.shape
x_2d = x.reshape(-1, x.shape[-1])  # 3D→2D

qx, input_scale = torch_npu.npu_dynamic_mx_quant(x_2d, dst_type=torch_npu.float8_e4m3fn)

output = torch_npu.npu_quant_matmul(
    qx,
    layer.weight.transpose(0, 1),
    layer.weight_scale_inv.transpose(0, 1),
    scale_dtype=torch_npu.float8_e8m0fnu,
    pertoken_scale=input_scale,
    pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
    bias=bias.to(torch.float32) if bias is not None else None,
    output_dtype=original_dtype,
    group_sizes=[1, 1, 32],
)

output_shape = list(input_shape[:-1]) + [output.shape[-1]]
output = output.reshape(output_shape)
```

## 离线加载实现模式（`modelslim_mxfp8_scheme.py`）

msmodelslim 导出的权重格式：

| 字段 | dtype | shape | 说明 |
|------|-------|-------|------|
| `weight` | `float8_e4m3fn` | `[out, in]` | 已量化，直接加载 |
| `weight_scale` | `uint8` | `[out, in/32]` | e8m0 偏移 +127，推理前 reshape 为 `[out, in/32//2, 2]` |
| `bias`（可选） | `float32` | `[out]` | |

```python
# process_weights_after_loading（离线）
weight_scale = layer.weight_scale.data
weight_scale = weight_scale.reshape(weight_scale.shape[0], -1, 2)
layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)

# apply_weights（推理，与在线 apply 相同逻辑）
# 注意：离线的 weight_scale 参数名叫 weight_scale，在线叫 weight_scale_inv
```

## MXFP8 vs MXFP4 对比

| | MXFP8 | MXFP4 (DualScale) |
|--|-------|-------------------|
| 量化 API | `npu_dynamic_mx_quant(..., dst_type=float8_e4m3fn)` | `npu_dynamic_dual_level_mx_quant(x)` |
| 矩阵乘 API | `npu_quant_matmul(..., group_sizes=[1,1,32])` | `npu_dual_level_quant_matmul(...)` |
| Scale 数量 | 1 级（l1 only） | 2 级（l0 + l1） |
| 权重 dtype | `float8_e4m3fn` | `float4_e2m1fn_x2` |
| 权重转置 | **是**（`.transpose(0,1)`） | 否 |
| NZ 格式转换 | 不需要 | **必须**（`npu_format_cast(qw.view(int8), 29)`） |
| dual_scale 转置 | 不需要 | **必须**（`squeeze(-1).transpose(0,1).contiguous()`） |
| CANN 最低版本 | ≥ 8.0.RC3 | 待确认 |

## 参考文件

- 在线实现：`sglang/.../quantization/mxfp8_npu.py`
- 离线实现：`sglang/.../quantization/modelslim_mxfp8_scheme.py`
- 参考：`MindIE-SD/mindiesd/quantization/layer.py` → `W8A8MXFP8QuantLinear`

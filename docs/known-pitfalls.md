# 已知陷阱详解

> 每个陷阱的完整根因分析、调试过程、修复方法。
> CLAUDE.md 中只保留标题 + 一句话结论 + 指向本文档对应锚点的链接。

## 量化不生效/乱码输出

先验证模型是否注册成功。若 `sgl_kernel_npu` 某 kernel 不存在会导致模型模块 import 失败，`ModelRegistry` 静默跳过，fallback 到 HF Transformers（无量化感知），FP8 权重被当 BF16 解读 → 乱码。

```bash
python3 -c "from sglang.srt.models.registry import ModelRegistry; print(list(ModelRegistry.models.keys()))"
python3 -c "from sglang.srt.models.qwen3 import Qwen3ForCausalLM; print('OK')"
```

修复：`sgl_kernel_npu` 非核心 kernel 的 import 改为 try/except + `None` fallback（见 `rotary_embedding/base.py`）。
若是 `sgl_kernel_npu` 版本太旧、缺某个 kernel 符号（如跑 Qwen3.5 缺 `split_qkvgate_gemma_rmsnorm_rope`），
需从源码升级 `sgl_kernel_npu`，编译安装过程见 [sgl-kernel-npu-build.md](sgl-kernel-npu-build.md)。

---

## 模块级 `import torch_npu` 会炸掉全平台 CI

`quantization/__init__.py` 无条件 import `ModelSlimConfig`，链路上任何文件顶层 `import torch_npu` 都会让 CUDA/CPU/AMD/XPU CI 在 import 时 `ModuleNotFoundError`（PR #22352 踩过两次：`linear_method_npu.py`、`modelslim_mxfp8.py`）。

**标准写法**（见这两个文件）：

```python
from sglang.srt.utils import is_npu
_is_npu = is_npu()
if _is_npu:
    import torch_npu
```

模块级用到 `torch_npu` 属性的常量（如 `_FLOAT8_E8M0FNU_DTYPE`）也要用 `if _is_npu else` 三元守卫；函数体内的 `torch_npu.xxx` 调用只在 NPU 运行时执行，无需改。

> ⚠️ **不要用 `current_platform.is_npu()` 做这个守卫**（旧写法，2026-06-16 已废弃）：新 upstream 把 `sglang/srt/platforms/` 重构成「插件发现的懒单例」，NPU 是 out-of-tree 插件，没装注册 `entry_point`（group `sglang.srt.platforms`）的 NPU 平台插件时，`current_platform` 会 fallback 到 base `SRTPlatform`、`is_npu()` 在**真 NPU 机器上也恒返回 False**（单例缓存、永久卡住）→ `torch_npu` 不 import → 量化哑掉。upstream 自己全用 util 版 `from sglang.srt.utils import is_npu`（直接 `torch.npu.is_available()`，核心文件如 `model_runner.py`/`model_loader/loader.py` 都是模块级 `_is_npu = is_npu()`），这才是可靠且与 upstream 一致的写法。

---

## `process_weights_after_loading` 中 transpose 不加 `.contiguous()`

`npu_grouped_matmul` 通过 strides 感知内存布局，`.contiguous()` 会物理重排数据破坏 block-scale 映射 → 乱码。用 `Parameter(qw.transpose(1, 2), requires_grad=False)` 直接包装非连续 view，不要在 transpose 后加 `.contiguous()`。

---

## vllm-ascend MoE MXFP8/MXFP4 均为 offline

`AscendW8A8MXFP8DynamicFusedMoEMethod`（`w8a8_mxfp8.py:178`）和 `AscendW4A4MXFP4DynamicFusedMoEMethod`（`w4a4_mxfp4.py:119`）的 `process_weights_after_loading` 只做 layout transform，没有 BF16→FP 在线转换。

**MoE W4A8_MXFP 在 vllm-ascend 没有 MoE scheme**（`quant_parser.py` 注册了字符串但无对应类）。

MoE 在线量化需自实现：online quant 参考 dense `NPUMXFP8LinearMethod`，routing pipeline 参考 `npu_fused_experts_w4a4`。

---

## FusedMoE vs EPMoE quant method 共享，但 dispatch_output 类型不同

`Fp8Config.get_quant_method` 对 FusedMoE 及其所有 EP 子类（`DeepEPMoE`/`NpuFuseEPMoE`/`MoriEPMoE`）返回同一个 method 实例。EPMoE 传入 `apply` 的不是 `StandardDispatchOutput`，需单独处理。当前 `NPUMXFP8FusedMoEMethod.apply` 仅支持 `StandardDispatchOutput`（TP-only），其他类型抛 `NotImplementedError`。

---

## W4A8_MXFP checkpoint 权重格式

当前 msmodelslim `ascendv1.py` 使用 `pack_fp4_to_uint8` 导出 W4A8_MXFP 权重，物理格式为 packed FP4 `uint8 [out,in//2]`；SGLang `ModelSlimMXFP4W4A8Scheme.create_weights` 必须使用相同 shape/dtype，post-load 只做 FRACTAL_NZ layout 转换和 transpose。旧 `qwen3-8b-dense-w4a8` checkpoint 可能仍是早期 `float8_e4m3fn [out,in]` 格式，它不代表当前导出契约，需重新导出或转换后再由当前 scheme 加载。

---

## 离线 W4A8 (`W4A8_MXFP`) 在 A5 上有两个不同报错，根因完全不同

（`junlin_qwen3_dense_w4a8`，2026-06-24/25，A5 + `Qwen3-8B-mxw4a8-pack-full-0421`，graph 模式 e2e 输出已验证正常）

### ① prefill 阶段 `x2 should be in ... nz format, but it is 2` = 旧 torch_npu 的 FP4 `npu_quant_matmul` bug（量化相关，已靠升级解决）

`NPUMXFP4W4A8OfflineLinearMethod` 照搬 vllm（`npu_format_cast(weight,29,customize_dtype=fp8,input_dtype=fp4)` → `.transpose(-1,-2)` → `npu_quant_matmul(x2_dtype=float4_e2m1fn_x2, group_sizes=[0,0,32])`）在 **`torch_npu 2.10.0.dev20260320`** 报此错；升级到 **`2.10.0.post1.dev20260624`** 后消失，NZ 写法正确。

该 A5 强制 `allow_internal_format=False`（设 True 被打回、无 getter）但 NZ 仍能造出且 matmul OK，**不是阻塞点**；`npu_dynamic_mx_quant` 原生返回 3D block scale `[tokens, in//64, 2]`，无需 vllm `maybe_normalize_mxfp_scale_layout`（那个只 MoE 用）。代码保持 vllm 对齐的 NZ 写法（cast29+transpose），**不要切 ND**。

### ② decode 阶段 `atb::OperationSetup` 段错误 = eager-decode 走 ATB `_npu_paged_attention`，与量化无关

（已坐实根因 + 真修复，2026-06-25）

升级 torch_npu 后改在 decode 崩。三段同步插桩定位：qkv 的 `npu_quant_matmul` 同步 OK、o_proj 的 matmul **还没跑**就在「进 o_proj 前的入口同步」崩 → fault 来自 qkv→o_proj 之间的 **decode attention**，不是 FP4 matmul。

**根因**：graph / eager 两模式在 `AscendAttnBackend` dispatch 到不同 attention 算子——
- graph decode（`forward_decode_graph`，`ascend_backend.py:2256`）走 `npu_fused_infer_attention_score`（**aclnn** op，吃 ND，不建 internal 张量）
- eager decode（`forward_decode` 默认分支 `ascend_backend.py:2617`，`use_fia=use_fa=False`）落到 `torch_npu._npu_paged_attention`（**ATB** op）

这台 A5 把 `allow_internal_format` 强制打回 False（`utils.py:114` 设 True 不生效），ATB `OperationSetup` 要建 FRACTAL_NZ internal workspace 张量、建不出来 → 段错误（报错命名空间 `atb::` 正对 ATB 算子，FIA 是 aclnn 不带此前缀）。

**两条解法都已在 A5 验证可跑**：
1. **别加 `--disable-cuda-graph`**（默认 graph 模式，FIA，离线 W4A8 输出正常）
2. 若确需 eager decode，**`ASCEND_USE_FIA=1`** 让 eager 路径也改走 FIA（`ascend_backend.py:2552`，aclnn）绕开 `_npu_paged_attention` → **A5 实测能跑**（这是一整套 FIA 模式，`memory_pool_npu.py:67`/`npu_graph_runner.py:113` 同读此 flag、KV cache 布局随之变，非单纯换算子）

此 attention bug **非 W4A8 特有**（与 linear 量化无关、在线/离线同理），属独立 NPU attention 问题，不在本 PR 交付范围。

**误判史**：我中途基于旧 torch_npu 误加的 ND commit `4c1a0f5a0d` 已回退（②的 atb 段错误一度被我也归给 torch_npu 版本/ND，实为 eager-decode attention，已订正）。`--disable-cuda-graph` 是我早期排查时让用户加的，结果它本身才是触发 ② 的开关。

---

## 在线 W4A8 FP4 崩「output y must be same shape as input x」的真凶 = fp4 dtype 来源用错

（`junlin_qwen3_dense_w4a8`，2026-07-06 二次诊断）

`mxfp_w4a8` 在线量化 `npu_dynamic_mx_quant(weight, dst_type=fp4, round_mode="round")` 在 torch_npu `2.10.0.post2.dev20260704`（CANN 9.1.0，A5）崩 561002。

**~~此前误诊为「upstream FP4 kernel 被整体打回、只能纯 torch RTN 绕过」（commit `db4dee06ea`）——已推翻。~~**

**真相**：**NPU op 的 fp4 dtype 参数**（`npu_dynamic_mx_quant(dst_type=)` / `npu_quant_matmul(x2_dtype=)` / `npu_format_cast(input_dtype=)`）只接受 `torch_npu.float4_e2m1fn_x2`（int enum，==`296`），拒绝 `torch.float4_e2m1fn_x2`（torch dtype 对象，存在但喂给 op 会在 op-plugin 层报错；传 `None` 报「Expected a value」）。

生产 `_get_float4_e2m1fn_x2_dtype()` 当时 `getattr(torch, "float4_e2m1fn_x2")` → 拿 torch dtype 对象 → 崩。

A5 probe（同 shape 分别拿 `torch_npu.float4_e2m1fn_x2`=296 vs `torch.float4_e2m1fn_x2` 当 dst_type）在**同一台 post2 机器**坐实：
- `dst=296` 走完整 quant→format_cast→matmul 链 **PASS**（含 weight `[4096,4096]→[4096,2048]`）
- `dst=torch.float4` 在 quant 第一步 **FAIL**

**kernel 完全没坏。** fp8 不中招是因为 `torch.float8_e8m0fnu`(dtype 对象)作 `scale_dtype` 被接受——**只有 fp4 dtype 挑剔**。post1 能跑 post2 崩：升级收紧的是 op 对 fp4 dst_type 的**类型接受度**，不是 kernel。

**修法（恢复 op 版）**：`_get_float4_e2m1fn_x2_dtype()` 改为 `is_npu()` 时优先 `getattr(torch_npu, "float4_e2m1fn_x2")`（函数体内惰性 import torch_npu，不炸跨平台 CI），torch fallback；然后 `git checkout` 回 op 版 `process_weights_after_loading`（`npu_dynamic_mx_quant(dst=fp4)`）、删纯 torch `_mxfp4_quantize_weight`。在线+离线的 matmul/format_cast 一并受益（离线 post2 也没验过、同隐患）。**A5 e2e 已验证：op 版 + dtype 修复后在线 `mxfp_w4a8` 输出连贯（2026-07-06，用户实测）。**

**两条教训**：
1. 第一次误诊「kernel regression」正因 probe **没忠实复现生产的 dtype 来源**——probe v1 用 `torch_npu.float4`(296) 全 PASS、生产用 `torch.float4` 会崩，差异全在 dtype 从哪个模块取。**probe 要逐字节照抄生产 call，dtype 取自哪也算 call 的一部分**；「先 probe 再下结论」还不够，probe 本身必须忠实。
2. PR #23650 review 定的「W4A8 全脱 torch_npu、dtype 走 `getattr(torch,...)`」对 **fp4 是错的**——fp4 dtype 必须来自 torch_npu（fp8/e8m0 无所谓）。

---

## A8W4 `npu_quant_matmul` 的 bias 必须是 BF16 二维 `[1,N]`

Qwen3.5 在线 W4A8 的首次图片请求会进入视觉塔 QKV；该层带一维 bias。若沿用其他量化 matmul 的 FP32 `[N]` 处理，A5 会在 `NPUMXFP4W4A8LinearMethod.apply` 报 `The dimension of bias should be 2. Actual bias dimension is: 1.`。MindIE-SD 的同类 W4A8 路径会将 bias 转为 BF16，并把 `[N]` 扩成 `[1,N]`。

在线和离线 W4A8 Dense 方法必须共享这一规范化逻辑。Qwen3 文本模型常见的无 bias 路径保持 `None`，不会受到数值或性能影响；不能把这个例外扩散到 MXFP8/W4A4 等其他量化方法。

---

## sglang 文档路径迁移过两次；长期分支里的 `docs_new/` 改动会被 merge 静默吞掉

（2026-07-06 踩第一次，`junlin_qwen3_dense_w4a8` PR #23650；2026-08-25 踩第二次，PR #32602）

upstream 先把文档从 `docs/`（`.md`）迁到 `docs_new/docs/`（`.mdx`），**之后又把 `docs_new/` 改回 `docs/`**。当前 main 只有 `docs/docs/**/*.mdx`，`docs_new/` 与当年那个 `scripts/ci/check_no_docs_changes.py` 守卫都已不存在。

**改文档一律去 `docs/docs/` 对应位置**，用 `.mdx`（HTML `<table>` + JSX `style={{color:'green'}}`，非 md 表）。改之前先 `git ls-tree -r upstream/main --name-only | grep <关键词>` 确认真实路径，不要照抄旧文档快照——Ascend 量化页面这一路走过 `docs/platforms/ascend/ascend_npu_quantization.md` → `docs_new/docs/hardware-platforms/ascend-npus/ascend_npu_quantization.mdx` → `docs/docs/hardware-platforms/ascend-npus/optimization/quantization.mdx`。

**最危险的是长期分支**：目录改名 + 文件重排后 git 的 rename 检测认不出来，合并 `upstream/main` 时**不报冲突**，只是把你在 `docs_new/` 下的改动一起删掉。合并后必须 `git diff upstream/main -- docs` 复核自己的文档改动还在不在，为空就说明被吞了，要手动搬到新路径。

---

## MoE MXFP8 `npu_grouped_matmul` 必须显式传 `x_dtype` + `weight_dtype`

仅传 `scale_dtype=FLOAT8_E8M0FNU_DTYPE` + `per_token_scale_dtype=FLOAT8_E8M0FNU_DTYPE` 不够——kernel 无法仅从 scale_dtype 推断「权重/激活是 fp8_e4m3fn 且 scales 是 UE8M0 block scale」，会走错 dequant 路径产生乱码。

对齐 vllm-ascend `A5DeviceAdaptor.get_quant_gmm2_kwargs` (`device_op.py:460-466`)，gmm1/gmm2 都加：

```python
x_dtype=torch_npu.float8_e4m3fn,
weight_dtype=torch_npu.float8_e4m3fn,
scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
per_token_scale_dtype=_FLOAT8_E8M0FNU_DTYPE,
```

注意：dense linear (`npu_quant_matmul`) **不** 需要 `x_dtype/weight_dtype`，它通过 `group_sizes=[1, 1, 32]` 显式带块大小，能从 tensor dtype 推断模式；MoE `npu_grouped_matmul` 没 `group_sizes`，必须靠 `x_dtype/weight_dtype` 显式声明。

---

## MoE MXFP8 的 gmm1 需使用 fused gmm+swiglu+quant

对齐 vllm-ascend A5 MXFP MoE 路径，gmm1 使用 `torch_npu.npu_grouped_matmul_swiglu_quant_v2`，而不是拆成 `npu_grouped_matmul` → `npu_swiglu` → `npu_dynamic_mx_quant`。

该 fused op 的 `group_list` 需要从 count-style（`expert_tokens_num_type=1`）转换为 cumulative-style（`group_list.cumsum(dim=0)`）；gmm2 仍使用 `npu_grouped_matmul` 并保留 count-style `expert_tokens`。

---

## MoE MXFP8 的 weight + scale 必须 `.transpose(1, 2)` 但 _不要_ `.contiguous()`

**真实机制是内核断言，不是带宽效应**（2026-07-21 A5 一次性 kernel 探针实测更正，脚本已删）。`npu_grouped_matmul_swiglu_quant_v2` 有一条 `CheckMXTranspose`：**weight 与 weight_scale 的 transpose 标志必须一致**。只给 weight 加 `.contiguous()`（或转 NZ）而 scale 仍是 transpose view，会**直接报错**，不是静默乱码：

```
AclNN_Parameter_Error(EZ1001): The transposition of weightScale/weight
should be equal, but actual transpositions are true/false.
CheckMXTranspose failed.
```

**两边一起 `.contiguous()` 数值是正确的**（cos 与 strided view 完全一致），只是更慢：128-expert 下 decode −6.2%、prefill −0.5%。所以结论「不要 contiguous」成立，但理由是实测更慢 + 容易踩标志不一致，而非早先写的「block-scale 映射错位 → 乱码」。

> ⚠️ 小 expert 数的 micro-benchmark 会给出完全相反的结论：E=4/top_k=2 时 contiguous 反而快 58%。定 layout 必须用真实 expert 数（128）测。

正确做法（对齐 vllm-ascend `AscendW8A8MXFP8DynamicFusedMoEMethod.process_weights_after_loading`，`w8a8_mxfp8.py:332-339`）：

```python
layer.w13_weight = Parameter(qw13.transpose(1, 2), requires_grad=False)         # [E, H, 2I] strided view
layer.w13_weight_scale = Parameter(s13.transpose(1, 2), requires_grad=False)    # [E, H//64, 2I, 2] strided view
```

内存仍是 `[E, N, K]` / `[E, N, K_blk//2, 2]` 物理布局，但逻辑 shape 已变为 `[E, K, N]` / `[E, K_blk//2, N, 2]`——同时满足 kernel 的「n-dim 相等」和「transposition 一致」两个约束。

`npu_dynamic_mx_quant` 对 2D 输入 `[N, K]` 直接吐 **3D** scale `[N, K_blk//2, 2]`（参考 `mxfp8_npu.py:144`，**不要手动 reshape**——已经 3D，stack 后是 4D，再 reshape 会 `too many values to unpack`）；`torch.stack` 拼出 weight 3D `[E, N, K]` + scale 4D `[E, N, K_blk//2, 2]`；transpose(1, 2) 后即为正确布局。

注意：dense linear 路径 (`NPUMXFP8LinearMethod`) 用 `.transpose(0, 1).contiguous()` 是 OK 的，因为 `npu_quant_matmul` 接受 contig 布局；MoE 的 `npu_grouped_matmul` 不接受。

踩坑历史：早先版本错误地加了 `.contiguous()`，跑通但输出乱码——纠正回 vllm-ascend 的 strided-view 布局后修复。（当时归因为「映射错位」，2026-07-21 探针证明真实约束是上面的 `CheckMXTranspose` 断言。）

### FRACTAL_NZ：cast 必须在 transpose _之前_

同一条断言决定了 NZ 怎么加。`npu_format_cast` 返回的是物理重排后的张量，transpose 标志为 false，所以**先 transpose 再 cast 会让 weight 与 scale 标志不一致而报错**——同文件 int8 MoE 方法（`moe_methods.py:378`/`:485`）的 `npu_format_cast(w.transpose(1, 2))` 写法**不能照抄到 MXFP8**（int8 没有 MX scale 要同步）。正确顺序与 dense W4A8（`linear_method_npu.py:443`/`:578`）一致：

```python
weight = npu_format_cast(weight)                 # [E, N, K] ND -> FRACTAL_NZ
Parameter(weight.transpose(1, 2), ...)           # 再 transpose，标志与 scale 一致
Parameter(scale.transpose(1, 2), ...)            # scale 保持 strided view，不动
```

A5 实测（Qwen3-30B-A3B，128 experts，torch_npu 2.10.0.post2）：**decode +1.4%、prefill +3.8%**，噪声底噪 0.2~0.3%，输出与改前完全一致。远不及 int8 路径报告的 ~10%，但两个 shape 方向一致。分支 `junlin_qwen3_moe_w8a8_nz`（commit `d419aa41f`）。

其他实测结论：

- `npu_format_cast` **直接接受 `float8_e4m3fn`**；vllm-ascend `fp8.py` 的 `uint8 view + customize_dtype` 写法在此 build 上失败（`Cannot find bin of op TransData ... FRACTAL_NZ_C0_32`）。
- NZ 权重配 ND 的 e8m0 block scale 数值正确，不会静默读错 scale。
- `_is_nz_aligned` 原先对 fp8 落到兜底 `return True`；已补 int8 同款规则（`k%16, n%32`，fp8 同为单字节）。

---

## strided-view（去 `.contiguous()`）在 w4a8 硬件上反而变慢

（`junlin_qwen3_dense_w4a8`，2026-06-16）

strided weight/scale view（w8a8 上实测 -6.6% 提升）在 **w4a8 分支的 NPU 上端到端比更新前 w4a8 慢**（多条路径），且是「慢」非「乱码」。故 commit `9d6e9583e` 把两处 `.contiguous()` 恢复回来：MXFP8 dense（`NPUMXFP8LinearMethod`，merge 带来的 strided）+ W4A8 dual-level（`NPUMXFP4W4A8LinearMethod`，原 perf commit `33dfc0b9b` 去掉的）。

**注**：`NPUMXFP4W4A8LinearMethod` 的 dual-level 路径已于 2026-06-25（commit `a1947f3133`）整体替换为单级真 W4A8（apply 复用离线 NZ 路径），此条对它已不适用；MXFP8 dense 那半仍有效。**bias 缓存（`layer.bias_fp32`）是纯增益、保留**。

strided 优化版存档在分支 **`junlin_qwen3_dense_w4a8_strided`**（72fa20005）。

**教训**：`.contiguous()` 去留是「硬件/kernel 相关」，不要跨分支照搬 w8a8 的 layout 优化，需各自 NPU benchmark。

---

## NPU kernel 类在 CPU 上构造即炸，CPU 单测写不出来

（2026-08-25 踩，PR #32601）

`HiddenStatesDynamicQuant.__init__` 曾直接 `self._op = torch.ops.npu.npu_dynamic_mx_quant`。没装 torch_npu 时 `torch.ops.npu` 是空 namespace，属性访问就 `AttributeError`，于是 `NPUW4A8MXFP4MoEMethod()` / `NPUW4A4MXFP4MoEMethod()` 构造不出来，连带持有它们的 `ModelSlimW4A8MXFP4MoE` / `ModelSlimW4A4MXFP4MoE` 也无法在 CPU 上实例化——`register_cpu_ci` 的 scheme 测试一行都写不了。

已改成首次调用时 `getattr(torch.ops.npu, self._op_name)`：dtype 校验仍在 `__init__` 里立刻报错，缺 torch_npu 依然在真正执行时炸得很响。**新写 NPU kernel 类时别在 `__init__` 里绑 `torch.ops.npu.*`**，否则等价于宣布这条路径没有 CPU 覆盖。

---

## 新增 MoE 量化方案前先查 `moe_quant_schemes`，重名条目会变死代码

（2026-08-25 踩，PR #32601 / #32602）

`ModelSlimConfig.get_moe_scheme` 用一张 `[(scheme_name, scheme_class), ...]` 列表按顺序匹配，命中第一条就返回。上游 [#30318](https://github.com/sgl-project/sglang/pull/30318) / [#30319](https://github.com/sgl-project/sglang/pull/30319) 已把 `W4A8_MXFP` → `ModelSlimW4A8MXFP4MoE`、`W4A4_MXFP4` → `ModelSlimW4A4MXFP4MoE` 注册进去；长期分支里再追加一条同名条目**不会冲突也不会报错**，只是排在后面永远匹配不到，整套自研 scheme 变成死代码。

同理，`hardware_backend/npu/quantization/moe_methods.py` 里上游已有 `NPUW4A8MXFP4MoEMethod` / `NPUW4A4MXFP4MoEMethod`。要加在线量化就在这些类里按权重 dtype 分支（BF16/FP16 = 在线，uint8 = 离线 checkpoint），不要另起一个平行类。

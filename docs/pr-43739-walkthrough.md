# PR #43739 代码走读：Ascend 在线 W4A8 MoE 与完成度加载

> **一句话：让 BF16/FP16 checkpoint 中的 MoE 专家接入已有的 FP4 权重 × FP8 激活执行链，并把权重量化从“全模型加载之后”提前到“每份本地逻辑权重加载完整时”。**

- PR：[Support online W4A8 MoE](https://github.com/sgl-project/sglang/pull/43739)。
- 走读日期：2026-11-06；源码固定于 `29d7388454355c2d25fb54261e38b1adc009b929`，当次 GitHub PR head、fork 分支与本地一致。
- 对照基线：PR base `4dc6070f589caf15586e85215b5bfd58c712331f`；净 diff 为 **17 个文件，+1456/-136**，其中 13 个源码文件、4 个测试文件。
- 本地阅读入口：`sglang/qwen3.5_moe_w4a8/`。下文源码路径省略该前缀，`srt/` 进一步省略 `python/sglang/`。
- 本文解释这个版本的实现，不维护另一份 PR/CI 状态表。合入依赖与验收见[目标计划](mxfp-refactoring-plan.md)，可用 worktree 见[分支表](branches.md)。PR body 中仍有旧 head/旧测试环境描述，不能用它们替代实际 diff。

## 1. 先厘清：这个 PR 到底改变了什么

### 1.1 三条主线

| 主线 | 修改前 | 修改后 |
| --- | --- | --- |
| 在线 MoE 入口 | `mxfp_w4a8` 遇到 `FusedMoE` 回退到 BF16/FP16 | 选择在线 W4A8 MoE method，复用 Ascend W4A8 kernel |
| 权重生命周期 | 为浮点源权重分配存储，加载后再量化 | meta 占位、按需分配，本地逻辑权重完成即量化，释放旧浮点 Parameter 的 backing |
| FP4 权重策略 | `round_mode="round"`，其余依赖算子默认值 | Linear/MoE 统一显式使用 `rint / scale_alg=2 / dst_type_max=7.25` |

附带修正围绕这条链路展开：Linear 本地 K 对齐回退、A8W4 bias、post-load 设备别名，以及 W13/W2 kernel 职责绑定。

### 1.2 三个容易混淆的边界

1. **W4A8 不是 W4A4，也不是 INT4 W4A8。** 权重为 MXFP4 E2M1，推理激活为 MXFP8 E4M3，block scale 为 UE8M0。
2. **在线不是每次 forward 都重新量化权重。** 权重在初始化加载期间转换；每次推理动态量化的是 activation。重载是新的加载周期。
3. **不是为 Qwen3.5 复制一套 MoE 算子。** 配置按 `LinearBase` / `FusedMoE` 选择方法，Qwen3、Qwen3.5 等模型复用底层能力；实际模型支持仍需分别验证。

正式 diff 不包含 #36426 的组合补丁、#32602 的在线 W4A4 功能，也不包含已归档的 TP 路由阈值/GDN 输入复用实验。**这里不主张 serving 提速。**

## 2. 建议阅读顺序：不要从算子参数开始

| 顺序 | 文件 / 符号 | 要回答的问题 |
| --- | --- | --- |
| ① | [`srt/layers/quantization/npu_mxfp4.py:76`][config] | 哪些层进入在线 W4A8，哪些保留浮点？ |
| ② | [`…/npu/quantization/online_moe_methods.py:76`][online-moe] | 在线 checkpoint 如何接入现成 MoE runner？ |
| ③ | [`…/npu/quantization/online_weight_loader.py:14`][loader] | 如何判断加载完成，何时量化，如何重载？ |
| ④ | [`srt/layers/moe/fused_moe_triton/layer.py:1270`][expert-ids] | 哪些专家真正属于本 rank，副本怎么计数？ |
| ⑤ | [`…/npu/quantization/moe_methods.py:224`][moe] | 在线浮点与离线 packed 权重在哪里汇合？ |
| ⑥ | [`srt/layers/moe/moe_runner/ascend.py:237`][runner] | dispatch 后如何执行 W13、激活和 W2？ |
| ⑦ | [`…/npu/quantization/linear_method_npu.py:357`][linear]、[`srt/model_loader/post_load.py:109`][post-load] | Dense 路径与外层生命周期如何配套？ |

其中 `ascend.py`、`moe_utils.py`、`moe/matmul.py` 等在本次净 diff 中没有修改。它们是理解执行链必需的**复用代码**，不是本 PR 新增的一套 runner/GMM。

整体链路：

```text
--quantization mxfp_w4a8
  └─ Mxfp4W4A8Config.get_quant_method()
      ├─ Linear → NPUMXFP4W4A8LinearMethod
      │    └─ NPUOnlineDenseWeightLoader
      └─ FusedMoE → NPUW4A8MXFP4OnlineMoEMethod
           └─ NPUOnlineMoEWeightLoader
                ├─ W13 独立完成度 tracker
                └─ W2  独立完成度 tracker

checkpoint shard
  → 原生 TP/EP loader 写入本地浮点源
  → 完整时 on_complete()
  → FP4 量化 + weight/scale layout 准备
  → post-load 只检查完成状态
  → Ascend runner 消费 prepared weights，动态量化激活
```

## 3. 配置入口：接入 MoE，同时保留明确的回退边界

入口是已有注册项 `mxfp_w4a8 → Mxfp4W4A8Config`，不是新建另一套 CLI，也不要求额外的 `--online-quantization` 开关。

### 3.1 Linear 的判断顺序

[`get_quant_method()`][config] 依次执行：

1. `is_layer_skipped(...)`：尊重 `ignored_layers` 和 fused-name mapping。
2. 检查 **本地 K** 是否满足 `K % 32 == 0`；不满足则记录 warning，返回 `UnquantizedLinearMethod`。
3. NPU 上返回在线 W4A8 Linear method；其他设备的量化路径明确报 `NotImplementedError`。

为什么要新加 [`local_input_size()`][local-k]？因为选择 quant method 时，`input_size_per_partition` 尚未建立：

```python
# 语义摘录
if isinstance(layer, RowParallelLinear):
    local_k = layer.input_size // get_group_rank_size(layer.tp_group)[1]
else:
    local_k = layer.input_size
```

- RowParallel 按 **layer 已绑定的 TP group** 切 K，不能随意取全局 TP 大小。
- ColumnParallel 切输出维，ReplicatedLinear 不切 K；两者不能再把 K 除一次 TP。
- 例如全局 K=4352、Row TP=16，本地 K=272，虽然全局 K 对齐，本地 K 却不对齐。
- 视觉 fc2 的 K=4304 会保留原浮点精度，不能宣传为“全视觉 W4A8”。

这里的 `% 32` 是当前 Linear 的提前回退条件，**不是完整的硬件支持证明**。后续 scale pair 布局仍有自己的形状约束。

### 3.2 MoE 的判断顺序

MoE 同样先检查 ignored layers，再在 NPU 上返回 `NPUW4A8MXFP4OnlineMoEMethod`；非 NPU 的在线 W4A8 MoE 不再静默回退。

这也不意味着所有非专家层都保持 BF16：本配置同时处理可量化的 Linear。router、attention、shared MLP、视觉塔具体落到什么精度，要沿模型构造和 ignored-layer 配置判断，不能从“MoE W4A8”这个名字推导。

## 4. 在线 MoE method：只接管加载与 kernel 选择

[`NPUW4A8MXFP4OnlineMoEMethod`][online-moe] 继承 `UnquantizedFusedMoEMethod`，乍看矛盾，实际是有意复用：**checkpoint 输入仍然是未量化权重，原生 shape、属性和 shard loader 正好适用。**

### 4.1 构造阶段不分配整层浮点权重

`create_weights()` 在 `torch.device("meta")` 下调用父类创建权重。对常见 gated MoE，令：

- `E`：本 rank 的专家存储槽数；
- `H`：hidden size；
- `I`：TP 后的 intermediate size。

原生浮点形状为：

```text
W13（gate + up）：[E, 2I, H]
W2 （down）     ：[E, H, I]
```

这时只有 Parameter 的 shape/dtype/loader metadata，没有对应的大块浮点存储，也还没有 weight scale。

特殊处理是 bias：若 `with_bias=True`，bias 会在原 load device 上实体化，并保留原生 loader，不进入 FP4 完成度 tracker。相关测试证明的是 **bias 分配/加载生命周期**，不能据此推断所有带 bias MoE 的 W4A8 算子数值支持。

### 4.2 runner 只允许 `auto` / `ascend`

`create_moe_runner()` 为两次矩阵乘分别建立 kernel：

```python
w13_kernel = NPUW4A8MXFP4MoEMethod(weight_prefix="w13")
w2_kernel  = NPUW4A8MXFP4MoEMethod(weight_prefix="w2")
```

通过已有的 [`create_ascend_moe_runner()`][moe-utils]，先把 kernel 挂到 layer，再构造 Ascend runner。限制 backend 是为了避免其他 runner 的 gate/up 排列约定混进这条路径。

实际 `FusedMoE` 初始化先 `create_weights()`、再 `create_moe_runner()`；checkpoint 加载发生在模型构造之后。因此 loader 中的完成回调是**调用时**才查询 `layer.w13_kernel` / `layer.w2_kernel`，不是创建 meta 参数时立即量化。

推理入口继续复用父类 `forward_npu()`，最终调用 `self.runner.run(dispatch_output, layer)`。继承 unquantized 类并不代表最终仍用 BF16 权重计算。

## 5. 核心：完成度 loader 是一个受约束的状态机

### 5.1 “逻辑权重完成”是什么意思

[`_NPUOnlineWeightLoader`][loader] 每次管理一个完整本地目标张量：Dense 的 `weight`，或 MoE 的 `w13_weight` / `w2_weight`。

**不是收到一个 checkpoint 文件就完成，也不是单个专家到齐就量化。** MoE 的 W13 必须覆盖本地全部目标专家槽的 gate/up；W2 独立等到自己的目标张量完整。

### 5.2 不重写 TP/EP，只包住原生 copy

`_load()` 的核心可以压缩成下面的语义伪代码：

```python
with per_weight_lock:
    validate_state_and_duplicate_shards()
    source = materialize_if_meta_or_reloading()
    with CopyNumelCounter() as counter:
        original_loader(source, checkpoint_shard, ...)
    loaded_numel += counter.copied_numel
    if loaded_numel == target_numel:
        on_complete()             # 量化并准备推理布局
        preserve_loader_attrs()
        release_old_source()
        state = "converted"
```

这里复用了上游 [`CopyNumelCounter`][counter]：通过 `TorchDispatchMode` 统计实际 `aten.copy_` **目标元素数**。原生 loader 先 narrow/切 TP shard，再执行 copy，因此计数对应本地写入，而不是 checkpoint 的全局 `numel()`。

但**计数不能单独保证正确性**。同一 shard 写两次也可能凑够总数，所以还要维护 `loaded_shards`：

- Dense：普通 shard ID，或 merged tuple 中各 shard ID；`None` 表示整份加载。
- MoE：`(destination_expert_slot, shard_id)`；整份 fused 加载使用 `{None}`。
- 拒绝重复 key，以及整份加载和分片加载的混用；零 copy 或累计超出目标也报错。

这不是通用的逐地址覆盖检测器。它依赖原生 loader 的 copy 语义和已知 shard 标识，不能解释成对任意外部 loader 都能证明“每个元素恰好写一次”。

### 5.3 状态转换

```text
loading（初始，源参数为 meta）
  ├─ 首次有效 shard → 分配浮点 source
  ├─ 部分 shard     → 继续累计，不能推理/通过 post-load
  ├─ 目标全部完成   → on_complete → converted
  └─ copy/转换失败  → failed（不自动重试）

converted
  ├─ 继续写入       → 拒绝：不能把重复加载当成新一轮
  └─ finish_post_load() → ready_reload

ready_reload
  ├─ 再次 post-load → 检查 prepared shape/dtype，不重复转换
  └─ 下轮有效 shard → 清空计数，重新分配浮点源 → loading
```

锁覆盖 copy 和 conversion，防止并发 checkpoint worker 同时把同一权重判定为完成。锁是**每份逻辑权重独立**的，不会把整个模型所有权重串行化。

### 5.4 为什么还要清空旧 Parameter

量化通常会用新的 packed Parameter 替换 `layer.weight` / `layer.w13_weight`。只替换 layer 属性不够：模型 loader 可能缓存了旧 `Parameter`，使原浮点存储仍被引用。

代码因此做两件事：

1. 将源参数的缺失属性复制到 prepared 参数，保留 registered weight loader 等 metadata。
2. 如果新旧 Parameter 不同，把旧 `source.data` 改为空 tensor，使**持有旧 Parameter 本身**不再留住原始大存储。

同时，wrapper 故意不依赖调用者传进来的 `param`，而是每次从 layer 查当前 source；旧 meta/浮点参数句柄不会把下一轮写入导向已经失效的对象。

注意结论范围：这处理的是代码里的旧 Parameter 引用；若外部另存了共享原 storage 的 tensor view，不能仅凭这行代码宣称所有别名都已释放。

### 5.5 post-load 从“做转换”变成“检查转换已完成”

在线 Dense/MoE method 的 `process_weights_after_loading()` 只调用 `finish_post_load()`：

- 未加载或只加载一部分，立即报错；
- `converted` / `ready_reload` 状态允许通过；
- 当前参数若仍是 meta，或 shape/dtype 不符合记录的 prepared layout，拒绝旁路替换。

MoE kernel 中同名的 `process_weights_after_loading(layer, prefix)` 则仍负责真正转换，只是它现在被 `on_complete()` **提前调用**。走读时务必分清这两层同名函数。

该检查不是权重内容校验：相同 shape/dtype 的原地篡改不在其检测范围内。正规 reload 必须走 registered loader；它也不是支持推理并发热更新的原子切换协议。

## 6. MoE 专家映射：先知道会写哪里，再分配

[`NPUOnlineMoEWeightLoader.weight_loader()`][loader] 收到逐专家权重后，先调用 [`_weight_loader_expert_ids()`][expert-ids]，得到真正的目标槽：

- 无本地目标：直接返回，**不分配 source，也不启动 reload**。
- 有本地目标：以目标槽和 `w1/w3/w2` 组成去重 key，然后让原生 loader 完成实际切片和写入。
- 一个逻辑专家因 EPLB 对应多个本地物理副本时，每个副本都属于目标，而不是只处理其中一个。

本 PR 将原有专家目标解析提取成可复用 helper；不是重新发明 EP 映射。原有规则继续保留，包括 `_sglang_require_global_experts`、shared slot，以及先过滤本 rank 的 routed physical slots、再映射本地 ID，避免把别的 rank 的 routed slot 误当成 shared slot。

**它如何降低加载峰值？** 已完成权重尽早变成低比特常驻态，浮点 source 不必等全模型加载完才释放。但当前粒度仍是完整本地 W13/W2，不是逐专家/逐 block 流式量化；多个逻辑张量交错加载时也可能同时保留未完成 source。因此不能把峰值写成“严格只有一份小 buffer”。

## 7. 权重准备：在线与离线在 packed 表示处汇合

### 7.1 在线分支只由浮点输入触发

[`NPUW4A8MXFP4MoEMethod.process_weights_after_loading()`][moe] 先看权重 dtype：

```text
FP16/BF16 source
  → 必要时搬到当前 NPU
  → npu_dynamic_mx_quant（FP4 权重策略）
  → packed FP4 + UE8M0 scales
  ┐
  ├→ prepare_w4a8_mxfp_weight → prepared weight/scale
  ┘
离线已打包 weight + scales
  → 跳过上述动态权重量化
```

在线统一显式传入：

```python
dst_type=fp4_dtype
round_mode="rint"
scale_alg=2
dst_type_max=7.25
```

**这组参数属于 FP4 权重，不属于 A8 激活。** 原有 MXFP8 activation 调用仍使用其自身策略；不能把 `7.25` 顺手复制到 FP8 路径，也不能因评审建议就声称旧默认参数必然产生精度错误。

### 7.2 把 scale 的轴讲清楚

以下是代码采用的逻辑布局，令任一专家矩阵的浮点源为 `[E,N,K]`，例子取满足 pair 组织要求的 K：

| 阶段 | weight | scale |
| --- | --- | --- |
| 浮点 source | `[E,N,K]`，BF16/FP16 | 无 |
| 动态 MXFP4 输出 | packed `[E,N,K/2]` | `[E,N,K/64,2]`，或平坦 `[E,N,K/32]` |
| 对齐 checkpoint 形状 | packed 权重不二次量化 | 四维在线 scale 先 `flatten(-2)` 成 `[E,N,K/32]` |
| 公共准备后 | FP4-aware NZ cast，再 `transpose(-1,-2)` | `view(E,N,K/64,2).transpose(1,2)` → `[E,K/64,N,2]` |

[`prepare_w4a8_mxfp_weight()`][moe-utils] 还处理 `int8` packed carrier 的 `uint8` view，以及 FP32 checkpoint scale 的解码。ModelSlim 的原始 UE8M0 bytes 不应被当作普通浮点 scale 再量化。

NZ cast 显式携带 `customize_dtype=torch.float8_e4m3fn` 和 `input_dtype=fp4_dtype`。这是后端表示契约，**不等于模型权重改成 W8**，也不等于对 offline packed 数据再次做 BF16→FP4 量化。准备后保留转置视图，不机械补 `.contiguous()`。

这里的 shape 说明不能替代真机物理 NZ/stride 验证；mock cast 只验证 Python 轴变换。公共 scale helper 还会拒绝奇数个 K/32 block，不能把前面的 Linear `% 32` 条件等同于所有 scale/layout 条件都满足。

## 8. 推理链：W13 可以融合，W2 只是 down projection

### 8.1 明确本 PR 的新增与已有能力

W4A8 grouped matmul、融合 GMM1/SwiGLU、Ascend runner 与 dispatcher 适配已经存在。本 PR 的主要工作是把在线准备好的权重交给它们，并明确两份 kernel 的职责：

```text
W13 kernel：use_fused_gmm1=True，构造 GroupedMatmulSwigluQuant
W2  kernel：use_fused_gmm1=False，不构造该融合组件
```

新增 `weight_prefix` 是 **keyword-only**，原有首个位置参数 `dynamic_quant_kwargs` 的含义不变。ModelSlim、compressed-tensors、DeepSeek FP4、FP8-config 的 FP4 experts 入口一起补上前缀，是为了保持共享 kernel 调用一致，不是给这些 checkpoint 换量化格式。

### 8.2 标准 SwiGLU 的执行示意

沿 [`AscendRunnerCore.run()`][runner]，典型融合路径为：

```text
tokens + top-k
  → dispatch / 按专家分组
  → [若没有输入 scale] 动态 MXFP8 量化
  → W13: FP8 × FP4 GMM + SwiGLU + MXFP8 requant
       输出 (FP8 intermediate, UE8M0 scale)
  → W2:  FP8 × FP4 GMM
       输出 FP16/BF16 hidden states
  → combine / 按 top-k 权重恢复 token 顺序
```

runner 构造和执行都按 **W13** 判断是否融合；W2 决定非融合链中的激活格式，但不能决定 GMM1 是否跳过独立 activation。当前上游 `_uses_fused_gmm1()` 对 `swiglu_limit > 0` 和 SiTU 禁用融合，保留这些激活语义；本 PR 的测试覆盖了这一点。

在线方法沿用父类 NPU forward，把 layer 作为包含权重/scale 属性的 payload 传给 runner。不要画成在线路径一定另建一个 `AscendQuantInfo` 对象；该类型描述的属性契约同样可由 layer 提供。

### 8.3 scale 是否存在决定要不要再次量化 activation

[`NPUW4A8MXFP4MoEMethod`][moe] 保留两类通信约定：

- normal dispatch：BF16；
- low-latency dispatch：MXFP8，携带 scale。

收到 scale 就复用已有量化 payload，并规范为 pair shape；没有 scale 才动态量化。no-A2A 路径则经 standard→ascend hooks 路由，本版本这条 W4A8 路径未设置额外 routing quant dtype，沿默认 BF16 路由，再在 GMM1 前量化。

这说明入口契约，不是对所有 DeepEP/Ascend Ops、跨节点、空 token 和 graph 组合的验收声明。特别不能把 normal/low-latency 都概括成“通信前已经 FP8”。

### 8.4 GMM1 / GMM2 的 group list 并不完全一样

普通 GMM 使用传入的 `group_list_type`：`1` 表示 COUNT，`0` 表示累计边界，并将 expert token 数转成 `int64`。

已有 [`GroupedMatmulSwigluQuant.forward()`][matmul] 则在输入为 COUNT 时做 `cumsum(0)`，因为 `npu_grouped_matmul_swiglu_quant_v2` 消费累计列表。例如：

```text
dispatcher COUNT： [2, 0, 3]
融合 GMM1 输入：   [2, 2, 5]
普通 GMM2 输入：   [2, 0, 3]，group_list_type=1
```

不能因为 dtype/shape 相同就把这两种语义互换。普通 W4A8 GMM 的量化参数也要成套看：FP8 `x_dtype`、FP4 `weight_dtype`、`antiquant_scale=[weight_scale]` 和 UE8M0 per-token scale，不能误套 INT8 的 scale 参数组织。

## 9. Dense 与 post-load：为何不是“只改 MoE”

### 9.1 Dense 复用同一个生命周期

[`NPUMXFP4W4A8LinearMethod`][linear] 也改为 meta 创建原生参数，使用 `NPUOnlineDenseWeightLoader` 包装原 loader；merged gate/up、QKV、Row/Column TP 仍由原生 loader 处理。

完成回调 `_quantize_weights()` 才量化权重并准备 NZ/scale。推理 `apply()` 将 `[...,K]` 展平为 `[M,K]`，动态量化成 MXFP8，调用 `npu_quant_matmul(..., x2_dtype=fp4_dtype, group_sizes=[0,0,32])`，最后恢复输出前导维度。

### 9.2 bias 修正同时覆盖在线和离线 Linear

新增 `_prepare_mxfp4_w4a8_bias()` 把 bias 规范成 **BF16 `[1,N]`**；`None` 原样透传。它替换在线路径之前的 FP32 bias 缓存，也替换离线 apply 中的 FP32 转换。

这里在线/离线共享的是一个小 helper，不应描述成“本 PR 已把两个 Linear 类完全合并成公共执行类”；那是另一项独立重构。

### 9.3 三行 post-load 修复为什么重要

权重现在在外层 post-load 之前就可能成为 NPU NZ 转置视图。若外层把 `torch.device("npu")` 和实际 `npu:0` 当成不同设备，就会误判为需要 staging/restore，额外触发不支持的 `copy_`。

[`stage_module_for_post_load()`][post-load] 新增：

```python
process_device = torch.empty(0, device=process_device).device
```

先将别名解析成实际设备，再比较 residency。这样已驻留权重不被无意义复制；真实跨设备 staging 仍沿用原机制。修的是**设备身份判断**，不是通过强转 ND、contiguous 或改量化参数掩盖 NZ copy 问题。

## 10. 测试应该怎么读，哪些结论不能扩大

### 10.1 四份改动测试分别守住什么

| 文件（相对 `test/registered/unit/`） | 主要约束 |
| --- | --- |
| [`layers/quantization/test_npu_mxfp4_w4a8.py`][test-contract]，新增 | config 选择与 ignored layers、本地 K、FP4 权重策略与 FP8 激活分离、bias/shape、在线/离线准备、W13/W2 前缀及融合决策、NPU op 惰性解析 |
| [`npu/quantization/test_online_mxfp4_loading.py`][test-loader]，新增 | 原生 TP/EP loader、merged/QKV/fused shard、独立完成、重复/缺失/失败、旧句柄源释放、重载、并发、非本地专家、EPLB/shared slot |
| `test_quantization_post_load.py`，扩充 | 等价设备不复制，保持 pointer/stride/offset/alias；hook 异常不被 restore copy 掩盖；CUDA staging 相关覆盖 |
| `eplb/test_expert_location_weight_loading.py`，调整 fixture | stand-in 绑定提取后的 helper，继续运行已有专家位置加载回归 |

`HiddenStatesDynamicQuant` 的改动也服务于这个边界：构造时只保存 op 名，真正调用时才解析 `torch.ops.npu`。这允许 CPU/mock 测试构造对象，**并不提供 CPU 版 NPU 量化算子**。

### 10.2 推荐现场展示的三个例子

1. **Dense 两个 shard**：只加载一半时 post-load 报错；第二个 shard 完成后只转换一次，旧 Parameter `numel()==0`，之后可以经 registered loader 重载。
2. **MoE 非本地专家**：输入不属于当前 rank 的专家时，W13 仍保持 meta；本地 W13 完成不要求 W2 已完成。
3. **W13/W2 故意设置相反融合标志**：runner 仍必须由 W13 决定是否运行独立 activation，防止构造与执行选择两条不同链。

这些例子分别对应生命周期、分布式映射和执行职责，比只展示“测试全绿”更能解释设计。

### 10.3 验证结论的边界

本次工作是源码走读，没有重新运行单测或 NPU 模型。现成测试命令见[本地 GPU 测试](agents/local-gpu-tests.md)，已有用户真机反馈和版本对应关系见[目标计划](mxfp-refactoring-plan.md#验收证据与边界)。

- 原生 loader 在 CPU 上运行，能检验切片、计数和状态机；mock 量化不能证明 FP4 数值或物理 NZ。
- CUDA staging 可以真实检验搬运/alias 行为，但不能替代 Ascend runtime、NPU graph 或 NPU allocator 验证。
- 加载峰值下降与最终驻留占用下降不是同一个结论；`load_model` allocator 窗口也不包含后续 KV、graph 和 serving warmup。
- 用户曾报告的模型分数、权重误差和加载峰值，应保留各自测试版本和场景边界，不自动继承到本文 head。

## 11. 收尾：评审时抓住这五个不变量

1. **先完整加载，再量化。** 完整指本地逻辑张量，不能用全局 checkpoint 大小或调用次数代替。
2. **量化之后不保留无用浮点 Parameter backing。** 但也不夸大为逐 block 流式或全部外部别名自动释放。
3. **权重 FP4 策略与激活 FP8 策略分开。** 离线 packed 输入不经过动态权重量化。
4. **W13 决定 GMM1 融合，W2 接收激活及其 scale。** group list、dtype、scale axes 必须成套匹配。
5. **生命周期检查不等于硬件验收。** meta、copy 计数、mock layout 和 CI 成功都不能单独证明模型精度或 serving 性能。

可以用这句话结束走读：

> 本 PR 的价值不在“另写一个 FP4 算子”，而在把在线 checkpoint 的加载生命周期，正确接到已有 W4A8 执行契约上：既让 MoE 真正用上 FP4 权重，也避免为了加载低比特模型而先同时保留完整浮点模型。

---

### 设计来源

- [加载生命周期评审](https://github.com/sgl-project/sglang/pull/43739#discussion_r4360300286)：参考 #29048 的完成度加载，**只复用生命周期思路，不引入 INT4 量化/scale 编码**。
- [权重策略评审](https://github.com/sgl-project/sglang/pull/43739#discussion_r4360510552)：明确 FP4 权重参数，并要求与 MXFP8 activation 分开。
- [格式与入口](quantization-overview.md)、[已知陷阱](known-pitfalls.md)：布局、TP、设备搬运与部署边界。

<!-- 固定源码链接属于本次走读的阅读版本，不作活跃分支状态表。 -->
[config]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/npu_mxfp4.py#L76-L131
[local-k]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/npu_mxfp4_utils.py#L11-L18
[online-moe]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/online_moe_methods.py#L76-L140
[loader]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/online_weight_loader.py#L14-L198
[expert-ids]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1270-L1323
[counter]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/online_quantization.py#L6-L23
[moe]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py#L224-L373
[moe-utils]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/moe_utils.py#L14-L86
[runner]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/moe_runner/ascend.py#L136-L308
[matmul]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/moe/matmul.py#L53-L88
[linear]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py#L357-L482
[post-load]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/model_loader/post_load.py#L109-L135
[test-contract]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/test/registered/unit/layers/quantization/test_npu_mxfp4_w4a8.py
[test-loader]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/test/registered/unit/npu/quantization/test_online_mxfp4_loading.py

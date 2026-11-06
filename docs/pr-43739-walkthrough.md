# PR #43739 功能走读：从启动参数到在线 W4A8 加载与推理

本文沿一条实际执行链阅读代码：**用户指定在线 W4A8 → 模型构造出对应的量化层 → checkpoint 写入这些层 → 权重加载完整后立即量化 → 请求使用准备好的权重推理。** 每一步都说明入口在哪里、里面做了什么、为什么这样做，以及如何进入下一步。

阅读场景：原始 BF16/FP16 checkpoint、Ascend NPU、`mxfp_w4a8`，以普通 `DefaultModelLoader` 加载和标准 no-A2A MoE 推理为主线。模型接线以 Qwen3.5 MoE 为例；Qwen3 也复用这些底层量化方法，不另写模型专属算子。下文短代码为语义摘录，`...` 表示省略的参数或分支，不是独立运行脚本。

- [PR #43739](https://github.com/sgl-project/sglang/pull/43739)，2026-11-06 核对的阅读版本：`29d7388454355c2d25fb54261e38b1adc009b929`。
- 源码链接固定到该版本；本地位于 `sglang/qwen3.5_moe_w4a8/`。正文路径相对其中的 `python/sglang/srt/`，测试路径除外。
- 这条链路包含已有上游代码；本 PR 的重点是在线 MoE 接入、加载完成即量化，以及相关权重策略/生命周期修正，不是重写整个 loader 或 runner。

## 功能一：`--quantization mxfp_w4a8` 如何让模型使用在线 W4A8

### 1. 从启动参数进入量化配置

用户提供的关键参数是：

```text
--quantization mxfp_w4a8
```

这时 checkpoint 仍然是浮点权重。“在线”表示 SGLang 在初始化时转换权重，不是要求用户先导出 FP4 checkpoint，也不需要再加一个 `--online-quantization`。

先沿已有加载框架往下看：

```text
ServerArgs.quantization
  → ModelConfig.from_server_args()
  → ModelConfig.quantization = "mxfp_w4a8"
  → ModelRunner.load_model()
  → load_model_with_memory_saver()
  → DefaultModelLoader.load_model()
       ├─ _get_quantization_config(...)
       ├─ _initialize_model(..., quant_config)
       └─ load_weights_and_postprocess(...)
```

[`ModelConfig.from_server_args()`][model-config] 把用户选择保存在模型配置里。[`DefaultModelLoader.load_model()`][default-loader] 随后取得量化配置，再把它交给模型构造函数。

配置的具体解析发生在 [`get_quant_config()`][get-quant-config]：它用 `model_config.quantization` 查询 registry。已有注册关系是：

```python
"mxfp_w4a8": Mxfp4W4A8Config
```

对普通未量化 checkpoint，没有必须读取的专用量化配置文件，就使用该配置类的默认实例。到这里还没有量化任何 tensor；得到的是一个**决定各层采用什么量化方法的配置对象**。

### 2. 配置怎样传到真正的专家层

[`_initialize_model()`][initialize] 将 `quant_config` 传给模型。模型再把它往 decoder layer、MLP/experts 传递。

Qwen3.5 MoE 的一条具体构造链是：

```text
Qwen3_5MoeForCausalLM
  → Qwen3_5LinearDecoderLayer / Qwen3_5AttentionDecoderLayer
  → Qwen2MoeSparseMoeBlock(..., quant_config=quant_config)
  → experts = get_moe_impl_class(quant_config)(..., quant_config=quant_config)
  → FusedMoE
```

可以从 [`qwen3_5.py` 中创建 MLP 的位置][qwen35-construct] 接到 [`qwen2_moe.py` 中创建 experts 的位置][qwen-experts]。

真正选择量化 method 的地方不是模型名判断，而是 [`FusedMoE.__init__()`][fused-init] 中的：

```python
self.quant_method = quant_config.get_quant_method(self, prefix)
```

于是调用进入本 PR 修改的 [`Mxfp4W4A8Config.get_quant_method()`][config]。省略无关分支后，逻辑是：

```python
if isinstance(layer, FusedMoE):
    if is_layer_skipped(prefix, ...):
        return UnquantizedFusedMoEMethod(...)
    if is_npu():
        return NPUW4A8MXFP4OnlineMoEMethod(self)
    raise NotImplementedError(...)
```

**这是在线 MoE 功能真正接入的地方。** 原先这里对 MoE 返回未量化 method；现在返回在线 W4A8 method。后续构造、加载和推理框架不必认识一个新的 Qwen3.5 专用量化实现，只需要继续调用 layer 的 `quant_method`。

同一配置也会被传给普通 Linear。`LinearBase` 同样调用 `get_quant_method()`，得到 `NPUMXFP4W4A8LinearMethod`。但它先处理两个例外：

- ignored layer 保留浮点；
- 本地 K 不满足 `% 32 == 0` 时保留浮点，例如 K=4304 的视觉 fc2。

[`local_input_size()`][local-k] 必须在此使用真实的本地 K：RowParallel 根据 **layer 绑定的 TP group** 除 TP，ColumnParallel/ReplicatedLinear 不除。原因是选择 method 时 `input_size_per_partition` 尚未建立，不能等到分配权重后才决定回退。

另外，“传了全局量化配置”不代表所有层都会量化。例如这里的 `Qwen2MoeSparseMoeBlock` 创建 router gate 时显式使用 `quant_config=None`；gate 保持浮点，experts 使用在线 W4A8。

### 3. 选中 method 后，先构造什么

回到 `FusedMoE.__init__()`。得到 method 后，框架按顺序调用：

```python
self.quant_method.create_weights(..., weight_loader=self.weight_loader, ...)
self.quant_method.create_moe_runner(self, self.moe_runner_config)
```

若模型选用 fused checkpoint loader，第一个调用传入的是 `weight_loader_fused`；在线 wrapper 同样包住实际传进来的原生 loader。

现在进入 [`NPUW4A8MXFP4OnlineMoEMethod.create_weights()`][online-moe]。它继承 `UnquantizedFusedMoEMethod`，因为 checkpoint 仍是 BF16/FP16，需要沿用父类的浮点 shape 和 loader 属性，但**不希望父类立刻分配真实浮点存储**。

因此实现是：

```python
loader = NPUOnlineMoEWeightLoader(layer, params_dtype, original_loader)

with torch.device("meta"):
    super().create_weights(
        ...,
        weight_loader=loader.weight_loader,
    )

loader.register_sources()
layer._npu_online_moe_loader = loader
```

对 gated MoE，令 `E` 为本 rank 的专家存储槽数，`H` 为 hidden size，`I` 为 TP 后 intermediate size，此时 layer 上得到：

```text
w13_weight：meta Parameter，shape=[E, 2I, H]，dtype=BF16/FP16
w2_weight ：meta Parameter，shape=[E, H, I]，dtype=BF16/FP16

每个 Parameter 的 weight_loader → 在线 wrapper
在线 wrapper 内保存的 original_loader → 原生 TP/EP loader
```

meta 只记录形状、类型和属性，没有这些大权重对应的真实存储。`register_sources()` 保存原始 shape 和目标 `numel`，以后据此判断是否加载完整。若存在 bias，它单独实体化并保留原 loader，不计入 FP4 权重的完成度。

接着执行 `create_moe_runner()`，在 layer 上安装两份 kernel：

```python
w13_kernel = NPUW4A8MXFP4MoEMethod(weight_prefix="w13")
w2_kernel  = NPUW4A8MXFP4MoEMethod(weight_prefix="w2")
```

已有 [`create_ascend_moe_runner()`][moe-utils] 先绑定 kernel，再创建 Ascend runner。在线方法只接受 `auto` / `ascend` backend，避免其他 runner 的 gate/up 排列规则混进来。

**这一阶段结束时：模型结构、meta 权重、加载回调和推理 kernel 都已经接好，但权重内容还没有读入，也没有做 FP4 量化。下一步才开始加载 checkpoint。**

## 功能二：checkpoint 如何加载、何时量化，以及为什么降低显存峰值

### 1. 入口是模型的 `load_weights()`，不是 post-load

继续沿 [`DefaultModelLoader`][default-loader] 往下走：

```text
load_weights_and_postprocess(model, weights, target_device)
  ├─ load_weights_only(...)
  │    └─ model.load_weights(weights)
  └─ postprocess_weights(...)
```

其中 `weights` 来自 `_get_all_weights(...)`，对模型暴露的是 `(name, loaded_weight)` 迭代流。模型负责把 checkpoint 命名转换为运行时参数名，而不是把整个文件原样交给量化 kernel。

以 [`Qwen3_5MoeForCausalLM.load_weights()`][qwen-load] 的逐专家分支为例：

```text
checkpoint 名称：model.layers.0.mlp.experts.2.gate_proj.weight
                          ↓ 名称映射
运行时目标：    model.layers.0.mlp.experts.w13_weight
附加定位信息：  expert_id=2，shard_id="w1"
```

`gate_proj`、`up_proj`、`down_proj` 分别映射到 `w1`、`w3`、`w2`；gate/up 共用 W13 目标参数。模型随后调用：

```python
param = params_dict[name_mapped]
weight_loader = param.weight_loader
weight_loader(param, loaded_weight, name_mapped,
              shard_id=shard_id, expert_id=expert_id)
```

**上一步已经把 `param.weight_loader` 换成了在线 wrapper，所以不改模型的这段加载循环，也能接管量化时机。** Qwen3.5 的 fused checkpoint 分支会先拆分/遍历对应专家，再进入相同的参数加载接口；不能把“一个文件读完”当作量化触发条件。

### 2. 第一块权重到来：先确定目标，再分配浮点 source

调用进入 [`NPUOnlineMoEWeightLoader.weight_loader()`][online-loader] 后，先确定它属于哪个逻辑目标：

```text
w1 / w3 / w13 → W13 tracker
w2            → W2 tracker
```

对逐专家输入，wrapper 先调用 [`layer._weight_loader_expert_ids()`][expert-ids] 查询目标存储槽。如果该专家不属于本 rank，直接返回：不会因为路过了一份非本地权重，就分配 source 或启动新一轮 reload。

若专家有多个本地 EPLB 副本，helper 返回多个目标；wrapper 按每个目标槽登记 shard key。原有 shared slot、global-expert 参数等映射规则仍由这份 helper 统一处理。

确定需要写入后，进入 `_NPUOnlineWeightLoader._load()`。第一次看到目标还是 meta，就调用 `_materialize_source()`：

```python
data = torch.empty(source_shape, dtype=params_dtype, device=load_device)
source = Parameter(data, requires_grad=False)
# 保留 loader 等属性，再把真实 source 挂回 layer
setattr(layer, weight_name, source)
```

`load_device` 在进入 meta 上下文前已经记录。因此这里创建的是原本应加载到的真实设备上的浮点 source，而不是再次创建 meta tensor。

此时只分配当前这份本地 W13 或 W2，其他尚未开始加载的逻辑权重仍可保持 meta。

### 3. 怎么写入：保留原生 TP/EP loader，不自己重做切片

source 建好后，wrapper 调用之前保存的原生 loader：

```text
在线 wrapper
  → FusedMoE.weight_loader(...)
  → _weight_loader_expert_ids(...)
  → _weight_loader_impl(...)
  → 原生 W13 / W2 的分片和 copy_
```

这层原生 loader 继续负责 TP 切片、专家落槽和 gate/up 放置。在线 wrapper 不需要复制一套模型加载规则，它只需知道这次**实际写入了多少本地元素**。

因此它用已有 [`CopyNumelCounter`][counter] 包住原生调用：

```python
counter = CopyNumelCounter()
with counter:
    self.load_weight(source, loaded_weight, *args, **kwargs)

copied = self.loaded_numel + counter.copied_numel
```

`CopyNumelCounter` 用 `TorchDispatchMode` 统计实际 `aten.copy_` 的目标元素数。原生 loader 已经完成 narrow/TP 切片，所以不会误用 checkpoint 全局 tensor 的大小来判断本地完成度。

与此同时还检查 shard key 是否重复。只计数而不去重，会把“同一块写了两遍”误当成“另一块也已到齐”。copy 和转换共用每份逻辑权重自己的锁，避免并发 worker 重复触发转换。

### 4. 哪一次调用触发量化：完整本地逻辑权重到齐的那一次

触发条件在 `_load()` 内：

```python
if copied == self.target_numel:
    self.on_complete()
```

这里的 `target_numel` 来自最初 meta 参数的完整本地 shape，**不是某一个输入 shard 的大小，也不是单个专家的大小**。

为什么选择这个粒度？原生 loader 仍要把后续 shard 写进 BF16/FP16 目标，而现有转换函数会把整个 layer 参数替换成低比特布局。若第一块刚到就替换，后面的浮点 shard 就没有原来的目标可写了。等待完整逻辑张量，才能不改原生分片加载规则，并直接复用整块 weight 的量化/布局准备函数；逐 shard 量化再拼接 packed weight/scale 是另一种实现，不是这次采用的方案。

用一个具体 W13 走完加载过程。假设本 rank 保存专家 2、3，`H=128`、TP 后 `I=64`：

```text
本地 W13 shape = [2, 128, 128]
目标元素数      = 2 × 128 × 128 = 32768
每个专家的 gate 或 up 本地切片 = [64,128] = 8192 个元素
```

假设按照下面的顺序到达：

| 本次输入 | 累计实际 copy 元素 | 结果 |
| --- | ---: | --- |
| 专家 2 的 gate / `w1` | 8192 | 第一次分配 W13 浮点 source，继续等待 |
| 专家 3 的 gate / `w1` | 16384 | 继续等待 |
| 专家 2 的 up / `w3` | 24576 | 专家 2 完整了，但整个本地 W13 未完整，不量化 |
| 专家 3 的 up / `w3` | 32768 | 立即量化整个本地 W13 |

所以，**既不是每收到一个 checkpoint shard 就量化，也不是每个专家到齐就量化，而是完整本地 W13 到齐时量化。** 如果一次加载本来就覆盖了完整目标，那一次调用就会触发转换。

W2 有自己的 shape、计数和回调。W13 完成不用等 W2，W2 也可以先完成；更不用等其他层或整个模型。

### 5. `on_complete()` 实际做什么：浮点权重变成 kernel 能直接使用的表示

MoE tracker 注册的回调相当于：

```python
layer.w13_kernel.process_weights_after_loading(layer, "w13")
# W2 对应 layer.w2_kernel.process_weights_after_loading(layer, "w2")
```

这里虽然函数名含 `after_loading`，但“loading”指这份逻辑权重已经完整；**它正在 checkpoint 加载循环内部被调用，不是等全模型 post-load。**

进入 [`NPUW4A8MXFP4MoEMethod.process_weights_after_loading()`][moe-kernel] 后：

1. 从 layer 取出当前完整浮点 source；若 source 在 CPU，先搬到当前 NPU。
2. 对 BF16/FP16 输入调用动态 MXFP4 权重量化。
3. 注册 packed weight 和 block scale。
4. 通过公共准备函数整理成 GMM 需要的 weight/scale layout。

核心权重量化调用为：

```python
packed, block_scale = torch.ops.npu.npu_dynamic_mx_quant(
    source,
    dst_type=fp4_dtype,
    round_mode="rint",
    scale_alg=2,
    dst_type_max=7.25,
)
```

这些显式参数是本 PR 的 FP4 **权重**策略：让 Linear/MoE 使用一致、可审查的权重转换设置，不再依赖原先的算子默认策略。推理时 FP8 activation 有自己的量化调用，不沿用 `scale_alg=2 / dst_type_max=7.25`。

若一般专家矩阵的浮点 shape 是 `[E,N,K]`，准备过程可以读成：

```text
BF16/FP16 [E,N,K]
  → packed FP4 [E,N,K/2] + UE8M0 block scale
  → 在线四维 scale 必要时 flatten(-2)，对齐 [E,N,K/32]
  → prepare_w4a8_mxfp_weight(...)
       weight：FP4-aware NZ cast，再 transpose(-1,-2)
       scale ：view(E,N,K/64,2)，再 transpose(1,2)
              得到 [E,K/64,N,2]
```

[`prepare_w4a8_mxfp_weight()`][moe-utils] 是已有公共准备函数。在线路径在这里与离线 packed 权重汇合，因此能继续复用同一套 W4A8 GMM。离线输入不是 FP16/BF16，不会走前面的动态权重量化分支。

这一步结束时，layer 上已经是推理使用的 FP4 语义权重和 scale。NZ cast 的 `customize_dtype`/`input_dtype` 是后端表示契约，不代表把 W4 改成 W8，也不是再次量化离线 packed 权重。

### 6. 转换返回 loader：释放浮点 source，这才构成完整的省显存链路

回调返回后，`_load()` 取出 layer 上的新 prepared Parameter，保留 loader 等属性，再清空旧 source 的 backing：

```python
prepared = getattr(self.layer, self.weight_name)
copy_missing_attrs(source, prepared)

if prepared is not source:
    source.data = torch.empty(0, device=source.device, dtype=source.dtype)

self.state = "converted"
```

为什么不只把 `layer.w13_weight` 换成新参数？因为模型加载循环可能缓存了 `params_dict`，外部仍握着旧 Parameter。只替换 layer 属性，不足以让这类引用释放原浮点存储。

wrapper 后续也不依赖调用者传入的旧 `param`，而是每次从 layer 查当前 source，因此旧句柄不会让下一次加载写错位置。

现在就能完整解释显存变化：

```text
修改前：
  模型构造时就分配各层浮点权重
  → 加载完整浮点模型
  → 最后逐份转换成 FP4

修改后：
  模型构造时大权重为 meta
  → 某份逻辑权重首次加载才分配浮点 source
  → 这份权重完整后立刻转成 FP4，并释放浮点 source
  → 继续处理后续权重
```

**省的是“完整浮点模型同时驻留”的加载峰值，不是把已经量化的最终模型再压缩一次。** 即使仍要暂存完整本地 W13，也不再需要预先为全模型 W13/W2 分配浮点存储。

按层加载时，后面的权重仍未分配，前面的权重已变成低比特常驻态；只有正在加载的权重需要浮点 source。若多份权重交错加载，可能同时存在多个未完成 source，所以不是严格的一份 buffer 上界，也不是逐 block 流式量化。

这里处理的是旧 Parameter 引用；外部另存的共享 storage view 不会被这段代码自动全部回收。

### 7. 普通 Linear 在加载链中如何处理

对应入口是 [`NPUMXFP4W4A8LinearMethod.create_weights()`][linear-kernel]。它同样在 meta 上创建原生参数，但使用 `NPUOnlineDenseWeightLoader`，完成回调改为：

```python
lambda: self._quantize_weights(layer)
```

其余思路不变：模型 `load_weights()` → 参数上的 wrapper → 原生 Linear loader → 累计本地 copy → 完整时量化。

不同点只是“完整目标”是什么：

- 普通 Row/Column Linear：完整本地 `weight`；一次原生加载可能就已填满。
- merged gate/up：两部分都填入合并目标后才完整。
- QKV：Q/K/V 所需本地部分全部写完才完整。

`_quantize_weights()` 使用相同 FP4 权重策略，准备 Linear 的 NZ 权重及 `[K/64,N,2]` scale。它不需要 MoE 的专家映射，也不需要等待任何其他层。

### 8. 模型加载结束：post-load 不再重复量化，而是验收完成状态

所有 `(name, tensor)` 都处理完，才回到 [`DefaultModelLoader.postprocess_weights()`][default-loader]：

```python
for module, quant_method in _modules_with_quant_method(model):
    with device_loading_context(module, target_device):
        quant_method.process_weights_after_loading(module)
```

此时在线 method 的实现只调用 `finish_post_load()`，检查：

- 是否已通过 wrapper 完整加载并转换；
- 当前参数是否仍为 meta，或 shape/dtype 与 prepared layout 不一致。

没有加载、只加载一部分、copy/转换失败，都会被拒绝。成功后进入 `ready_reload`；下一轮经 registered loader 写入时，再清零计数、创建浮点 source 并重复上述过程。未经过 post-load 就继续写入已转换权重，则作为重复加载拒绝。

这里要分清两层同名函数：

```text
在线 method.process_weights_after_loading(layer)
  → 全模型加载后的完成状态检查

W4A8 kernel.process_weights_after_loading(layer, prefix)
  → 单份逻辑权重完成时，由 callback 提前执行的量化与布局准备
```

由于权重此时已经可能是 NPU NZ 转置视图，本 PR 还修正了外层 [`stage_module_for_post_load()`][post-load]：先用 `torch.empty(0, device=process_device).device` 将 `npu` 等别名解析成实际设备，避免把已经在 `npu:0` 的权重误判为需要搬运，额外触发 restore `copy_`。真正的跨设备 staging 仍保留。

**到这里，初始化完成：推理只需消费 prepared weights，不再重新量化 checkpoint。**

## 功能三：请求进入 MoE 后，如何实际执行 W4A8

### 1. 从模型 forward 找到 experts 入口

输入到 Qwen3.5 MoE block 时，先经过 router 和 top-k。沿 [`Qwen2MoeSparseMoeBlock._forward_router_experts()`][qwen-forward]，核心调用为：

```python
router_logits, _ = self.gate(hidden_states)
topk_output = self.topk(hidden_states, router_logits)
return self.experts(hidden_states, topk_output)
```

shared expert 若未融合，仍由模型的原有旁路计算并相加；若已融合，则沿原有额外专家槽规则处理。下面只跟进 routed experts 主链。

[`FusedMoE.forward_impl()`][fused-forward] 将工作拆成三个部分：

```text
dispatcher.dispatch(hidden_states, topk_output)
  → run_moe_core(dispatch_output)
       → quant_method.apply(layer=self, dispatch_output=...)
  → dispatcher.combine(combine_input)
```

在线 MoE method 没有重写一份前向计算，而是继承父类 `apply()`/NPU forward。NPU 分支最终调用：

```python
self.runner.run(dispatch_output, layer)
```

这里传入的 `layer` 已含 `w13_weight/w2_weight` 及各自 scale，是 runner 消费的量化 payload；不要求在线路径另外构造一个 `AscendQuantInfo` 实例。

### 2. token 如何变成按专家分组的 GMM 输入

进入已有 [`MoeRunner.run()`][runner-entry]，调用顺序是：

```text
pre_permute(dispatch_output, layer, config, running_state)
  → AscendRunnerCore.run(runner_input, layer, running_state)
  → post_permute(runner_output, layer, config, running_state)
```

在本文主线的 no-A2A 场景，`StandardDispatcher` 主要传递 hidden states 和 top-k 结果。**NPU 的 token 展开/排序实际发生在 [`pre_permute_standard_to_ascend()`][pre-permute]，不是在一个另外实现的 Ascend dispatcher 里。**

这个 hook 调用 `npu_moe_init_routing(...)`，把 token 的各条专家路由整理为按专家分组的行，并得到：

```text
hidden_states        ：按专家分组后的输入行
hidden_states_scale  ：若路由阶段已量化则有值，否则为 None
expert_tokens        ：各专家的 token 数
expanded_row_idx 等  ：还原 token 顺序时需要的映射
```

映射和 top-k 权重存入 `running_state["ascend_routing"]`，供最后还原使用。当前这条 W4A8 no-A2A 路径使用默认的不量化路由，保留浮点输入；接下来才在 GMM1 前量化为 FP8。

### 3. W13：先量化输入，再融合执行 gate/up、SwiGLU 和 requant

执行进入 [`AscendRunnerCore.run()`][ascend-core]。标准 SwiGLU 场景下，runner 先依据 **W13 kernel** 判断能否使用融合 GMM1：

```python
if _uses_fused_gmm1(layer.w13_kernel, config):
    hidden_states, pertoken_scale = (
        layer.w13_kernel.apply_fused_gmm1_swiglu(...)
    )
else:
    hidden_states = layer.w13_kernel.apply(...)
    hidden_states, pertoken_scale = activation(...)
```

这就是构造阶段为什么要给 W13/W2 分别绑定前缀：W13 有融合 GMM1 职责，W2 没有；W2 不应构造融合组件，也不能决定要不要跳过独立 activation。`weight_prefix` 是 keyword-only，不改变原有 `dynamic_quant_kwargs` 位置参数含义。

顺着 [`apply_fused_gmm1_swiglu()`][moe-kernel] 往下看：

1. 如果输入没有 scale，调用 `HiddenStatesDynamicQuant`，把浮点 activation 转成 MXFP8 E4M3，生成 UE8M0 scale。
2. 如果输入已经携带 scale，则复用量化后的 activation，并规范 scale shape，不重新量化一次。
3. 把输入、加载期准备好的 W13 和 weight scale 交给 `GroupedMatmulSwigluQuant.forward()`。

底层调用已有 `npu_grouped_matmul_swiglu_quant_v2`，一次完成：

```text
FP8 input × FP4 W13
  → gate/up 两路结果
  → SwiGLU
  → 再量化为 MXFP8
  → 返回 (FP8 intermediate, UE8M0 scale)
```

因此融合路径后面不能再跑一次独立 SwiGLU。这里的 activation 是 A8，不是 FP4；W4A8 的“4”只描述权重。

### 4. 为什么 GMM1 还要转换 expert token 列表

[`GroupedMatmulSwigluQuant.forward()`][gmm-wrapper] 接收 `group_list_type`。标准路由给的是每个专家的 COUNT，但融合算子需要累计边界，所以 wrapper 会转换：

```text
COUNT：    [2, 0, 3]
累计边界：[2, 2, 5]
```

对应代码是：

```python
group_list = expert_tokens.cumsum(0) if group_list_type == 1 else expert_tokens
```

这个转换只服务于融合 GMM1；它没有把 runner 手里的原始 COUNT 改掉。下一步普通 GMM2 仍可以使用 COUNT，并明确传 `group_list_type=1`。

### 5. W2：直接使用 GMM1 输出的 FP8 与 scale

回到 runner，W13/激活完成后统一进入：

```python
hidden_states = layer.w2_kernel.apply(
    ...,
    hidden_states=hidden_states,
    pertoken_scale=pertoken_scale,
    weight_prefix="w2",
    output_dtype=original_dtype,
    group_list_type=group_list_type,
)
```

[`NPUW4A8MXFP4MoEMethod.apply()`][moe-kernel] 调用公共 `w4a8_mxfp_gmm()`。它遵守同一个规则：有 input scale 就使用现成的量化输入；没有才动态量化。

在融合主链中，W13 已经输出 FP8 和 scale，因此 W2 不重复量化，直接用：

```text
FP8 intermediate
+ FP4 W2
+ W2 的 UE8M0 weight scale
+ intermediate 的 UE8M0 per-token scale
  → npu_grouped_matmul
  → 浮点 hidden states
```

普通 GMM 参数中，weight scale 走 `antiquant_scale=[weight_scale]`，激活 scale 走 `per_token_scale`，并显式指定 FP8/FP4 dtype。最终按 runner 的 `original_dtype` 返回：浮点输入为 FP16 时输出 FP16，否则为 BF16。

### 6. 专家输出如何还原成 token 输出

GMM2 输出仍按专家分组。`MoeRunner.run()` 随后调用 [`post_permute_ascend_to_standard()`][post-permute]，取出前面保存的路由信息：

```python
state = running_state.pop("ascend_routing")
final_hidden_states = npu_moe_finalize_routing(
    runner_output.hidden_states,
    state["expanded_row_idx"],
    topk_weights=state["topk_weights"],
    topk_ids=state["topk_ids"],
)
```

这一步完成 token 顺序还原和 top-k 加权求和，返回 `StandardCombineInput`。再回到 `FusedMoE.forward_impl()`，标准 dispatcher 的 combine 取出最终 tensor；所需的 TP/EP 归约继续由现有层/模型规则处理。

因此本文标准路径真正的计算顺序是：

```text
模型 router / top-k
  → StandardDispatcher 传递输入
  → pre-permute：按专家分组
  → W13：输入量化 + GMM/SwiGLU/requant
  → W2：GMM
  → post-permute：还原顺序 + top-k 加权求和
  → StandardDispatcher combine
  → 返回模型后续计算
```

### 7. 两个分支如何接回同一条主线

**非融合激活。** 当前 `_uses_fused_gmm1()` 遇到正的 `swiglu_limit` 或 SiTU 时禁用融合，因为普通 fused SwiGLU 不能代替这些激活语义。流程变为 `W13.apply()` → 独立 activation → `W2.apply()`；W2 仍按是否有 scale 决定是否动态量化。构造 runner 和实际执行必须使用相同的 W13 判断。

**带通信的 dispatch。** 传输 dispatcher 在已有 pre/post hooks 中适配为相同的 runner 输入。W4A8 kernel 配置 normal dispatch 输出 BF16，low-latency 输出 MXFP8+scale；因此 normal 输入通常在 GMM1 前量化，low-latency 可直接复用已有 scale。本文不把不同通信路径画成一条固定的“通信前已量化”流程。

### 8. 普通 Linear 的推理入口更短

普通 Linear 的 forward 调用 [`NPUMXFP4W4A8LinearMethod.apply(layer, x, bias)`][linear-kernel]：

```text
x[...,K]
  → reshape 为 [M,K]
  → npu_dynamic_mx_quant(dst_type=float8_e4m3fn)
  → npu_quant_matmul(
        FP8 x, prepared FP4 weight, weight_scale,
        pertoken_scale=activation_scale,
        x2_dtype=fp4_dtype, group_sizes=[0,0,32], ...)
  → reshape 回 [...,N]
```

这里不再触碰 checkpoint 或加载完成度。bias 由 `_prepare_mxfp4_w4a8_bias()` 规范为 BF16 `[1,N]`；本 PR 同时修正在线/离线 W4A8 Linear 的这个调用参数，但并没有把两个 Linear 类合并为一个类。

## 收尾：把三项功能连起来理解

这个实现的职责可以归纳成三个问题：

| 问题 | 对应入口 | 实现思路 |
| --- | --- | --- |
| 哪些层使用在线 W4A8？ | `Mxfp4W4A8Config.get_quant_method()` | 模型传递配置，层按类型取得 method，保留 ignored/对齐回退 |
| 什么时候、对哪份权重量化？ | 参数上的 completion-tracked `weight_loader` | 复用原生 TP/EP 写入，完整本地 Linear/W13/W2 到齐即转换并释放浮点源 |
| 请求到来后怎么算？ | Linear `apply()` / MoE `quant_method.apply()` | 消费准备好的 FP4 权重，动态量化 FP8 激活，复用已有 Ascend matmul/runner |

**所以这不是“加载完一个 checkpoint 文件就量化”，而是利用已有的层级加载入口，提前结束每份完整本地浮点权重的生命周期，再把它交给现成的 W4A8 执行链。**

### 测试与验证边界

实现对应两份核心新增测试：

- [`test_npu_mxfp4_w4a8.py`][test-contract]：配置选型、权重/激活策略、布局、W13/W2 runner 绑定等。
- [`test_online_mxfp4_loading.py`][test-loader]：原生 TP/EP 加载、完成时机、重复/缺失输入、source 释放、重载和并发等。

已有 post-load 和 EPLB 测试也随对应接口调整。CPU/mock 可以验证调用、轴变换和状态机，不能替代物理 NZ、NPU 算子数值和模型验收；`% 32` 的提前回退条件也不代表所有 scale pair/layout 条件已满足。带 bias MoE 的分配测试不等于其数值支持证明；registered reload 也不是推理并发热更新协议。

本次只重写走读，没有重新执行模型或单测。加载峰值、模型精度等历史证据与版本边界见[目标计划](mxfp-refactoring-plan.md#验收证据与边界)，不在本文复制状态表。正式 PR 不含 #36426 组合补丁、在线 W4A4 功能或已归档的 serving 性能实验。

<!-- 以下链接固定到本次走读版本，不维护活跃分支状态。 -->
[model-config]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/configs/model_config.py#L973-L1015
[default-loader]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/model_loader/loader.py#L1006-L1100
[get-quant-config]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/model_loader/weight_utils.py#L269-L375
[initialize]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/model_loader/loader.py#L295-L322
[qwen35-construct]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/models/qwen3_5.py#L1143-L1201
[qwen-experts]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/models/qwen2_moe.py#L318-L370
[fused-init]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L531-L625
[config]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/npu_mxfp4.py#L76-L131
[local-k]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/npu_mxfp4_utils.py#L11-L18
[online-moe]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/online_moe_methods.py#L76-L140
[moe-utils]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/moe_utils.py#L14-L86
[qwen-load]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/models/qwen3_5.py#L2123-L2382
[online-loader]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/online_weight_loader.py#L14-L198
[expert-ids]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1236-L1323
[counter]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/quantization/online_quantization.py#L6-L23
[moe-kernel]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py#L224-L373
[linear-kernel]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/quantization/linear_method_npu.py#L357-L482
[post-load]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/model_loader/post_load.py#L109-L135
[qwen-forward]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/models/qwen2_moe.py#L693-L722
[fused-forward]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/fused_moe_triton/layer.py#L1767-L1852
[runner-entry]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/moe_runner/runner.py#L195-L269
[pre-permute]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/moe_runner/ascend.py#L348-L399
[ascend-core]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/moe_runner/ascend.py#L138-L308
[gmm-wrapper]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/hardware_backend/npu/moe/matmul.py#L53-L88
[post-permute]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/python/sglang/srt/layers/moe/moe_runner/ascend.py#L596-L623
[test-contract]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/test/registered/unit/layers/quantization/test_npu_mxfp4_w4a8.py
[test-loader]: https://github.com/sgl-project/sglang/blob/29d7388454355c2d25fb54261e38b1adc009b929/test/registered/unit/npu/quantization/test_online_mxfp4_loading.py

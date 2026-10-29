# Ascend 量化与部署陷阱

只保留仍有用的机制、识别条件和处理边界；历史日志/地址/逐次测试记录不在此累积。能力与路径见 [量化概览](quantization-overview.md)，验收结论见 [目标计划](mxfp-refactoring-plan.md)。

## 环境与部署

### Qwen-VL 图片预热与 Transformers 接口不匹配

同步 SGLang 后也要核对其依赖约束。曾有部署仍为 Transformers 5.12.1，而 checkout 要求 5.19.0，resize 的 `image`/`images` 契约不同导致 warmup 失败。先对齐实际安装版本，不复制旧补丁或仅改一个关键字。#43309 已修复的逻辑不要重复提交。

### 模型注册失败可能只是导入失败

缺失 `sgl_kernel_npu` 接口可能让模型模块被 registry 跳过，最后只显示“不支持模型”。直接导入实际模型模块查看最早 traceback，并核对安装 wheel 与 SGLang 的配套版本。不要把任意 checkout 文件复制进 site-packages 伪装匹配 wheel；构建与安装入口见 [kernel 文档](sgl-kernel-npu-build.md)。

### Attention、异步 fault 与服务来源

- 历史 Ascend 950 默认 ATB attention 在 warmup/解码出现故障，本仓 LLM 脚本用 `ASCEND_USE_FIA=1`。不能把该故障直接归为 FP4 布局错误。
- Python traceback 常落在下一次同步点，不一定是 fault kernel；优先读取已有 plog：在 `llm/` 执行 `bash diagnostics/npu_fault_kernel.sh <scheduler-PID>`。
- 客户端安装路径/SHA 不证明运行中服务的 SHA。修改环境变量也不会改变已启动服务，必须重启才生效。
- 用户环境的 NPU `Critical` 状态不作为不能运行或卡顿根因的证据；单次 wall 异常同样不能定性硬件故障。

<a id="wan-modelslim-repack"></a>

## Wan ModelSlim 与 attention

- 显式选择正确的 `quant_model_description.json`，不能按字母顺序误选 FLOAT 描述。以当前 msmodelslim 导出格式为准，检查两套 transformer 的权重、模板和描述完整性。
- repack 必须同步转换旋转张量及其描述键；只改权重名或只改 JSON 都会破坏加载契约。工具版本见 [分支表](branches.md#g2-runtime-record)。
- Linear 量化与 MXFP8 attention 是两个开关/路径。出现 SDPA 或没有 rotation key 不自动证明 Linear 量化失败。
- 自动 FA 验收要看到 `Using MXFP8 quantized Ascend Flash Attention.`，不能仅凭后端名 `fa` 或视频生成成功判断。图片缩放后 token 数、对齐和后端限制要按实际 shape 检查；不要把某张图的对齐尺寸推广到所有图片。

<a id="mxfp-shared-execution-adapters"></a>

## 权重、scale 与执行契约

### 在线 / 离线与单级 / 双级

- 在线从 BF16/FP16 准备权重；离线 ModelSlim MXFP4 为 packed `uint8 [out, in/2]`，不得再 cast/pack 或再次量化。
- Dense 在线 W4A4 双级 L0/L1、MoE W4A4 单级、W4A8 的 FP4/FP8 组合是不同契约。不能因都叫 MXFP4 就复用同一个 scale 变换。
- checkpoint 参数名、Parameter 类型、loader 属性与分片轴必须保留。共享执行仅消费准备后的张量，不接管各来源的加载语义。
- scale rank 相同不代表轴相同；在线量化原始四维 scale 与转置/配对后的 GMM 四维 scale 必须区分。准备应可识别完成状态，避免二次 transpose。
- bias 的 dtype/shape 属于算子契约：W4A8 Dense 当前使用 BF16 `[1,N]`。如缓存准备后 bias，原参数替换、原地更新与重载都需失效处理。

### partial-scale 与 TP

MX block 为 32，pair 按 2 个 block 组织。K=4304 需要 `ceil(K/32)=135` 个 scale block，pair 容量为 136；不能丢尾块或直接 reshape 奇数 block。RowParallel 的 TP 分片要按**全局 K 起点**取 scale，并计入左侧 block 偏移，不能仅用本地 K 整除。占位 shape 修正不等于 kernel 支持任意 K，padding 和实际分片都要核对。

### transpose、NZ 与 FP4 dtype

- FP4 API 对 enum 与 torch dtype 的接受范围随算子/版本而异；历史接口不能用 `torch_npu.float4_e2m1fn_x2` 简单替代 torch dtype（或反向替代）。核对实际生产调用与安装版本。
- W4A8 常用 NZ 权重转置视图，W4A4 MoE 的融合调用可能使用 ND；不要统一强转 NZ。weight/scale 的方向、stride、transpose 标志必须匹配具体 API。
- `.contiguous()` 不是通用修复，也不是一概禁止：应证明表示与转置参数一致，再测数值/性能。不要把旧版本报错扩大成所有版本的支持结论。
- 双级量化 L0 scale 应保留算子所需的二维 `[M,1]`/`[1,N]` 形状；`squeeze(-1)` 可能把 `[N,1]` 变成错误的一维，使用明确 reshape。N=1 边界也要覆盖。

### 在线 W4A8 post-load 恢复 NZ 权重时报 `copy_` 不支持

`torch.device("npu")` 与 `torch.device("npu:0")` 的对象比较不同，即使都指向当前设备。外层 `stage_module_for_post_load` 若未解析实际设备，会把常驻 NZ 转置视图误判为需要 staging/恢复，并触发不支持的 `copy_`。

处理：先解析实际 process device，等价设备不搬运、不改 storage/stride/offset/alias；真正跨设备 offload 仍需恢复。不要用改 ND/contiguous、改量化参数掩盖问题。回归应穿过外层 context 并检查 hook 异常不被额外 copy 掩盖，不能只测试量化方法内部。

来源：10-26 用户加载结束后的报错及对应 CPU/mock 回归；不把 mock 的禁止 copy 当作 NPU 验收，后续证据见目标计划。

<a id="modelslim-mixed-gate-up"></a>

### 混精与 fused loader

- router 的 `*gate` 与 MLP `*gate_proj` 不同；过宽的 `*gate*` 会误伤目标层。按实际层名和描述核对，不按模型名猜精度。
- 同一 fused 参数的 shard 必须有一致格式；Dense 异构 gate/up 可拆成现有两套 ColumnParallelLinear 并保留各自 loader，MoE 专家 fused layout 另行处理。
- 不修改描述 JSON 假装权重已转换；FLOAT、FP4、INT8 的真实存储才是依据。

<a id="w4a8-alignment-fallback"></a>

### 在线 W4A8 对齐回退

Qwen 视觉 fc2 的 K=4304 不满足当前 W4A8 调用对齐要求时保留 BF16，不能将它计为全视觉 W4A8。判断使用 TP 后真实本地 shape，不扩大为所有 Linear 或模型的禁用条件。离线能力以 checkpoint scheme 和实际 kernel 为准。

## MoE 与跨平台导入

- GMM 的 group list 是 COUNT 还是 CUMSUM 必须逐调用确认；融合 GMM1/SwiGLU 与 GMM2 可以不同，不能只复制一个 dtype/shape。
- dispatcher 的 payload、scale、原输出 dtype 及 DeepEP normal/low-latency 契约分别检查。上游已重构过 Ascend dispatcher，不按旧类名推断当前路径。
- `vllm-ascend` 已存在 W4A8 MoE 参考；应查当前 checkout 的 `w4a8_mxfp4.py`，不沿用“只有 Dense”的旧结论。
- 跨平台顶层 `torch_npu` import 受 `sglang.srt.utils.is_npu()` 保护；不能用插件未加载时恒假的 platform 判断代替。CPU 可构造的类不要在 `__init__` 提前绑定缺失 NPU op，按调用惰性解析。
- 在线 W4A4 `"mxfp4"` 配置需正确标明动态模式（`is_static_cfg=False`）；已在上游注册的模块不要重复装饰器注册。

<a id="npu-mtp-state-contract"></a>

## MTP / GDN 状态与边界

- SSM/intermediate state 的 BF16 dtype、`(HV,V,K)` layout 需与部署 wheel 配套。相关源头：kernel #747/#808、SGLang #39589/#40419；不能只升级一侧。用 K≠V 的输入检测轴交换，方阵可能掩盖错误。
- prefill、decode、verify 的状态索引/容量和更新顺序分别核对。曾经重启后恢复正常的 graph/乱码反馈不证明 graph 是根因；不默认关闭 graph 或重排 update/replay。
- cache_loc 曾按 `batch*16` 发射，而实际 view 更小，写入同 storage 后续区域；kernel #824 已修复。要使用 tensor 的真实 numel/stride，并检查 view 的存储边界。越界写入恰好相同值时，单纯输出相等仍可能漏检。
- 旧 text-replacement 补丁已删除；恢复历史复现从 Git 获取，不对新版部署自动应用。

## 文档与验证

- 上游文档当前在 `docs/docs/*.mdx`；修改前查实际目录，合并后检查 rename 是否吞掉修改。MDX 用 `{/* ... */}`，不是 HTML 注释。
- 独立算子探针只在空闲 NPU 上运行，不能与 serving 抢同一卡。CPU/mock、NPU kernel、graph replay、模型精度与 serving 性能是不同证据层级。
- 新机器/软件栈重新核对完整链路，旧微基准阈值不作为生产策略。G6 的负收益和停止条件见 [归档](mxfp-refactoring-plan.md#g6-performance-archive)。

# sgl-kernel-npu 源码编译安装指南（Ascend / CANN）

> 记录一次从源码编译 `sgl-kernel-npu` 并安装成功的完整过程，供后人复现。
> 触发场景：sglang 跑新模型（如 **Qwen3.5**）报 `XxxForConditionalGeneration has no SGlang implementation`，
> 真因往往是已装的 `sgl_kernel_npu` 版本太旧、缺某个 kernel 符号，模块 import 失败被 `ModelRegistry` 静默跳过
> （详见 [known-pitfalls.md](known-pitfalls.md) 的「量化不生效/乱码输出」一节）。升级 `sgl_kernel_npu` 即可。

本次验证环境：

| 项 | 值 |
|---|---|
| 芯片 | Ascend950PR（A5）※ 见文末 SOC 注意事项 |
| CANN | `cann-9.1.T560`（beta 版，头文件树不全，需打补丁） |
| Python | 3.11.10 |
| 目标 | 让 sglang 支持 Qwen3.5（提供 `split_qkvgate_gemma_rmsnorm_rope` 等 kernel）|

---

## 1. 克隆仓库并初始化子模块

```bash
git clone <sgl-kernel-npu 仓库地址>
cd sgl-kernel-npu
git submodule update --init --recursive
```

## 2. 直接 `bash build.sh` 会遇到的两个报错（及修法）

`build.sh` 默认编译全部模块（deepep / kernels / attentions / memory-saver）。在上述环境下会依次撞到两个**环境相关**的编译错误，都不是代码 bug。

### 报错 A：`-Wframe-larger-than` 被 `-Werror` 提升为致命错误

```
block_sparse_attention_tiling.cpp: warning: the frame size of 89680 bytes is larger than 32768 bytes [-Wframe-larger-than=]
gmake[1]: *** [.../optiling.dir/all] Error 2
```

**根因**：`csrc/attentions/csrc/ops/cmake/intf_pub.cmake` 给所有算子加了 `-Wframe-larger-than=32768`，
而 `block_sparse_attention/op_host/CMakeLists.txt` 又加了 `-Werror`。某些 gcc 版本给该 tiling 函数分配的栈帧 > 32KB，
warning 就被 `-Werror` 变成 fatal。栈帧大只是良性提示（服务器默认 8MB 栈完全够）。

**修法**：调大阈值（一处生效于所有算子）：

```bash
sed -i 's/-Wframe-larger-than=32768/-Wframe-larger-than=262144/' \
    csrc/attentions/csrc/ops/cmake/intf_pub.cmake
```

> 注意：attentions 有独立且会缓存的 build 目录 `csrc/attentions/build/build`。改完 flag 后若报错不变、
> warning 里仍显示旧的 `32768`，说明用了缓存，删掉该 build 目录重编即可。

### 报错 B：CANN（T560 beta）profiling 头文件放错目录

```
prof_api.h:21: fatal error: prof_common.h: No such file or directory
# 补上 prof_common.h 后又：
prof_common.h:15: fatal error: aprof_pub.h: No such file or directory
```

**根因**：编译走的 include 根是 `$CANN/include/experiment/`，但该 beta 版把 `msprof` 的一整套头文件
（`prof_common.h`、`aprof_pub.h` …）漏放进 `include/experiment/msprof/profiling/`，只留了一个 `prof_api.h`。
完整的一套实际在 `$CANN/aarch64-linux/include/experiment/msprof/toolchain/`（和 `aarch64-linux/pkg_inc/profiling/`）。
这些头文件用引号 `#include`（先找同目录），把它们凑到 `prof_api.h` 旁边即可。

**修法**（写系统 CANN 目录，需 sudo；把整套 toolchain 头文件一次性软链过去，避免逐个打地鼠）：

```bash
CANN=/usr/local/Ascend/cann-9.1.T560   # 换成你的实际 CANN 路径
sudo ln -sf $CANN/aarch64-linux/include/experiment/msprof/toolchain/*.h \
            $CANN/include/experiment/msprof/profiling/
# 若个别头（如 aprof_pub.h）不在 toolchain，补链 pkg_inc/profiling：
sudo ln -sf $CANN/aarch64-linux/pkg_inc/profiling/*.h \
            $CANN/include/experiment/msprof/profiling/
```

> `experiment/platform`、`experiment/slog` 那两条只是 `-Wmissing-include-dirs` **warning**（-I 目录不存在，无害），
> 不用管。只有真正被 `#include` 的头缺失才会 fatal。
>
> 定位任意缺失头文件的通法：`find /usr/local/Ascend -name "<缺的头>.h" 2>/dev/null`，找到后同法 symlink。
> 非 T560 的正式版 CANN 通常不需要这一步。

## 3. 编译

```bash
bash build.sh
```

也可只编需要的模块（`build.sh -h` 查看），例如 `-a kernels` 只编 `sgl_kernel_npu`。
但注意当前 `build.sh` 的 `main()` **无条件调用 `build_attentions_kernels`**，
若只为跳过报错的 attentions（多数 dense 模型如 Qwen3.5 用不到 block_sparse_attention），
需手动注释掉 `main()` 里的 `build_attentions_kernels` 与 `make_attentions_package` 两行。

## 4. 安装

产物 wheel 在 `output/`：

```bash
ls output/*.whl   # sgl_kernel_npu-*.whl / deep_ep-*.whl / attentions-*.whl / torch_memory_saver-*.whl
pip install --force-reinstall --no-deps output/sgl_kernel_npu-*.whl
# 按需安装其余 wheel
```

## 5. 验证

```bash
# a. 目标 kernel 符号可导入
python -c "from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import split_qkvgate_gemma_rmsnorm_rope; print('kernel OK')"

# b. sglang 模型模块能 import（这才是注册成功的关键；绕过 ModelRegistry 的静默捕获）
cd <sglang 路径>
python -c "import sglang.srt.models.qwen3_5; print('qwen3_5 register OK')"
```

看到 `register OK` 后，再启动服务即可，`no SGlang implementation` 的假报错会消失。

---

## SOC / A5 注意事项 ⚠️

`build.sh` **不探测硬件**，SOC 目标是默认 / 写死的：

- 顶层默认 `SOC_VERSION=Ascend910_9382`（**A3**）。
- `build_kernels()` 里 `-DSOC_VERSION=Ascend910_9382` 是**写死**的。
- attentions 的 `build_ascendc_ops.sh` `--compute-unit` 默认 `ascend910b`（**A2**）。

命名对照：A2=`Ascend910B1`，A3=`Ascend910_9382`，**A5=`Ascend950`**。

本次是在 A5（Ascend950PR）机器上按**默认 A3 目标**编译并安装成功的——纯 triton kernel（如 `split_qkvgate_gemma_rmsnorm_rope`）
是运行时 JIT，与 build 时 SOC 无关，所以能用。但**编译型 AscendC 算子**若在 A5 上按 A3 目标编，运行时可能对不上。
若要为 A5 正确编译，需三处 SOC 口径统一到 950：`bash build.sh Ascend950` + 改 `build_kernels()` 写死行为 `Ascend950`
+ attentions 传 `-c ascend950`。此项本次**暂搁置**，待运行时确实报缺某编译型算子再针对性处理。

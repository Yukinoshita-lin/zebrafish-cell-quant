# P0 修复 与 双加速路径（C++ / GPU）实现报告

> 日期：2026-10-16
> 目标：解决审计 P0 三项；实现"C++ 加速"与"GPU 加速"两条路径、实测择优、
>       硬件缺失自动回退、C++ 对拍、GPU 批量化、DLL 缺失自动回退。

---

## 一、P0 三项修复

### P0-① 编译开关（PERF-1）→ `build_accel.py`
原生 `Makefile`/`CMakeLists.txt` 默认 `GPU=0 CUDNN=0 OPENCV=0 AVX=0 OPENMP=0`。
新增 `build_accel.py`，显式开启正确开关并产出**命名可区分**的两个库：

| 变体 | 产物 | 开关 |
|------|------|------|
| C++ CPU 加速 | `darknet_cpu.dll` / `libdarknet_cpu.so` | AVX + OpenMP + OpenCV，CUDA=OFF |
| GPU 加速 | `darknet_gpu.dll` / `libdarknet_gpu.so` | CUDA + CuDNN + CUDNN_HALF + OpenCV |

命名区分是关键：两个版本可共存，运行时才能选择与回退。

本机实测（dry-run）：`cmake ✓ / gcc ✓ / mingw32-make ✓ / opencv ✓ / GPU(4060) ✓ / nvcc ✗`
→ **GPU 变体被自动跳过并给出明确原因**，不失败、不中断。

### P0-② 缓冲区溢出（SEC-1 / SEC-2）→ 修改 C 源码
| 位置 | 问题 | 修复 |
|------|------|------|
| `src/utils.c` `find_replace` / `trim` / `find_replace_extension` | 固定 8192 缓冲 + 无长度检查 `sprintf`，超长路径写越界 | 内部缓冲改为 `strlen(str)+1` 精确分配；新增 `*_n` 显式长度版（用 `snprintf` 截断）；旧签名保留为兼容包装（上限与现有调用方 `char[4096]` 一致） |
| `src/utils.c` `trim` | 全空白字符串时 `end` 回退到 `p` 之前 → **越界读** | 增加 `end > p` 边界与全空分支 |
| `src/image.c`、`src/image_opencv.cpp`、`src/http_stream.cpp` `labelstr[4096]` | `strcat` 拼接任意个类名 → 栈溢出 | 新增 `safe_strcat(dst, size, src)` 统一带边界拼接，三处全部改用；`sprintf` → `snprintf` |
| `src/http_stream.cpp` | `utils.h` 在第 768 行才引入，而 597 行已用到 `safe_strcat` | 在文件顶部提前引入 `utils.h` |

**编译验证**：`gcc -fsyntax-only` 通过 `utils.c`、`image.c`（仅有上游固有的 `X_OK` 重定义告警）。

### P0-③ 命令注入（SEC-5）→ 脚本改造
| 脚本 | 修复 |
|------|------|
| `scripts/get_openimages_dataset.py` | `os.system("gsutil cp ...%s.jpg" % bbox[0])` → `subprocess.run([...], check=False)` 参数化 |
| `data/labels/make_labels.py` | 4 处 `os.system` → `subprocess.run` 列表参数；同时移除 Python 3.13 已删除的 `pipes` 依赖 |
| `scripts/voc_label.py` | `os.system("cat ...")` → 纯 Python 合并（顺带修好在 Windows 上 `cat` 不存在导致必失败的问题） |

---

## 二、双加速路径架构

```
                 ┌─────────────────────────────────────┐
                 │  choose_backend() 实测择优 + 缓存    │
                 └──────────────┬──────────────────────┘
                                │ 回退链（tier 由低到高）
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
  tier0: gpu              tier1: cpp_cpu          tier2: classical_python
  darknet_gpu.dll         darknet_cpu.dll         纯 Python + OpenCV
  CUDA/CuDNN              AVX/OpenMP              无需任何 DLL
  批量推理(batch=8)        批内逐张               经典斑点检测
        │                       │                       │
        └─────── 任一级加载/推理失败 → 自动降级 ────────┘
```

### 核心文件
| 文件 | 职责 |
|------|------|
| `zebrafish_quant/native.py` | 可指定库路径的 ctypes 绑定（**不用** `darknet.py`，因其在 import 期硬编码加载单个库，无法二选一、无法优雅降级） |
| `zebrafish_quant/backends.py` | 三级后端统一接口 + 发现 + 可用性判定 |
| `zebrafish_quant/benchmark.py` | 实测吞吐、选最快、缓存、跳过失败后端 |
| `zebrafish_quant/crosscheck.py` | 对拍：逐块计数差异 + 框贪心匹配 + 中心偏移/尺寸比 |
| `zebrafish_quant/tiling_pipeline.py` | 端到端：切块→后端→坐标回映→全局NMS→掩码计数→CSV/可视化 |
| `build_accel.py` | 构建两个加速变体 |
| `run_quant.py` | CLI：`--backend auto/gpu/cpp_cpu/classical_python`、`--batch-size`、`--crosscheck` |

---

## 三、三项关键机制

### 1. GPU 批量化
`NativeBackend.detect_batch()` 走 darknet 的 `network_predict_batch`，
一次前向处理 `batch_size` 张（默认 8），摊薄 kernel launch 与 PCIe 开销。
批量大小可用 `--batch-size` / 环境变量 `ZQ_GPU_BATCH` 调整。

### 2. DLL 缺失自动回退
- `find_library()` 找不到库 → `GpuBackend()/CpuBackend()` 返回 `None`（不抛异常）
- `available_backends()` 只保留通过可用性检查的后端
- `ClassicalBackend` 永远可用 → **保证流水线在任何机器上都能跑出结果**
- `NativeDarknet` 加载失败抛 `NativeLoadError`，被 `choose_backend` 捕获并记录后跳过
- CLI 强制指定不可用的 `--backend gpu` 时，打印警告并退回 auto（不会崩）

### 3. C++ 对拍
`crosscheck(A, B, tiles)` 输出：逐块计数差、匹配框数、一致率、
中心偏移（均值/最大）、尺寸比均值、是否通过阈值（默认 90%）。
用于验证"加速没有改变结果"——加速若改变结果就不是优化而是 bug。

---

## 四、测试结果

| 测试 | 结果 |
|------|------|
| `tests/test_zebrafish_quant_smoke.py`（回归） | **33/33 通过** |
| `tests/test_backends_and_fallback.py`（新增） | **26/26 通过** |
| 8 个模块导入 | 全部 OK，且**未加载任何 darknet DLL** |
| `run_quant.py --help` | 正常 |
| `build_accel.py --dry-run` | 环境探测正确，GPU 因缺 nvcc 被优雅跳过 |

### 关键验证点（重点）
**全局 NMS 去重确实有效**——这是切块方案最容易出错的地方（重叠区同一细胞会被相邻块各检一次）：

| overlap | 原始检测 | 全局NMS后 | 掩码内计数 |
|---------|---------|----------|-----------|
| 0 | 9 | 8 | 8 |
| 64 | 10 | **8** | 8 |

两种重叠设置去重后**收敛到同一个 8**，证明跨块重复已被正确消除。

> 注：合成图真实细胞数为 4，经典后端输出 8 —— 差值来自**经典算法把肠道管壁纹理误检为细胞**，
> 属于已文档化的弱基线局限（正因如此才需要训练 YOLO 来抑制这类伪影），
> **不是**重复计数。

**对拍有效性**：自比对一致率 100%；注入"整体平移 40px"的桩后端后一致率降为 0 并被判定不通过
→ 证明对拍能真正发现实现间的不一致。

---

## 五、仍需你决定的一件事

本机 **GPU 硬件是好的**（RTX 4060 Laptop 8GB，驱动 591.86），
但 **CUDA Toolkit 未安装**（`nvcc` 不存在）→ GPU 变体无法编译，
因此 GPU 路径与"GPU vs C++ 提速比"在本机**尚无法实测**。

已具备的条件：`cmake 4.4.3 / gcc 15.2 / mingw32-make / OpenCV`，
即 **C++ CPU 变体现在就能编译**。

建议下一步（三选一）：
1. 我先运行 `python build_accel.py --only cpu` 编译 C++ 加速版（需几分钟），
   然后立刻可做 **C++ vs 经典后端对拍**，拿到真实 CPU 提速数据；
2. 你安装 CUDA Toolkit 后我再编 GPU 版，补齐 GPU 批量化实测；
3. 两者都做，最终 `auto` 模式在真机上实测择优。

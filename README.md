# 斑马鱼中后肠荧光细胞深度学习定量

在斑马鱼荧光显微图中**定位中后肠区域，并只在该区域内定量荧光炎症细胞（中性粒细胞）**的全自动工具链。

本项目 Fork 自 [darknet](https://github.com/AlexeyAB/darknet)（保留其完整的 C/Python 源码与构建体系，见 `docs/darknet-upstream-README.md`），但**实际使用的训练与推理主线是 PyTorch U-Net 分割 + 经典检测计数**，并提供一个开箱即用的图形界面。

---

## 工作链路

```
单帧荧光图 (2752×2208)
   │  取绿色通道，÷4 下采样 → 688×552
   ▼
U-Net 分割  ──► 中后肠掩码        ← 唯一需要训练的环节（验收线 test Dice ≥ 0.85）
   │
   ▼
经典检测（草稿框）+ 掩码过滤 ──► 细胞计数
```

关键认知：**计数环节本身不训练**，它复用已有的检测框（`annotations/drafts/`），模型只负责"这块区域算不算肠"。

## 快速开始

### 方式一：图形界面（推荐，零环境依赖）

下载 Release 中的 `ZebraQuant-Setup-1.1.1.exe`（约 242 MB，单文件），双击安装：
桌面/开始菜单会出现 **斑马鱼荧光细胞定量工具** 图标，点击即用。界面五页：

| 页面 | 功能 |
|---|---|
| 数据 | 选择标注数据文件夹（含 `manifest.csv` + `masks/`），自动校验并显示统计 |
| 训练 | 一键训练（4 套预设 + 自定义参数），实时进度条 / ETA / 日志 |
| 指标 | Dice 训练曲线（raw/ema/0.85 验收线）与报告摘要 |
| 计数 | 端到端「AI 分割掩码 + 经典检测」计数，输出逐图 CSV |
| 环境 | 依赖自检 + PyTorch 一键安装 |

**首次运行为什么会进入「环境」页**：安装包只内置界面与数据处理所需的基础库，
PyTorch 作为可选项按需安装（CUDA 版约 2.5 GB，放进安装包会让体积膨胀 12 倍）。
torch 安装包已内含 CUDA 13 运行库，**无需单独安装 CUDA Toolkit**：

- 有 NVIDIA GPU → 点「安装 CUDA 版」（训练快约 37 倍）
- 没有 GPU 或只想试用 → 点「安装 CPU 版」（约 200 MB）

装完自动重新检测，之后训练/计数即可正常使用。

### 方式二：源码运行 GUI

```bash
pip install torch opencv-python-headless numpy PySide6-Essentials   # GPU 版 torch 见 docs/新人指南
python gui_app.py
```

### 方式三：命令行

```bash
# 长训练前必做：确认增强没让掩码错位（1 分钟，避免几小时算力白费）
python tools_check_aug.py

# 训练（推荐配置，约 84 分钟 @ RTX 4060）
UNET_EPOCHS=400 UNET_BASE=32 UNET_DEPTH=4 UNET_NORM=gn UNET_BATCH=4 \
UNET_SCALE=4 UNET_LR=0.002 UNET_AUG_DECAY=1 python tools_train_unet_v3.py

# 端到端计数
V2_MODEL_PATH=unet_model_v4_gn.pt V2_TAG=v4 python tools_v2_count.py
```

完整参数表、失效模式诊断（`tools_diagnose.py`）、提升手段实测对比见 **[docs/新人指南-如何训练模型.md](docs/新人指南-如何训练模型.md)**。

## 仓库结构

```
gui_app.py               图形界面主程序（PySide6）
build_package.py         便携运行时组装脚本（制作安装包用）
tools_train_unet_v3.py   U-Net 训练（深监督 + EMA + 强增强，断点续训）
tools_v2_count.py        端到端计数（U-Net 掩码 + 草稿框过滤）
tools_diagnose.py        失效模式诊断（阈值扫描 / 灾难失败 / 过欠分割）
tools_*.py               增强自检 / EMA 测试 / 分辨率天花板 / 集成评估等
zebrafish_quant/         图像 IO、经典荧光检测、切块、计数等基础库
annotations/
  manifest.csv           190 张图的标注清单（train 150 / test 40，按图划分无泄漏）
  masks/                 人工标注的中后肠掩码（2752×2208，白=前景）
  drafts/                经典检测草稿框（YOLO 格式）
  unet_model_*.pt        训练好的分割权重（最佳 0.8320，4 模型集成 0.8349）
  unet_metrics_*.txt     各轮训练报告
docs/                    新人指南 + darknet 上游文档
cfg/ src/ include/ ...   darknet 原生链路（保留，非当前主线）
```

## 数据与标注协议

- **训练集 150 / 测试集 40**，按图像划分（非按切块），无同图泄漏；测试集为盲法标注
- 三态分层：处理 104 / 造模 48 / 对照 38（造模组源图全量纳入，是最难也最重要的难例）
- 掩码口径：中肠 + 后肠，排除前肠肠球与肠外区域
- 标注流程：LabelMe 画 polygon → `tools_collect_masks.py` 汇总为 `masks/` + `manifest.csv`
- 前景极度稀疏（仅占画面 0.32%），损失必须用加权 BCE(`pos_weight=20`) + SoftDice

## 当前结果

| 指标 | 数值 |
|---|---|
| test Dice（单模型最佳 v4_gn） | 0.8320 |
| test Dice（4 模型集成） | **0.8349** |
| train Dice（集成） | 0.8666（已过 0.85 线） |
| 计数相关 r / 绝对偏差 | 0.922 / 3.3 |
| 验收状态 | 差 0.015 未达标；瓶颈是泛化（150 张训练图不足），**补充人工标注是最有效手段** |

## 环境注意

- 本项目在 **RTX 4060 Laptop (8GB)** 上开发：GroupNorm 下 batch 不得超过 4（8 会 OOM）；评估必须分块
- 判断 GPU 不要看 `torch.__version__` 后缀，直接 `torch.cuda.is_available()`
- 掩码上采样必须 `INTER_NEAREST`（÷4 口径下 nearest 0.9699 vs linear 0.9274）

## 许可

本项目以 **Apache License 2.0** 发布，版权所有 © 2026 **Yukinoshita-lin**（见 `LICENSE`）。
`annotations/` 下的数据集（掩码、清单、草稿框）与训练好的模型权重同样采用 Apache 2.0——
二次使用时请保留版权声明与许可文件，并用显著方式标明所做的修改。

上游致谢（非共同版权方）：
- 本仓库 fork 自 [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)（public domain），
  保留其 C/Python 源码与构建体系；`3rdparty/`（stb / pthreads / getopt）均为 public domain 组件
- 方法参考：Thapa & Stachura 2021（荧光细胞深度学习检测），但本数据集为 6MP 高分辨、
  细胞仅 5~10 px，架构上做了针对性调整（详见 `DATASET_AND_TRAINING_READINESS.md`）

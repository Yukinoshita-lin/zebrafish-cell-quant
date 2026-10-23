#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2026 Yukinoshita-lin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
build_package.py —— 组装 ZebraQuant 便携运行时（打包 setup.exe 的前置步骤）

产物：F:/dist_stage/ZebraQuant/
  python/   便携 Python（基础解释器 + venv 的 site-packages，无 pyvenv.cfg，可整体拷贝）
  app/      项目运行时文件（GUI + 训练/计数脚本 + 默认数据集）
  zebra.ico 应用图标（由 cv2 绘制）

**torch 不打包**：CUDA 版 PyTorch 约 3GB，会让安装包臃肿。改为安装后由 GUI
「环境」页按需一键安装（CPU 版约 200MB / CUDA 版约 2.5GB，源为 PyTorch 官方）。
安装包含：PySide6 / numpy / opencv / pip（供环境页安装 torch）/ 默认数据集。

排除：__pycache__、*.pyc、断点续训状态（unet_resume_*.pt）、
      旧可视化输出（*_vis/）、darknet 原生链路文件（DLL/build 目录等）。
"""
from __future__ import annotations

import os
import shutil
import struct
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent
VENV = Path("C:/Users/32403/.workbuddy/binaries/python/envs/default")
BASE_PY = Path("C:/Users/32403/.workbuddy/binaries/python/versions/3.13.12")
STAGE = Path("F:/dist_stage/ZebraQuant")

# app/ 需要携带的项目文件（相对路径或 glob）
APP_ITEMS = [
    "gui_app.py",
    "tools_collect_masks.py",
    "tools_train_unet_v3.py",
    "tools_train_unet.py",
    "tools_v1_count.py",
    "tools_v2_count.py",
    "tools_diagnose.py",
    "tools_gen_draft_boxes.py",
    "tools_pack_annotation_data.py",
    "tools_check_aug.py",
    "tools_test_ema.py",
    "zebrafish_quant",
    "docs",
    "annotations",
]
# annotations/ 内排除项
ANN_EXCLUDE_DIRS = {"qc", "review", "v1_vis", "v2_vis", "v3_vis", "v3bn_vis",
                    "ens4_vis", "unet_vis_v3", "__pycache__"}
ANN_EXCLUDE_GLOBS = ["unet_resume_*.pt", "*.log", "*.pyc"]


def ignore_patterns(patterns):
    def _ignore(_dir, names):
        import fnmatch
        return {n for n in names
                if any(fnmatch.fnmatch(n, p) for p in patterns)}
    return _ignore


# site-packages 中不打包的大件/无关组件：
#   torch/torchgen/functorch/nvidia/triton —— CUDA 版 PyTorch 约 3GB，
#       改为应用内「环境」页按需一键安装（CPU 版约 200MB / CUDA 版约 2.5GB）
#   pyinstaller 及其依赖 —— 仅打包工具链需要，运行时无关
# 注意：dist-info 目录也必须排除，否则 pip 会认为 torch 已安装而跳过下载
SP_EXCLUDE_DIR_GLOBS = ["torch", "torch*", "nvidia*", "functorch*",
                        "triton*", "pyinstaller*", "altgraph*", "pefile*"]
SP_EXCLUDE_FILES = ["*.pyc", "*.pyo", "pefile*"]


def copy_tree(src: Path, dst: Path, exclude_dir_globs=None,
              exclude_files=None, exclude_dirs=None):
    import fnmatch
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        if exclude_dirs:
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
        if exclude_dir_globs:
            dirs[:] = [d for d in dirs
                       if not any(fnmatch.fnmatch(d, g)
                                  for g in exclude_dir_globs)]
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            if exclude_files:
                if any(fnmatch.fnmatch(f, p) for p in exclude_files):
                    continue
            shutil.copy2(Path(root) / f, target / f)
            n += 1
    return n


def make_icon(path: Path):
    """cv2 绘制简洁图标：青色圆底 + 白色细胞点，写成 ICO（内嵌 PNG）。"""
    import cv2
    import numpy as np
    S = 256
    img = np.zeros((S, S, 4), np.uint8)
    cv2.circle(img, (S // 2, S // 2), 118, (155, 201, 53, 255), -1)
    cv2.circle(img, (S // 2, S // 2), 118, (30, 110, 86, 255), 6)
    rng = np.random.RandomState(7)
    pts = [(96, 108), (128, 92), (160, 118), (112, 148), (146, 152),
           (88, 132), (172, 140), (132, 170), (104, 176)]
    for x, y in pts:
        r = int(rng.randint(9, 16))
        cv2.circle(img, (x, y), r, (255, 255, 255, 255), -1)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    data = buf.tobytes()
    header = struct.pack("<HHH", 0, 1, 1)
    entry = struct.pack("<BBBBHHII", 0, 0, 0, 0, 1, 32, len(data), 6 + 16)
    path.write_bytes(header + entry + data)
    return True


def main():
    if STAGE.exists():
        # 覆盖式组装：不做整树删除（避免误删风险），同名文件直接覆盖
        print(f"[提示] 目标已存在，按覆盖方式更新: {STAGE}")
    print(f"[1/4] 复制基础解释器 {BASE_PY} -> {STAGE / 'python'}")
    copy_tree(BASE_PY, STAGE / "python",
              exclude_dirs={"__pycache__", "include", "libs", "tcl",
                            "Scripts", "test", "tests"},
              exclude_files=["*.pyc", "*.pyo", "*.a", "*.lib"])

    print("[2/4] 合并 site-packages（排除 torch/CUDA/pyinstaller，约 3GB）")
    sp_src = VENV / "Lib" / "site-packages"
    sp_dst = STAGE / "python" / "Lib" / "site-packages"
    copy_tree(sp_src, sp_dst,
              exclude_dir_globs=SP_EXCLUDE_DIR_GLOBS + ["__pycache__"],
              exclude_files=SP_EXCLUDE_FILES)

    print("[3/4] 复制项目运行时 -> app/")
    app = STAGE / "app"
    app.mkdir(parents=True, exist_ok=True)
    for item in APP_ITEMS:
        s = SRC_ROOT / item
        if not s.exists():
            print(f"  [跳过] 不存在: {item}")
            continue
        if s.is_dir():
            if item == "annotations":
                copy_tree(s, app / "annotations",
                          exclude_dirs=ANN_EXCLUDE_DIRS,
                          exclude_files=["*.pyc"])
                # glob 排除（resume/log）
                import fnmatch
                for p in (app / "annotations").iterdir():
                    if any(fnmatch.fnmatch(p.name, g)
                           for g in ANN_EXCLUDE_GLOBS):
                        p.unlink()
            else:
                copy_tree(s, app / item, exclude_dirs={"__pycache__"},
                          exclude_files=["*.pyc"])
        else:
            shutil.copy2(s, app / item)

    print("[4/4] 生成图标")
    icon = STAGE / "zebra.ico"
    if make_icon(icon):
        print(f"  图标 -> {icon}")

    def size(p: Path):
        total = 0
        for root, _d, files in os.walk(p):
            for f in files:
                total += (Path(root) / f).stat().st_size
        return total / 1e9

    print("\n===== 组装完成 =====")
    print(f"  python/  {size(STAGE / 'python'):.2f} GB")
    print(f"  app/     {size(STAGE / 'app'):.2f} GB")
    print(f"  合计     {size(STAGE):.2f} GB")
    print(f"  位置     {STAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

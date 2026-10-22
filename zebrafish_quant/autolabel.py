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
autolabel.py - 基于经典荧光斑点检测的"草稿标注"生成

用途（解决需求 6 的标注瓶颈）：
    需求要求人工标注 80~120 张图的"每一个炎症细胞点位"，逐个数点 + 画框极耗时。
    本模块先用经典方法（top-hat + Otsu + 连通域）自动生成**草稿标注**，
    人工只需做"删错/补漏"的校正，可把单图标注时间从十几分钟降到 1~2 分钟。

⚠️ 重要前提：
    草稿标注是**弱监督**，质量有限（会把碎片/噪点标成细胞，也会漏掉弱信号）。
    它不能直接当作 ground truth 用于验收评估（需求 7 的 Precision/Recall），
    但非常适合：
      (a) 作为人工标注的起点（强烈推荐，人工必须复核）
      (b) 作为"自训练/伪标签"冷启动，训练后模型再反过来清洗伪标签
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Blob:
    cx: float
    cy: float
    w: float
    h: float
    area: float
    peak: float        # 斑点峰值强度，可用于过滤弱噪点


def detect_fluorescent_blobs(image, *,
                             min_area: int = 8,
                             max_area: int = 8000,
                             top_hat_kernel: int = 25,
                             percentile: float = 95.0,
                             min_peak: int = 0) -> List[Blob]:
    """检测 EGFP 荧光斑点。

    流程：
      1. 信号 = G - max(R, B)，抑制明场残留与非特异性背景
      2. top-hat 去掉照明不均造成的背景起伏
      3. Otsu 与分位数取较大者作为阈值（对弱信号图更稳）
      4. 连通域 -> 按面积过滤 -> 输出等效框
    """
    if image.ndim == 3:
        b, g, r = cv2.split(image)
        sig = cv2.subtract(g, cv2.max(r, b))
    else:
        sig = image

    ksize = max(5, top_hat_kernel | 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    th = cv2.morphologyEx(sig, cv2.MORPH_TOPHAT, k)

    v = th[th > 0]
    if v.size < 50:
        return []

    t_otsu, _ = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = max(float(t_otsu), float(np.percentile(v, percentile)), 10.0)
    _, bw = cv2.threshold(th, t, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    blobs: List[Blob] = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        # 用连通域自身包围盒，比等效圆更贴合实际形状
        x, y, w, h = (int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                      int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        peak = float(sig[max(0, y):y + h, max(0, x):x + w].max()) if w and h else 0.0
        if peak < min_peak:
            continue
        blobs.append(Blob(cx=cx, cy=cy, w=float(w), h=float(h),
                          area=area, peak=peak))
    return blobs


def to_yolo_lines(blobs: List[Blob], tile_w: int, tile_h: int,
                  class_id: int = 0) -> List[str]:
    """转成 YOLO 标注行：class cx cy w h（相对切块归一化）。"""
    lines = []
    for b in blobs:
        cx = min(max(b.cx / tile_w, 0.0), 1.0)
        cy = min(max(b.cy / tile_h, 0.0), 1.0)
        w = min(max(b.w / tile_w, 0.0), 1.0)
        h = min(max(b.h / tile_h, 0.0), 1.0)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def blobs_to_yolo(blobs: List[Blob], tile_w: int, tile_h: int,
                  class_id: int = 0) -> str:
    return "\n".join(to_yolo_lines(blobs, tile_w, tile_h, class_id))


def median_blob_diameter(blobs: List[Blob]) -> float:
    """斑点中位直径，用于 anchor 设计与 sanity check。"""
    if not blobs:
        return 0.0
    d = [max(b.w, b.h) for b in blobs]
    return float(np.median(d))

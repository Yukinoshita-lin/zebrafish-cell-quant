#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_gen_draft_boxes.py - 为全部标注图像生成经典算法草稿检测框

背景：用户（领域专家）只做中后肠掩码标注，不做逐细胞框标注。
本脚本用经典荧光检测（top-hat + Otsu + 连通域，zebrafish_quant.autolabel）
在**全图原分辨率**上生成草稿框，YOLO 格式，供：
  (a) 人工轻复核后训练检测模型；
  (b) 与掩码配合做"经典检测 + 掩码内计数"的 v1 计数方案。

注意：草稿是弱监督，含噪点/碎片，不能直接当验收真值。

输入：annotations/manifest.csv（由 tools_collect_masks.py 生成）
输出：annotations/drafts/{name}.txt（YOLO: 0 cx cy w h，全图归一化）
      annotations/draft_stats.txt（数量/直径分布统计）
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "annotations")
DRAFTS = os.path.join(OUT, "drafts")

sys.path.insert(0, ROOT)
from zebrafish_quant.autolabel import detect_fluorescent_blobs
from zebrafish_quant.imaging import imread_robust


def main() -> int:
    mf = os.path.join(OUT, "manifest.csv")
    if not os.path.exists(mf):
        print(f"[错误] 先生成 manifest: {mf}")
        return 1
    rows = list(csv.DictReader(open(mf, encoding="utf-8-sig")))
    os.makedirs(DRAFTS, exist_ok=True)
    print(f"待处理图像: {len(rows)}")

    total, per_img, diams = 0, [], []
    for i, r in enumerate(rows):
        src = r["src_tif"]
        if not os.path.exists(src):
            print(f"[警告] 源图缺失: {src}")
            continue
        img = imread_robust(src)
        h, w = img.shape[:2]
        blobs = detect_fluorescent_blobs(img)
        lines = []
        for b in blobs:
            cx = min(max(b.cx / w, 0.0), 1.0)
            cy = min(max(b.cy / h, 0.0), 1.0)
            bw = min(max(b.w / w, 0.0), 1.0)
            bh = min(max(b.h / h, 0.0), 1.0)
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            diams.append(max(b.w, b.h))
        with open(os.path.join(DRAFTS, r["name"] + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        total += len(lines)
        per_img.append(len(lines))
        if (i + 1) % 40 == 0:
            print(f"  进度 {i + 1}/{len(rows)}，累计 {total} 框")

    stats = f"""经典算法草稿框统计（全图原分辨率 2752x2208）
图像数:        {len(per_img)}
总框数:        {total}
每图框数:      均值 {np.mean(per_img):.1f}  中位 {np.median(per_img):.0f}  范围 [{min(per_img)}, {max(per_img)}]
框直径(px):    P10 {np.percentile(diams, 10):.1f}  中位 {np.percentile(diams, 50):.1f}  P90 {np.percentile(diams, 90):.1f}
"""
    with open(os.path.join(OUT, "draft_stats.txt"), "w", encoding="utf-8") as f:
        f.write(stats)
    print("\n" + stats)
    print(f"草稿框 -> {DRAFTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

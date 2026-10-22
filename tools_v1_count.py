#!/usr/bin/env python
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
tools_v1_count.py - v1 计数方案：经典检测 + 人工掩码过滤

对应需求 4.3"仅掩码内计数"的可运行第一版：
  1. 经典荧光检测（autolabel.detect_fluorescent_blobs，全图原分辨率）
  2. 读人工标注的中后肠掩码（annotations/masks/）
  3. counting.count_within_mask 按"框中心落在掩码内"筛选
  4. 输出逐图计数 CSV + 分组统计 + 抽样叠加可视化

局限（写进交付文档）：经典检测有噪点/漏检，计数未经人工校正；
分割掩码为人工标注（替代 AI 分割），Dice 指标待 U-Net 训练后补评。
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from zebrafish_quant.autolabel import detect_fluorescent_blobs
from zebrafish_quant.counting import count_within_mask
from zebrafish_quant.imaging import imread_robust
from zebrafish_quant.inference import Detection
from tools_collect_masks import imread_unicode

ANN = os.environ.get("UNET_ANN_DIR") or os.path.join(ROOT, "annotations")
OUT_VIS = os.path.join(ANN, "v1_vis")


def blobs_to_detections(blobs, w: int, h: int) -> List[Detection]:
    dets = []
    for b in blobs:
        dets.append(Detection(label="cell", confidence=min(1.0, b.peak / 255.0),
                              cx=b.cx, cy=b.cy, w=b.w, h=b.h))
    return dets


def main() -> int:
    mf = os.path.join(ANN, "manifest.csv")
    rows = list(csv.DictReader(open(mf, encoding="utf-8-sig")))
    os.makedirs(OUT_VIS, exist_ok=True)
    print(f"v1 计数：{len(rows)} 张")

    results = []
    for i, r in enumerate(rows):
        img = imread_robust(r["src_tif"])
        h, w = img.shape[:2]
        blobs = detect_fluorescent_blobs(img)
        dets = blobs_to_detections(blobs, w, h)
        mask = imread_unicode(r["mask_png"])
        if mask is None or mask.shape[:2] != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
        cr = count_within_mask(dets, mask, min_confidence=0.0)
        results.append({
            "name": r["name"], "split": r["split"], "stratum": r["stratum"],
            "batch": r["batch"], "group": r["group"],
            "n_detected": len(dets), "n_inside_mask": cr.total,
            "n_outside": len(cr.outside),
            "mask_area_pct": r["mask_area_pct"],
        })
        if (i + 1) % 40 == 0:
            print(f"  进度 {i + 1}/{len(rows)}")

    out = os.path.join(ANN, "v1_counts.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(results)

    # 分组统计
    def agg(key):
        groups = {}
        for r in results:
            groups.setdefault(r[key], []).append(r)
        return groups

    print("\n=== v1 计数结果（掩码内炎症细胞数）===")
    lines = []
    for key in ("split", "stratum", "group"):
        lines.append(f"\n按{key}:")
        for g, lst in sorted(agg(key).items()):
            cnts = [x["n_inside_mask"] for x in lst]
            dets = [x["n_detected"] for x in lst]
            lines.append(f"  {g:28s} n={len(lst):3d}  掩码内计数 均值{np.mean(cnts):7.1f} "
                         f"中位{np.median(cnts):6.0f}  全图检测 均值{np.mean(dets):7.1f}")
    summary = "\n".join(lines)
    print(summary)
    with open(os.path.join(ANN, "v1_summary.txt"), "w", encoding="utf-8") as f:
        f.write("v1 计数：经典检测 + 人工中后肠掩码过滤\n" + summary + "\n")
    print(f"\n计数明细 -> {out}")

    # 抽样可视化：造模 vs 对照 各 2 张
    import cv2
    for stratum in ("造模", "对照"):
        cand = [r for r in results if r["stratum"] == stratum][:2]
        for r in cand:
            row = next(x for x in rows if x["name"] == r["name"])
            img = imread_robust(row["src_tif"])
            h, w = img.shape[:2]
            mask = imread_unicode(row["mask_png"])
            ov = img.copy()
            ov[mask > 0] = (0.3 * ov[mask > 0] +
                            0.7 * np.array([0, 0, 255])).astype(np.uint8)
            blobs = detect_fluorescent_blobs(img)
            for b in blobs:
                x1, y1 = int(b.cx - b.w / 2), int(b.cy - b.h / 2)
                x2, y2 = int(b.cx + b.w / 2), int(b.cy + b.h / 2)
                inside = mask[max(0, y1):y2, max(0, x1):x2].mean() > 0
                color = (0, 255, 255) if inside else (0, 128, 0)
                cv2.rectangle(ov, (x1, y1), (x2, y2), color, 2)
            sc = 1400 / w
            small = cv2.resize(ov, (1400, int(h * sc)))
            imwrite = cv2.imencode(".png", small)[1]
            imwrite.tofile(os.path.join(OUT_VIS, f"v1_{stratum}_{r['name']}.png"))
    print(f"可视化 -> {OUT_VIS}（黄=掩码内计数，绿=掩码外排除）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

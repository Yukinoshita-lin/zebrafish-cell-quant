#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_draft_review.py - 草稿框人工复核预备（按计数口径聚焦）

核心口径：**只有中心落在中后肠掩码内的框参与计数**（需求 4.3）。
因此人工复核只需做两件事：
  1. 核验掩码内的草稿框：删假阳（噪点/碎片/粘连块）、补漏标细胞；
  2. 在掩码范围内扫一眼有无完全没被框到的细胞。
掩码外的框不影响计数，无需逐框核对（图上以暗绿色 contextual 显示）。

产出：
  annotations/review/{name}_roi.png   掩码区复核图（外扩 80px）
  annotations/review_triage.csv      每图复核量与可疑框统计
  annotations/review/复核说明.txt     复核协议
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from zebrafish_quant.imaging import imread_robust, green_channel
from tools_collect_masks import imread_unicode

ANN = os.path.join(ROOT, "annotations")
REVIEW = os.path.join(ANN, "review")
MARGIN = 80           # 掩码外扩像素（复核视野）


def load_boxes(name: str, w: int, h: int) -> List[tuple]:
    path = os.path.join(ANN, "drafts", name + ".txt")
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path, encoding="utf-8"):
        p = line.split()
        if len(p) != 5:
            continue
        cx, cy, bw, bh = map(float, p[1:])
        boxes.append((cx * w, cy * h, bw * w, bh * h))
    return boxes


def main() -> int:
    import cv2
    rows = list(csv.DictReader(open(os.path.join(ANN, "manifest.csv"),
                                    encoding="utf-8-sig")))
    os.makedirs(REVIEW, exist_ok=True)
    print(f"复核预备：{len(rows)} 张（聚焦掩码内框）")

    triage = []
    for i, r in enumerate(rows):
        img = imread_robust(r["src_tif"])
        h, w = img.shape[:2]
        mask = imread_unicode(r["mask_png"])
        if mask is None:
            continue
        boxes = load_boxes(r["name"], w, h)
        g = green_channel(img)
        areas = np.array([b[2] * b[3] for b in boxes]) if boxes else np.array([1.0])
        med = float(np.median(areas))
        ys, xs = np.where(mask > 0)
        x0, x1 = max(0, xs.min() - MARGIN), min(w, xs.max() + MARGIN)
        y0, y1 = max(0, ys.min() - MARGIN), min(h, ys.max() + MARGIN)

        n_in = n_susp = n_ctx = 0
        for (cx, cy, bw, bh) in boxes:
            a = bw * bh
            sig = g[max(0, int(cy - bh / 2)):int(cy + bh / 2),
                    max(0, int(cx - bw / 2)):int(cx + bw / 2)]
            low_sig = sig.size == 0 or sig.mean() < 15
            suspicious = (a > 3 * med) or (a < 0.3 * med) or low_sig
            xi, yi = int(cx), int(cy)
            in_mask = 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0
            if in_mask:
                n_in += 1
                if suspicious:
                    n_susp += 1
            elif x0 <= cx <= x1 and y0 <= cy <= y1:
                n_ctx += 1
        # 复核图：掩码内框=黄（须核验）/可疑=红边；掩码外 ROI 框=暗绿（仅上下文）
        crop = img[y0:y1, x0:x1].copy()
        for (cx, cy, bw, bh) in boxes:
            xi, yi = int(cx), int(cy)
            in_mask = 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0
            in_roi = x0 <= cx <= x1 and y0 <= cy <= y1
            if not in_roi:
                continue
            p1 = (int(cx - bw / 2 - x0), int(cy - bh / 2 - y0))
            p2 = (int(cx + bw / 2 - x0), int(cy + bh / 2 - y0))
            if in_mask:
                a = bw * bh
                sig = g[max(0, int(cy - bh / 2)):int(cy + bh / 2),
                        max(0, int(cx - bw / 2)):int(cx + bw / 2)]
                low_sig = sig.size == 0 or sig.mean() < 15
                suspicious = (a > 3 * med) or (a < 0.3 * med) or low_sig
                color = (0, 0, 255) if suspicious else (0, 255, 255)
                cv2.rectangle(crop, p1, p2, color, 2)
            else:
                cv2.rectangle(crop, p1, p2, (0, 90, 0), 1)
        mc = mask[y0:y1, x0:x1]
        cnts, _ = cv2.findContours(mc, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts, -1, (255, 0, 255), 2)
        ok, buf = cv2.imencode(".png", crop)
        buf.tofile(os.path.join(REVIEW, f"{r['name']}_roi.png"))

        triage.append({
            "name": r["name"], "split": r["split"], "stratum": r["stratum"],
            "n_total_boxes": len(boxes),
            "n_inside_mask": n_in,          # 真正需要核验的框
            "n_suspicious_in_mask": n_susp,
            "n_context_outside": n_ctx,     # 上下文，不需核对
            "median_area_px": f"{med:.0f}",
            "est_review_min": round(n_in * 2.5 / 60 + 0.5, 1),   # ~2.5s/框+扫漏
        })
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(rows)}")

    out = os.path.join(ANN, "review_triage.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(triage[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(triage)

    tot_in = sum(t["n_inside_mask"] for t in triage)
    tot_susp = sum(t["n_suspicious_in_mask"] for t in triage)
    tot_min = sum(t["est_review_min"] for t in triage)
    test_in = sum(t["n_inside_mask"] for t in triage if t["split"] == "test")
    test_min = sum(t["est_review_min"] for t in triage if t["split"] == "test")
    print(f"\n复核范围统计（按计数口径聚焦后）：")
    print(f"  全部 190 图：掩码内框 {tot_in} 个（可疑 {tot_susp}），"
          f"预计 {tot_min:.0f} 分钟")
    print(f"  其中测试集 40 图：掩码内框 {test_in} 个，预计 {test_min:.0f} 分钟")

    readme = f"""中后肠草稿框复核说明（计数口径版）
{'=' * 52}
口径：只有中心落在中后肠掩码内的框参与计数（需求 4.3），
      掩码外的框不影响计数结果，无需逐框核对。

每张复核图（annotations/review/{{图名}}_roi.png）：
  黄色框 = 掩码内的草稿框，【需要核验】
  红色框 = 掩码内且可疑（面积异常/低信号区），【优先核验】
  暗绿框 = 掩码外的框，仅上下文，不用管
  紫线   = 人工中后肠掩码边界

复核动作（每图约 2~4 分钟）：
  1. 黄框逐个过：是细胞->保留；噪点/碎片/明场残留->删；
     一个框里两个细胞->拆成两个（画两个框）。
  2. 紫线范围内扫一眼：有明显细胞但没框的 -> 补框。
  3. 拿不准的 -> 保留并在复核记录里备注。

记录方式：新建 复核记录.csv，每图一行：
  name, corrected_count（修正后的掩码内细胞数）, reviewer, note

预计工作量：测试集 40 图约 {test_min:.0f} 分钟；
全部 190 图约 {tot_min:.0f} 分钟（可按组分给多人并行）。
"""
    with open(os.path.join(REVIEW, "复核说明.txt"), "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\n复核图 -> {REVIEW}（{len(triage)} 张）")
    print(f"分级清单 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

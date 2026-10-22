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
reporting.py - 结果导出（CSV）与可视化

对应需求：4.2 输出可视化图（原图叠加半透明肠道轮廓）
          4.3 用圆点/方框标记每一个识别到的炎症细胞
          4.4 CSV 汇总表 + 批量保存可视化结果图（原图+轮廓+打点），便于溯源

CSV 字段严格按需求 4.4：
      图片文件名、肠道内炎症细胞总数、分割是否通过人工复核
其余审计字段统一追加在后面，保证"必需字段在前、顺序稳定"，
下游 GraphPad/SPSS 或 pandas 读取不受影响。
"""
from __future__ import annotations

import csv
import os
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

from .inference import Detection

# 需求 4.4 规定的必需字段（顺序固定）
CSV_REQUIRED_FIELDS = ["图片文件名", "肠道内炎症细胞总数", "分割是否通过人工复核"]
# 审计/溯源附加字段
CSV_EXTRA_FIELDS = ["处理状态", "异常标记", "掩码外排除数", "掩码面积像素", "备注", "相对路径"]
CSV_FIELDS = CSV_REQUIRED_FIELDS + CSV_EXTRA_FIELDS

# 可视化配色（BGR）
CONTOUR_FILL = (60, 220, 60)      # 肠道轮廓/填充：绿色半透明
CONTOUR_LINE = (40, 255, 40)      # 轮廓线
CELL_DOT = (0, 60, 255)           # 计数的细胞：红色圆点（醒目）
CELL_BOX = (0, 120, 255)
EXCLUDED_DOT = (200, 200, 200)    # 被掩码排除的：灰色（便于人工核验是否误排）


def write_csv(rows: Iterable[dict], csv_path: str) -> str:
    """写 CSV（UTF-8 with BOM，保证 Excel 直接打开不乱码）。

    rows 中缺失字段自动补默认值，多出字段被忽略，
    这样调用方只关心业务字段也不会报错。
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or ".", exist_ok=True)
    rows = list(rows)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in CSV_FIELDS}
            writer.writerow(out)
    return csv_path


def make_result_row(*, image_path: str, count: int, manual_review: bool = False,
                    status: str = "ok", anomaly: str = "", excluded: int = 0,
                    mask_area: int = 0, note: str = "",
                    root: Optional[str] = None) -> dict:
    """构造一行结果。保持文件名与原始图片一一对应（需求 4.1）。"""
    rel = os.path.relpath(image_path, root) if root else image_path
    return {
        "图片文件名": os.path.basename(image_path),
        "肠道内炎症细胞总数": int(count),
        "分割是否通过人工复核": "是" if manual_review else "否",
        "处理状态": status,
        "异常标记": anomaly,
        "掩码外排除数": int(excluded),
        "掩码面积像素": int(mask_area),
        "备注": note,
        "相对路径": rel,
    }


def draw_overlay(image_bgr: np.ndarray,
                 mask: Optional[np.ndarray],
                 counted: Sequence[Detection],
                 excluded: Optional[Sequence[Detection]] = None,
                 *, alpha: float = 0.28,
                 dot_radius: int = 4,
                 draw_boxes: bool = True) -> np.ndarray:
    """在原图分辨率上叠加：半透明肠道轮廓 + 每个细胞一个圆点。

    与原始实现的差别：原 darknet_images.image_detection 把框画在
    **resize 后的 416x416** 图上再返回，输出图分辨率丢失、无法用于溯源核查。
    这里所有绘制都在**原图坐标系**完成（Detection 已由 inference 还原）。

    mask        : 中后肠掩码（需求 4.2 半透明轮廓）
    counted     : 计数的细胞（红点）——需求 4.3
    excluded    : 被掩码排除的检测（灰点），仅用于人工核验是否误排
    """
    canvas = image_bgr.copy()

    # 1) 半透明肠道区域 + 轮廓
    if mask is not None and mask.size:
        m = (mask > 0).astype(np.uint8)
        if m.any():
            color_layer = np.zeros_like(canvas)
            color_layer[:] = CONTOUR_FILL
            fill = cv2.bitwise_and(color_layer, color_layer, mask=m * 255)
            canvas = cv2.addWeighted(canvas, 1.0, fill, alpha, 0)

            cnts, _ = cv2.findContours(m * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, cnts, -1, CONTOUR_LINE, 2)

    h, w = canvas.shape[:2]

    # 2) 被排除的检测（灰，先画，避免遮挡计数的点）
    for d in (excluded or []):
        cx, cy = int(round(d.cx)), int(round(d.cy))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(canvas, (cx, cy), max(2, dot_radius - 1), EXCLUDED_DOT, -1)

    # 3) 计数的细胞：圆点 + 可选方框
    for d in counted:
        cx, cy = int(round(d.cx)), int(round(d.cy))
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(canvas, (cx, cy), dot_radius, CELL_DOT, -1)
            cv2.circle(canvas, (cx, cy), dot_radius + 1, (255, 255, 255), 1)
        if draw_boxes:
            x0 = int(round(d.cx - d.w / 2))
            y0 = int(round(d.cy - d.h / 2))
            x1 = int(round(d.cx + d.w / 2))
            y1 = int(round(d.cy + d.h / 2))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), CELL_BOX, 1)

    return canvas


def save_overlay(image_bgr: np.ndarray, out_path: str, **kwargs) -> str:
    """生成并保存可视化结果图。文件名与原图一一对应（需求 4.4）。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    overlay = draw_overlay(image_bgr, **kwargs)
    # 用 imencode 写盘，规避 Windows 中文路径写图失败
    ext = os.path.splitext(out_path)[1] or ".png"
    ok, buf = cv2.imencode(ext, overlay)
    if not ok:
        raise IOError(f"可视化图编码失败: {out_path}")
    buf.tofile(out_path)
    return out_path

"""
counting.py - 掩码约束下的炎症细胞计数

对应需求：4.3（仅在掩码内计数、掩码外忽略、区分噪点、处理轻微粘连）
          7  Precision/Recall >= 0.85、平均相对误差 <= 15%

关键实现点：
  1. 掩码约束：以检测框**中心**是否落在掩码内作为判据。
     需求原文"仅在AI分割出的中后肠掩码区域内部识别荧光细胞点，
     掩码外（肠外、前肠肠球）的荧光信号直接忽略"——中心判据
     对边界细胞最稳健，也不会因框略微越界而误杀。
  2. 噪点抑制：置信度过低的检测直接丢弃（阈值由 PR 曲线在验证集上确定）。
  3. 轻微粘连拆分：检测框面积显著大于中位数时，在框内做局部极大值检测，
     若出现多个足够分离的峰，则拆成多个细胞（减少漏检）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .imaging import green_channel
from .inference import Detection


@dataclass
class CountResult:
    """单张图的计数结果。"""
    inside: List[Detection]        # 参与计数的（掩码内）
    outside: List[Detection]       # 被掩码排除的（肠外/前肠），仅用于审计可视化
    total: int

    @property
    def count(self) -> int:
        return self.total


def point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    """点是否落在掩码内。越界返回 False（安全默认：越界即排除）。"""
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or yi < 0 or xi >= w or yi >= h:
        return False
    return bool(mask[yi, xi] > 0)


def count_within_mask(detections: Sequence[Detection], mask: np.ndarray,
                      min_confidence: float = 0.0) -> CountResult:
    """按掩码筛分检测结果。

    min_confidence 用于噪点抑制；默认 0（信任上游 detect 的 thresh）。
    """
    inside: List[Detection] = []
    outside: List[Detection] = []

    for d in detections:
        if d.confidence < min_confidence:
            continue                       # 噪点：不计数，也不计入 outside
        if point_in_mask(mask, d.cx, d.cy):
            inside.append(d)
        else:
            outside.append(d)

    return CountResult(inside=inside, outside=outside, total=len(inside))


def _median_area(dets: Sequence[Detection]) -> float:
    if not dets:
        return 0.0
    areas = [max(1.0, d.w * d.h) for d in dets]
    return float(np.median(areas))


def split_adhesions(image_bgr: np.ndarray, detections: Sequence[Detection],
                    *, area_ratio: float = 1.8,
                    min_peak_distance_ratio: float = 0.45) -> List[Detection]:
    """处理"轻微粘连"：把一个框里挤在一起的多个细胞拆开。

    判据：框面积 > area_ratio * 中位数面积时，认为可能是粘连团。
    做法：在框内绿色通道上做局部极大值检测：
        - 阈值分割 -> 距离变换 -> 取局部极大值 -> 按最小间距去重
        - 若找到 >=2 个峰，用这些峰替换原检测，峰数即细胞数

    若只找到 1 个峰（说明只是"大细胞"而非粘连），保留原检测不变。
    这样既减少漏检，又不会把单个大细胞误拆成多个（避免重复计数）。
    """
    if len(detections) < 2:
        return list(detections)

    green = green_channel(image_bgr)
    med_area = _median_area(detections)
    if med_area <= 0:
        return list(detections)

    out: List[Detection] = []
    h_img, w_img = green.shape[:2]

    for d in detections:
        area = max(1.0, d.w * d.h)
        if area <= area_ratio * med_area:
            out.append(d)
            continue

        # 框内裁剪（留出 1 像素安全边）
        x0 = max(0, int(d.cx - d.w / 2))
        y0 = max(0, int(d.cy - d.h / 2))
        x1 = min(w_img, int(d.cx + d.w / 2) + 1)
        y1 = min(h_img, int(d.cy + d.h / 2) + 1)
        if x1 <= x0 or y1 <= y0:
            out.append(d)
            continue

        patch = green[y0:y1, x0:x1]
        if patch.size == 0:
            out.append(d)
            continue

        peaks = _find_peaks(patch)
        if len(peaks) < 2:
            out.append(d)
            continue

        # 峰间距过近视为同一个细胞
        cell_d = max(4.0, min(d.w, d.h) * min_peak_distance_ratio)
        kept = _dedupe_peaks(peaks, cell_d)
        if len(kept) < 2:
            out.append(d)
            continue

        for (px, py) in kept:
            out.append(Detection(
                label=d.label, confidence=d.confidence,
                cx=x0 + px, cy=y0 + py,
                w=d.w / np.sqrt(len(kept)), h=d.h / np.sqrt(len(kept)),
                cx_net=d.cx_net, cy_net=d.cy_net,
                w_net=d.w_net, h_net=d.h_net,
            ))
    return out


def _find_peaks(patch: np.ndarray) -> List[Tuple[float, float]]:
    """在绿色通道 patch 上找局部极大值（荧光点中心）。"""
    if patch.max() <= 0:
        return []
    # 自适应阈值：只保留较亮的部分
    thr = max(30, float(patch.max()) * 0.55)
    _, bw = cv2.threshold(patch, thr, 255, cv2.THRESH_BINARY)
    bw = bw.astype(np.uint8)
    if bw.sum() == 0:
        return []

    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 5)
    # 局部极大值：膨胀后与原图相等的位置即峰值
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.dilate(dist, k)
    peak_mask = (dist >= dilated) & (dist > 0.5)
    ys, xs = np.nonzero(peak_mask)
    vals = dist[ys, xs]
    order = np.argsort(-vals)
    return [(float(xs[i]), float(ys[i])) for i in order]


def _dedupe_peaks(peaks: Sequence[Tuple[float, float]],
                  min_dist: float) -> List[Tuple[float, float]]:
    """按最小间距去重，保留最强的峰。"""
    kept: List[Tuple[float, float]] = []
    for p in peaks:
        if all(np.hypot(p[0] - k[0], p[1] - k[1]) >= min_dist for k in kept):
            kept.append(p)
    return kept


def summarize_counts(results: Sequence[CountResult]) -> dict:
    """批量汇总：总数、均值、零值图数量等，供日志与复核使用。"""
    totals = [r.total for r in results]
    if not totals:
        return {"images": 0, "sum": 0, "mean": 0.0, "min": 0, "max": 0, "zero_images": 0}
    return {
        "images": len(totals),
        "sum": int(np.sum(totals)),
        "mean": float(np.mean(totals)),
        "min": int(np.min(totals)),
        "max": int(np.max(totals)),
        "zero_images": int(np.sum(np.array(totals) == 0)),
    }

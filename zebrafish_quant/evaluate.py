"""
evaluate.py - 验收指标计算

对应需求：7 验收指标
    细胞检测：Precision >= 0.85、Recall >= 0.85
    肠道分割：Dice >= 0.85
    计数误差：AI 计数 vs 人工标准，平均相对误差 <= 15%

⚠️ 与参考论文口径的差异（必须在评估报告里写明）：
  Thapa & Stachura (2021) 报告的 8.65% 是**汇总口径**
    |ΣAI - Σ人工| / Σ人工 = |8579-7896|/7896
  需求写的是**平均相对误差**，即逐图 |AI_i - 人工_i|/人工_i 再取平均。

  两者关键差别：汇总口径允许不同图片的**高估与低估相互抵消**，
  从而系统性低估真实误差。极端例子：AI=[12,18] vs 人工=[10,20]，
    逐图口径 = (0.2+0.1)/2 = 15.0%（不达标）
    汇总口径 = |30-30|/30 = 0.0%（看似完美）
  同一组数据给出完全相反的验收结论。

  本模块两个口径都算（count_relative_error 同时返回），
  交付前**必须与需求方确认**采用哪一个。

⚠️ 匹配口径：需求 6 的标注内容是"标注肠道内部每一个炎症细胞**点位**"，
  即 GT 是点而非框。点标注下 IoU 无意义，因此默认用**点-框/点-点距离**匹配；
  若 GT 是框，用 match_boxes_iou。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .inference import Detection
from .segment import dice_coefficient


@dataclass
class PRF:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict:
        return {
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "Precision": round(self.precision, 4),
            "Recall": round(self.recall, 4),
            "F1": round(self.f1, 4),
        }


def _safe_prf(tp: int, fp: int, fn: int) -> PRF:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return PRF(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1)


def match_points_to_detections(gt_points: Sequence[Tuple[float, float]],
                               detections: Sequence[Detection],
                               radius: float = 12.0) -> PRF:
    """GT 为点位时的匹配：每个 GT 点在 radius 内找最近的未使用检测。

    一对一匹配（贪心，按距离升序），避免一个检测匹配多个 GT。
    返回 TP/FP/FN：
      TP = 被匹配上的 GT 数
      FP = 未匹配任何 GT 的检测数
      FN = 未匹配任何检测的 GT 数
    """
    gt = list(gt_points)
    dets = list(detections)
    n_gt, n_det = len(gt), len(dets)
    if n_gt == 0:
        return _safe_prf(0, n_det, 0)
    if n_det == 0:
        return _safe_prf(0, 0, n_gt)

    dists = []
    for gi, (gx, gy) in enumerate(gt):
        for di, d in enumerate(dets):
            dist = float(np.hypot(d.cx - gx, d.cy - gy))
            if dist <= radius:
                dists.append((dist, gi, di))
    dists.sort(key=lambda t: t[0])

    used_g, used_d = set(), set()
    for _, gi, di in dists:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)

    tp = len(used_g)
    return _safe_prf(tp, n_det - tp, n_gt - tp)


def _box_iou(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax0, ay0, ax1, ay1 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
    bx0, by0, bx1, by1 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_boxes_iou(gt_boxes: Sequence[Tuple[float, float, float, float]],
                    detections: Sequence[Detection],
                    iou_threshold: float = 0.5) -> PRF:
    """GT 为框时按 IoU 匹配（贪心，IoU 降序）。

    注：参考论文用的是 IoU=0.2；若想与论文口径对齐，把阈值设为 0.2。
    """
    gt = list(gt_boxes)
    dets = list(detections)
    n_gt, n_det = len(gt), len(dets)
    if n_gt == 0:
        return _safe_prf(0, n_det, 0)
    if n_det == 0:
        return _safe_prf(0, 0, n_gt)

    pairs = []
    for gi, gb in enumerate(gt):
        for di, d in enumerate(dets):
            iou = _box_iou(gb, d.box_original)
            if iou >= iou_threshold:
                pairs.append((iou, gi, di))
    pairs.sort(key=lambda t: -t[0])

    used_g, used_d = set(), set()
    for _, gi, di in pairs:
        if gi in used_g or di in used_d:
            continue
        used_g.add(gi)
        used_d.add(di)

    tp = len(used_g)
    return _safe_prf(tp, n_det - tp, n_gt - tp)


@dataclass
class CountError:
    per_image_mean: float      # 需求口径：逐图相对误差的均值
    per_image_median: float
    aggregate: float           # 论文口径：汇总相对误差
    n: int
    mae: float
    rmse: float

    def as_dict(self) -> dict:
        return {
            "逐图平均相对误差": round(self.per_image_mean, 4),
            "逐图中位相对误差": round(self.per_image_median, 4),
            "汇总相对误差": round(self.aggregate, 4),
            "样本数": self.n,
            "MAE": round(self.mae, 4),
            "RMSE": round(self.rmse, 4),
        }


def count_relative_error(ai_counts: Sequence[float],
                         human_counts: Sequence[float]) -> CountError:
    """计数误差。两种口径都返回。

    逐图口径对 human=0 的图做特殊处理：
      - 若 AI 也为 0 -> 该图误差 0（正确）
      - 若 AI > 0    -> 该图误差记为 1.0（100%），避免除零的同时不放过假阳性
    """
    ai = np.asarray(list(ai_counts), dtype=np.float64)
    hu = np.asarray(list(human_counts), dtype=np.float64)
    if ai.shape != hu.shape:
        raise ValueError(f"长度不一致: ai={ai.shape}, human={hu.shape}")
    if ai.size == 0:
        return CountError(0.0, 0.0, 0.0, 0, 0.0, 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(hu > 0, np.abs(ai - hu) / hu,
                       np.where(ai > 0, 1.0, 0.0))
    rel = np.nan_to_num(rel, nan=0.0, posinf=1.0, neginf=0.0)

    agg = abs(ai.sum() - hu.sum()) / hu.sum() if hu.sum() > 0 else (
        0.0 if ai.sum() == 0 else 1.0)
    diff = ai - hu
    return CountError(
        per_image_mean=float(rel.mean()),
        per_image_median=float(np.median(rel)),
        aggregate=float(agg),
        n=int(ai.size),
        mae=float(np.abs(diff).mean()),
        rmse=float(np.sqrt((diff ** 2).mean())),
    )


def evaluate_dice(pred_masks: Sequence, gt_masks: Sequence) -> dict:
    """批量 Dice（需求 7：肠道分割 Dice >= 0.85）。"""
    vals = [dice_coefficient(p, g) for p, g in zip(pred_masks, gt_masks)]
    if not vals:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "pass_ratio": 0.0}
    arr = np.asarray(vals)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "pass_ratio": round(float((arr >= 0.85).mean()), 4),
    }


def acceptance_report(dice_stats: dict, prf: PRF, count_err: CountError) -> dict:
    """按需求 7 汇总是否达标。"""
    return {
        "Dice>=0.85": dice_stats.get("mean", 0.0) >= 0.85,
        "Precision>=0.85": prf.precision >= 0.85,
        "Recall>=0.85": prf.recall >= 0.85,
        "平均相对误差<=0.15": count_err.per_image_mean <= 0.15,
        "明细": {
            "Dice": dice_stats,
            "检测": prf.as_dict(),
            "计数误差": count_err.as_dict(),
        },
    }

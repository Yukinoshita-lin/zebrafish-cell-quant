"""
crosscheck.py - 后端对拍（differential testing）

"对拍"目的：
    C++ 原生路径与 GPU 路径（以及与纯 Python 参照实现）必须给出**一致**的结果。
    加速若改变了结果，就不是优化而是 bug。本模块在真实切块上比对两个后端：

      * 逐块计数差异
      * 检测框按中心距离贪心匹配后的 TP / 仅A有 / 仅B有
      * 匹配上的框的中心偏移与尺寸比（衡量定位一致性）
      * 一致性结论（agreement_ratio 是否达标）

典型用法：
    crosscheck(gpu_backend, cpp_backend, tiles)   # 验证 GPU 与 C++ 结果一致
    crosscheck(cpp_backend, classical_backend, tiles)  # 与经典参照比对
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .backends import Backend
from .inference import Detection


def _greedy_match(a: Sequence[Detection], b: Sequence[Detection],
                  radius: float) -> List[tuple]:
    """按中心距离贪心一对一匹配，返回 [(a_idx, b_idx, dist), ...]。"""
    pairs = []
    for i, da in enumerate(a):
        for j, db in enumerate(b):
            d = float(np.hypot(da.cx - db.cx, da.cy - db.cy))
            if d <= radius:
                pairs.append((d, i, j))
    pairs.sort(key=lambda t: t[0])
    used_a, used_b, out = set(), set(), []
    for d, i, j in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        out.append((i, j, d))
    return out


def crosscheck(backend_a: Backend, backend_b: Backend,
               tiles: List[np.ndarray],
               *, thresh: float = 0.30, nms: float = 0.20,
               radius: float = 8.0,
               agreement_threshold: float = 0.90) -> Dict:
    """对拍两个后端。要求两者均已 load()。"""
    res_a = backend_a.detect_tiles(tiles, thresh, nms)
    res_b = backend_b.detect_tiles(tiles, thresh, nms)

    per_tile = []
    tot_a = tot_b = tot_matched = 0
    offsets, ratios = [], []

    for k in range(len(tiles)):
        da, db = res_a[k], res_b[k]
        m = _greedy_match(da, db, radius)
        matched = len(m)
        tot_a += len(da)
        tot_b += len(db)
        tot_matched += matched
        for i, j, d in m:
            offsets.append(d)
            wa, wb = da[i].w, db[j].w
            if wa > 0 and wb > 0:
                ratios.append(max(wa, wb) / min(wa, wb))
        per_tile.append({
            "tile": k, "count_a": len(da), "count_b": len(db),
            "matched": matched,
            "only_a": len(da) - matched, "only_b": len(db) - matched,
        })

    denom = max(tot_a, tot_b, 1)
    agreement = tot_matched / denom if denom else 1.0
    mean_abs_diff = float(np.mean([abs(t["count_a"] - t["count_b"])
                                   for t in per_tile])) if per_tile else 0.0

    return {
        "backend_a": backend_a.name,
        "backend_b": backend_b.name,
        "tiles": len(tiles),
        "total_a": tot_a,
        "total_b": tot_b,
        "matched": tot_matched,
        "agreement_ratio": round(float(agreement), 4),
        "mean_abs_count_diff": round(mean_abs_diff, 4),
        "mean_center_offset_px": round(float(np.mean(offsets)), 3) if offsets else 0.0,
        "max_center_offset_px": round(float(np.max(offsets)), 3) if offsets else 0.0,
        "mean_size_ratio": round(float(np.mean(ratios)), 3) if ratios else 0.0,
        "passed": bool(agreement >= agreement_threshold),
        "threshold": agreement_threshold,
        "match_radius_px": radius,
        "per_tile": per_tile,
    }


def format_report(rep: Dict) -> str:
    lines = [
        f"对拍: {rep['backend_a']}  vs  {rep['backend_b']}",
        f"  切块数            : {rep['tiles']}",
        f"  检测总数 A / B    : {rep['total_a']} / {rep['total_b']}",
        f"  匹配上的框        : {rep['matched']}  (半径 {rep['match_radius_px']}px)",
        f"  一致率            : {rep['agreement_ratio']:.2%} "
        f"(阈值 {rep['threshold']:.0%})",
        f"  逐块计数平均绝对差: {rep['mean_abs_count_diff']}",
        f"  中心偏移 均值/最大: {rep['mean_center_offset_px']} px / "
        f"{rep['max_center_offset_px']} px",
        f"  尺寸比均值        : {rep['mean_size_ratio']}",
        f"  结论              : {'✅ 通过' if rep['passed'] else '❌ 不一致'}",
    ]
    return "\n".join(lines)

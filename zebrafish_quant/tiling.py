"""
tiling.py - 大图切块（原生分辨率）与坐标映射

为什么必须有这个模块（实测结论）：
    本数据集图像为 2752x2208 (6.08 MP)，炎症细胞中位直径仅 4.7~10.5 px。
    若按 darknet_images.py 的做法把整图 resize 到 416：
        缩放 0.151 -> 中位细胞变成 0.7~1.6 px（亚像素，任何检测器都看不见）
    即便放大到 832 输入，细胞仍只有 1.4~3.2 px，依然不可靠。

    因此必须在**原生分辨率**上切块送入网络：
        416 切块 -> 42 块/图，细胞保持原始 4.7~10.5 px（可检测）
        608 切块 -> 20 块/图

    这也是参考论文（Thapa & Stachura 2021）做法不能直接照搬的原因：
    论文图像放大倍率/画幅与本数据集不同，细胞相对尺寸大得多。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple

import cv2
import numpy as np


@dataclass
class Tile:
    """一个切块及其在原图中的位置。"""
    image: np.ndarray
    x0: int          # 原图左上角 x
    y0: int          # 原图左上角 y
    index: int

    def to_global(self, cx: float, cy: float) -> Tuple[float, float]:
        """切块内坐标 -> 原图坐标。"""
        return cx + self.x0, cy + self.y0


def iter_tiles(image: np.ndarray, tile_size: int = 416,
               overlap: int = 0) -> Iterator[Tile]:
    """滑动窗口切块。overlap>0 时对边界细胞更友好（跨块细胞至少完整出现一次）。

    边缘处向外平铺：最后一块会回退以保证不越界，
    因此块与块之间可能重叠覆盖边缘区域（这正是 overlap 想解决的）。
    """
    h, w = image.shape[:2]
    stride = max(1, tile_size - overlap)
    idx = 0

    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    # 保证覆盖到底边/右边
    if ys and ys[-1] != h - tile_size:
        ys.append(max(0, h - tile_size))
    if xs and xs[-1] != w - tile_size:
        xs.append(max(0, w - tile_size))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile_size, h)
            x1 = min(x0 + tile_size, w)
            patch = image[y0:y1, x0:x1]
            if patch.shape[0] < tile_size or patch.shape[1] < tile_size:
                # 边缘块补齐（避免尺寸不一致导致训练报错）
                pad_y = tile_size - patch.shape[0]
                pad_x = tile_size - patch.shape[1]
                patch = cv2.copyMakeBorder(patch, 0, pad_y, 0, pad_x,
                                           cv2.BORDER_CONSTANT, value=0)
            yield Tile(image=patch, x0=x0, y0=y0, index=idx)
            idx += 1


def tile_has_signal(tile: np.ndarray, min_frac: float = 0.0002) -> bool:
    """快速判断切块内是否含有荧光信号，用于丢弃纯背景块。

    纯背景块占比很高（6MP 图中鱼体只占一部分），
    丢弃它们可把 42 块/图 降到 10~15 块/图，显著节省训练与推理时间。
    """
    if tile.ndim == 3:
        b, g, r = cv2.split(tile)
        sig = cv2.subtract(g, cv2.max(r, b))
    else:
        sig = tile
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    th = cv2.morphologyEx(sig, cv2.MORPH_TOPHAT, k)
    v = th[th > 0]
    if v.size == 0:
        return False
    t = max(float(np.percentile(v, 95)), 12.0)
    return bool((th > t).mean() > min_frac)


def count_tiles(w: int, h: int, tile_size: int = 416, overlap: int = 0) -> int:
    stride = max(1, tile_size - overlap)
    nx = max(1, len(range(0, max(1, w - tile_size + 1), stride)))
    ny = max(1, len(range(0, max(1, h - tile_size + 1), stride)))
    if (w - tile_size) % stride:
        nx += 1
    if (h - tile_size) % stride:
        ny += 1
    return nx * ny


def nms_boxes_global(boxes: List[Tuple[float, float, float, float, float]],
                     iou_threshold: float = 0.3) -> List[Tuple[float, float, float, float, float]]:
    """对由多个切块汇总而来的检测框做全局 NMS。

    切块推理后同一细胞可能在相邻块中重复出现（尤其带 overlap 时），
    必须映射回原图坐标后做一次全局 NMS，否则会**重复计数**。
    boxes: [(cx, cy, w, h, score), ...]（原图坐标）
    """
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: -boxes[i][4])
    keep: List[int] = []
    suppressed = [False] * len(boxes)

    def iou(a, b) -> float:
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        ax0, ay0, ax1, ay1 = ax - aw / 2, ay - ah / 2, ax + aw / 2, ay + ah / 2
        bx0, by0, bx1, by1 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
        ix0, iy0 = max(ax0, bx0), max(ay0, by0)
        ix1, iy1 = min(ax1, bx1), min(ay1, by1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return inter / union if union > 0 else 0.0

    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if suppressed[j] or j == i:
                continue
            if iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed[j] = True
    return [boxes[i] for i in keep]

"""
segment.py - 中后肠掩码获取与掩码运算

对应需求：4.2 肠道区域分割（生成掩码 / 排除前肠肠球 / 可视化轮廓）
          4.5 人工微调掩码
          7  验收指标 Dice >= 0.85

设计要点：
  * 原 darknet 是纯检测器，无法输出掩码 —— 这是最大的功能性缺口。
    本模块定义 MaskProvider 统一接口，使"分割模型"成为可插拔组件：
      - ClassicalGutMask()  : 无需训练的经典基线，保证流水线今天就能跑通
      - 训练好的 U-Net / YOLO-seg / Mask-RCNN : 实现同一 predict() 即可替换
    这样在拿到分割模型前后，上层 pipeline / 计数 / 导出代码完全不用改。
"""
from __future__ import annotations

import os
from typing import Optional, Protocol

import cv2
import numpy as np

from .imaging import green_channel


class MaskProvider(Protocol):
    """掩码提供者统一接口。

    predict(image_bgr) -> uint8 掩码，与原图同高宽，取值 {0, 255}，
    255 表示"中后肠内部"（参与计数的区域）。
    """

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        ...


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    """Dice 系数（需求 7：肠道分割 Dice >= 0.85）。

    输入可以是 {0,255} 或 {0,1} / bool，内部统一二值化。
    """
    a = (mask_a > 0).astype(np.float64)
    b = (mask_b > 0).astype(np.float64)
    inter = float((a * b).sum())
    return (2.0 * inter + eps) / (a.sum() + b.sum() + eps)


def iou(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-6) -> float:
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return (inter + eps) / (union + eps)


def mask_contours(mask: np.ndarray) -> list:
    """提取掩码轮廓，用于叠加半透明肠道轮廓（需求 4.2）。"""
    m = (mask > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(cnts)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """保留最大连通域，去掉散落的伪影。"""
    m = (mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if num <= 1:
        return np.zeros_like(m)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8) * 255


def clean_mask(mask: np.ndarray, *, open_k: int = 5, close_k: int = 15) -> np.ndarray:
    """形态学清理：开运算去小噪点，闭运算填补肠道内部空洞。"""
    m = (mask > 0).astype(np.uint8) * 255
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    return m


def save_mask(mask: np.ndarray, path: str) -> None:
    """保存掩码 PNG。人工微调（需求 4.5）时，用户可直接用图像软件改这张图。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, (mask > 0).astype(np.uint8) * 255)


def load_mask(path: str, target_shape: Optional[tuple] = None) -> np.ndarray:
    """加载人工修正后的掩码，并按目标尺寸还原（保证与原图对齐）。"""
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"掩码读取失败: {path}")
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    if target_shape is not None and raw.shape[:2] != target_shape[:2]:
        raw = cv2.resize(raw, (target_shape[1], target_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    return (raw > 127).astype(np.uint8) * 255


class ClassicalGutMask:
    """经典基线分割器（无需训练，保证流水线可运行）。

    思路：荧光图中肠道区域本身有一定自体荧光/背景结构，用绿色通道
    低阈值 + 形态学闭运算连成管状区域，再取最大连通域。

    ⚠️ 局限（必须在交付文档中写明）：
      该基线**不能保证 Dice >= 0.85**，也无法可靠排除前肠肠球。
      它的作用是让端到端流程先跑通；要满足验收指标，必须训练
      U-Net / Mask-RCNN 之类的分割模型并替换本类。
    """

    def __init__(self, *, thresh: Optional[int] = None,
                 open_k: int = 5, close_k: int = 25):
        self.thresh = thresh
        self.open_k = open_k
        self.close_k = close_k

    def predict(self, image_bgr: np.ndarray) -> np.ndarray:
        g = green_channel(image_bgr)

        if self.thresh is not None:
            t = self.thresh
        else:
            # Otsu 自适应；若图像几乎无信号则退回固定低阈值
            t, _ = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = max(int(t) * 0.35, 8)

        _, m = cv2.threshold(g, int(t), 255, cv2.THRESH_BINARY)
        m = clean_mask(m, open_k=self.open_k, close_k=self.close_k)
        m = largest_component(m)
        return m

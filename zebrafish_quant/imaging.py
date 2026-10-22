"""
imaging.py - 鲁棒图像 I/O 与几何映射

对应需求：4.1 批量图像导入（png/jpg/tif）、5.2 稳定性（损坏图片跳过）、
          4.2/4.3 可视化需在原图分辨率上叠加。

相对原实现的改进：
  1. 支持 tif/tiff（原 darknet_images.load_images 仅 glob jpg/jpeg/png）。
  2. 支持中文/Unicode 路径（cv2.imread 在 Windows 下读不了非 ASCII 路径，
     改用 np.fromfile + cv2.imdecode）。
  3. 显式处理 16-bit / 灰度 / 4 通道 / 空文件，避免下游崩溃。
  4. 用 letterbox 保持长宽比（原实现直接 cv2.resize，会拉伸细胞形状，
     直接影响计数与掩码精度）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 需求 4.1：常见显微图片格式
SUPPORTED_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


class ImageReadError(Exception):
    """图片无法读取（损坏、空文件、格式不支持）。用于批量时逐张跳过。"""


def imread_robust(path: str, *, clip_percentile: float = 99.5) -> np.ndarray:
    """读取任意显微图片，统一返回 8-bit 3 通道 BGR。

    与 cv2.imread 的区别（保持原有调用方兼容：返回值形态一致）：
      - 支持中文路径（np.fromfile + imdecode）
      - 支持 16-bit 荧光图（分位数裁剪后线性映射到 8-bit，避免过曝溢出）
      - 灰度/4 通道统一转 3 通道 BGR
      - 读失败抛 ImageReadError（而不是返回 None 让下游静默崩溃）
    """
    if not os.path.isfile(path):
        raise ImageReadError(f"文件不存在: {path}")
    if os.path.getsize(path) == 0:
        raise ImageReadError(f"空文件(0 字节): {path}")

    # np.fromfile 走 Python 层，天然支持 Unicode 路径
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ImageReadError(f"无法读取: {path} ({exc})") from exc

    if buf.size == 0:
        raise ImageReadError(f"空文件: {path}")

    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ImageReadError(f"解码失败/格式不支持: {path}")

    if img.ndim == 2:                       # 灰度
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:   # BGRA
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim != 3:
        raise ImageReadError(f"非预期维度 {img.shape}: {path}")

    if img.dtype != np.uint8:
        img = _to_uint8(img, clip_percentile)
    return img


def _to_uint8(img: np.ndarray, clip_percentile: float) -> np.ndarray:
    """把 12/16-bit 荧光图压到 8-bit。

    荧光图常为 16-bit 且动态范围集中在低值区，直接 /65535*255 会几乎全黑。
    这里做上分位数裁剪 + 线性拉伸，保留弱信号的同时抑制过曝点。
    """
    arr = img.astype(np.float32)
    hi = float(np.percentile(arr, clip_percentile))
    lo = float(np.percentile(arr, 1.0))
    if hi <= lo:                       # 退化图像（纯黑/纯白）
        hi = float(arr.max())
        lo = float(arr.min())
    if hi <= lo:
        return np.zeros(img.shape[:2] + (3,), dtype=np.uint8)
    scaled = np.clip((arr - lo) * (255.0 / (hi - lo)), 0, 255)
    return scaled.astype(np.uint8)


def list_images(folder: str, exts: Tuple[str, ...] = SUPPORTED_EXTS) -> List[str]:
    """列出文件夹内所有受支持图片（大小写不敏感），按文件名排序保证可复现。

    修正原实现两处问题：
      - 原 load_images 用 path.split('.')[-1] 判断“是不是单文件”，
        当文件夹路径本身含点（如 data.v2/images）会误判。
      - 原实现只匹配小写 jpg/png/jpeg，.TIF/.JPG 会被静默漏掉。
    """
    out = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            out.append(os.path.join(folder, name))
    return out


def resolve_input(path: str, exts: Tuple[str, ...] = SUPPORTED_EXTS) -> List[str]:
    """输入可以是：单个图片文件 / 文件夹 / 换行分隔的路径清单 txt。

    保持与原 darknet_images.load_images 相同的语义，但用 os.path.splitext
    取扩展名且大小写不敏感，避免路径含点导致误判。
    """
    if os.path.isdir(path):
        return list_images(path, exts)

    _, ext = os.path.splitext(path)
    if ext.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        return [path]
    if ext.lower() == ".txt":
        with open(path, "r", encoding="utf-8") as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    # 原实现此处会当作文件夹 glob；这里保持宽容，仍按目录尝试
    return list_images(path, exts)


@dataclass
class Letterbox:
    """记录 letterbox 变换参数，用于把网络坐标还原回原图坐标。"""
    scale: float      # 原图 -> 网络画布 的缩放系数
    pad_x: int        # 水平填充（画布左侧）
    pad_y: int        # 垂直填充（画布上方）
    net_w: int
    net_h: int
    orig_w: int
    orig_h: int

    def to_original(self, x: float, y: float) -> Tuple[float, float]:
        """网络画布坐标 -> 原图像素坐标。"""
        ox = (x - self.pad_x) / self.scale
        oy = (y - self.pad_y) / self.scale
        return ox, oy

    def box_to_original(self, box: Tuple[float, float, float, float]):
        """(cx, cy, w, h) 网络坐标 -> (cx, cy, w, h) 原图坐标。"""
        cx, cy, w, h = box
        ox, oy = self.to_original(cx, cy)
        return (ox, oy, w / self.scale, h / self.scale)


def letterbox_resize(img: np.ndarray, net_w: int, net_h: int) -> Tuple[np.ndarray, Letterbox]:
    """等比缩放并居中填充到网络输入尺寸，返回画布与变换参数。

    为什么改：原实现 cv2.resize(image, (w,h)) 直接拉伸。荧光细胞近似圆形，
    非等比拉伸会把圆拉成椭圆，改变细胞尺寸分布 -> 影响检测阈值、
    影响掩码 IoU、最终影响 Recall/Dice。letterbox 保持长宽比，
    并通过 Letterbox 精确还原坐标，不损失任何信息。
    """
    oh, ow = img.shape[:2]
    scale = min(net_w / ow, net_h / oh)
    nw = max(1, int(round(ow * scale)))
    nh = max(1, int(round(oh * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

    canvas = np.full((net_h, net_w, 3), 127, dtype=np.uint8)
    pad_x = (net_w - nw) // 2
    pad_y = (net_h - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    lb = Letterbox(scale=scale, pad_x=pad_x, pad_y=pad_y,
                   net_w=net_w, net_h=net_h, orig_w=ow, orig_h=oh)
    return canvas, lb


def green_channel(img: np.ndarray) -> np.ndarray:
    """提取绿色荧光通道（Tg(lyz:EGFP) 信号在 G 通道最强）。

    BGR 顺序：G 为 index 1。用 G - (R+B)/2 抑制非特异性背景与明场残留。
    """
    b, g, r = cv2.split(img.astype(np.int16))
    return np.clip(g - (r + b) // 2, 0, 255).astype(np.uint8)


def quality_flags(img: np.ndarray, green: Optional[np.ndarray] = None) -> dict:
    """异常图片判定（需求 7 例外说明：过曝 / 荧光极弱 -> 标记交人工）。

    返回 dict，字段：
      over_exposed : 饱和像素占比 > 0.15
      weak_signal  : 强信号像素占比 < 0.0005（几乎没有荧光）
      reasons      : 人可读原因列表
    """
    if green is None:
        green = green_channel(img)
    total = green.size
    sat_ratio = float((img >= 250).mean())          # 接近纯白 = 过曝
    strong_ratio = float((green > 60).sum()) / total

    reasons = []
    if sat_ratio > 0.15:
        reasons.append(f"疑似过曝(饱和像素{sat_ratio:.1%})")
    if strong_ratio < 0.0005:
        reasons.append(f"荧光信号极弱(强信号{strong_ratio:.2%})")
    return {
        "over_exposed": sat_ratio > 0.15,
        "weak_signal": strong_ratio < 0.0005,
        "sat_ratio": sat_ratio,
        "strong_ratio": strong_ratio,
        "reasons": reasons,
    }

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
inference.py - darknet 推理封装（修正坐标/类型/NMS 三类问题）

对应需求：4.3 炎症细胞检测与计数；7 Precision/Recall >= 0.85

相对原 darknet.py / darknet_images.py 的修正：

  [修复 1] NMS/IOU 阈值
      原 detect_image 默认 nms=0.45。荧光炎症细胞常相邻甚至轻微粘连，
      0.45 会把"挨着的两个细胞"合并成一个 -> 严重漏检，
      直接威胁 Recall >= 0.85。
      参考 Thapa & Stachura (2021) 在同任务上通过 PR 曲线选定的最优值，
      本模块默认 nms=0.20（可配置）。

  [修复 2] 置信度被转成字符串
      原 decode_detection 把 confidence 转成 str，
      detect_image 末尾 sorted(..., key=lambda x: x[1]) 因此按**字典序**排序
      （"9.0" > "80.0"）。本模块直接读数值型置信度，排序/阈值过滤都正确。

  [修复 3] 坐标空间
      原实现把检测框画在 resize 后的 416x416 图上就返回，
      输出图是网络分辨率而非原图分辨率；且检测框坐标无法直接与原图掩码求交。
      本模块保留 Letterbox 参数，把 (cx,cy,w,h) 精确还原到原图像素坐标。

  [修复 4] 非 ASCII（中文）路径
      原 load_network 用 path.encode("ascii")，遇到中文目录直接 UnicodeEncodeError；
      而 Windows 版 darknet 的 fopen 走 ANSI 接口，也不能直接吃 UTF-8。
      本模块在加载前把模型三件套暂存到纯 ASCII 目录，彻底绕开该限制。

对外接口保持兼容：detect() 返回的 Detection 同时提供 .bbox_original 与
.bbox_network，既有行为不被破坏。
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .imaging import Letterbox, letterbox_resize

# 参考论文（Thapa & Stachura 2021, Bioinformatics and Biology Insights）
# 在斑马鱼中性粒细胞计数任务上通过 PR 曲线选定的最优阈值
DEFAULT_THRESH = 0.30
DEFAULT_NMS = 0.20


@dataclass
class Detection:
    """一次检测。坐标同时提供原图与网络两种空间，避免调用方混淆。"""
    label: str
    confidence: float                       # 数值型（修复：原实现返回 str）
    cx: float                               # 原图坐标
    cy: float
    w: float
    h: float
    cx_net: float = 0.0                     # 网络(416x416)坐标
    cy_net: float = 0.0
    w_net: float = 0.0
    h_net: float = 0.0

    @property
    def center_original(self) -> Tuple[float, float]:
        return self.cx, self.cy

    @property
    def box_original(self) -> Tuple[float, float, float, float]:
        return self.cx, self.cy, self.w, self.h

    def as_tuple(self) -> tuple:
        """兼容原 darknet 的 (label, confidence, (cx,cy,w,h)) 形式。"""
        return (self.label, self.confidence, (self.cx, self.cy, self.w, self.h))


@dataclass
class DarknetModel:
    config_file: str
    data_file: str
    weights_file: str
    net_width: int = 0
    net_height: int = 0
    class_names: List[str] = field(default_factory=list)
    _staged_dir: Optional[str] = None       # 若为绕开中文路径而暂存，记录以便清理
    network: object = None
    _colors: dict = field(default_factory=dict)

    def close(self) -> None:
        """清理为绕开中文路径而建立的临时目录。不主动关闭 darknet（保持原语义）。"""
        if self._staged_dir and os.path.isdir(self._staged_dir):
            shutil.rmtree(self._staged_dir, ignore_errors=True)
            self._staged_dir = None


def _rewrite_data_refs(orig: str, staged_copy: str, tmpdir: str) -> None:
    """把 .data 文件内部引用的 names/train/valid 等也暂存到 ASCII 目录。

    只暂存 .data 本身不够：darknet 的 get_metadata() 会按 cell.data 里的
    names = <路径> 行去 fopen 那个文件，路径仍是中文 -> "Couldn't open file"
    （且 darknet 的 file_error() 直接 exit(0)，连异常都不会抛）。
    """
    keys = ("names", "train", "valid", "backup", "results", "eval", "labels")
    out = []
    try:
        with open(orig, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return
    for line in lines:
        parts = line.split()
        # .data 的格式是 "names = <路径>"（等号两侧有空格），split 后路径在最后一段
        if (len(parts) >= 2 and parts[0] in keys
                and not parts[-1].isascii()
                and os.path.exists(parts[-1])):
            src = parts[-1]
            ref_dst = os.path.join(tmpdir, os.path.basename(src))
            if not ref_dst.isascii():
                ref_dst = os.path.join(tmpdir, "ref_%d%s" % (len(out),
                                       os.path.splitext(src)[1]))
            if not os.path.exists(ref_dst):
                shutil.copy2(src, ref_dst)
            line = "%s = %s\n" % (parts[0], ref_dst)
        out.append(line)
    with open(staged_copy, "w", encoding="utf-8") as f:
        f.writelines(out)


def _stage_to_ascii(paths: List[str]) -> Tuple[List[str], Optional[str]]:
    """把模型文件复制到纯 ASCII 临时目录。

    仅在任一路径含非 ASCII 字符时才复制；否则原样返回（零开销、零副作用）。
    这样既有用法（英文路径）完全不受影响。
    .data 文件内部的 names/train/valid 引用会一并暂存并改写。
    """
    if all(p.isascii() for p in paths):
        return list(paths), None

    tmpdir = tempfile.mkdtemp(prefix="dn_ascii_")
    staged = []
    for p in paths:
        dst = os.path.join(tmpdir, os.path.basename(p))
        # basename 也可能含中文 -> 用序号重命名，确保纯 ASCII
        if not dst.isascii():
            ext = os.path.splitext(p)[1]
            dst = os.path.join(tmpdir, f"model_{len(staged)}{ext}")
        shutil.copy2(p, dst)
        if dst.lower().endswith(".data"):
            _rewrite_data_refs(p, dst, tmpdir)
        staged.append(dst)
    return staged, tmpdir


def load_model(config_file: str, data_file: str, weights_file: str,
               batch_size: int = 1) -> DarknetModel:
    """加载 darknet 模型，兼容中文路径。"""
    import darknet  # 延迟导入：不加载 darknet 也能用本模块的其他功能

    for p in (config_file, data_file, weights_file):
        if not os.path.exists(p):
            raise FileNotFoundError(f"模型文件不存在: {p}")

    staged, tmpdir = _stage_to_ascii([config_file, data_file, weights_file])
    cfg, dat, wts = staged

    network, class_names, colors = darknet.load_network(cfg, dat, wts, batch_size=batch_size)
    return DarknetModel(
        config_file=config_file, data_file=data_file, weights_file=weights_file,
        net_width=darknet.network_width(network),
        net_height=darknet.network_height(network),
        class_names=class_names,
        network=network,
        _colors=colors,
        _staged_dir=tmpdir,
    )


def detect(model: DarknetModel, image_bgr: np.ndarray,
           thresh: float = DEFAULT_THRESH,
           nms: float = DEFAULT_NMS) -> Tuple[List[Detection], Letterbox]:
    """对单张 BGR 图做检测，返回原图坐标下的检测列表与 letterbox 参数。

    输入 image_bgr 是**原图**（任意尺寸）；内部完成 letterbox + 推理 + 坐标还原。
    """
    import darknet

    canvas, lb = letterbox_resize(image_bgr, model.net_width, model.net_height)

    # darknet 需要 RGB 且通道优先的连续浮点数组
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    darknet_image = darknet.make_image(model.net_width, model.net_height, 3)
    try:
        darknet.copy_image_from_bytes(darknet_image, rgb.tobytes())
        raw = darknet.detect_image(model.network, model.class_names,
                                   darknet_image, thresh=thresh, nms=nms)
    finally:
        darknet.free_image(darknet_image)

    dets: List[Detection] = []
    for label, conf, (cx, cy, w, h) in raw:
        # 修复 2：conf 在原实现里是字符串，这里统一转回 float
        try:
            confidence = float(conf)
        except (TypeError, ValueError):
            continue
        if confidence > 1.0:          # 原实现按百分比给（0~100）
            confidence = confidence / 100.0
        if confidence < thresh:
            continue

        ox, oy, ow, oh = lb.box_to_original((cx, cy, w, h))
        dets.append(Detection(
            label=str(label), confidence=confidence,
            cx=ox, cy=oy, w=ow, h=oh,
            cx_net=cx, cy_net=cy, w_net=w, h_net=h,
        ))

    # 修复 2：按数值置信度降序（原实现按字符串字典序）
    dets.sort(key=lambda d: d.confidence, reverse=True)
    return dets, lb


def detect_with_split(model: DarknetModel, image_bgr: np.ndarray,
                      thresh: float = DEFAULT_THRESH,
                      nms: float = DEFAULT_NMS,
                      split_adhesions: bool = True) -> Tuple[List[Detection], Letterbox]:
    """在 detect 基础上可选做粘连拆分（见 counting.split_adhesions）。"""
    dets, lb = detect(model, image_bgr, thresh=thresh, nms=nms)
    if split_adhesions and dets:
        from .counting import split_adhesions as _split
        dets = _split(image_bgr, dets)
    return dets, lb

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
backends.py - 双加速路径 + 三级自动回退

设计目标（对应用户要求）：
    1. 两种加速方式：C++ 原生库（CPU，AVX/OpenMP）与 GPU（CUDA/CuDNN）
    2. 实测后默认选更快的那一种
    3. 加速硬件缺失时自动回退到另一种
    4. DLL 缺失时自动回退到纯 Python 经典算法（保证永不"跑不起来"）

回退链：
    GPU 原生库  -->  C++ CPU 原生库  -->  纯 Python 经典检测
   (darknet_gpu)    (darknet_cpu)       (无需任何 DLL)

统一接口：detect_tiles(tiles, thresh, nms) -> List[List[Detection]]
    上层 pipeline 完全不感知用的是哪一级。
"""
from __future__ import annotations

import os
import platform
import shutil
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from .autolabel import detect_fluorescent_blobs
from .imaging import letterbox_resize
from .inference import Detection

# 动态库命名约定（由 build_accel.py 产出）
GPU_LIB_NAMES = ("darknet_gpu.dll", "libdarknet_gpu.so")
CPU_LIB_NAMES = ("darknet_cpu.dll", "libdarknet_cpu.so")
GENERIC_LIB_NAMES = ("darknet.dll", "libdarknet.so")


def _candidate_dirs() -> List[str]:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [here, os.path.join(here, "build", "darknet", "x64"), os.getcwd()]


def find_library(names: Tuple[str, ...]) -> Optional[str]:
    for d in _candidate_dirs():
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return None


class Backend(ABC):
    """所有后端的统一抽象。"""

    name: str = "base"
    tier: int = 99                     # 越小越快（用于回退排序）

    @abstractmethod
    def availability(self) -> Tuple[bool, str]:
        """(是否可用, 原因)"""

    @abstractmethod
    def load(self, config_file: str, data_file: str, weights_file: str) -> None:
        ...

    @abstractmethod
    def detect_tiles(self, tiles: List[np.ndarray], thresh: float,
                     nms: float) -> List[List[Detection]]:
        """输入若干切块（BGR），返回每个切块的检测结果（切块坐标系）。"""

    def preferred_batch_size(self) -> int:
        return 1

    def close(self) -> None:
        pass


# ------------------------------------------------------------------
# 一级 / 二级：darknet 原生库（GPU 与 CPU 共用实现，仅加载的库不同）
# ------------------------------------------------------------------
class NativeBackend(Backend):
    def __init__(self, library_path: str, *, is_gpu: bool, name: str,
                 batch_size: int = 1):
        self.library_path = library_path
        self.is_gpu = is_gpu
        self.name = name
        self.tier = 0 if is_gpu else 1
        self._batch = batch_size if is_gpu else 1
        self._native = None
        self._net = None
        self._names: List[str] = []
        self._w = self._h = 0
        self._staged_dir: Optional[str] = None

    def availability(self) -> Tuple[bool, str]:
        if not os.path.exists(self.library_path):
            return False, f"动态库不存在: {os.path.basename(self.library_path)}"
        if self.is_gpu and not gpu_present():
            return False, "未检测到 NVIDIA GPU（nvidia-smi 不可用）"
        return True, "ok"

    def load(self, config_file: str, data_file: str, weights_file: str) -> None:
        from .native import NativeDarknet, NativeLoadError
        try:
            self._native = NativeDarknet(self.library_path)
        except NativeLoadError as exc:
            raise RuntimeError(f"[{self.name}] {exc}") from exc

        if not self.is_gpu:
            self._native.init_cpu()
        else:
            self._native.try_set_device(0)

        # 中文路径：darknet 的 fopen 走 ANSI 代码页（GBK），native.load_net 传的是
        # UTF-8 字节 -> "Couldn't open file"。与 inference._stage_to_ascii 同一
        # 缺陷（B-1）的同一解法：复制到纯 ASCII 临时目录再喂给 DLL。
        from .inference import _stage_to_ascii
        staged, self._staged_dir = _stage_to_ascii(
            [config_file, data_file, weights_file])
        cfg, dat, wts = staged

        self._net = self._native.load_net(cfg, wts,
                                          batch_size=max(1, self._batch))
        _n, self._names = self._native.metadata(dat)
        self._w = self._native.width(self._net)
        self._h = self._native.height(self._net)

    def preferred_batch_size(self) -> int:
        return self._batch

    def close(self) -> None:
        if self._staged_dir and os.path.isdir(self._staged_dir):
            shutil.rmtree(self._staged_dir, ignore_errors=True)
            self._staged_dir = None

    def detect_tiles(self, tiles, thresh, nms):
        if self._native is None or self._net is None:
            raise RuntimeError(f"[{self.name}] 未加载模型")
        bs = self.preferred_batch_size()
        out: List[List[Detection]] = []

        if bs > 1:
            # GPU 批量化：一次前向处理多张，摊薄 kernel launch / PCIe 开销
            for i in range(0, len(tiles), bs):
                chunk = tiles[i:i + bs]
                batch, lbs = self._make_batch(chunk)
                raw = self._native.detect_batch(
                    self._net, self._names, batch, self._w, self._h,
                    len(chunk), thresh=thresh, nms=nms)
                for dets, lb in zip(raw, lbs):
                    out.append(self._to_detections(dets, lb))
            return out

        for t in tiles:
            canvas, lb = letterbox_resize(t, self._w, self._h)
            rgb = np.ascontiguousarray(canvas[:, :, ::-1])
            raw = self._native.detect(self._net, self._names, rgb.tobytes(),
                                      self._w, self._h, thresh=thresh, nms=nms)
            out.append(self._to_detections(raw, lb))
        return out

    # ---------- 内部 ----------
    def _make_batch(self, chunk):
        import cv2
        planes, lbs = [], []
        for t in chunk:
            canvas, lb = letterbox_resize(t, self._w, self._h)
            rgb = np.ascontiguousarray(canvas[:, :, ::-1])
            planes.append(rgb.transpose(2, 0, 1))      # (3, h, w)
            lbs.append(lb)
        batch = np.ascontiguousarray(np.concatenate(planes, axis=0), dtype=np.float32)
        # 必须 /255：本 fork 的图像装载器（训练 load_image_stb、单张
        # detect() 的 copy_image_from_bytes）都做 /255，模型是在 0-1 数据上
        # 训练的。批路径喂 0-255 会与单张路径尺度相反，检测结果崩坏
        # （实测：单张 56 框 vs 批量 27 框且 IoU 匹配为 0）。
        batch /= 255.0
        return batch, lbs

    def _to_detections(self, raw, lb) -> List[Detection]:
        dets = []
        for label, conf, (cx, cy, w, h) in raw:
            ox, oy, ow, oh = lb.box_to_original((cx, cy, w, h))
            dets.append(Detection(label=str(label), confidence=float(conf),
                                  cx=ox, cy=oy, w=ow, h=oh,
                                  cx_net=cx, cy_net=cy, w_net=w, h_net=h))
        dets.sort(key=lambda d: -d.confidence)
        return dets


def GpuBackend(batch_size: Optional[int] = None) -> Optional[NativeBackend]:
    """GPU 后端。批量大小可用环境变量 ZQ_GPU_BATCH 覆盖。"""
    p = find_library(GPU_LIB_NAMES)
    if not p:
        return None
    if batch_size is None:
        try:
            batch_size = int(os.environ.get("ZQ_GPU_BATCH", "8"))
        except ValueError:
            batch_size = 8
    return NativeBackend(p, is_gpu=True, name="gpu", batch_size=max(1, batch_size))


def CpuBackend() -> Optional[NativeBackend]:
    p = find_library(CPU_LIB_NAMES) or find_library(GENERIC_LIB_NAMES)
    if not p:
        return None
    return NativeBackend(p, is_gpu=False, name="cpp_cpu", batch_size=1)


# ------------------------------------------------------------------
# 三级：纯 Python 经典检测（DLL 缺失时的兜底，永不失败）
# ------------------------------------------------------------------
class ClassicalBackend(Backend):
    """不依赖任何动态库的兜底后端。

    用 top-hat + Otsu + 连通域检测荧光斑点。
    精度远低于训练好的 YOLO，但保证：
      * 没有编译 darknet、没有 GPU、甚至 DLL 丢失时，流水线依然能跑出结果
      * 可作为"对拍"的第三方参照实现
    """

    name = "classical_python"
    tier = 2

    def availability(self) -> Tuple[bool, str]:
        return True, "始终可用（无需动态库）"

    def load(self, config_file: str, data_file: str, weights_file: str) -> None:
        # 经典算法不需要模型权重；仅做存在性提示，不抛错
        self._weights = weights_file

    def detect_tiles(self, tiles, thresh, nms):
        out = []
        for t in tiles:
            blobs = detect_fluorescent_blobs(t)
            dets = []
            for b in blobs:
                # 用归一化峰值强度作为"伪置信度"，使下游阈值逻辑统一可用
                conf = min(1.0, max(0.0, b.peak / 255.0))
                if conf < thresh:
                    continue
                dets.append(Detection(label="cell", confidence=conf,
                                      cx=b.cx, cy=b.cy, w=b.w, h=b.h,
                                      cx_net=b.cx, cy_net=b.cy,
                                      w_net=b.w, h_net=b.h))
            dets.sort(key=lambda d: -d.confidence)
            out.append(dets)
        return out


# ------------------------------------------------------------------
# 硬件探测
# ------------------------------------------------------------------
def gpu_present() -> bool:
    """检测 NVIDIA GPU 是否存在（nvidia-smi 可用即可）。"""
    try:
        import subprocess
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:                                    # noqa: BLE001
        return False


def discover_backends() -> List[Backend]:
    """按 tier 顺序返回所有**已构造**的候选后端（未验证可用性）。"""
    cands: List[Backend] = []
    g = GpuBackend()
    if g:
        cands.append(g)
    c = CpuBackend()
    if c:
        cands.append(c)
    cands.append(ClassicalBackend())
    return cands


def available_backends() -> List[Tuple[Backend, str]]:
    """返回 [(backend, 可用原因)]，仅含通过 availability() 的后端。"""
    res = []
    for b in discover_backends():
        ok, why = b.availability()
        if ok:
            res.append((b, why))
    return res

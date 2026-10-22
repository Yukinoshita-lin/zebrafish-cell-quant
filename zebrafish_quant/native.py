"""
native.py - 可指定动态库路径的 darknet ctypes 绑定

为什么不用现成的 darknet.py：
    darknet.py 在**模块导入时**就硬编码加载 libdarknet.so / darknet.dll，
    既无法选择"CPU 版还是 GPU 版"两个不同的库，也无法在 DLL 缺失时优雅降级
    （import 直接抛 OSError）。要实现"双加速路径 + 自动回退"，必须自己控制
    加载哪一个库。

本模块提供：
    * NativeDarknet(library_path) —— 显式加载指定 .dll/.so
    * detect()     单张推理
    * detect_batch() 批量推理（GPU 提速的关键：一次前向处理多张）
    * 只读字符串解析（不再依赖 darknet.py 的 decode_detection，
      避免其把置信度转成 str 导致排序错乱的缺陷 B-2）
"""
from __future__ import annotations

import ctypes as ct
import os
import platform
from typing import List, Optional, Tuple

import numpy as np


class BOX(ct.Structure):
    _fields_ = (("x", ct.c_float), ("y", ct.c_float),
                ("w", ct.c_float), ("h", ct.c_float))


FloatPtr = ct.POINTER(ct.c_float)
IntPtr = ct.POINTER(ct.c_int)


class DETECTION(ct.Structure):
    _fields_ = (
        ("bbox", BOX), ("classes", ct.c_int), ("best_class_idx", ct.c_int),
        ("prob", FloatPtr), ("mask", FloatPtr), ("objectness", ct.c_float),
        ("sort_class", ct.c_int), ("uc", FloatPtr), ("points", ct.c_int),
        ("embeddings", FloatPtr), ("embedding_size", ct.c_int),
        ("sim", ct.c_float), ("track_id", ct.c_int),
    )


DETECTIONPtr = ct.POINTER(DETECTION)


class DETNUMPAIR(ct.Structure):
    _fields_ = (("num", ct.c_int), ("dets", DETECTIONPtr))


DETNUMPAIRPtr = ct.POINTER(DETNUMPAIR)


class IMAGE(ct.Structure):
    _fields_ = (("w", ct.c_int), ("h", ct.c_int),
                ("c", ct.c_int), ("data", FloatPtr))


class METADATA(ct.Structure):
    _fields_ = (("classes", ct.c_int), ("names", ct.POINTER(ct.c_char_p)))


class NativeLoadError(Exception):
    """动态库加载失败（文件缺失、依赖缺失、架构不匹配）。"""


def _preload_dll_dir(lib_path: str) -> None:
    """Windows：把库所在目录加入 PATH / DLL 搜索路径，
    确保 pthreadVC2.dll、cudnn64_*.dll 等依赖可被找到。"""
    if not lib_path or os.name != "nt":
        return
    d = os.path.dirname(os.path.abspath(lib_path))
    if d and d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = d + os.path.pathsep + os.environ.get("PATH", "")


class NativeDarknet:
    """绑定到一个具体的 darknet 动态库。"""

    def __init__(self, library_path: str):
        if not library_path or not os.path.exists(library_path):
            raise NativeLoadError(f"动态库不存在: {library_path}")
        self.path = os.path.abspath(library_path)
        _preload_dll_dir(self.path)
        try:
            if os.name == "nt":
                self.lib = ct.CDLL(self.path, winmode=0, mode=ct.RTLD_GLOBAL)
            else:
                self.lib = ct.CDLL(self.path, mode=ct.RTLD_GLOBAL)
        except OSError as exc:
            raise NativeLoadError(
                f"加载失败（可能是依赖 DLL 缺失或 32/64 位不匹配）: {self.path} :: {exc}"
            ) from exc
        self._bind()

    # ---------- 原型绑定 ----------
    def _bind(self) -> None:
        lib = self.lib
        lib.network_width.argtypes = (ct.c_void_p,)
        lib.network_width.restype = ct.c_int
        lib.network_height.argtypes = (ct.c_void_p,)
        lib.network_height.restype = ct.c_int

        lib.load_network_custom.argtypes = (ct.c_char_p, ct.c_char_p, ct.c_int, ct.c_int)
        lib.load_network_custom.restype = ct.c_void_p

        lib.get_metadata.argtypes = (ct.c_char_p,)
        lib.get_metadata.restype = METADATA

        lib.make_image.argtypes = (ct.c_int, ct.c_int, ct.c_int)
        lib.make_image.restype = IMAGE
        lib.free_image.argtypes = (IMAGE,)
        lib.copy_image_from_bytes.argtypes = (IMAGE, ct.c_char_p)

        lib.network_predict_ptr.argtypes = (ct.c_void_p, FloatPtr)
        lib.network_predict_ptr.restype = FloatPtr

        lib.get_network_boxes.argtypes = (ct.c_void_p, ct.c_int, ct.c_int, ct.c_float,
                                          ct.c_float, IntPtr, ct.c_int, IntPtr, ct.c_int)
        lib.get_network_boxes.restype = DETECTIONPtr

        lib.do_nms_sort.argtypes = (DETECTIONPtr, ct.c_int, ct.c_int, ct.c_float)
        lib.do_nms_obj.argtypes = (DETECTIONPtr, ct.c_int, ct.c_int, ct.c_float)
        lib.free_detections.argtypes = (DETECTIONPtr, ct.c_int)

        lib.network_predict_batch.argtypes = (ct.c_void_p, IMAGE, ct.c_int, ct.c_int,
                                              ct.c_int, ct.c_float, ct.c_float,
                                              IntPtr, ct.c_int, ct.c_int)
        lib.network_predict_batch.restype = DETNUMPAIRPtr
        lib.free_batch_detections.argtypes = (DETNUMPAIRPtr, ct.c_int)
        lib.set_batch_network.argtypes = (ct.c_void_p, ct.c_int)
        # net->batch 查询（ctypes 侧不知道，由 load_net 记录）
        self._batch_of: dict = {}

    # ---------- 网络 ----------
    def load_net(self, cfg: str, weights: str, batch_size: int = 1) -> ct.c_void_p:
        net = self.lib.load_network_custom(cfg.encode(), weights.encode(), 0, batch_size)
        if not net:
            raise NativeLoadError(f"load_network_custom 失败: {cfg} / {weights}")
        self._batch_of[net] = batch_size
        return net

    def metadata(self, data_file: str) -> Tuple[int, List[str]]:
        md = self.lib.get_metadata(data_file.encode())
        names = [md.names[i].decode() for i in range(md.classes)]
        return md.classes, names

    def width(self, net) -> int:
        return self.lib.network_width(net)

    def height(self, net) -> int:
        return self.lib.network_height(net)

    # ---------- 推理 ----------
    def detect(self, net, names: List[str], rgb_bytes: bytes,
               w: int, h: int, thresh: float = 0.30,
               nms: float = 0.20) -> List[tuple]:
        """单张推理。rgb_bytes 为 RGB 顺序、长度 w*h*3 的原始字节。

        返回 [(label, confidence(float), (cx, cy, bw, bh)), ...]，
        坐标位于 (w,h) 输入空间。置信度保持 float（修复 B-2）。

        注意：若该网络是以 batch>1 加载的（GPU 批量推理），这里会先把
        net->batch 切成 1 再推理、之后还原 —— network_predict_gpu() 按
        net->batch 计算输入 memcpy 长度，直接用 batch=8 的网络跑单图会
        越界读（access violation）。
        """
        img = self.lib.make_image(w, h, 3)
        try:
            self.lib.copy_image_from_bytes(img, rgb_bytes)
            batch = self._batch_of.get(net, 1)
            if batch != 1:
                self.lib.set_batch_network(net, 1)
            try:
                self.lib.network_predict_ptr(net, img.data)
                pnum = ct.pointer(ct.c_int(0))
                dets = self.lib.get_network_boxes(net, w, h, thresh, 0.5,
                                                  None, 0, pnum, 0)
                n = pnum[0]
                if nms:
                    self.lib.do_nms_sort(dets, n, len(names), nms)
                out = self._extract(dets, n, names, thresh)
                self.lib.free_detections(dets, n)
                return out
            finally:
                if batch != 1:
                    self.lib.set_batch_network(net, batch)
        finally:
            self.lib.free_image(img)

    def detect_batch(self, net, names: List[str], batch_rgb: np.ndarray,
                     w: int, h: int, batch_size: int,
                     thresh: float = 0.30, nms: float = 0.20) -> List[List[tuple]]:
        """批量推理（GPU 提速关键：一次前向处理多张）。

        batch_rgb: shape (batch_size*3, h, w) 的 float32 连续数组（通道优先拼接），
                   或 (batch_size, 3, h, w)。内部统一处理。
        返回：每张图一个检测结果列表。

        安全约束：net->input_state_gpu / input_pinned_cpu 在 load 时按加载批量
        一次性分配，之后 set_batch_network 只会改 batch 计数、不重分配这两个
        缓冲区。因此这里按「加载时的批量」切片，绝不超过上限。
        """
        arr = np.ascontiguousarray(batch_rgb, dtype=np.float32)
        if arr.ndim == 4:
            arr = arr.reshape(-1, h, w)
        max_batch = self._batch_of.get(net, 1) or 1
        plane = 3 * h * w
        results: List[List[tuple]] = []
        for off in range(0, batch_size, max_batch):
            n = min(max_batch, batch_size - off)
            flat = np.ascontiguousarray(arr[off * plane:(off + n) * plane],
                                        dtype=np.float32)
            ptr = flat.ctypes.data_as(FloatPtr)
            batch_img = IMAGE(w, h, 3, ptr)
            pairs = self.lib.network_predict_batch(net, batch_img, n, w, h,
                                                   thresh, 0.5, None, 0, 0)
            try:
                for i in range(n):
                    num = pairs[i].num
                    d = pairs[i].dets
                    if nms and num:
                        self.lib.do_nms_sort(d, num, len(names), nms)
                    results.append(self._extract(d, num, names, thresh))
            finally:
                self.lib.free_batch_detections(pairs, n)
        return results

    # ---------- 内部 ----------
    @staticmethod
    def _extract(dets, n: int, names: List[str], thresh: float) -> List[tuple]:
        out = []
        for j in range(n):
            for idx, name in enumerate(names):
                p = dets[j].prob[idx]
                if p > thresh:
                    b = dets[j].bbox
                    out.append((name, float(p), (b.x, b.y, b.w, b.h)))
        # 按数值置信度降序（修复 B-2：darknet.py 用字符串排序）
        out.sort(key=lambda t: -t[1])
        return out

    def init_cpu(self) -> None:
        try:
            self.lib.init_cpu()
        except Exception:                                # noqa: BLE001
            pass

    def try_set_device(self, gpu_index: int = 0) -> bool:
        """尝试切换 GPU；CPU 版库调用会失败，用于运行时探测。"""
        try:
            self.lib.cuda_set_device(gpu_index)
            return True
        except Exception:                                # noqa: BLE001
            return False

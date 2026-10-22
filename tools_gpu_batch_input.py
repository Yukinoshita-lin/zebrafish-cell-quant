#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_gpu_batch_input.py - 比对两条路径喂给网络输入层的宿主缓冲区内容

detect()      : make_image + copy_image_from_bytes(img, t0.tobytes())  → img.data
detect_batch(): numpy float32 CHW (t0.transpose(2,0,1))                → flat

两者都应是 416*416*3 个 0-255 float、CHW 布局。若内容不一致 → 输入构造是根因。
"""
from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from tools_gpu_verify import collect_images, load_tiles  # noqa: E402
from zebrafish_quant.native import NativeDarknet, FloatPtr  # noqa: E402

CFG = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell.cfg")
WEIGHTS = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell_last.weights")
GPU_DLL = os.path.join(ROOT, "darknet_gpu.dll")


def main() -> int:
    paths = collect_images(1)
    tiles = load_tiles(paths, 416, 416)
    w = h = 416
    t0 = tiles[0]
    print(f"tile: shape={t0.shape} dtype={t0.dtype} "
          f"min={t0.min()} max={t0.max()}")

    gpu = NativeDarknet(GPU_DLL)
    gpu.try_set_device(0)
    from zebrafish_quant.inference import _stage_to_ascii
    staged = _stage_to_ascii([CFG, WEIGHTS])
    cfg_a, wts_a = staged[0]
    net = gpu.load_net(cfg_a, wts_a, batch_size=1)

    # 路径 1：make_image + copy_image_from_bytes
    img = gpu.lib.make_image(w, h, 3)
    gpu.lib.copy_image_from_bytes(img, t0.tobytes())
    n = w * h * 3
    buf1 = np.ctypeslib.as_array(ctypes_array_from(img.data, n)).copy()

    # 路径 2：numpy float32 CHW
    flat = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32).reshape(-1)

    print(f"buf1 (copy_image_from_bytes): n={buf1.size} min={buf1.min():.1f} "
          f"max={buf1.max():.1f} sum={buf1.sum():.1f}")
    print(f"flat (numpy transpose)      : n={flat.size} min={flat.min():.1f} "
          f"max={flat.max():.1f} sum={flat.sum():.1f}")
    diff = np.abs(buf1 - flat)
    print(f"max|diff|={diff.max():.6f}  相等元素比例={np.mean(diff == 0):.6f}")
    if diff.max() > 0:
        idx = int(np.argmax(diff))
        print(f"  最大差异位置 {idx}: buf1={buf1[idx]} flat={flat[idx]}")
        # 看看是否是端序/布局问题：把 flat 当 HWC 解释后与 buf1 比
        flat_hwc = np.ascontiguousarray(t0, dtype=np.float32).reshape(-1)
        d2 = np.abs(buf1 - flat_hwc)
        print(f"  [对照] flat 按 HWC 解释: max|diff|={d2.max():.6f}")
    gpu.lib.free_image(img)
    return 0


def ctypes_array_from(ptr, n):
    import ctypes as ct
    return (ct.c_float * n).from_address(ct.cast(ptr, ct.c_void_p).value or 0)


if __name__ == "__main__":
    raise SystemExit(main())

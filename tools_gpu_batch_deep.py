#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
tools_gpu_batch_deep.py - 同进程内判别：network_predict_batch 坏在前向还是取框

流程（全部 batch=1 加载，杜绝切批干扰）：
  1. network_predict_ptr(net, img)          → 前向 #1
  2. get_network_boxes(...) + nms + extract → 参照 A（应=56）
  3. network_predict_batch(net, img, 1,...) → 前向 #2 + 批量取框 → B
  4. get_network_boxes(...) 再读一次前向 #2 的输出 → C
  5. network_predict_ptr(net, img)          → 前向 #3
  6. get_network_boxes(...)                  → D（检验前向是否被 #2 破坏）

判定：
  B=27 且 C=56 → 批量取框函数(make/fill_network_boxes_batch)有 bug
  B=27 且 C=27 → network_predict_batch 内部的前向被破坏（与 #1 的前向不同）
  D≠56        → 前向 #2 破坏了网络状态（如 workspace/缓冲区）
"""
from __future__ import annotations

import ctypes as ct
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CFG = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell.cfg")
WEIGHTS = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell_last.weights")
DATA = os.path.join(ROOT, "dataset", "cell.data")
GPU_DLL = os.path.join(ROOT, "darknet_gpu.dll")

from tools_gpu_verify import match_sets, collect_images, load_tiles  # noqa: E402
from zebrafish_quant.native import (NativeDarknet, IMAGE, FloatPtr,  # noqa: E402
                                    DETECTIONPtr)
from zebrafish_quant.inference import _stage_to_ascii  # noqa: E402


def boxes_via_get_network_boxes(gpu, net, names, w, h, thresh, nms):
    """复刻 detect() 的取框路径（不经 network_predict，只用取框函数）。"""
    pnum = ct.pointer(ct.c_int(0))
    dets = gpu.lib.get_network_boxes(net, w, h, thresh, 0.5, None, 0, pnum, 0)
    n = pnum[0]
    if nms:
        gpu.lib.do_nms_sort(dets, n, len(names), nms)
    out = gpu._extract(dets, n, names, thresh)
    gpu.lib.free_detections(dets, n)
    return out


def main() -> int:
    paths = collect_images(1)
    tiles = load_tiles(paths, 416, 416)
    w = h = 416
    thresh, nms = 0.30, 0.20

    gpu = NativeDarknet(GPU_DLL)
    gpu.try_set_device(0)
    cfg_a, data_a, wts_a = _stage_to_ascii([CFG, DATA, WEIGHTS])[0]
    n_cls, names = gpu.metadata(data_a)
    net = gpu.load_net(cfg_a, wts_a, batch_size=1)   # batch=1，无切批
    print("net loaded batch=1")

    t0 = tiles[0]

    # 1+2: 前向 #1 + 非批量取框 → 参照 A
    img = gpu.lib.make_image(w, h, 3)
    gpu.lib.copy_image_from_bytes(img, t0.tobytes())
    gpu.lib.network_predict_ptr(net, img.data)
    A = boxes_via_get_network_boxes(gpu, net, names, w, h, thresh, nms)
    print(f"A (predict#1 + get_network_boxes)      : n={len(A)}")

    # 3: network_predict_batch(n=1) → 前向 #2 + 批量取框 → B
    arr = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32) / 255.0
    flat = np.ascontiguousarray(arr, dtype=np.float32)
    ptr = flat.ctypes.data_as(FloatPtr)
    batch_img = IMAGE(w, h, 3, ptr)
    pairs = gpu.lib.network_predict_batch(net, batch_img, 1, w, h,
                                          thresh, 0.5, None, 0, 0)
    try:
        num = pairs[0].num
        d = pairs[0].dets
        if nms and num:
            gpu.lib.do_nms_sort(d, num, len(names), nms)
        B = gpu._extract(d, num, names, thresh)
    finally:
        gpu.lib.free_batch_detections(pairs, 1)
    print(f"B (predict_batch n=1)                  : n={len(B)}")

    # 4: 前向 #2 的输出还在，用非批量取框再读 → C
    C = boxes_via_get_network_boxes(gpu, net, names, w, h, thresh, nms)
    print(f"C (前向#2 输出 + get_network_boxes)     : n={len(C)}")

    # 5+6: 前向 #3 + 非批量取框 → D
    gpu.lib.network_predict_ptr(net, img.data)
    D = boxes_via_get_network_boxes(gpu, net, names, w, h, thresh, nms)
    print(f"D (predict#3 + get_network_boxes)      : n={len(D)}")

    # 交叉比对
    print(f"\nmatched: A-B={len(match_sets(A, B))}  A-C={len(match_sets(A, C))}  "
          f"A-D={len(match_sets(A, D))}  B-C={len(match_sets(B, C))}")
    gpu.lib.free_image(img)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

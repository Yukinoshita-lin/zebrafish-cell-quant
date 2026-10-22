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
tools_gpu_batch_bisect.py - 二分定位 network_predict_batch(n=1) 结果错误的原因

实验矩阵（每个组合都是全新进程，避免相互污染）：
  M1: batch=1 加载 → detect() 单张            （已知正确，作参照）
  M2: batch=1 加载 → detect_batch(n=1)        （不触发 set_batch_network）
  M3: batch=8 加载 → detect() 单张            （已知正确：切批到1再还原）
  M4: batch=8 加载 → detect_batch(n=1)        （触发 set_batch_network 8→1）
  M5: batch=8 加载 → detect() 一次后 → detect_batch(n=1)  （模拟验证脚本时序）

判定：
  M2 正确 & M4 错误 → set_batch_network 切批是根因
  M2 错误           → 批量取框路径(make/fill_network_boxes_batch)本身有 bug
  M4 正确 & M5 错误 → 反复切批造成状态损坏
"""
from __future__ import annotations

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
from zebrafish_quant.native import NativeDarknet  # noqa: E402
from zebrafish_quant.inference import _stage_to_ascii  # noqa: E402


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "M4"
    paths = collect_images(2)
    tiles = load_tiles(paths, 416, 416)
    w = h = 416
    thresh, nms = 0.30, 0.20

    gpu = NativeDarknet(GPU_DLL)
    gpu.try_set_device(0)
    cfg_a, data_a, wts_a = _stage_to_ascii([CFG, DATA, WEIGHTS])[0]
    n_cls, names = gpu.metadata(data_a)

    t0 = tiles[0]
    ref = None

    if mode in ("M1", "M3"):
        net = gpu.load_net(cfg_a, wts_a, batch_size=1 if mode == "M1" else 8)
        r = gpu.detect(net, names, t0.tobytes(), w, h, thresh=thresh, nms=nms)
        print(f"[{mode}] detect(): n={len(r)}")
        return 0

    if mode == "M2":
        net = gpu.load_net(cfg_a, wts_a, batch_size=1)
        arr = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32)
        r = gpu.detect_batch(net, names, arr, w, h, 1, thresh=thresh, nms=nms)
        print(f"[M2] detect_batch(n=1, batch=1 load): n={len(r[0])}")
        return 0

    if mode == "M4":
        net = gpu.load_net(cfg_a, wts_a, batch_size=8)
        arr = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32)
        r = gpu.detect_batch(net, names, arr, w, h, 1, thresh=thresh, nms=nms)
        print(f"[M4] detect_batch(n=1, batch=8 load, 无预热): n={len(r[0])}")
        return 0

    if mode == "M5":
        net = gpu.load_net(cfg_a, wts_a, batch_size=8)
        r_d = gpu.detect(net, names, t0.tobytes(), w, h, thresh=thresh, nms=nms)
        print(f"[M5] detect() first: n={len(r_d)}")
        arr = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32)
        r = gpu.detect_batch(net, names, arr, w, h, 1, thresh=thresh, nms=nms)
        print(f"[M5] detect_batch(n=1) after: n={len(r[0])} "
              f"matched={len(match_sets(r[0], r_d))}")
        return 0

    if mode == "M6":
        # batch=1 加载，但走 network_predict_batch 的 C 路径且 batch_size=1
        # （net->batch 已为 1，不触发 set_batch_network）—— 同 M2，保留
        pass

    print(f"unknown mode {mode}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

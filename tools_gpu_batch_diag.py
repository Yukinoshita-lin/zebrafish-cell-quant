#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_gpu_batch_diag.py - GPU batch=8 批量推理异常定位（诊断脚本）

现象：tools_gpu_verify.py 中 GPU 批量(batch=8) 检出 234 框，GPU/CPU 单张各 617 框，
且 GPU批量 vs 单张 匹配数 = 0（IoU>=0.5 一个都匹配不上）。

判别实验设计：
  A. 「同图重复」：把 tile[0] 复制 8 份做 batch 前向。
     - 若 8 个槽位结果都与 ref_single[0] 一致 → 批量前向+取框链路正确，
       问题在验证脚本的批量输入构造（布局/数据类型）。
     - 若仅槽位 0 一致、其余崩坏 → 批量前向对 batch>1 的图计算错误。
     - 若全部不一致 → 连 batch=8 的首图都错（输入推送/缓冲区问题）。
  B. 「异图混合」：8 张不同图做 batch 前向，逐槽位与各自单张结果比对，
     打印每槽位框数，定位是前向错还是取框索引错。
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
IMAGES_DIR = os.path.join(ROOT, "dataset", "images")
GPU_DLL = os.path.join(ROOT, "darknet_gpu.dll")

sys.path.insert(0, os.path.join(ROOT, "tools"))  # noqa: E402
from tools_gpu_verify import iou, match_sets, collect_images, load_tiles  # noqa: E402
from zebrafish_quant.native import NativeDarknet  # noqa: E402
from zebrafish_quant.inference import _stage_to_ascii  # noqa: E402


def main() -> int:
    paths = collect_images(8)
    tiles = load_tiles(paths, 416, 416)
    w = h = 416

    gpu = NativeDarknet(GPU_DLL)
    gpu.try_set_device(0)
    cfg_a, data_a, wts_a = _stage_to_ascii([CFG, DATA, WEIGHTS])[0]
    n_cls, names = gpu.metadata(data_a)
    print(f"classes={n_cls} names={names}")

    # 以 batch=8 加载（与验证脚本一致）
    net = gpu.load_net(cfg_a, wts_a, batch_size=8)
    print(f"net loaded: batch=8")

    thresh, nms = 0.30, 0.20

    # ---- 单张参考（每张一次，内部切批到 1 再还原）----
    ref = []
    for t in tiles:
        ref.append(gpu.detect(net, names, t.tobytes(), w, h, thresh=thresh, nms=nms))
    print("ref single box counts:", [len(r) for r in ref])

    # ---- 实验 A：同图重复 8 次 ----
    print("\n=== 实验 A：tile[0] 重复 8 次 batch 前向 ===")
    t0 = tiles[0]
    planes = [np.ascontiguousarray(t0.transpose(2, 0, 1))] * 8
    arr = np.ascontiguousarray(np.concatenate(planes, axis=0),
                               dtype=np.float32) / 255.0   # 0-1 约定
    print(f"input arr: shape={arr.shape} dtype={arr.dtype} "
          f"min={arr.min():.3f} max={arr.max():.3f}")
    resA = gpu.detect_batch(net, names, arr, w, h, 8, thresh=thresh, nms=nms)
    print("batch-same box counts:", [len(r) for r in resA])
    for i, r in enumerate(resA):
        m = match_sets(r, ref[0])
        print(f"  slot {i}: n={len(r):3d}  vs ref[0](n={len(ref[0])})  "
              f"matched={len(m)}")

    # ---- 实验 B：8 张异图 ----
    print("\n=== 实验 B：8 张不同图 batch 前向 ===")
    planes = [np.ascontiguousarray(t.transpose(2, 0, 1)) for t in tiles]
    arr = np.ascontiguousarray(np.concatenate(planes, axis=0),
                               dtype=np.float32) / 255.0   # 0-1 约定
    resB = gpu.detect_batch(net, names, arr, w, h, 8, thresh=thresh, nms=nms)
    print("batch-mixed box counts:", [len(r) for r in resB])
    for i, (r, rf) in enumerate(zip(resB, ref)):
        m = match_sets(r, rf)
        print(f"  slot {i}: n={len(r):3d}  vs ref[{i}](n={len(rf)})  "
              f"matched={len(m)}")

    # ---- 实验 C：n=1 走 batch 路径（切批到 1 的 network_predict_batch）----
    print("\n=== 实验 C：单图走 network_predict_batch(n=1) ===")
    t0 = tiles[0]
    arr1 = np.ascontiguousarray(t0.transpose(2, 0, 1), dtype=np.float32)
    resC = gpu.detect_batch(net, names, arr1, w, h, 1, thresh=thresh, nms=nms)
    print("n=1 via batch path:", [len(r) for r in resC],
          " matched vs ref[0]:", len(match_sets(resC[0], ref[0])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

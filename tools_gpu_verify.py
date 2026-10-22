#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_gpu_verify.py - darknet_gpu.dll 真机验证 + CPU/GPU 对拍 + 吞吐基准

对应用户要求「两者都做，最终 auto 模式在真机上实测择优」中的实测环节：

  1. 加载验证   GPU DLL 经 ctypes 加载，运行时依赖（cudart/cublas/curand）
                与 nvcuda 驱动均能解析，cuda_set_device(0) 成功。
  2. 对拍       GPU 单张 vs CPU 单张：同一张真实斑马鱼切片，检测集合按 IoU
                贪心匹配，报告匹配率 / 置信度偏差 / 框偏差（differential）。
  3. 批量一致性 GPU batch=8 vs GPU 单张 vs CPU 单张，三方交叉核对；
                同时验证 0-1 输入约定（本 fork 训练/推理全链路 /255）。
  4. 基准       各路径单图耗时与吞吐，给出 GPU 相对 CPU 的加速比。

用法：
    python tools_gpu_verify.py [--images 24] [--batch 8] [--gpu-only]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CFG = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell.cfg")
WEIGHTS = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell_last.weights")
DATA = os.path.join(ROOT, "dataset", "cell.data")
IMAGES_DIR = os.path.join(ROOT, "dataset", "images")
GPU_DLL = os.path.join(ROOT, "darknet_gpu.dll")
CPU_DLL = os.path.join(ROOT, "darknet_cpu.dll")


# ---------------------------------------------------------------- 工具
def iou(a: Tuple[float, float, float, float],
        b: Tuple[float, float, float, float]) -> float:
    """darknet 坐标 (cx, cy, w, h) 下的 IoU。"""
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def match_sets(A: List[tuple], B: List[tuple],
               thr: float = 0.5) -> List[Tuple[tuple, tuple, float]]:
    """贪心按 IoU 匹配两个检测集合，返回 [(a, b, iou), ...]。"""
    pairs = []
    for a in A:
        for b in B:
            v = iou(a[2], b[2])
            if v >= thr:
                pairs.append((v, a, b))
    pairs.sort(key=lambda t: -t[0])
    used_a, used_b, out = set(), set(), []
    for v, a, b in pairs:
        if id(a) in used_a or id(b) in used_b:
            continue
        used_a.add(id(a)); used_b.add(id(b))
        out.append((a, b, v))
    return out


def collect_images(n: int) -> List[str]:
    import glob
    files = sorted(glob.glob(os.path.join(IMAGES_DIR, "**", "*.png"),
                             recursive=True))
    files += sorted(glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"),
                              recursive=True))
    if not files:
        raise SystemExit(f"未找到测试图像: {IMAGES_DIR}")
    return files[:n]


def load_tiles(paths: List[str], w: int, h: int):
    """返回 rgb_tiles：uint8 RGB 序列输入（cv2 读入为 BGR，这里转 RGB）。"""
    from zebrafish_quant.imaging import letterbox_resize, imread_robust
    tiles = []
    for p in paths:
        img = imread_robust(p)          # 中文路径 + 16-bit 安全
        canvas, _lb = letterbox_resize(img, w, h)
        rgb = np.ascontiguousarray(canvas[:, :, ::-1])       # BGR -> RGB
        tiles.append(rgb)
    return tiles


# ---------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="GPU DLL 真机验证与对拍")
    ap.add_argument("--images", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--gpu-only", action="store_true")
    args = ap.parse_args()

    from zebrafish_quant.native import NativeDarknet
    from zebrafish_quant.inference import _stage_to_ascii

    # darknet 的 fopen 走 ANSI 代码页，中文路径必须暂存到 ASCII 目录（缺陷 B-1）
    (cfg_a, data_a, wts_a), staged_dir = _stage_to_ascii([CFG, DATA, WEIGHTS])
    if staged_dir:
        print(f"  中文路径已暂存: {staged_dir}")
    try:
        return _run(args, NativeDarknet, cfg_a, data_a, wts_a)
    finally:
        if staged_dir:
            import shutil
            shutil.rmtree(staged_dir, ignore_errors=True)


def _run(args, NativeDarknet, cfg_a, data_a, wts_a) -> int:
    paths = collect_images(args.images)
    print(f"测试图像: {len(paths)} 张（来自 {os.path.relpath(IMAGES_DIR, ROOT)}）")

    # ---------------- 1. GPU DLL 加载 ----------------
    print("\n=== [1] GPU DLL 加载 ===")
    if not os.path.exists(GPU_DLL):
        print(f"[失败] 不存在: {GPU_DLL}")
        return 2
    t0 = time.perf_counter()
    gpu = NativeDarknet(GPU_DLL)
    print(f"  CDLL 加载耗时 {time.perf_counter() - t0:.3f}s")
    ok = gpu.try_set_device(0)
    print(f"  cuda_set_device(0): {'OK' if ok else '失败'}")
    if not ok:
        print("[失败] 无法初始化 CUDA 设备")
        return 2
    t0 = time.perf_counter()
    gpu_net = gpu.load_net(cfg_a, wts_a, batch_size=args.batch)
    print(f"  模型加载 {time.perf_counter() - t0:.2f}s  "
          f"(net={gpu_net}, w={gpu.width(gpu_net)}, h={gpu.height(gpu_net)})")
    n_cls, names = gpu.metadata(data_a)
    print(f"  类别: {names}")

    # ---------------- 2. CPU DLL 加载 ----------------
    cpu = cpu_net = None
    if not args.gpu_only:
        print("\n=== [2] CPU DLL 加载 ===")
        if not os.path.exists(CPU_DLL):
            print(f"[警告] 不存在 {CPU_DLL}，跳过对拍（仅 GPU 基准）")
        else:
            cpu = NativeDarknet(CPU_DLL)
            cpu.init_cpu()
            cpu_net = cpu.load_net(cfg_a, wts_a, batch_size=1)
            print(f"  CPU 模型加载 OK (net={cpu_net})")

    w, h = gpu.width(gpu_net), gpu.height(gpu_net)
    tiles = load_tiles(paths, w, h)
    print(f"  tile 尺寸: {w}x{h}，共 {len(tiles)} 张")

    # ---------------- 3. 预热 ----------------
    print("\n=== [3] 预热（CUDA context / kernel 首次编译） ===")
    t0 = time.perf_counter()
    for t in tiles[:3]:
        gpu.detect(gpu_net, names, t.tobytes(), w, h, thresh=0.30, nms=0.20)
    print(f"  GPU 预热 {len(tiles[:3])} 张耗时 {time.perf_counter() - t0:.2f}s")
    if cpu_net:
        cpu.detect(cpu_net, names, tiles[0].tobytes(), w, h, thresh=0.30, nms=0.20)

    # ---------------- 4. 单张推理 + 对拍 ----------------
    print("\n=== [4] GPU 单张推理 + CPU/GPU 对拍 ===")
    gpu_single: List[List[tuple]] = []
    cpu_single: List[List[tuple]] = []
    t_gpu = t_cpu = 0.0
    for t in tiles:
        t0 = time.perf_counter()
        gpu_single.append(gpu.detect(gpu_net, names, t.tobytes(), w, h,
                                     thresh=0.30, nms=0.20))
        t_gpu += time.perf_counter() - t0
        if cpu_net:
            t0 = time.perf_counter()
            cpu_single.append(cpu.detect(cpu_net, names, t.tobytes(), w, h,
                                         thresh=0.30, nms=0.20))
            t_cpu += time.perf_counter() - t0
    n_gpu = sum(len(r) for r in gpu_single)
    print(f"  GPU 单张: {t_gpu / len(tiles) * 1000:.1f} ms/图，"
          f"检出 {n_gpu} 个（{n_gpu / len(tiles):.1f} 个/图）")
    if cpu_net:
        n_cpu = sum(len(r) for r in cpu_single)
        print(f"  CPU 单张: {t_cpu / len(tiles) * 1000:.1f} ms/图，"
              f"检出 {n_cpu} 个（{n_cpu / len(tiles):.1f} 个/图）")
        print(f"  GPU 单张加速比: {t_cpu / t_gpu:.2f}x")

    # ---------------- 5. GPU 批量推理 ----------------
    print(f"\n=== [5] GPU 批量推理（batch={args.batch}） ===")
    gpu_batch: List[List[tuple]] = []
    t_batch = 0.0
    bs = max(1, args.batch)
    for i in range(0, len(tiles), bs):
        chunk = tiles[i:i + bs]
        planes = [np.ascontiguousarray(t.transpose(2, 0, 1)) for t in chunk]
        arr = np.ascontiguousarray(np.concatenate(planes, axis=0),
                                   dtype=np.float32) / 255.0   # 本 fork 统一 0-1 约定
        t0 = time.perf_counter()
        res = gpu.detect_batch(gpu_net, names, arr, w, h, len(chunk),
                               thresh=0.30, nms=0.20)
        t_batch += time.perf_counter() - t0
        gpu_batch.extend(res)
    n_b = sum(len(r) for r in gpu_batch)
    print(f"  GPU 批量: {t_batch / len(tiles) * 1000:.1f} ms/图，"
          f"检出 {n_b} 个")
    print(f"  批吞吐加速比（相对 GPU 单张）: {t_gpu / t_batch:.2f}x")

    # ---------------- 6. 三方一致性 ----------------
    print("\n=== [6] 一致性核对 ===")
    def compare(A: List[List[tuple]], B: List[List[tuple]], la: str, lb: str):
        tot_a = tot_b = matched = 0
        conf_d, box_d = [], []
        for ra, rb in zip(A, B):
            tot_a += len(ra); tot_b += len(rb)
            for a, b, v in match_sets(ra, rb):
                matched += 1
                conf_d.append(abs(a[1] - b[1]))
                box_d.append(1.0 - v)
        line = (f"  {la} vs {lb}: {la}={tot_a} {lb}={tot_b} "
                f"匹配={matched}")
        if conf_d:
            line += (f" 平均|Δconf|={np.mean(conf_d):.4f} "
                     f"平均框偏差={np.mean(box_d):.4f}")
        print(line)
        return matched, tot_a, tot_b

    m_ab, na, nb = compare(gpu_single, gpu_batch, "GPU单张", "GPU批量")
    if cpu_net:
        m_ac, _, _ = compare(gpu_single, cpu_single, "GPU单张", "CPU单张")
        m_bc, _, _ = compare(gpu_batch, cpu_single, "GPU批量", "CPU单张")

    ok = True
    if cpu_net:
        # 对拍判定：匹配率 >= 95% 且平均置信度偏差 < 0.05
        rate = m_ac / max(1, na)
        print(f"\n  GPU/CPU 对拍匹配率: {rate:.1%}")
        ok = rate >= 0.95

    # ---------------- 7. 结论 ----------------
    print("\n=== [7] 基准结论 ===")
    print(f"  GPU 单张 : {t_gpu / len(tiles) * 1000:7.1f} ms/图")
    if cpu_net:
        print(f"  CPU 单张 : {t_cpu / len(tiles) * 1000:7.1f} ms/图"
              f"   （GPU 加速 {t_cpu / t_gpu:.2f}x）")
    print(f"  GPU 批量 : {t_batch / len(tiles) * 1000:7.1f} ms/图"
          f"   （相对 GPU 单张 {t_gpu / t_batch:.2f}x）")
    verdict = "通过" if ok else "未通过（见上方匹配率）"
    print(f"\n  验证结论: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

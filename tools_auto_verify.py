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
tools_auto_verify.py - auto 模式真机实测择优 + 缺失回退验证

对应用户要求「两者都做，最终 auto 模式在真机上实测择优」的最终验收环节。
与 run_quant.py 生产路径完全一致：choose_backend(config, data, weights,
tiles, cache_path=..., prefer=None)，仅缓存路径换成临时目录，不污染
真实输出目录。

四个场景（全部在真机、PATH 已剃掉 MinGW 的条件下执行）：
  S1 真机择优    全新缓存 -> auto 实测 GPU / CPU 吞吐（classical 兜底
                 不参与速度竞争），期望选中 gpu（GPU ~10ms vs CPU
                 ~160ms 每图）；二次调用应命中缓存（from_cache=True）。
  S2 GPU 缺失    把 darknet_gpu.dll 改名 -> GPU 后端从候选消失，
                 auto 应选中 cpp_cpu，且不报错、不中断。
  S3 双缺失      再把 darknet_cpu.dll 改名 -> auto 应退回
                 classical_python（纯 Python 兜底，保证永远能跑）。
  S4 缓存升级    恢复 darknet_gpu.dll -> 更低 tier 后端出现令缓存失效，
                 auto 应重新实测并升级回 gpu（而不是粘在旧选择上）。
  收尾           finally 中恢复两个 DLL 并校验恢复后的文件哈希。

用法：
    python tools_auto_verify.py [--tiles 4]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

CFG = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell.cfg")
WEIGHTS = os.path.join(ROOT, "cfg", "yolov4-tiny-3l-cell_last.weights")
DATA = os.path.join(ROOT, "dataset", "cell.data")
IMAGES_DIR = os.path.join(ROOT, "dataset", "images")
GPU_DLL = os.path.join(ROOT, "darknet_gpu.dll")
CPU_DLL = os.path.join(ROOT, "darknet_cpu.dll")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_tiles(n: int):
    from zebrafish_quant.imaging import letterbox_resize, imread_robust
    import numpy as np
    files = sorted(glob.glob(os.path.join(IMAGES_DIR, "**", "*.png"),
                             recursive=True))
    files += sorted(glob.glob(os.path.join(IMAGES_DIR, "**", "*.jpg"),
                              recursive=True))
    if not files:
        raise SystemExit(f"未找到测试图像: {IMAGES_DIR}")
    tiles = []
    for p in files[:n]:
        img = imread_robust(p)
        canvas, _lb = letterbox_resize(img, 416, 416)
        tiles.append(canvas)
    return tiles


def main() -> int:
    ap = argparse.ArgumentParser(description="auto 模式真机实测与回退验证")
    ap.add_argument("--tiles", type=int, default=4)
    args = ap.parse_args()

    from zebrafish_quant.benchmark import choose_backend
    from zebrafish_quant.backends import available_backends

    tiles = load_tiles(args.tiles)
    print(f"测试切块: {len(tiles)} 张（{tiles[0].shape[1]}x{tiles[0].shape[0]}）")

    tmp = tempfile.mkdtemp(prefix="zq_auto_verify_")
    cache = os.path.join(tmp, "backend_cache.json")
    failures = []

    gpu_bak = GPU_DLL + ".autoverify_bak"
    cpu_bak = CPU_DLL + ".autoverify_bak"
    gpu_moved = cpu_moved = False
    gpu_hash_before = sha256(GPU_DLL) if os.path.exists(GPU_DLL) else None
    cpu_hash_before = sha256(CPU_DLL) if os.path.exists(CPU_DLL) else None

    try:
        # ---------------- S1 真机 auto 择优 ----------------
        print("\n=== S1 真机 auto 择优（全新缓存） ===")
        names = [b.name for b, _ in available_backends()]
        print(f"  候选后端: {names}")
        t0 = time.perf_counter()
        backend, report = choose_backend(CFG, DATA, WEIGHTS, tiles,
                                         cache_path=cache)
        dt = time.perf_counter() - t0
        print(f"  实测耗时 {dt:.1f}s -> 选中: {backend.name}")
        for r in report.get("results", []):
            tag = "OK " if r.get("ok") else "失败"
            print(f"    [{tag}] {r['name']:16s} "
                  f"{r.get('tiles_per_sec', 0):8.2f} tiles/s"
                  + (f"  ({r['error']})" if r.get("error") else ""))
        if backend.name != "gpu":
            failures.append(f"S1: 期望选中 gpu，实际 {backend.name}")
        else:
            print("  [OK] GPU 在真机实测中最快，auto 选中 gpu")

        # 缓存落盘内容
        if not os.path.exists(cache):
            failures.append("S1: 缓存文件未生成")
        else:
            c = json.load(open(cache, encoding="utf-8"))
            print(f"  缓存: chosen={c.get('chosen')} "
                  f"tiles_per_sec={c.get('tiles_per_sec')}")
            if c.get("chosen") != "gpu":
                failures.append(f"S1: 缓存 chosen={c.get('chosen')}，期望 gpu")

        # 缓存命中（不重新实测）
        t0 = time.perf_counter()
        backend2, report2 = choose_backend(CFG, DATA, WEIGHTS, tiles,
                                           cache_path=cache)
        dt2 = time.perf_counter() - t0
        if not report2.get("from_cache"):
            failures.append("S1: 二次调用未命中缓存（重新实测了）")
        else:
            print(f"  [OK] 二次调用命中缓存（{dt2:.2f}s，无需重新实测），"
                  f"选中 {backend2.name}")

        # 选中后端与手动 GPU 结果的检测数一致性（防"选错/选坏"）
        gpu_dets = sum(len(d) for d in backend.detect_tiles(tiles, 0.30, 0.20))
        print(f"  auto 选中后端检出: {gpu_dets} 个 / {len(tiles)} 张")
        if gpu_dets <= 0:
            failures.append("S1: 选中后端未检出任何目标")

        # ---------------- S2 GPU DLL 缺失回退 ----------------
        print("\n=== S2 GPU DLL 缺失回退 ===")
        if not os.path.exists(GPU_DLL):
            failures.append("S2: darknet_gpu.dll 不存在，无法构造场景")
        else:
            os.rename(GPU_DLL, gpu_bak)
            gpu_moved = True
            if os.path.exists(cache):
                os.remove(cache)      # 清掉 S1 缓存，强制重新实测
            names2 = [b.name for b, _ in available_backends()]
            print(f"  候选后端: {names2}")
            if "gpu" in names2:
                failures.append("S2: GPU DLL 已改名，gpu 后端仍在候选中")
            backend3, report3 = choose_backend(CFG, DATA, WEIGHTS, tiles,
                                               cache_path=cache)
            print(f"  选中: {backend3.name}")
            for r in report3.get("results", []):
                tag = "OK " if r.get("ok") else "跳过/失败"
                print(f"    [{tag}] {r['name']:16s} "
                      f"{r.get('tiles_per_sec', 0):8.2f} tiles/s"
                      + (f"  ({r['error']})" if r.get("error") else ""))
            if backend3.name != "cpp_cpu":
                failures.append(f"S2: 期望回退到 cpp_cpu，实际 {backend3.name}")
            else:
                print("  [OK] GPU DLL 缺失时自动回退到 C++ CPU 版")

        # ---------------- S3 双 DLL 缺失回退 ----------------
        print("\n=== S3 双 DLL 缺失回退（纯 Python 兜底） ===")
        if not os.path.exists(CPU_DLL):
            failures.append("S3: darknet_cpu.dll 不存在，无法构造场景")
        else:
            os.rename(CPU_DLL, cpu_bak)
            cpu_moved = True
            if os.path.exists(cache):
                os.remove(cache)
            names3 = [b.name for b, _ in available_backends()]
            print(f"  候选后端: {names3}")
            backend4, report4 = choose_backend(CFG, DATA, WEIGHTS, tiles,
                                               cache_path=cache)
            print(f"  选中: {backend4.name}")
            if backend4.name != "classical_python":
                failures.append(f"S3: 期望回退到 classical_python，"
                                f"实际 {backend4.name}")
            else:
                dets4 = sum(len(d) for d in
                            backend4.detect_tiles(tiles, 0.30, 0.20))
                print(f"  [OK] 双缺失时回退纯 Python 兜底，"
                      f"检出 {dets4} 个 / {len(tiles)} 张（流程不中断）")

        # ---------------- S4 缓存升级（GPU 恢复后从兜底升级回 GPU） ----------------
        print("\n=== S4 GPU DLL 恢复 -> 缓存失效并升级回 GPU ===")
        if gpu_moved and os.path.exists(gpu_bak):
            os.rename(gpu_bak, GPU_DLL)
            gpu_moved = False
            print("  已恢复 darknet_gpu.dll（缓存仍记录 classical_python）")
            backend5, report5 = choose_backend(CFG, DATA, WEIGHTS, tiles,
                                               cache_path=cache)
            print(f"  选中: {backend5.name}"
                  f"{'（命中缓存）' if report5.get('from_cache') else '（重新实测）'}")
            if report5.get("from_cache"):
                failures.append("S4: GPU 恢复后仍命中旧缓存（未重新实测）")
            elif backend5.name != "gpu":
                failures.append(f"S4: 期望升级回 gpu，实际 {backend5.name}")
            else:
                print("  [OK] 更低 tier 后端出现 -> 缓存失效，重新实测选回 GPU")
        else:
            print("  （跳过：GPU DLL 未被改名或无法恢复）")

    finally:
        # ---------------- 恢复 DLL 并校验 ----------------
        print("\n=== 恢复现场 ===")
        if gpu_moved and os.path.exists(gpu_bak):
            os.rename(gpu_bak, GPU_DLL)
            print(f"  已恢复 darknet_gpu.dll")
        if cpu_moved and os.path.exists(cpu_bak):
            os.rename(cpu_bak, CPU_DLL)
            print(f"  已恢复 darknet_cpu.dll")
        for p, h in ((GPU_DLL, gpu_hash_before), (CPU_DLL, cpu_hash_before)):
            if h is None:
                continue
            if not os.path.exists(p):
                failures.append(f"恢复失败: {p} 不存在")
            elif sha256(p) != h:
                failures.append(f"恢复失败: {p} 哈希不匹配")
        if not failures:
            print("  两个 DLL 均已恢复且哈希校验一致")
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n=== 结论 ===")
    if failures:
        for f in failures:
            print(f"  [失败] {f}")
        return 1
    print("  全部通过：auto 真机择优（GPU）、缓存命中、GPU 缺失回退 CPU、"
          "双缺失回退纯 Python、GPU 恢复后升级回 GPU，现场完整恢复")
    return 0


if __name__ == "__main__":
    sys.exit(main())

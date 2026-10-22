#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_quant.py - 斑马鱼肠道炎症细胞自动定量（双加速路径 + 自动回退）

三种后端，自动择优、自动回退：
    gpu                darknet_gpu.dll / libdarknet_gpu.so  (CUDA + CuDNN，批量推理)
    cpp_cpu            darknet_cpu.dll / libdarknet_cpu.so  (C++ 原生 + AVX + OpenMP)
    classical_python   无需任何 DLL                         (兜底，保证永远能跑)

默认 auto：实测各可用【原生】后端（gpu / cpp_cpu）吞吐后选最快，结果缓存到
<output>/backend_cache.json；classical_python 是兜底，不参与速度竞争（精度远低于
YOLO，仅在原生后端全部不可用时启用，也可用 --backend 强制选择）。
硬件缺失（无 GPU）、CUDA 库缺失、DLL 缺失 —— 均自动降级，不会中断。

用法：
    python run_quant.py --input "F:/AI斑马鱼/MD8-20" --output out ^
        --config cfg/yolov4-tiny-3l-cell.cfg --data dataset/cell.data ^
        --weights backup/cell_final.weights

    python run_quant.py ... --backend gpu     # 强制指定
    python run_quant.py ... --crosscheck      # 对拍验证各后端一致性
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="斑马鱼荧光图像肠道部位炎症细胞自动定量（双加速路径）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="图片文件夹/单图/清单txt")
    p.add_argument("--output", required=True, help="结果输出目录")
    p.add_argument("--config", required=True, help="darknet .cfg")
    p.add_argument("--data", required=True, help="darknet .data")
    p.add_argument("--weights", required=True, help="darknet .weights")

    p.add_argument("--backend", default="auto",
                   choices=["auto", "gpu", "cpp_cpu", "classical_python"],
                   help="auto=实测后选最快")
    p.add_argument("--batch-size", type=int, default=8, help="GPU 批量大小")
    p.add_argument("--crosscheck", action="store_true",
                   help="对拍：验证各后端结果一致性后退出")

    p.add_argument("--tile-size", type=int, default=416)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--global-nms", type=float, default=0.30,
                   help="跨块去重 IoU 阈值")
    p.add_argument("--thresh", type=float, default=0.30)
    p.add_argument("--nms", type=float, default=0.20,
                   help="检测 NMS；荧光细胞相邻，0.45 会严重漏检")
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument("--no-split", action="store_true", help="关闭粘连拆分")

    p.add_argument("--manual-masks", default="")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--log", default="")
    return p


def _collect_images(input_path: str) -> list:
    """收集图片；原始数据常按处理组分子目录，不足时递归兜底。"""
    from zebrafish_quant import imaging
    imgs = imaging.resolve_input(input_path)
    if len(imgs) <= 1 and os.path.isdir(input_path):
        imgs = []
        for root, _d, files in os.walk(input_path):
            for f in sorted(files):
                if f.lower().endswith(imaging.SUPPORTED_EXTS):
                    imgs.append(os.path.join(root, f))
        imgs.sort()
    return imgs


def _load_tiles_for_bench(input_path: str, tile_size: int, overlap: int, n: int = 8):
    """取少量真实切块作为基准样本（贴近真实数据分布）。"""
    import numpy as np
    from zebrafish_quant import imaging
    from zebrafish_quant.tiling import iter_tiles, tile_has_signal

    tiles = []
    for ip in _collect_images(input_path)[:3]:
        try:
            img = imaging.imread_robust(ip)
        except Exception:                                  # noqa: BLE001
            continue
        for t in iter_tiles(img, tile_size, overlap):
            if tile_has_signal(t.image):
                tiles.append(t.image)
                break
        if len(tiles) >= n:
            break
    if not tiles:                                          # 兜底：合成样本
        tiles = [np.zeros((tile_size, tile_size, 3), np.uint8) for _ in range(2)]
    return tiles[:n]


def _run_crosscheck(args) -> int:
    from zebrafish_quant.backends import available_backends
    from zebrafish_quant.crosscheck import crosscheck, format_report

    tiles = _load_tiles_for_bench(args.input, args.tile_size, args.overlap, n=6)
    loaded = []
    for b, why in available_backends():
        try:
            b.load(args.config, args.data, args.weights)
            loaded.append(b)
        except Exception as exc:                           # noqa: BLE001
            print(f"  [跳过对拍] {b.name}: {type(exc).__name__}: {exc}")

    print(f"\n可用后端: {[b.name for b in loaded]}")
    if len(loaded) < 2:
        print("\n至少需要 2 个可用后端才能对拍。")
        print("当前不足 2 个 —— 这正是『DLL 缺失自动回退』后的正常状态；")
        print("用 build_accel.py 编译出 darknet_cpu.dll / darknet_gpu.dll 后，")
        print("即可与经典后端对拍，验证加速未改变结果。")
        if loaded:                                         # 与自身对拍，验证管线自洽
            rep = crosscheck(loaded[0], loaded[0], tiles,
                             thresh=args.thresh, nms=args.nms)
            print("\n（自对拍，用于验证管线确定性）")
            print(format_report(rep))
            return 0 if rep["passed"] else 1
        return 1

    ok = True
    for i in range(len(loaded)):
        for j in range(i + 1, len(loaded)):
            rep = crosscheck(loaded[i], loaded[j], tiles,
                             thresh=args.thresh, nms=args.nms)
            print("\n" + format_report(rep))
            ok = ok and rep["passed"]
    print("\n对拍总体结论:", "OK 全部一致" if ok else "FAIL 存在不一致")
    return 0 if ok else 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    os.makedirs(args.output, exist_ok=True)

    from zebrafish_quant.backends import available_backends
    from zebrafish_quant.benchmark import choose_backend
    from zebrafish_quant.pipeline import configure_logging
    from zebrafish_quant.segment import ClassicalGutMask
    from zebrafish_quant.tiling_pipeline import TilingConfig, TilingPipeline

    configure_logging(args.log or os.path.join(args.output, "run.log"))
    os.environ["ZQ_GPU_BATCH"] = str(args.batch_size)

    if args.crosscheck:
        return _run_crosscheck(args)

    tiles = _load_tiles_for_bench(args.input, args.tile_size, args.overlap)

    prefer = None if args.backend == "auto" else args.backend
    cache = os.path.join(args.output, "backend_cache.json")

    names = [b.name for b, _ in available_backends()]
    if prefer and prefer not in names:
        print(f"[警告] 指定后端 {prefer} 不可用（库缺失或硬件缺失），"
              f"自动改用可用后端: {names}", file=sys.stderr)
        prefer = None

    try:
        backend, _report = choose_backend(
            args.config, args.data, args.weights, tiles,
            thresh=args.thresh, nms=args.nms,
            cache_path=cache, prefer=prefer)
    except RuntimeError as exc:
        print(f"[错误] 后端选择失败：{exc}", file=sys.stderr)
        return 3

    cfg = TilingConfig(
        tile_size=args.tile_size, overlap=args.overlap,
        global_nms=args.global_nms, thresh=args.thresh, nms=args.nms,
        min_confidence=args.min_confidence,
        split_adhesion=not args.no_split, resume=not args.no_resume,
        strict=args.strict)

    pipeline = TilingPipeline(backend, ClassicalGutMask(), cfg)

    def progress(i, n, name):
        if i % 5 == 0 or i == n:
            print(f"  进度 {i}/{n}", flush=True)

    try:
        rep = pipeline.process_folder(args.input, args.output,
                                      manual_mask_dir=args.manual_masks or None,
                                      progress=progress)
    finally:
        backend.close()

    print("\n===== 批量处理完成 =====")
    print(f"  后端          : {backend.name}")
    for k, v in rep.summary().items():
        print(f"  {k}: {v}")
    print(f"  结果 CSV: {os.path.join(args.output, 'quantification_results.csv')}")
    if rep.failures:
        print(f"  失败清单: {os.path.join(args.output, 'failed_images.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

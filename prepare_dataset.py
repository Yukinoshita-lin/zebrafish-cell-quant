#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
prepare_dataset.py - 由原始大图构建"原生分辨率切块"训练数据集

背景（实测结论，必须切块）：
    原始图 2752x2208，细胞中位直径仅 4.7~10.5 px。
    整图缩放到 416 后细胞只剩 0.7~1.6 px，检测器看不见。
    => 在原生分辨率上切 416x416 块，细胞保持原始尺寸。

产出：
    dataset/images/<stem>__t<idx>.png      切块图
    dataset/labels/<stem>__t<idx>.txt      YOLO 草稿标注（需人工复核！）
    dataset/cell.names / cell.data
    dataset/train.txt / val.txt / test.txt
    cfg/yolov4-tiny-3l-cell.cfg            已改为 1 类 + 重算 anchors

关键正确性：
    * 划分按**图像**而非切块（同一张图的切块绝不跨集），避免数据泄漏。
    * 只保留有信号的切块，丢弃纯背景块。
    * 输出目录独立于原始数据，绝不改动/写入原始图像目录。

用法：
    python prepare_dataset.py --source "F:/AI斑马鱼/MD8-20" --out dataset --max-images 20
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

from zebrafish_quant.autolabel import detect_fluorescent_blobs, to_yolo_lines
from zebrafish_quant.imaging import imread_robust, resolve_input
from zebrafish_quant.tiling import iter_tiles, tile_has_signal

CLASS_ID = 0
CLASS_NAME = "cell"
IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def collect_images(source: str) -> list:
    """递归收集图片。

    原始数据按"化合物分组/处理组"分子目录存放（如 MD8-20/Control/Snap-*.tif），
    所以必须递归；同时按路径排序保证划分可复现。
    """
    if os.path.isfile(source):
        return [source]
    if not os.path.isdir(source):
        raise SystemExit(f"源路径不存在: {source}")
    out = []
    for root, _dirs, files in os.walk(source):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                out.append(os.path.join(root, f))
    return sorted(out)


def kmeans_anchors(boxes: np.ndarray, n: int = 9, iters: int = 100,
                   seed: int = 0) -> list:
    """IoU 距离 k-means 聚类得到 anchors（darknet 需要）。

    boxes: (N,2) 的 (w,h)，已归一化到 0~1（相对切块）。
    """
    if len(boxes) == 0:
        return []
    rng = np.random.default_rng(seed)
    n = min(n, len(boxes))
    centroids = boxes[rng.choice(len(boxes), n, replace=False)].copy()

    def iou_dist(b, c):
        inter = np.minimum(b[:, None, 0], c[None, :, 0]) * np.minimum(b[:, None, 1], c[None, :, 1])
        union = (b[:, None, 0] * b[:, None, 1]) + (c[None, :, 0] * c[None, :, 1]) - inter
        return 1.0 - inter / np.maximum(union, 1e-9)

    for _ in range(iters):
        d = iou_dist(boxes, centroids)
        assign = d.argmin(axis=1)
        new = centroids.copy()
        for k in range(n):
            m = assign == k
            if m.any():
                new[k] = boxes[m].mean(axis=0)
        if np.allclose(new, centroids):
            break
        centroids = new

    order = np.argsort(centroids[:, 0] * centroids[:, 1])
    return [centroids[i] for i in order]


def build(source: str, out_dir: str, *, tile_size: int = 416, overlap: int = 64,
          max_images: int = 0, skip_empty: bool = True, seed: int = 42,
          base_cfg: str = "cfg/yolov4-tiny-3l.cfg") -> dict:
    images = collect_images(source)
    images = [p for p in images if os.path.splitext(p)[1].lower() in
              (".tif", ".tiff", ".png", ".jpg", ".jpeg")]
    if not images:
        raise SystemExit(f"未找到图片: {source}")
    if max_images:
        images = images[:max_images]

    # 不做整目录 rmtree（既有安全风险，也会触发批量删除保护）。
    # 采用"覆盖写 + 全量重写清单"：未被本次写入的旧文件不会被任何
    # train/val/test 清单引用，因此自动失效，无需删除。
    img_dir = os.path.join(out_dir, "images")
    lbl_dir = os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # 按图像划分（关键：避免同图切块跨集泄漏）
    rng = random.Random(seed)
    idxs = list(range(len(images)))
    rng.shuffle(idxs)
    n = len(idxs)
    n_tr = int(round(n * 0.70))
    n_va = int(round(n * 0.15))
    split_of = {}
    for r, i in enumerate(idxs):
        split_of[i] = "train" if r < n_tr else ("val" if r < n_tr + n_va else "test")

    lists = {"train": [], "val": [], "test": []}
    all_wh = []
    stats = {"images": n, "tiles": 0, "kept": 0, "blobs": 0, "failed": 0}

    for i, path in enumerate(images):
        try:
            img = imread_robust(path)
        except Exception as exc:                      # noqa: BLE001
            print(f"  [跳过] {os.path.basename(path)}: {exc}")
            stats["failed"] += 1
            continue

        # 用"父目录名_文件名"做 stem，避免不同分组下同名文件互相覆盖
        parent = os.path.basename(os.path.dirname(path)) or "root"
        stem = f"{parent}_{os.path.splitext(os.path.basename(path))[0]}"
        sp = split_of[i]
        for tile in iter_tiles(img, tile_size=tile_size, overlap=overlap):
            stats["tiles"] += 1
            if skip_empty and not tile_has_signal(tile.image):
                continue
            blobs = detect_fluorescent_blobs(tile.image)
            if not blobs:
                continue

            name = f"{stem}__t{tile.index:03d}"
            h, w = tile.image.shape[:2]
            all_wh.extend([[min(b.w / w, 1.0), min(b.h / h, 1.0)] for b in blobs])

            ok, buf = cv2.imencode(".png", tile.image)
            buf.tofile(os.path.join(img_dir, name + ".png"))
            with open(os.path.join(lbl_dir, name + ".txt"), "w", encoding="utf-8") as fh:
                fh.write("\n".join(to_yolo_lines(blobs, w, h, CLASS_ID)))

            lists[sp].append(os.path.abspath(os.path.join(img_dir, name + ".png")))
            stats["kept"] += 1
            stats["blobs"] += len(blobs)
        print(f"  [{i+1}/{n}] {stem} -> {sp}（累计切块 {stats['kept']}）")

    for sp, files in lists.items():
        with open(os.path.join(out_dir, f"{sp}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(files) + ("\n" if files else ""))

    with open(os.path.join(out_dir, "cell.names"), "w", encoding="utf-8") as fh:
        fh.write(CLASS_NAME + "\n")
    with open(os.path.join(out_dir, "cell.data"), "w", encoding="utf-8") as fh:
        fh.write(
            f"classes = 1\n"
            f"train   = {os.path.abspath(os.path.join(out_dir,'train.txt'))}\n"
            f"valid   = {os.path.abspath(os.path.join(out_dir,'val.txt'))}\n"
            f"names   = {os.path.abspath(os.path.join(out_dir,'cell.names'))}\n"
            f"backup  = backup/\n"
        )

    anchors = kmeans_anchors(np.asarray(all_wh, dtype=np.float64)) if all_wh else []
    px = [(round(a[0] * tile_size, 1), round(a[1] * tile_size, 1)) for a in anchors]
    stats["anchors_px"] = px

    if anchors and os.path.exists(base_cfg):
        _patch_cfg(base_cfg, "cfg/yolov4-tiny-3l-cell.cfg", anchors, tile_size)
        stats["cfg"] = "cfg/yolov4-tiny-3l-cell.cfg"

    return stats


def _patch_cfg(src: str, dst: str, anchors, tile_size: int) -> None:
    """把 base cfg 改成 1 类 + 新 anchors。

    规则：
      * [net]      width/height = tile_size
      * [yolo]     classes = 1；anchors = 由数据聚类得到；mask 保持
      * 每个 [yolo] **紧邻其前的那个 [convolutional]** 的
            filters = (classes + 5) * len(mask) = 6 * len(mask)

    ⚠️ 旧实现的错误：用"全局第 N 个 yolo 对应全局第 N 个 filters"的近似，
       当 cfg 中 conv 与 yolo 数量不等（yolov4-tiny-3l 为 3 个 yolo）
       时会漏改最后一个，导致运行时报
       "filters= in the [convolutional]-layer doesn't correspond to classes=
        or mask= in [yolo]-layer"。
       这里改为严格的"紧邻前一个 convolutional"配对。
    """
    lines = open(src, encoding="utf-8").read().splitlines()

    out: list = []
    sec = None
    conv_filters_idx = None      # 最近一个 [convolutional] 的 filters 行在 out 中的位置
    pending_conv_idx = None      # 进入 [yolo] 时记录，待读到 mask 后配对
    pairs: list = []             # [(filters 行位置, mask 长度)]

    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            newsec = s.strip("[]").lower()
            if newsec == "yolo":
                pending_conv_idx = conv_filters_idx
            elif newsec == "convolutional":
                conv_filters_idx = None
            sec = newsec
            out.append(ln)
            continue

        key = s.split("=")[0].strip() if "=" in s else ""

        if sec == "net" and key in ("width", "height"):
            out.append(f"{key}={tile_size}")
            continue
        if sec == "convolutional" and key == "filters":
            conv_filters_idx = len(out)
            out.append(ln)
            continue
        if sec == "yolo":
            if key == "classes":
                out.append("classes=1")
                continue
            if key == "anchors":
                a = ", ".join(f"{an[0]*tile_size:.0f},{an[1]*tile_size:.0f}"
                              for an in anchors)
                out.append(f"anchors = {a}")
                continue
            if key == "mask":
                m = [int(x) for x in s.split("=")[1].split(",")]
                out.append("mask = " + ",".join(str(x) for x in m))
                if pending_conv_idx is not None:
                    pairs.append((pending_conv_idx, len(m)))
                continue

        out.append(ln)

    # 按配对改写 filters = (1 + 5) * len(mask)
    for idx, n_mask in pairs:
        out[idx] = f"filters={(1 + 5) * n_mask}"

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    open(dst, "w", encoding="utf-8").write("\n".join(out) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="构建切块训练数据集")
    ap.add_argument("--source", required=True, help="原始图文件夹/单图/清单txt")
    ap.add_argument("--out", default="dataset", help="输出目录")
    ap.add_argument("--tile-size", type=int, default=416)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--max-images", type=int, default=0, help="0=全部")
    ap.add_argument("--base-cfg", default="cfg/yolov4-tiny-3l.cfg")
    args = ap.parse_args()

    print(f"源: {args.source}")
    stats = build(args.source, args.out, tile_size=args.tile_size,
                  overlap=args.overlap, max_images=args.max_images,
                  base_cfg=args.base_cfg)
    print("\n===== 数据集构建完成 =====")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n⚠️  dataset/labels 是**草稿标注**（经典方法生成），")
    print(f"    用于训练前必须人工复核，不能直接当作验收 ground truth。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

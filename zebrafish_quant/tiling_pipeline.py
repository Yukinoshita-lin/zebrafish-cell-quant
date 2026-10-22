"""
tiling_pipeline.py - 原生分辨率切块 + 可插拔后端的端到端定量流水线

把前面所有关键修正串起来：
    大图(2752x2208) -> 切块(416, 不缩放) -> 后端检测(GPU/C++/Python 自动选)
    -> 坐标回映到原图 -> 全局 NMS(去跨块重复) -> 粘连拆分
    -> 掩码约束计数 -> 原图分辨率可视化 -> CSV

这是生产路径。直接检测整图不可用（细胞会缩成亚像素），
因此切块是强制的；跨块重复则由 nms_boxes_global 消除。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import cv2
import numpy as np

from . import imaging
from .backends import Backend
from .counting import count_within_mask, split_adhesions
from .imaging import ImageReadError
from .inference import Detection
from .reporting import make_result_row, save_overlay, write_csv
from .segment import ClassicalGutMask, MaskProvider, load_mask, save_mask
from .tiling import iter_tiles, nms_boxes_global, tile_has_signal

logger = logging.getLogger("zebrafish_quant")


@dataclass
class TilingConfig:
    tile_size: int = 416
    overlap: int = 64              # >0 可缓解跨块细胞被裁切
    global_nms: float = 0.30       # 全局去重的 IoU 阈值
    thresh: float = 0.30
    nms: float = 0.20              # 后端内部的 NMS（论文标定值）
    min_confidence: float = 0.0
    split_adhesion: bool = True
    skip_empty_tiles: bool = True
    save_masks: bool = True
    save_overlays: bool = True
    resume: bool = True
    strict: bool = False


@dataclass
class BatchReport:
    rows: List[dict] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)
    processed: int = 0
    skipped: int = 0
    tiles_done: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> dict:
        counts = [r["肠道内炎症细胞总数"] for r in self.rows]
        return {
            "成功图片": self.processed,
            "失败/跳过": len(self.failures) + self.skipped,
            "处理切块": self.tiles_done,
            "细胞总数": int(sum(counts)) if counts else 0,
            "每张均值": round(float(np.mean(counts)), 2) if counts else 0.0,
            "耗时秒": round(self.elapsed_sec, 1),
        }


class TilingPipeline:
    def __init__(self, backend: Backend,
                 mask_provider: Optional[MaskProvider] = None,
                 config: Optional[TilingConfig] = None):
        self.backend = backend
        self.mask_provider = mask_provider or ClassicalGutMask()
        self.cfg = config or TilingConfig()

    # ---------- 单张 ----------
    def process_image(self, image_path: str, output_dir: str,
                      manual_mask_dir: Optional[str] = None) -> dict:
        image = imaging.imread_robust(image_path)
        flags = imaging.quality_flags(image)
        anomaly = "；".join(flags["reasons"])

        # 1) 掩码（优先人工修正）
        manual_review = False
        mask = None
        if manual_mask_dir:
            cand = os.path.join(manual_mask_dir,
                                os.path.splitext(os.path.basename(image_path))[0] + ".png")
            if os.path.exists(cand):
                mask = load_mask(cand, target_shape=image.shape)
                manual_review = True
        if mask is None:
            mask = self.mask_provider.predict(image)
        mask = self._align(mask, image.shape)
        mask_area = int((mask > 0).sum())

        # 2) 切块 + 后端检测
        tiles, patches = [], []
        for t in iter_tiles(image, self.cfg.tile_size, self.cfg.overlap):
            if self.cfg.skip_empty_tiles and not tile_has_signal(t.image):
                continue
            tiles.append(t)
            patches.append(t.image)

        boxes: List[tuple] = []
        if patches:
            results = self.backend.detect_tiles(patches, self.cfg.thresh, self.cfg.nms)
            for t, dets in zip(tiles, results):
                for d in dets:
                    gx, gy = t.to_global(d.cx, d.cy)
                    boxes.append((gx, gy, d.w, d.h, d.confidence))

        # 3) 全局 NMS：消除跨块重复（重叠区同一细胞会被相邻块各检一次）
        boxes = nms_boxes_global(boxes, self.cfg.global_nms)
        dets = [Detection(label="cell", confidence=s, cx=x, cy=y, w=w, h=h)
                for (x, y, w, h, s) in boxes]

        # 4) 粘连拆分（在原图上做，信息最全）
        if self.cfg.split_adhesion and dets:
            dets = split_adhesions(image, dets)

        # 5) 掩码约束计数
        res = count_within_mask(dets, mask, min_confidence=self.cfg.min_confidence)

        # 6) 落盘
        stem = os.path.splitext(os.path.basename(image_path))[0]
        vis_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        if self.cfg.save_masks and not manual_review:
            mdir = os.path.join(output_dir, "masks")
            os.makedirs(mdir, exist_ok=True)
            save_mask(mask, os.path.join(mdir, stem + ".png"))
        if self.cfg.save_overlays:
            save_overlay(image, os.path.join(vis_dir, stem + "_result.png"),
                         mask=mask, counted=res.inside, excluded=res.outside)

        note = ""
        if mask_area == 0:
            note = "分割掩码为空，未检出中后肠区域，需人工复核"
        elif flags["over_exposed"] or flags["weak_signal"]:
            note = "图像质量异常，建议人工复核"

        return make_result_row(image_path=image_path, count=res.total,
                               manual_review=manual_review, status="ok",
                               anomaly=anomaly, excluded=len(res.outside),
                               mask_area=mask_area, note=note), len(patches)

    @staticmethod
    def _align(mask, shape) -> np.ndarray:
        if mask is None:
            return np.zeros(shape[:2], np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != shape[:2]:
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255

    # ---------- 批量 ----------
    def process_folder(self, input_path: str, output_dir: str,
                       manual_mask_dir: Optional[str] = None,
                       progress: Optional[Callable[[int, int, str], None]] = None
                       ) -> BatchReport:
        t0 = time.time()
        os.makedirs(output_dir, exist_ok=True)
        images = imaging.resolve_input(input_path)
        # 原始数据常分子目录，递归兜底
        if len(images) <= 1 and os.path.isdir(input_path):
            images = []
            for root, _d, files in os.walk(input_path):
                for f in sorted(files):
                    if f.lower().endswith(imaging.SUPPORTED_EXTS):
                        images.append(os.path.join(root, f))
            images.sort()

        rep = BatchReport()
        vis_dir = os.path.join(output_dir, "visualizations")

        for idx, path in enumerate(images, 1):
            name = os.path.basename(path)
            if self.cfg.resume and os.path.exists(
                    os.path.join(vis_dir, os.path.splitext(name)[0] + "_result.png")):
                rep.skipped += 1
                if progress:
                    progress(idx, len(images), name)
                continue
            try:
                row, ntiles = self.process_image(path, output_dir, manual_mask_dir)
                rep.rows.append(row)
                rep.processed += 1
                rep.tiles_done += ntiles
                logger.info("[%d/%d] %s -> %s 个细胞（%d 块）",
                            idx, len(images), name,
                            row["肠道内炎症细胞总数"], ntiles)
            except ImageReadError as exc:
                self._fail(rep, name, str(exc), "图片读取失败")
                logger.warning("[%d/%d] 跳过损坏图片 %s: %s", idx, len(images), name, exc)
            except Exception as exc:                       # noqa: BLE001
                if self.cfg.strict:
                    raise
                self._fail(rep, name, f"{type(exc).__name__}: {exc}", "处理异常")
                logger.exception("[%d/%d] 处理失败 %s，已跳过", idx, len(images), name)
            if progress:
                progress(idx, len(images), name)

        rep.elapsed_sec = time.time() - t0
        write_csv(rep.rows + rep.failures,
                  os.path.join(output_dir, "quantification_results.csv"))
        if rep.failures:
            write_csv(rep.failures, os.path.join(output_dir, "failed_images.csv"))
        logger.info("批量完成: %s", rep.summary())
        return rep

    @staticmethod
    def _fail(rep: BatchReport, name: str, msg: str, status: str) -> None:
        rep.failures.append({
            "图片文件名": name, "肠道内炎症细胞总数": "",
            "分割是否通过人工复核": "否", "处理状态": status,
            "异常标记": "", "掩码外排除数": "", "掩码面积像素": "",
            "备注": msg, "相对路径": name,
        })

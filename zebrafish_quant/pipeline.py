"""
pipeline.py - 批量处理流水线

对应需求：4.1 批量导入 / 4.5 日志输出 / 5.2 稳定性
          4.5 人工微调掩码 / 4.4 结果导出

核心设计（针对"1024 张批量不崩溃、坏图跳过不中断"）：
  * 每张图独立 try/except 包裹：任何单张失败都被捕获、记入失败清单、
    写入日志，然后**继续**处理下一张，绝不中断整批。
  * 支持断点续跑：已生成结果图的默认跳过（resume=True）。
  * 人工微调闭环：若 masks 目录里存在同名掩码，则优先用它，
    并把"分割是否通过人工复核"置为"是"（需求 4.4/4.5）。
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from . import imaging
from .counting import CountResult, count_within_mask, split_adhesions, summarize_counts
from .imaging import ImageReadError
from .inference import DEFAULT_NMS, DEFAULT_THRESH, DarknetModel, detect
from .reporting import make_result_row, save_overlay, write_csv
from .segment import MaskProvider, ClassicalGutMask, load_mask, save_mask

logger = logging.getLogger("zebrafish_quant")


@dataclass
class PipelineConfig:
    # 检测
    thresh: float = DEFAULT_THRESH
    nms: float = DEFAULT_NMS
    min_confidence: float = 0.0      # 噪点抑制的额外下限
    split_adhesion: bool = True       # 需求 4.3：处理轻微粘连

    # 输出
    save_masks: bool = True
    save_overlays: bool = True
    overlay_alpha: float = 0.28

    # 稳定性
    resume: bool = True               # 已处理过的跳过
    strict: bool = False              # True 时遇错即抛（便于调试），默认 False=跳过继续


@dataclass
class BatchReport:
    rows: List[dict] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)
    processed: int = 0
    skipped: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> dict:
        counts = [r["肠道内炎症细胞总数"] for r in self.rows]
        return {
            "成功": self.processed,
            "失败/跳过": len(self.failures) + self.skipped,
            "图片总数": self.processed + len(self.failures) + self.skipped,
            "细胞总数": int(sum(counts)) if counts else 0,
            "均值": round(float(np.mean(counts)), 2) if counts else 0.0,
            "耗时秒": round(self.elapsed_sec, 1),
        }


class QuantPipeline:
    """中后肠炎症细胞批量定量流水线。"""

    def __init__(self, model: DarknetModel,
                 mask_provider: Optional[MaskProvider] = None,
                 config: Optional[PipelineConfig] = None):
        self.model = model
        self.mask_provider = mask_provider or ClassicalGutMask()
        self.cfg = config or PipelineConfig()

    # ---------- 单张 ----------
    def process_image(self, image_path: str, output_dir: str,
                      manual_mask_dir: Optional[str] = None) -> dict:
        """处理单张图，返回结果行。异常由调用方捕获（或本方法内部已隔离）。"""
        image = imaging.imread_robust(image_path)

        # 1) 异常判定（需求 7 例外说明）
        flags = imaging.quality_flags(image)
        anomaly = "；".join(flags["reasons"])

        # 2) 掩码：优先人工修正版（需求 4.5）
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
            mask = self._align_mask(mask, image.shape)

        mask_area = int((mask > 0).sum())

        # 3) 检测（内部已还原到原图坐标）
        dets, _lb = detect(self.model, image,
                           thresh=self.cfg.thresh, nms=self.cfg.nms)
        if self.cfg.split_adhesion and dets:
            dets = split_adhesions(image, dets)

        # 4) 掩码约束计数（需求 4.3）
        result = count_within_mask(dets, mask, min_confidence=self.cfg.min_confidence)

        # 5) 落盘
        stem = os.path.splitext(os.path.basename(image_path))[0]
        vis_dir = os.path.join(output_dir, "visualizations")
        mask_dir = os.path.join(output_dir, "masks")
        os.makedirs(vis_dir, exist_ok=True)

        if self.cfg.save_masks and not manual_review:
            os.makedirs(mask_dir, exist_ok=True)
            save_mask(mask, os.path.join(mask_dir, stem + ".png"))

        if self.cfg.save_overlays:
            save_overlay(
                image,
                os.path.join(vis_dir, stem + "_result.png"),
                mask=mask,
                counted=result.inside,
                excluded=result.outside,
                alpha=self.cfg.overlay_alpha,
            )

        note = ""
        if mask_area == 0:
            note = "分割掩码为空，未检出中后肠区域，需人工复核"
        elif flags["over_exposed"] or flags["weak_signal"]:
            note = "图像质量异常，建议人工复核"

        return make_result_row(
            image_path=image_path,
            count=result.total,
            manual_review=manual_review,
            status="ok",
            anomaly=anomaly,
            excluded=len(result.outside),
            mask_area=mask_area,
            note=note,
        )

    @staticmethod
    def _align_mask(mask: np.ndarray, shape: tuple) -> np.ndarray:
        """确保掩码与原图尺寸一致（防御不同分割模型输出尺寸差异）。"""
        if mask is None:
            return np.zeros(shape[:2], dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape[:2] != shape[:2]:
            import cv2
            mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        return (mask > 127).astype(np.uint8) * 255

    # ---------- 批量 ----------
    def process_folder(self, input_path: str, output_dir: str,
                       manual_mask_dir: Optional[str] = None,
                       progress: Optional[Callable[[int, int, str], None]] = None
                       ) -> BatchReport:
        """批量处理。单张失败不影响整批（需求 5.2）。"""
        started = time.time()
        os.makedirs(output_dir, exist_ok=True)
        images = imaging.resolve_input(input_path)
        logger.info("待处理图片 %d 张", len(images))

        report = BatchReport()
        vis_dir = os.path.join(output_dir, "visualizations")

        for idx, path in enumerate(images, 1):
            name = os.path.basename(path)

            # 断点续跑
            if self.cfg.resume:
                done = os.path.join(vis_dir,
                                    os.path.splitext(name)[0] + "_result.png")
                if os.path.exists(done):
                    report.skipped += 1
                    logger.debug("跳过（已处理）: %s", name)
                    if progress:
                        progress(idx, len(images), name)
                    continue

            try:
                row = self.process_image(path, output_dir, manual_mask_dir)
                report.rows.append(row)
                report.processed += 1
                logger.info("[%d/%d] %s -> %s 个细胞",
                            idx, len(images), name, row["肠道内炎症细胞总数"])
            except ImageReadError as exc:
                # 损坏/不支持：按需求 5.2 跳过并记录，不中断
                self._record_failure(report, name, str(exc), "图片读取失败")
                logger.warning("[%d/%d] 跳过损坏图片 %s: %s", idx, len(images), name, exc)
            except Exception as exc:                # noqa: BLE001 - 批量必须兜底
                if self.cfg.strict:
                    raise
                self._record_failure(report, name, f"{type(exc).__name__}: {exc}", "处理异常")
                logger.exception("[%d/%d] 处理失败 %s，已跳过", idx, len(images), name)

            if progress:
                progress(idx, len(images), name)

        report.elapsed_sec = time.time() - started

        # 汇总产物
        csv_path = os.path.join(output_dir, "quantification_results.csv")
        write_csv(report.rows + report.failures, csv_path)
        logger.info("结果 CSV: %s", csv_path)

        if report.failures:
            fail_csv = os.path.join(output_dir, "failed_images.csv")
            write_csv(report.failures, fail_csv)
            logger.warning("失败 %d 张，详见 %s", len(report.failures), fail_csv)

        logger.info("批量完成: %s", report.summary())
        return report

    @staticmethod
    def _record_failure(report: BatchReport, name: str, message: str, status: str) -> None:
        report.failures.append({
            "图片文件名": name,
            "肠道内炎症细胞总数": "",
            "分割是否通过人工复核": "否",
            "处理状态": status,
            "异常标记": "",
            "掩码外排除数": "",
            "掩码面积像素": "",
            "备注": message,
            "相对路径": name,
        })


def configure_logging(log_path: Optional[str] = None, level: int = logging.INFO) -> None:
    """日志同时输出到控制台与文件（需求 4.5：记录处理数量、失败图片名称）。"""
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    root = logging.getLogger("zebrafish_quant")
    root.setLevel(level)
    root.handlers.clear()

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)) or ".", exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

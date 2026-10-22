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
zebrafish_quant - 斑马鱼荧光图像肠道部位炎症细胞自动定量

在**不改动 darknet 既有代码与对外接口**的前提下补齐需求缺口。
所有模块都可独立导入：不需要 darknet 动态库时（如跑测试、评估），
import 本包不会触发 libdarknet 加载。

模块地图：
    imaging     鲁棒 I/O（tif/中文路径/16bit/灰度）+ letterbox 几何映射
    inference   darknet 封装（修正 NMS/字符串置信度/坐标空间/中文模型路径）
    segment     中后肠掩码（可插拔 MaskProvider）+ Dice 等掩码运算
    counting    掩码约束计数 + 轻微粘连拆分
    reporting   CSV 导出 + 原图分辨率可视化叠加
    evaluate    Dice / Precision / Recall / 计数相对误差（两种口径）
    pipeline    批量流水线（单张隔离、坏图跳过、断点续跑、人工掩码闭环）
"""
from __future__ import annotations

__version__ = "0.3.0"

# 轻量模块先导出（不依赖 libdarknet）
from . import (imaging, segment, counting, reporting, evaluate,  # noqa: F401
               tiling, autolabel, native, backends, benchmark,
               crosscheck, tiling_pipeline)
from .counting import CountResult, count_within_mask, split_adhesions
from .evaluate import (PRF, CountError, acceptance_report, count_relative_error,
                       evaluate_dice, match_boxes_iou, match_points_to_detections)
from .imaging import ImageReadError, Letterbox, imread_robust, letterbox_resize, resolve_input
from .reporting import CSV_FIELDS, draw_overlay, make_result_row, save_overlay, write_csv
from .segment import (ClassicalGutMask, MaskProvider, clean_mask, dice_coefficient,
                      largest_component, load_mask, mask_contours, save_mask)

__all__ = [
    "imaging", "segment", "counting", "reporting", "evaluate",
    "tiling", "autolabel", "native", "backends", "benchmark",
    "crosscheck", "tiling_pipeline",
    "ImageReadError", "Letterbox", "imread_robust", "letterbox_resize", "resolve_input",
    "ClassicalGutMask", "MaskProvider", "clean_mask", "dice_coefficient",
    "largest_component", "load_mask", "mask_contours", "save_mask",
    "CountResult", "count_within_mask", "split_adhesions",
    "CSV_FIELDS", "draw_overlay", "make_result_row", "save_overlay", "write_csv",
    "PRF", "CountError", "acceptance_report", "count_relative_error",
    "evaluate_dice", "match_boxes_iou", "match_points_to_detections",
]

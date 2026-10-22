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
tools_res_ceiling.py - 测量下采样带来的 Dice 天花板

关键问题：当前输入 ÷4（688x552），原图 2752x2208。
若"把人工掩码 ÷4 再上采样回原图"都达不到 Dice 0.85，
说明瓶颈是分辨率而非模型，再大的网络也白搭。

做法：用人工掩码（=理论完美预测）走一遍降采样-上采样链路，
      与原图掩码比 Dice，得到分辨率天花板。
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ANN = os.path.join(ROOT, "annotations")


def main() -> int:
    import cv2
    from tools_collect_masks import imread_unicode
    from zebrafish_quant.segment import dice_coefficient

    rows = list(__import__("csv").DictReader(
        open(os.path.join(ANN, "manifest.csv"), encoding="utf-8-sig")))
    test = [r for r in rows if r["split"] == "test"][:12]
    if not test:
        test = rows[:12]

    print(f"测试样本 {len(test)} 张")
    print("思路: 人工掩码 ÷S -> 上采样回原图 -> 与原图人工掩码比 Dice")
    print("      这是「完美预测」在下采样口径下的得分，即分辨率天花板\n")
    print(f"{'倍率':>5} {'下采样尺寸':>12} {'Dice均值':>9} {'中位':>8} "
          f"{'最小':>8} {'>=0.85占比':>11}")

    results = {}
    for scale in [2, 3, 4, 6, 8]:
        ds, up_nn, up_ln = [], [], []
        for r in test:
            gt = imread_unicode(os.path.join(ANN, "masks", r["name"] + ".png"))
            h, w = gt.shape
            small = cv2.resize(gt, (w // scale, h // scale),
                               interpolation=cv2.INTER_AREA)
            b = (small > 127).astype(np.uint8) * 255
            p_nn = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST)
            p_ln = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
            ds.append((w // scale, h // scale))
            up_nn.append(dice_coefficient(p_nn, gt))
            up_ln.append(dice_coefficient(p_ln, gt))
        nn = np.array(up_nn); ln = np.array(up_ln)
        results[scale] = (nn, ln)
        frac = float((nn >= 0.85).mean()) * 100
        print(f"{scale:>5} {f'{ds[0][0]}x{ds[0][1]}':>12} {nn.mean():>9.4f} "
              f"{np.median(nn):>8.4f} {nn.min():>8.4f} {frac:>10.0f}%")

    print("\n插值方式对比（nearest vs linear 上采样）:")
    for scale in results:
        nn, ln = results[scale]
        print(f"  ÷{scale}: nearest {nn.mean():.4f}  linear {ln.mean():.4f}")

    print("\n=== 结论 ===")
    for scale in [4, 2]:
        nn = results[scale][0]
        if nn.mean() >= 0.9:
            print(f"÷{scale} 分辨率天花板 {nn.mean():.4f} —— 远高于 0.85，"
                  f"说明瓶颈在模型/训练，不是分辨率。")
        elif nn.mean() >= 0.8:
            print(f"÷{scale} 分辨率天花板 {nn.mean():.4f} —— 接近 0.85，"
                  f"分辨率会明显吃掉余量，建议提升到 ÷2。")
        else:
            print(f"÷{scale} 分辨率天花板仅 {nn.mean():.4f} —— "
                  f"已低于验收线，必须提升分辨率。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
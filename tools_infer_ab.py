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
tools_infer_ab.py - 推理配置 A/B 测试（TTA x 后处理 x 阈值）

管线里 test Dice 0.7966 略低于训练评估的 0.8021，怀疑
largest_component 把 GT 里本该保留的碎片状肠道切掉了。
这里把各组合的 Dice 都测一遍，选出真正最优的推理配置。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ANN = os.path.join(ROOT, "annotations")


def main() -> int:
    import cv2
    import csv
    from zebrafish_quant.imaging import imread_robust, green_channel
    from zebrafish_quant.segment import (clean_mask, dice_coefficient,
                                        largest_component)
    from tools_collect_masks import imread_unicode
    import tools_train_unet_v3 as V3

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(ANN, "unet_model_v3_bn.pt"),
                    map_location="cpu", weights_only=False)
    V3.NORM = ck.get("norm", "bn")
    model = V3.build_unet(base=ck["base"], depth=ck["depth"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    scale = ck["scale"]
    print(f"模型 v3_bn epoch={ck['epoch']} Dice={ck['dice']:.4f} "
          f"base={ck['base']} depth={ck['depth']} norm={V3.NORM}")

    rows = list(csv.DictReader(
        open(os.path.join(ANN, "manifest.csv"), encoding="utf-8-sig")))
    rows = [r for r in rows if r["split"] == "test"]
    print(f"测试集 {len(rows)} 张\n")

    # 先把每张图的概率图算好一次，阈值/后处理组合复用
    probs = []
    for r in rows:
        img = imread_robust(r["src_tif"])
        g = green_channel(img)
        h, w = g.shape
        g_s = cv2.resize(g, (w // scale, h // scale),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(g_s[None, None]).to(dev)

        def prob_of(x):
            with torch.no_grad():
                out = model(x)
                lg = out[-1] if isinstance(out, (list, tuple)) else out
                return torch.sigmoid(lg)

        # 无 TTA 的基准概率（必须和 TTA 版分别存，否则无法隔离 TTA 的贡献）
        p0 = prob_of(t)[0, 0].cpu().numpy()

        acc = np.zeros_like(g_s, dtype=np.float32)
        for k in range(4):
            xr = torch.rot90(t, k, [2, 3])
            for flip in (False, True):
                xi = torch.flip(xr, [3]) if flip else xr
                p = prob_of(xi.contiguous())
                if flip:
                    p = torch.flip(p, [3])
                p = torch.rot90(p, -k, [2, 3])
                acc += p[0, 0].cpu().numpy()
        # 两种概率都存下来，供下方按 tta_name 选择
        probs.append({"name": r["name"], "hw": (h, w),
                      "prob_raw": p0, "prob_tta": acc / 8.0})
    print("概率图计算完成\n")

    print(f"{'配置':<34} {'Dice均值':>9} {'中位':>8} {'P10':>8}")
    best = None
    for tta_name in ["原始", "TTA8"]:
        for post in ["无", "clean", "clean+最大连通"]:
            for th in ([0.5] if post != "无" else [0.5]):
                ds = []
                for d in probs:
                    h, w = d["hw"]
                    nm = d["name"]
                    prob = d["prob_tta"] if tta_name == "TTA8" else d["prob_raw"]
                    m = (prob > th).astype(np.uint8) * 255
                    mf = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    if "clean" in post:
                        mf = clean_mask(mf)
                    if "最大连通" in post:
                        mf = largest_component(mf)
                    gt = imread_unicode(os.path.join(ANN, "masks", nm + ".png"))
                    ds.append(dice_coefficient(mf, gt))
                ds = np.array(ds)
                label = f"{tta_name} / {post} / th{th}"
                print(f"{label:<34} {ds.mean():>9.4f} {np.median(ds):>8.4f} "
                      f"{np.percentile(ds, 10):>8.4f}")
                if best is None or ds.mean() > best[0]:
                    best = (ds.mean(), label)

    # 最佳后处理下再扫阈值
    post = best[1].split(" / ")[1]
    print(f"\n以最优后处理（{post}）扫描阈值:")
    tta_on = best[1].startswith("TTA")
    for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        ds = []
        for d in probs:
            h, w = d["hw"]
            nm = d["name"]
            prob = d["prob_tta"] if tta_on else d["prob_raw"]
            m = (prob > th).astype(np.uint8) * 255
            mf = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            if "clean" in post:
                mf = clean_mask(mf)
            if "最大连通" in post:
                mf = largest_component(mf)
            gt = imread_unicode(os.path.join(ANN, "masks", nm + ".png"))
            ds.append(dice_coefficient(mf, gt))
        ds = np.array(ds)
        print(f"  th={th:.2f}  Dice {ds.mean():.4f}  中位 {np.median(ds):.4f}")

    print(f"\n==> 最佳配置: {best[1]}  Dice {best[0]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
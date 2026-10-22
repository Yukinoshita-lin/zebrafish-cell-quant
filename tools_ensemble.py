#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_ensemble.py - 多模型集成评估（补最后 0.02 Dice 的最便宜手段）

已有两个架构不同的模型：
  v3_bn  BatchNorm,   epoch285, Dice 0.8021
  v4_gn  GroupNorm,   epoch220, Dice 0.8320
两者训练轮数/归一化都不同，属"多样性"较强的集成候选。

对比：单模型 vs 集成（TTA8 + clean 后处理，全分辨率口径）
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ANN = os.path.join(ROOT, "annotations")

MODELS = {
    "v3_bn": "unet_model_v3_bn.pt",
    "v4_gn": "unet_model_v4_gn.pt",
    "v5_gn2": "unet_model_v5_gn2.pt",
    "v6_gn3": "unet_model_v6_gn3.pt",   # ÷3 微调，scale 与其余不同
}


def load(path):
    import tools_train_unet_v3 as V3
    ck = torch.load(os.path.join(ANN, path), map_location="cpu",
                    weights_only=False)
    V3.NORM = ck.get("norm", "bn")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = V3.build_unet(base=ck["base"], depth=ck["depth"]).to(dev)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def tta_prob(model, g_s, dev):
    """8 向二面体变换平均概率"""
    t = torch.from_numpy(g_s[None, None]).to(dev)
    acc = np.zeros_like(g_s, dtype=np.float32)
    with torch.no_grad():
        for k in range(4):
            xr = torch.rot90(t, k, [2, 3])
            for flip in (False, True):
                xi = torch.flip(xr, [3]) if flip else xr
                out = model(xi.contiguous())
                lg = out[-1] if isinstance(out, (list, tuple)) else out
                p = torch.sigmoid(lg)
                if flip:
                    p = torch.flip(p, [3])
                p = torch.rot90(p, -k, [2, 3])
                acc += p[0, 0].cpu().numpy()
    return acc / 8.0


def main() -> int:
    import cv2
    import csv
    from zebrafish_quant.imaging import imread_robust, green_channel
    from zebrafish_quant.segment import clean_mask, dice_coefficient
    from tools_collect_masks import imread_unicode

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {}
    for name, fn in MODELS.items():
        m, ck = load(fn)
        loaded[name] = (m, ck)
        print(f"载入 {name}: epoch={ck['epoch']} dice={ck['dice']:.4f} "
              f"norm={ck.get('norm','bn')}")

    rows = list(csv.DictReader(
        open(os.path.join(ANN, "manifest.csv"), encoding="utf-8-sig")))
    rows = [r for r in rows if r["split"] == "test"]
    print(f"\n测试集 {len(rows)} 张，算 TTA 概率...\n")

    # 统一到公共尺寸（÷4 口径）后才能跨分辨率平均
    REF_H, REF_W = 552, 688
    cache = []
    for r in rows:
        img = imread_robust(r["src_tif"])
        g = green_channel(img)
        h, w = g.shape
        d = {"name": r["name"], "hw": (h, w)}
        for name, (m, ck) in loaded.items():
            sc = ck["scale"]
            g_s = cv2.resize(g, (w // sc, h // sc),
                             interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            p = tta_prob(m, g_s, dev)
            if p.shape != (REF_H, REF_W):      # ÷3 等其他分辨率 -> 重采样
                p = cv2.resize(p, (REF_W, REF_H), interpolation=cv2.INTER_LINEAR)
            d[name] = p
        cache.append(d)
    print("完成\n")

    def score(get_prob, th=0.5):
        ds = []
        for d in cache:
            h, w = d["hw"]
            p = get_prob(d)
            m = (p > th).astype(np.uint8) * 255
            mf = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mf = clean_mask(mf)
            gt = imread_unicode(os.path.join(ANN, "masks", d["name"] + ".png"))
            ds.append(dice_coefficient(mf, gt))
        return np.array(ds)

    print(f"{'方案':<32} {'Dice均值':>9} {'中位':>8} {'P10':>8} {'达标率':>8}")
    results = {}
    combos = {
        "v3_bn 单模型 (÷4)": lambda d: d["v3_bn"],
        "v4_gn 单模型 (÷4)": lambda d: d["v4_gn"],
        "v5_gn2 单模型 (÷4)": lambda d: d["v5_gn2"],
        "v6_gn3 单模型 (÷3微调)": lambda d: d["v6_gn3"],
        "集成 v4+v6 (÷4+÷3 跨分辨率)": lambda d: (d["v4_gn"] + d["v6_gn3"]) / 2,
        "集成 v4+v5+v6": lambda d: (d["v4_gn"] + d["v5_gn2"]
                              + d["v6_gn3"]) / 3,
        "集成 v3+v4+v5 (原最佳)": lambda d: (d["v3_bn"] + d["v4_gn"]
                                    + d["v5_gn2"]) / 3,
        "集成 全部4个模型": lambda d: (d["v3_bn"] + d["v4_gn"] + d["v5_gn2"]
                              + d["v6_gn3"]) / 4,
        "集成 v4+v5+v6 (偏v4,v6)": lambda d: 0.4 * d["v4_gn"] + 0.3 * d["v5_gn2"]
                                  + 0.3 * d["v6_gn3"],
    }
    for k, fn in combos.items():
        results[k] = score(fn)
    for k, v in results.items():
        print(f"{k:<32} {v.mean():>9.4f} {np.median(v):>8.4f} "
              f"{np.percentile(v, 10):>8.4f} {(v >= 0.85).mean() * 100:>7.0f}%")

    best_k = max(results, key=lambda k: results[k].mean())
    best = results[best_k]
    print(f"\n==> 最佳: {best_k}  Dice {best.mean():.4f}")

    print(f"\n在最佳方案上扫阈值:")
    for th in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
        v = score(combos[best_k], th)
        print(f"  th={th:.2f}  Dice {v.mean():.4f}  "
              f"中位 {np.median(v):.4f}  达标 {(v >= 0.85).mean() * 100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
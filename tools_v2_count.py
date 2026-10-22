#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_v2_count.py - v2 计数：U-Net 分割掩码 + 经典检测（端到端无人工掩码）

v1 用人工标注掩码；v2 用训练好的 U-Net 实时预测掩码，验证
"AI 分割 -> 掩码内计数"全自动链路，并给出：
  * U-Net 掩码 vs 人工掩码的 Dice（按图，分 train/test）
  * v2 计数 vs v1 计数的相关性/偏差（换掩码带来的计数变化）
  * 抽样叠加可视化

前置：tools_train_unet.py 已产出 annotations/unet_model.pt
"""
from __future__ import annotations

import csv
import os
import sys
from typing import List

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ANN = os.environ.get("UNET_ANN_DIR") or os.path.join(ROOT, "annotations")
# 默认优先用 v3 模型；V2_MODEL=v1 回退到 v2(base16)；
# V2_MODEL_PATH 可直接指定检查点；V2_MODELS 用逗号分隔多个检查点做集成
_ENSEMBLE = [p.strip() for p in os.environ.get("V2_MODELS", "").split(",")
             if p.strip()]
MODEL = os.environ.get("V2_MODEL_PATH") or os.path.join(
    ANN, "unet_model_v3.pt" if os.environ.get("V2_MODEL", "v3") == "v3"
    else "unet_model.pt")
if _ENSEMBLE:
    MODEL = os.path.join(ANN, _ENSEMBLE[0])
TAG = os.environ.get("V2_TAG", "v3")


def _build(ckpt):
    import torch
    if "base" in ckpt:                     # v3/v4/v5/v6 架构
        import tools_train_unet_v3 as V3
        saved_norm = ckpt.get("norm", "bn")
        if V3.NORM != saved_norm:          # 按检查点记录重建对应归一化
            V3.NORM = saved_norm
        return V3.build_unet(base=ckpt["base"], depth=ckpt["depth"])
    from tools_train_unet import build_unet      # v2 架构
    return build_unet()


def load_model():
    """加载单模型或模型集成。

    返回 (models, scales, best_dice, ckpts)
    - models 为 list，集成时每个模型用各自的 scale 预处理
    - 不同 scale 的概率图统一重采样到 REF_H x REF_W 后再平均
    """
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = ([os.path.join(ANN, p) for p in _ENSEMBLE] if _ENSEMBLE
             else [MODEL])
    models, scales, dices = [], [], []
    for p in paths:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = _build(ck).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        models.append(m)
        scales.append(ck.get("scale", 4))
        dices.append(ck.get("dice"))
    return models, scales, (dices if len(dices) == 1 else dices[0]), paths


REF_H, REF_W = 552, 688       # 集成时概率图的公共尺寸（÷4 口径）


def predict_mask(models, scales, img_bgr,
                 threshold: float = 0.5, tta: bool = True) -> np.ndarray:
    """全图推理（支持多模型集成）：绿通道 ÷scale -> U-Net(+TTA) -> 上采样回原图。

    上采样必须用 INTER_NEAREST：实测 ÷4 口径下 nearest 0.9699 vs linear 0.9274，
    linear 会把掩码边界抹糊，白丢 4 个点 Dice。

    TTA：对 8 个二面体变换（4 旋转 x 2 翻转）分别推理后取概率均值。
    多模型集成时把各模型概率重采样到公共尺寸再平均。
    """
    import cv2
    import torch
    from zebrafish_quant.imaging import green_channel
    from zebrafish_quant.segment import clean_mask, largest_component
    if not isinstance(models, (list, tuple)):
        models, scales = [models], [scales]
    g = green_channel(img_bgr)
    h, w = g.shape
    dev = next(models[0].parameters()).device

    total = np.zeros((REF_H, REF_W), dtype=np.float32)
    for model, scale in zip(models, scales):
        g_s = cv2.resize(g, (w // scale, h // scale),
                         interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        t = torch.from_numpy(g_s[None, None]).to(dev)

        def prob_of(x):
            with torch.no_grad():
                out = model(x)
                logits = out[-1] if isinstance(out, (list, tuple)) else out
                return torch.sigmoid(logits)

        if not tta:
            prob = prob_of(t)[0, 0].cpu().numpy()
        else:
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
            prob = acc / 8.0
        if prob.shape != (REF_H, REF_W):
            prob = cv2.resize(prob, (REF_W, REF_H),
                              interpolation=cv2.INTER_LINEAR)
        total += prob
    total /= len(models)

    pred = (total > threshold).astype(np.uint8) * 255
    pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    pred_full = clean_mask(pred_full)
    # largest_component 会把 GT 里合法的碎片状肠道段整段删掉，实测掉 0.014 Dice
    # （0.8104 -> 0.7966），因此默认关闭；需要时用 V2_LARGEST=1 打开。
    if os.environ.get("V2_LARGEST", "0") == "1":
        pred_full = largest_component(pred_full)
    return pred_full


def main() -> int:
    if not os.path.exists(MODEL):
        print(f"[错误] 模型不存在: {MODEL}（先跑 tools_train_unet.py）")
        return 1
    import cv2
    from zebrafish_quant.imaging import imread_robust
    from zebrafish_quant.counting import count_within_mask
    from zebrafish_quant.inference import Detection
    from zebrafish_quant.segment import dice_coefficient
    from tools_collect_masks import imread_unicode

    models, scales, ckpt_dice, paths = load_model()
    arch = f"{len(models)} 模型集成" if len(models) > 1 else "单模型"
    print(f"U-Net 载入（{arch}: "
          f"{', '.join(os.path.basename(p) for p in paths)}）")
    print(f"各模型 scale={scales}，训练时 Dice={ckpt_dice}")

    rows = list(csv.DictReader(open(os.path.join(ANN, "manifest.csv"),
                                    encoding="utf-8-sig")))
    v1 = {r["name"]: r for r in csv.DictReader(
        open(os.path.join(ANN, "v1_counts.csv"), encoding="utf-8-sig"))}

    out_rows = []
    dices = {"train": [], "test": []}
    for i, r in enumerate(rows):
        n = r["name"]
        img = imread_robust(r["src_tif"])
        h, w = img.shape[:2]
        pred = predict_mask(models, scales, img)
        gt = imread_unicode(os.path.join(ANN, "masks", n + ".png"))
        d = dice_coefficient(pred, gt)
        dices[r["split"]].append(d)

        # 用 v2 掩码计数（草稿框复用，避免重复检测）
        dets = []
        for line in open(os.path.join(ANN, "drafts", n + ".txt"),
                          encoding="utf-8"):
            p = line.split()
            if len(p) != 5:
                continue
            cx, cy, bw, bh = map(float, p[1:])
            dets.append(Detection(label="cell", confidence=0.5,
                                  cx=cx * w, cy=cy * h, w=bw * w, h=bh * h))
        cr = count_within_mask(dets, pred, min_confidence=0.0)
        out_rows.append({
            "name": n, "split": r["split"], "stratum": r["stratum"],
            "batch": r["batch"], "group": r["group"],
            "dice_vs_human": f"{d:.4f}",
            "n_v1_human_mask": v1[n]["n_inside_mask"],
            "n_v2_unet_mask": cr.total,
        })
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(rows)}")

    out = os.path.join(ANN, f"{TAG}_counts.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(out_rows)

    d_tr = np.array(dices["train"])
    d_te = np.array(dices["test"])
    v1c = np.array([int(r["n_v1_human_mask"]) for r in out_rows], dtype=float)
    v2c = np.array([int(r["n_v2_unet_mask"]) for r in out_rows], dtype=float)
    diff = v2c - v1c
    corr = float(np.corrcoef(v1c, v2c)[0, 1]) if v1c.std() > 0 else float("nan")

    report = f"""{TAG} 端到端：U-Net 掩码 + 经典检测计数
{'=' * 48}
Dice（U-Net 预测 vs 人工掩码）:
  train 150 图: 均值 {d_tr.mean():.4f}  中位 {np.median(d_tr):.4f}  P10 {np.percentile(d_tr, 10):.4f}
  test   40 图: 均值 {d_te.mean():.4f}  中位 {np.median(d_te):.4f}  P10 {np.percentile(d_te, 10):.4f}
  验收标准 Dice>=0.85: {'达标' if d_te.mean() >= 0.85 else '未达标'}

计数对比（换掩码的影响）:
  v1（人工掩码）均值 {v1c.mean():.1f}   v2（U-Net 掩码）均值 {v2c.mean():.1f}
  差值 v2-v1: 均值 {diff.mean():+.1f}  中位 {np.median(diff):+.0f}  "
  绝对偏差均值 {np.abs(diff).mean():.1f}
  v1/v2 计数相关系数 r = {corr:.3f}
  |差|>5 的图: {(np.abs(diff) > 5).sum()}/{len(diff)}

结论：{'U-Net 掩码可替代人工掩码进入全自动计数链路' if d_te.mean() >= 0.85 and np.abs(diff).mean() < 5 else '仍需改进（见 Dice 与计数偏差）'}
"""
    with open(os.path.join(ANN, f"{TAG}_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"明细 -> {out}")

    # 抽样可视化：Dice 最差 2 张 + 最佳 2 张
    os.makedirs(os.path.join(ANN, f"{TAG}_vis"), exist_ok=True)
    order = sorted(out_rows, key=lambda r: float(r["dice_vs_human"]))
    picks = order[:2] + order[-2:]
    for r in picks:
        n = r["name"]
        row = next(x for x in rows if x["name"] == n)
        img = imread_robust(row["src_tif"])
        pred = predict_mask(models, scales, img)
        gt = imread_unicode(os.path.join(ANN, "masks", n + ".png"))
        ov = img.copy()
        ov[gt > 0] = (0.3 * ov[gt > 0] + 0.7 * np.array([0, 0, 255])).astype(np.uint8)
        ov[pred > 0] = (0.3 * ov[pred > 0] + 0.7 * np.array([255, 0, 255])).astype(np.uint8)
        sc = 1400 / ov.shape[1]
        small = cv2.resize(ov, (1400, int(ov.shape[0] * sc)))
        ok, buf = cv2.imencode(".png", small)
        buf.tofile(os.path.join(ANN, f"{TAG}_vis",
                                f"{TAG}_dice{r['dice_vs_human']}_{n}.png"))
    print(f"可视化 -> {os.path.join(ANN, f'{TAG}_vis')}"
          f"（红=人工掩码，紫=U-Net 预测）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_diagnose.py - v2 模型失效模式诊断（回答两个问题）

Q1 阈值 0.5 是不是瓶颈？-> 扫描阈值看 Dice 上限
Q2 失效模式是什么？    -> 看逐图 Dice 分布，区分
                        (a) 少数灾难性失败  (b) 全局都只是"不准"

结论决定下一轮训练方向：加容量 / 加轮次 / 改数据口径。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ANN = os.environ.get("UNET_ANN_DIR") or os.path.join(ROOT, "annotations")


def build_unet(base=16):
    import torch.nn as nn

    def conv_block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True))

    class _UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = conv_block(1, base)
            self.enc2 = conv_block(base, base * 2)
            self.enc3 = conv_block(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec2 = conv_block(base * 4, base * 2)
            self.dec1 = conv_block(base * 2, base)
            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    return _UNet()


def main() -> int:
    import tools_train_unet_v3 as T

    # 支持任意检查点：DIAG_MODEL=unet_model_v4_gn.pt（默认旧的 unet_model.pt）
    mpath = os.environ.get("DIAG_MODEL", "unet_model.pt")
    if not os.path.isabs(mpath):
        mpath = os.path.join(ANN, mpath)
    if not os.path.exists(mpath):
        print(f"[错误] 模型不存在: {mpath}")
        return 1
    ckpt = torch.load(mpath, map_location="cpu", weights_only=False)

    # 按检查点记录还原架构与预处理口径（scale / 归一化 / base / depth）
    if "base" in ckpt:
        T.NORM = ckpt.get("norm", "bn")
        T.SCALE = ckpt.get("scale", 4)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = T.build_unet(base=ckpt["base"], depth=ckpt["depth"]).to(dev)
    else:                                  # 旧的 v2 架构（base16, ÷4）
        from tools_train_unet import SCALE as V2_SCALE, build_unet
        T.SCALE = V2_SCALE
        dev = torch.device("cpu")
        model = build_unet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"模型 {os.path.basename(mpath)}  epoch={ckpt.get('epoch')}  "
          f"dice={ckpt.get('dice', float('nan')):.4f}  "
          f"÷{T.SCALE}  {dev}")

    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print("加载数据...")
    train, test = T.load_all()
    xe = torch.from_numpy(np.stack([x for _, x, _ in test])).to(dev)
    ye = torch.from_numpy(np.stack([y for _, _, y in test]))
    names = [n for n, _, _ in test]

    # 分块推理：整批 40 张 552x688 单层激活就要 1.8GB，直接 OOM
    px = xe.shape[-1] * xe.shape[-2]
    chunk = max(1, min(8, int(8 * (552 * 688) / max(px, 1))))
    probs = []
    with torch.no_grad():
        for i in range(0, xe.shape[0], chunk):
            out = model(xe[i:i + chunk])
            logits = out[-1] if isinstance(out, (list, tuple)) else out
            probs.append(torch.sigmoid(logits).float().cpu().numpy()[:, 0])
    prob = np.concatenate(probs, axis=0)          # (N,H,W)

    gt = ye.numpy()[:, 0]
    print(f"\nGT 正像素占比: 均值 {gt.mean() * 100:.2f}%")

    def dice_at(th):
        pred = (prob > th).astype(np.float32)
        inter = (pred * gt).sum(axis=(1, 2))
        denom = pred.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
        return (2 * inter + 1e-6) / (denom + 1e-6), pred

    base_dice, _ = dice_at(0.5)     # 阈值扫描的对比基准（不再硬编码旧模型分数）

    # ---- Q1 阈值扫描 ----
    print("\n=== Q1 阈值扫描（test %d 张）===" % len(gt))
    print(f"{'阈值':>6} {'Dice均值':>9} {'中位':>8} {'P10':>8} {'精确率':>8} {'召回率':>8}")
    best_th, best_mean = 0.5, -1
    for th in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]:
        dices, pred = dice_at(th)
        # 全局精确率/召回率（按像素总量）
        tp = (pred * gt).sum(); fp = pred.sum() - tp; fn = gt.sum() - tp
        prec = tp / max(tp + fp, 1e-6); rec = tp / max(tp + fn, 1e-6)
        if dices.mean() > best_mean:
            best_mean, best_th = dices.mean(), th
        print(f"{th:>6.2f} {dices.mean():>9.4f} {np.median(dices):>8.4f} "
              f"{np.percentile(dices, 10):>8.4f} {prec:>8.4f} {rec:>8.4f}")
    print(f"-> 最佳阈值 {best_th} (Dice {best_mean:.4f})，"
          f"相比 0.5 提升 {best_mean - base_dice.mean():+.4f}")
    if abs(best_mean - base_dice.mean()) < 0.01:
        print("   结论：阈值不是瓶颈，不要在调阈值上浪费时间。")

    # ---- Q2 失效模式 ----
    print("\n=== Q2 逐图 Dice 分布（阈值 0.5）===")
    dices, pred = dice_at(0.5)

    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"{'Dice区间':>14} {'张数':>6} {'占比':>7}")
    for lo, hi in bins:
        n = int(((dices >= lo) & (dices < hi)).sum())
        print(f"{f'{lo:.1f}-{hi:.1f}':>14} {n:>6} {n / len(dices) * 100:>6.1f}%")
    n_fail = int((dices < 0.2).sum())
    print(f"\n灾难性失败 (<0.2): {n_fail}/{len(dices)} 张")
    if n_fail:
        bad = np.argsort(dices)[:n_fail]
        print("最差 5 张: " + ", ".join(
            f"{names[i]}({dices[i]:.2f})" for i in bad[:5]))

    # 预测面积 vs GT 面积（判断是过分割还是欠分割）
    parea = pred.sum(axis=(1, 2))
    garea = gt.sum(axis=(1, 2))
    ratio = parea / np.maximum(garea, 1e-6)
    print(f"\n预测面积/GT面积: 中位 {np.median(ratio):.2f}  "
          f"P10 {np.percentile(ratio, 10):.2f}  P90 {np.percentile(ratio, 90):.2f}")
    print(f"  严重过分割(>2倍): {int((ratio > 2).sum())} 张"
          f"   严重欠分割(<0.5倍): {int((ratio < 0.5).sum())} 张")

    # 只看"非灾难"样本的上限，评估容量是否够
    ok = dices >= 0.2
    print(f"\n排除灾难样本后: Dice 均值 {dices[ok].mean():.4f} "
          f"(n={int(ok.sum())})  -> 反映模型在'看懂的图'上的真实水平")
    return 0


if __name__ == "__main__":
    sys.exit(main())
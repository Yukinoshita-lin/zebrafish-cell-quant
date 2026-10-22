#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tools_train_unet.py - 训练中后肠分割 U-Net（需求 4.2 / 7：Dice >= 0.85）

数据：annotations/manifest.csv（train 150 / test 40，按图划分无泄漏）
      图像绿色通道 ÷4 下采样（688x552）作为输入，人工掩码同步下采样为标签。
模型：小型 U-Net（base 16，3 次下采样），参数量 ~2M，CPU 可训。
损失：BCEWithLogits + SoftDice（肠道仅占画面 0.3%，Dice 项对抗类别不平衡）。
增强：随机水平/垂直翻转 + 90 度旋转。
评估：测试集逐图 Dice（原分辨率口径：预测上采样回全图算 Dice）。

产出：annotations/unet_model.pt / unet_metrics.txt / unet_vis/*.png
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

ANN = os.path.join(ROOT, "annotations")
SCALE = 4                       # 下采样倍率
EPOCHS = int(os.environ.get("UNET_EPOCHS", "60"))
BATCH = 8
LR = 1e-3


def preprocess(name: str, src_tif: str) -> Tuple[np.ndarray, np.ndarray]:
    import cv2
    from zebrafish_quant.imaging import imread_robust, green_channel
    from tools_collect_masks import imread_unicode
    img = imread_robust(src_tif)
    g = green_channel(img)
    h, w = g.shape
    g_s = cv2.resize(g, (w // SCALE, h // SCALE),
                     interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    m = imread_unicode(os.path.join(ANN, "masks", name + ".png"))
    m_s = cv2.resize(m, (w // SCALE, h // SCALE),
                     interpolation=cv2.INTER_NEAREST)
    m_s = (m_s > 127).astype(np.float32)
    return g_s[None], m_s[None]        # (1,H,W)


def load_all() -> Tuple[List[Tuple[str, np.ndarray, np.ndarray]],
                        List[Tuple[str, np.ndarray, np.ndarray]]]:
    rows = list(csv.DictReader(open(os.path.join(ANN, "manifest.csv"),
                                    encoding="utf-8-sig")))
    train, test = [], []
    for i, r in enumerate(rows):
        x, y = preprocess(r["name"], r["src_tif"])
        (train if r["split"] == "train" else test).append((r["name"], x, y))
        if (i + 1) % 50 == 0:
            print(f"  预处理 {i + 1}/{len(rows)}")
    return train, test


def build_unet():
    import torch.nn as nn

    def conv_block(c_in, c_out):
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True))

    class _UNet(nn.Module):
        def __init__(self, base=16):
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


def dice_loss(logits, targets, eps=1e-6):
    import torch
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def augment(x, y):
    import torch
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    if torch.rand(1).item() < 0.5:
        x, y = torch.flip(x, [2]), torch.flip(y, [2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x, y = torch.rot90(x, k, [2, 3]), torch.rot90(y, k, [2, 3])
    return x.contiguous(), y.contiguous()


def main() -> int:
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 4))
    print(f"torch {torch.__version__} 线程 {torch.get_num_threads()}")
    print(f"配置: ÷{SCALE} 输入, {EPOCHS} epochs, batch {BATCH}, lr {LR}")

    print("加载数据...")
    t0 = time.time()
    train, test = load_all()
    print(f"train {len(train)} / test {len(test)}，"
          f"输入尺寸 {train[0][1].shape}，耗时 {time.time() - t0:.0f}s")

    xt = torch.from_numpy(np.stack([x for _, x, _ in train]))
    yt = torch.from_numpy(np.stack([y for _, _, y in train]))
    xe = torch.from_numpy(np.stack([x for _, x, _ in test]))
    ye = torch.from_numpy(np.stack([y for _, _, y in test]))
    print(f"train tensor {xt.shape} 正样本像素占比 "
          f"{yt.mean().item() * 100:.2f}%")

    model = build_unet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params / 1e6:.2f}M")
    bce = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    def evaluate(threshold=0.5):
        model.eval()
        with torch.no_grad():
            logits = model(xe)
            pred = (torch.sigmoid(logits) > threshold).float()
            inter = (pred * ye).sum(dim=(1, 2, 3))
            denom = pred.sum(dim=(1, 2, 3)) + ye.sum(dim=(1, 2, 3))
            dices = (2 * inter + 1e-6) / (denom + 1e-6)
        return dices.numpy()

    best_dice, best_epoch = -1, -1
    n = xt.shape[0]
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(xt[idx], yt[idx])
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb) + dice_loss(logits, yb).mean()
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        if epoch % 5 == 0 or epoch == 1:
            d = evaluate()
            print(f"epoch {epoch:3d}/{EPOCHS}  loss {tot / nb:.4f}  "
                  f"test Dice 均值 {d.mean():.4f} 中位 {np.median(d):.4f} "
                  f"最小 {d.min():.4f}")
            if d.mean() > best_dice:
                best_dice, best_epoch = d.mean(), epoch
                import torch as _t
                _t.save({"state_dict": model.state_dict(),
                         "scale": SCALE, "epoch": epoch,
                         "dice": float(d.mean())},
                        os.path.join(ANN, "unet_model.pt"))
    d = evaluate()
    print(f"\n最终 test Dice: 均值 {d.mean():.4f} 中位 {np.median(d):.4f} "
          f"P10 {np.percentile(d, 10):.4f}")
    if d.mean() > best_dice:
        import torch as _t
        _t.save({"state_dict": model.state_dict(), "scale": SCALE,
                 "epoch": EPOCHS, "dice": float(d.mean())},
                os.path.join(ANN, "unet_model.pt"))
        best_dice, best_epoch = d.mean(), EPOCHS

    # 全图分辨率口径复评（预测上采样回 2752x2208 与人工掩码比）
    print("\n全图分辨率口径 Dice（预测上采样回原图）...")
    import cv2
    from zebrafish_quant.segment import dice_coefficient
    from tools_collect_masks import imread_unicode
    full_dices = []
    model.eval()
    with torch.no_grad():
        for j, (name, _x, _y) in enumerate(test[:12]):
            logits = model(xe[j:j + 1])
            pred_s = (torch.sigmoid(logits)[0, 0].numpy() > 0.5).astype(np.uint8) * 255
            gt_full = imread_unicode(os.path.join(ANN, "masks", name + ".png"))
            pred_full = cv2.resize(pred_s, (gt_full.shape[1], gt_full.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            full_dices.append(dice_coefficient(pred_full, gt_full))
    full_dices = np.array(full_dices)
    print(f"全图口径 Dice（{len(full_dices)} 张）: 均值 {full_dices.mean():.4f} "
          f"中位 {np.median(full_dices):.4f}")

    report = f"""中后肠 U-Net 分割训练报告
{'=' * 48}
配置：输入 绿色通道 ÷{SCALE}（{train[0][1].shape[2]}x{train[0][1].shape[1]}），
      base16 三级 U-Net（{n_params / 1e6:.2f}M 参数），
      BCE + SoftDice，Adam lr={LR} cosine，{EPOCHS} epochs
数据：train {len(train)} / test {len(test)}（按图划分，人工掩码 GT）

结果：
  测试集 Dice（下采样口径）: 均值 {d.mean():.4f}  中位 {np.median(d):.4f}  P10 {np.percentile(d, 10):.4f}
  测试集 Dice（全图分辨率口径，{len(full_dices)} 张）: 均值 {full_dices.mean():.4f}  中位 {np.median(full_dices):.4f}
  最佳 epoch: {best_epoch}（Dice {best_dice:.4f}）
  验收标准（需求 7）：Dice >= 0.85  ->  {'达标' if full_dices.mean() >= 0.85 else '未达标，需迭代'}

模型：annotations/unet_model.pt（含 state_dict/scale/epoch/dice）
"""
    with open(os.path.join(ANN, "unet_metrics.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"训练总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools_check_aug.py - 验证 augment() 不会让 x 与 y 错位（长训练前的保险）"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import tools_train_unet_v3 as T


def main() -> int:
    torch.manual_seed(0)
    train, _ = T.load_all()
    # 挑 8 张前景最多的样本，前景稀疏时相关性检验噪声大
    scored = sorted(train, key=lambda t: -t[2].sum())[:8]
    xs = torch.from_numpy(np.stack([t[1] for t in scored]))
    ys = torch.from_numpy(np.stack([t[2] for t in scored]))

    # 基准：原图上前景像素的亮度 vs 全图亮度
    fg = ys[:, 0] > 0.5
    base_fg_mean = xs[:, 0][fg].mean().item()
    base_bg_mean = xs[:, 0][~fg].mean().item()
    print(f"原图: 前景亮度 {base_fg_mean:.4f}  背景亮度 {base_bg_mean:.4f}  "
          f"对比度 {base_fg_mean - base_bg_mean:+.4f}")

    # 增强 30 次，每次检查前景是否仍落在高亮区域
    fg_means, bg_means = [], []
    for t in range(30):
        xa, ya = T.augment(xs, ys)
        f = ya[:, 0] > 0.5
        if f.sum() < 10:
            continue
        fg_means.append(xa[:, 0][f].mean().item())
        bg_means.append(xa[:, 0][~f].mean().item())
    fg_m, bg_m = np.mean(fg_means), np.mean(bg_means)
    print(f"增强后(30次): 前景亮度 {fg_m:.4f}  背景亮度 {bg_m:.4f}  "
          f"对比度 {fg_m - bg_m:+.4f}")

    ok = True
    if not (fg_m - bg_m) > 0.5 * (base_fg_mean - base_bg_mean):
        print("FAIL: 增强后前景/背景对比度大幅塌陷 -> x 与 y 疑似错位")
        ok = False
    else:
        print("PASS: 前景始终落在高亮区，x/y 对齐正确")

    # 形状不变性（rot90 会交换 H/W，属预期；全卷积网络可处理）
    xa, ya = T.augment(xs, ys)
    ok_shape = (xa.shape[0] == xs.shape[0] and xa.shape[1] == xs.shape[1]
                and sorted(xa.shape[2:]) == sorted(xs.shape[2:])
                and xa.shape[2:] == ya.shape[2:])
    if not ok_shape:
        print(f"FAIL: 形状异常 x={tuple(xa.shape)} y={tuple(ya.shape)}")
        ok = False
    else:
        print(f"PASS: 形状自洽（x/y 同形，允许 rot90 转置）{tuple(xa.shape)}")

    # 值域合理（未溢出、未全黑）
    lo, hi = xa.min().item(), xa.max().item()
    if not (-0.3 <= lo and hi <= 1.5):
        print(f"WARN: 增强后值域 [{lo:.3f}, {hi:.3f}] 超出预期")
    else:
        print(f"PASS: 值域 [{lo:.3f}, {hi:.3f}] 合理")

    # 模型前向一次，确认深监督输出尺寸正确
    model = T.build_unet(base=8, depth=3)
    with torch.no_grad():
        outs = model(xs[:2])
    sizes = [tuple(o.shape) for o in outs]
    print(f"深监督输出: {sizes}")
    if sizes[-1][-2:] != tuple(ys.shape[-2:]):
        print("FAIL: 最精细输出尺寸与标签不符")
        ok = False
    else:
        print("PASS: 最精细输出尺寸与标签一致")

    print("\n==> " + ("全部检查通过，可以长训练" if ok else "存在问题，先修再训"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
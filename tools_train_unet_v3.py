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
tools_train_unet_v3.py - 中后肠分割 U-Net 第二轮训练（针对 v2 欠拟合诊断）

v2 诊断结论（tools_diagnose.py）：
  - 阈值扫描 0.3~0.7 Dice 几乎不变 -> 阈值不是瓶颈
  - 灾难性失败 0/40 -> 数据/标注可学，没有"看不懂"的图
  - 逐图 Dice 42.5% 落在 0.4~0.6 -> 全员"半对"，典型欠拟合
=> 处方：加容量 + 加轮次 + 加正则

相对 v2 的四处升级（前三项几乎不增加算力）：
  1. 深监督 deep supervision：每个解码层接辅助分割头，
     迫使编码器浅层就学到判别性特征，小数据集上通常 +0.05 Dice
  2. 权重 EMA：平滑 SGD/Adam 噪声，低数据量下稳定提升
  3. 强增强：仿射(旋转/平移/缩放/剪切) + 亮度对比度抖动 + 弹性，
     对付荧光强度与鱼体姿态差异
  4. 容量：base 可配（默认 32，v2 是 16），depth 可配

环境变量：
  UNET_EPOCHS  轮次（默认 150）
  UNET_BASE    base 通道（默认 32）
  UNET_DEPTH   编码层数（默认 3）
  UNET_SCALE   下采样倍率（默认 4）
  UNET_LR      学习率（默认 2e-3）
  UNET_BENCH   设为 1 则只跑 2 轮测速并退出

产出：annotations/unet_model_v3.pt / unet_metrics_v3.txt / unet_vis_v3/*.png
"""
from __future__ import annotations

import copy
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
ANN = os.environ.get("UNET_ANN_DIR") or os.path.join(ROOT, "annotations")

SCALE = int(os.environ.get("UNET_SCALE", "4"))
EPOCHS = int(os.environ.get("UNET_EPOCHS", "150"))
BASE = int(os.environ.get("UNET_BASE", "32"))
DEPTH = int(os.environ.get("UNET_DEPTH", "3"))
LR = float(os.environ.get("UNET_LR", "2e-3"))
BATCH = int(os.environ.get("UNET_BATCH", "8"))
BENCH = os.environ.get("UNET_BENCH", "") == "1"
RESUME = os.environ.get("UNET_RESUME", "1") == "1"   # 断点续训（防机器休眠中断）
NORM = os.environ.get("UNET_NORM", "bn")             # bn | gn
AUG_DECAY = os.environ.get("UNET_AUG_DECAY", "1") == "1"   # 训练后期衰减增强强度
SEED = int(os.environ.get("UNET_SEED", "0"))              # 换种子可获得集成多样性
# 从已有权重初始化（卷积权重与分辨率无关，可跨 SCALE 直接加载），
# 用于在更高分辨率上做短时微调；与 RESUME 不同，不恢复优化器/轮次
INIT = os.environ.get("UNET_INIT", "")

# 备用标签保留：DeepLab 式 skip，保证前景梯度能回传
IGNORE_IDX = 255
LABEL_WEIGHT = 20.0          # 正样本像素只占 0.32%，必须加权


def preprocess(name: str, src_tif: str):
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
    return g_s[None], m_s[None]


def load_all():
    import csv
    rows = list(csv.DictReader(open(os.path.join(ANN, "manifest.csv"),
                                    encoding="utf-8-sig")))
    train, test = [], []
    for i, r in enumerate(rows):
        x, y = preprocess(r["name"], r["src_tif"])
        (train if r["split"] == "train" else test).append((r["name"], x, y))
        if (i + 1) % 50 == 0:
            print(f"  预处理 {i + 1}/{len(rows)}", flush=True)
    return train, test


def build_unet(base=BASE, depth=DEPTH):
    """base/depth 可配的 U-Net，带深监督头。返回模型（输出为 粗->细 列表）"""
    import torch.nn as nn

    def make_norm(c):
        """GroupNorm 不依赖 batch 统计，训练/推理行为一致。
        v3 用 BatchNorm 时 eval 精度在 epoch 间剧烈震荡（0.55~0.80），
        根因就是 BN 的 running 统计与真实 batch 统计失配。"""
        return nn.GroupNorm(min(8, c), c) if NORM == "gn" \
            else nn.BatchNorm2d(c)

    class ConvBNReLU(nn.Sequential):
        def __init__(self, c_in, c_out):
            super().__init__(
                nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
                make_norm(c_out),
                nn.ReLU(inplace=True))

    def conv_block(c_in, c_out):
        return nn.Sequential(ConvBNReLU(c_in, c_out), ConvBNReLU(c_out, c_out))

    class UNetV3(nn.Module):
        """depth=3: enc1(b) enc2(2b) enc3(4b)   解码到原尺寸
           depth=4: 追加 enc4(8b) 并多一级解码（显存/时间约 +35%）"""

        def __init__(self):
            super().__init__()
            chs = [base * (2 ** i) for i in range(depth + 1)]
            self.pool = nn.MaxPool2d(2)
            self.encs = nn.ModuleList()
            self.encs.append(conv_block(1, chs[0]))
            for i in range(1, depth + 1):
                self.encs.append(conv_block(chs[i - 1], chs[i]))
            self.ups = nn.ModuleList()
            self.decs = nn.ModuleList()
            for i in range(depth, 0, -1):
                self.ups.append(nn.ConvTranspose2d(chs[i], chs[i - 1], 2, stride=2))
                self.decs.append(conv_block(chs[i - 1] * 2, chs[i - 1]))
            # 深监督：解码各级一个头（输出保持 粗->细 顺序，outs[-1] 为最终预测）
            self.heads = nn.ModuleList([nn.Conv2d(chs[depth - 1 - i], 1, 1)
                                        for i in range(depth)])
            self.depth = depth

        def forward(self, x):
            feats = []
            h = x
            for enc in self.encs:
                h = enc(h)
                feats.append(h)
                h = self.pool(h)
            outs = []
            h = feats[-1]
            for i, (up, dec) in enumerate(zip(self.ups, self.decs)):
                skip = feats[self.depth - 1 - i]
                h = up(h)
                if h.shape[-2:] != skip.shape[-2:]:
                    h = torch.nn.functional.interpolate(
                        h, size=skip.shape[-2:], mode="nearest")
                h = dec(torch.cat([h, skip], dim=1))
                outs.append(self.heads[i](h))
            return outs            # outs[i] 逐级变精细，outs[-1] 为最终输出

    return UNetV3()


# ---------------- 损失 ----------------
def dice_loss(logits, targets, eps=1e-6):
    p = torch.sigmoid(logits)
    inter = (p * targets).sum(dim=(2, 3))
    denom = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def seg_loss(outs, yb, device):
    """加权 BCE + SoftDice，深监督权重 1/(2^depth)..1 递减"""
    import torch.nn as nn
    pos_weight = torch.tensor(LABEL_WEIGHT, device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    total, n = 0.0, len(outs)
    wsum = 0.0
    for i, o in enumerate(outs):
        w = 1.0 / (2 ** i)               # i=0 最粗权重最低，末级最精细权重 1.0
        up = torch.nn.functional.interpolate(
            o, size=yb.shape[-2:], mode="bilinear", align_corners=False) \
            if o.shape[-2:] != yb.shape[-2:] else o
        total = total + w * (bce(up, yb) + dice_loss(up, yb).mean())
        wsum += w
    return total / wsum


# ---------------- 增强 ----------------
def augment(x, y, strength=1.0):
    """strength=1.0 全强度；训练后期线性衰减到 0，让模型在接近真实的
    干净输入上收敛（强增强全程不变会让 train/eval 分布始终错位）。"""
    import torch
    import torch.nn.functional as F
    if strength <= 0:
        return x.contiguous(), (y > 0.5).float().contiguous()
    # 1) 几何：翻转 + 90 度旋转
    if torch.rand(1).item() < 0.5 * strength:
        x, y = torch.flip(x, [3]), torch.flip(y, [3])
    if torch.rand(1).item() < 0.5 * strength:
        x, y = torch.flip(x, [2]), torch.flip(y, [2])
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        x = torch.rot90(x, k, [2, 3]); y = torch.rot90(y, k, [2, 3])
    # 2) 仿射：旋转 ±12度 / 平移 ±6% / 缩放 0.9~1.1 / 剪切 ±5度
    if torch.rand(1).item() < 0.7 * strength:
        ang = (torch.rand(1).item() - 0.5) * 24 * np.pi / 180 * strength
        shear = (torch.rand(1).item() - 0.5) * 10 * np.pi / 180 * strength
        sc = 1.0 + (0.9 + torch.rand(1).item() * 0.2 - 1.0) * strength
        tx = (torch.rand(1).item() - 0.5) * 0.12 * strength
        ty = (torch.rand(1).item() - 0.5) * 0.12 * strength
        cos, sin = np.cos(ang) * sc, np.sin(ang) * sc
        sh = np.tan(shear) * sc
        theta = torch.tensor(
            [[cos, -sin + sh * cos, tx], [sin, cos + sh * sin, ty]],
            dtype=torch.float32, device=x.device).unsqueeze(0).expand(x.shape[0], -1, -1)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                          align_corners=False)
        y = F.grid_sample(y, grid, mode="nearest", padding_mode="border",
                          align_corners=False)
    # 3) 光度：亮度/对比度抖动 + 轻噪声（荧光强度批次差异很大）
    if torch.rand(1).item() < 0.8 * strength:
        gain = 1.0 + (0.8 + torch.rand(1).item() * 0.5 - 1.0) * strength
        bias = (torch.rand(1).item() - 0.5) * 0.12 * strength
        x = x * gain + bias
    if torch.rand(1).item() < 0.4 * strength:
        x = x + torch.randn_like(x) * (0.02 + torch.rand(1).item() * 0.04) * strength
    return x.contiguous(), (y > 0.5).float().contiguous()


class EMA:
    """指数滑动平均权重，用于评估与保存"""

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for se, p in zip(self.shadow.state_dict().values(),
                         model.state_dict().values()):
            if se.dtype.is_floating_point:
                se.mul_(d).add_(p.detach(), alpha=1 - d)
            else:
                se.copy_(p)


def evaluate(model, xe, ye, threshold=0.5, chunk=None):
    """分块推理：一次性喂 40 张 552x688 会让单层激活达 1.8GB 直接 OOM。
    chunk 按像素预算自适应——分辨率越高（如 ÷3）每张占用越大，块要相应变小。"""
    if chunk is None:
        px = xe.shape[-1] * xe.shape[-2]
        chunk = max(1, min(8, int(8 * (552 * 688) / max(px, 1))))
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, xe.shape[0], chunk):
            outs = model(xe[i:i + chunk])
            logits = outs[-1] if isinstance(outs, (list, tuple)) else outs
            pred = (torch.sigmoid(logits) > threshold).float()
            yb = ye[i:i + chunk]
            inter = (pred * yb).sum(dim=(1, 2, 3))
            denom = pred.sum(dim=(1, 2, 3)) + yb.sum(dim=(1, 2, 3))
            out.append((2 * inter + 1e-6) / (denom + 1e-6))
    model.train()
    return torch.cat(out).cpu().numpy()


def main() -> int:
    USE_CUDA = os.environ.get("UNET_CUDA", "1") == "1" \
        and torch.cuda.is_available()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if USE_CUDA else "cpu")
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True     # 固定输入尺寸下选最快卷积算法
    threads = max(1, os.cpu_count() or 4)
    torch.set_num_threads(threads)
    print(f"torch {torch.__version__}", flush=True)
    print(f"设备: {device}"
          + (f"（{torch.cuda.get_device_name(0)}, "
             f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB）"
             if USE_CUDA else f"（CPU {threads} 线程）"), flush=True)
    print(f"配置: ÷{SCALE} 输入, base={BASE}, depth={DEPTH}, "
          f"{EPOCHS} epochs, batch {BATCH}, lr {LR}, AMP={USE_CUDA}", flush=True)

    print("加载数据...", flush=True)
    t0 = time.time()
    train, test = load_all()
    print(f"train {len(train)} / test {len(test)}，"
          f"输入 {train[0][1].shape}，耗时 {time.time() - t0:.0f}s", flush=True)

    xt = torch.from_numpy(np.stack([x for _, x, _ in train])).to(device)
    yt = torch.from_numpy(np.stack([y for _, _, y in train])).to(device)
    xe = torch.from_numpy(np.stack([x for _, x, _ in test])).to(device)
    ye = torch.from_numpy(np.stack([y for _, _, y in test])).to(device)
    print(f"正样本像素占比 {yt.mean().item() * 100:.2f}%"
          f"，数据常驻显存 {torch.cuda.memory_allocated() / 1e9:.2f} GB"
          if USE_CUDA else f"正样本像素占比 {yt.mean().item() * 100:.2f}%",
          flush=True)

    model = build_unet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量 {n_params / 1e6:.2f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)
    warm = max(1, EPOCHS // 20)

    def lr_lambda(ep):
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(1, EPOCHS - warm)
        return 0.5 * (1 + np.cos(np.pi * prog))     # cosine 衰减

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, decay=0.995)

    n = xt.shape[0]
    epochs = 2 if BENCH else EPOCHS
    best_dice, best_epoch = -1.0, -1
    hist = []
    start_epoch = 1

    # ---- 从已有权重初始化（跨分辨率微调）----
    if INIT:
        ipath = INIT if os.path.isabs(INIT) else os.path.join(ANN, INIT)
        ic = torch.load(ipath, map_location="cpu", weights_only=False)
        model.load_state_dict(ic["state_dict"])
        print(f"从 {os.path.basename(ipath)} 初始化权重"
              f"（原 scale={ic.get('scale')} epoch={ic.get('epoch')} "
              f"Dice={ic.get('dice', float('nan')):.4f} -> 现 scale={SCALE}）",
              flush=True)

    # ---- 断点续训 ----
    resume_path = os.path.join(ANN, "unet_resume_v3.pt")
    if RESUME and not BENCH and os.path.exists(resume_path):
        try:
            rc = torch.load(resume_path, map_location="cpu", weights_only=False)
            if rc.get("base") == BASE and rc.get("depth") == DEPTH \
                    and rc.get("scale") == SCALE and rc.get("epochs") == EPOCHS \
                    and rc.get("norm", "bn") == NORM:
                model.load_state_dict(rc["model"])
                opt.load_state_dict(rc["opt"])
                sched.load_state_dict(rc["sched"])
                ema.shadow.load_state_dict(rc["ema"])
                start_epoch = rc["epoch"] + 1
                best_dice = rc.get("best_dice", -1.0)
                best_epoch = rc.get("best_epoch", -1)
                hist = [tuple(h) for h in rc.get("hist", [])]
                print(f"断点续训：从 epoch {start_epoch}/{EPOCHS} 恢复，"
                      f"历史最佳 Dice {best_dice:.4f} (epoch {best_epoch})", flush=True)
            else:
                print("续训检查点配置不匹配，改为从头训练", flush=True)
        except Exception as e:
            print(f"续训检查点读取失败（{e}），从头训练", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        te = time.time()
        model.train()
        # 增强退火：最后 25% 训练把强度线性降到 0
        if AUG_DECAY and not BENCH:
            prog_aug = (epoch - 1) / max(1, EPOCHS - 1)
            aug_strength = float(np.clip(1.0 - prog_aug / 0.75, 0.0, 1.0))
        else:
            aug_strength = 1.0
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            xb, yb = augment(xt[idx], yt[idx], aug_strength)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_CUDA):
                outs = model(xb)
            # 损失统一在 fp32 计算：BCE/SoftDice 对精度敏感，fp16 会不稳
            outs = [o.float() for o in outs]
            loss = seg_loss(outs, yb, device)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            tot += loss.item(); nb += 1
        sched.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            d_raw = evaluate(model, xe, ye)
            d_ema = evaluate(ema.shadow, xe, ye)
            best_d_now = max(d_raw.mean(), d_ema.mean())
            print(f"epoch {epoch:3d}/{epochs}  loss {tot / nb:.4f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  "
                  f"Dice raw {d_raw.mean():.4f} / ema {d_ema.mean():.4f}  "
                  f"({time.time() - te:.0f}s)", flush=True)
            hist.append((epoch, d_raw.mean(), d_ema.mean()))
            if best_d_now > best_dice:
                src = ema.shadow if d_ema.mean() >= d_raw.mean() else model
                torch.save({"state_dict": {k: v.detach().cpu().clone()
                                          for k, v in src.state_dict().items()},
                            "scale": SCALE, "base": BASE, "depth": DEPTH,
                            "norm": NORM, "seed": SEED,
                            "epoch": epoch, "dice": float(best_d_now)},
                           os.path.join(ANN, "unet_model_v3.pt"))
                best_dice, best_epoch = float(best_d_now), epoch
            # 每轮保存续训状态（休眠/断电可恢复）
            if not BENCH:
                torch.save({"model": {k: v.detach().cpu().clone()
                                      for k, v in model.state_dict().items()},
                            "opt": opt.state_dict(),
                            "sched": sched.state_dict(),
                            "ema": {k: v.detach().cpu().clone()
                                    for k, v in ema.shadow.state_dict().items()},
                            "epoch": epoch, "best_dice": best_dice,
                            "best_epoch": best_epoch, "hist": hist,
                            "base": BASE, "depth": DEPTH, "scale": SCALE,
                            "epochs": EPOCHS},
                           resume_path)
        else:
            print(f"epoch {epoch:3d}/{epochs}  loss {tot / nb:.4f}  "
                  f"({time.time() - te:.0f}s)", flush=True)

    if BENCH:
        print(f"\n[基准测试] {epochs} 轮耗时 {time.time() - t0:.0f}s "
              f"-> 外推 {EPOCHS} 轮约 "
              f"{(time.time() - t0) / max(1, epochs) * EPOCHS / 60:.0f} 分钟")
        return 0

    # ---- 最终评估：阈值扫描 + 全图口径 ----
    print("\n最终评估...")
    ckpt = torch.load(os.path.join(ANN, "unet_model_v3.pt"),
                      map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    probs = []
    _px = xe.shape[-1] * xe.shape[-2]
    _chunk = max(1, min(8, int(8 * (552 * 688) / max(_px, 1))))
    with torch.no_grad():
        for i in range(0, xe.shape[0], _chunk):   # 分块，避免 OOM
            probs.append(torch.sigmoid(
                model(xe[i:i + _chunk])[-1]).float().cpu().numpy()[:, 0])
    prob_all = np.concatenate(probs, axis=0)
    gt = ye.cpu().numpy()[:, 0]

    print(f"{'阈值':>6} {'Dice均值':>9} {'中位':>8} {'P10':>8}")
    best_th, best_mean = 0.5, -1
    for th in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
        pred = (prob_all > th).astype(np.float32)
        inter = (pred * gt).sum(axis=(1, 2))
        denom = pred.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
        dd = (2 * inter + 1e-6) / (denom + 1e-6)
        print(f"{th:>6.2f} {dd.mean():>9.4f} {np.median(dd):>8.4f} "
              f"{np.percentile(dd, 10):>8.4f}")
        if dd.mean() > best_mean:
            best_mean, best_th = float(dd.mean()), th
    print(f"-> 最佳阈值 {best_th}（Dice {best_mean:.4f}）")

    pred = (prob_all > 0.5).astype(np.float32)
    inter = (pred * gt).sum(axis=(1, 2))
    denom = pred.sum(axis=(1, 2)) + gt.sum(axis=(1, 2))
    dices = (2 * inter + 1e-6) / (denom + 1e-6)
    parea = pred.sum(axis=(1, 2)); garea = gt.sum(axis=(1, 2))
    ratio = parea / np.maximum(garea, 1e-6)
    print(f"面积比中位 {np.median(ratio):.2f}  "
          f"过分割(>2x) {int((ratio > 2).sum())} 张  "
          f"欠分割(<0.5x) {int((ratio < 0.5).sum())} 张")
    print("Dice 分布: " + "  ".join(
        f"{lo:.1f}-{hi:.1f}:{int(((dices >= lo) & (dices < hi)).sum())}张"
        for lo, hi in [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]))

    # ---- 可视化 ----
    import cv2
    vis_dir = os.path.join(ANN, "unet_vis_v3")
    os.makedirs(vis_dir, exist_ok=True)
    from tools_collect_masks import imread_unicode
    from zebrafish_quant.segment import dice_coefficient
    order = np.argsort(dices)
    picks = list(order[:3]) + list(order[-3:])
    full_d = []
    for j in picks:
        nm = test[j][0]
        pm = (prob_all[j] > 0.5).astype(np.uint8) * 255
        gt_full = imread_unicode(os.path.join(ANN, "masks", nm + ".png"))
        pred_full = cv2.resize(pm, (gt_full.shape[1], gt_full.shape[0]),
                               interpolation=cv2.INTER_NEAREST)
        fd = dice_coefficient(pred_full, gt_full)
        full_d.append(fd)
        rgb = np.dstack([gt_full, pred_full, np.zeros_like(gt_full)])
        cv2.imwrite(os.path.join(vis_dir, f"{nm}_d{dices[j]:.2f}.png"), rgb)
    print(f"可视化 {len(picks)} 张 -> unet_vis_v3/"
          f"（绿=GT 红=预测 黄=重合）")

    report = f"""中后肠 U-Net v3 训练报告（针对 v2 欠拟合的第二轮）
{'=' * 56}
背景：v2 诊断（tools_diagnose.py）显示阈值非瓶颈、0 张灾难失败、
     42.5% 样本挤在 Dice 0.4~0.6 -> 判定欠拟合，非数据问题。

v3 改动：深监督 + 权重EMA + 仿射/光度增强 + base {BASE} depth {DEPTH}
配置：输入 ÷{SCALE}（{train[0][1].shape[2]}x{train[0][1].shape[1]}），
      {n_params / 1e6:.2f}M 参数，AdamW lr={LR} + warmup+cosine，
      加权BCE(pos_weight={LABEL_WEIGHT}) + SoftDice，{EPOCHS} epochs
数据：train {len(train)} / test {len(test)}（按图划分，人工掩码 GT）

结果：
  测试集 Dice @0.5: 均值 {dices.mean():.4f}  中位 {np.median(dices):.4f}  P10 {np.percentile(dices, 10):.4f}
  最佳阈值 {best_th} -> Dice {best_mean:.4f}（较 @0.5 {best_mean - dices.mean():+.4f}）
  极端 4 张全图口径 Dice: 均值 {np.mean(full_d):.4f}
  最佳 epoch: {best_epoch}（Dice {best_dice:.4f}）
  v2 基线 @0.5 = 0.5798 -> v3 提升 {dices.mean() - 0.5798:+.4f}
  验收标准（需求 7）：Dice >= 0.85 -> {'达标' if dices.mean() >= 0.85 else '仍未达标'}

训练曲线（epoch, raw, ema）:
{chr(10).join(f'  {e:4d}  {a:.4f}  {b:.4f}' for e, a, b in hist)}

模型：annotations/unet_model_v3.pt
"""
    with open(os.path.join(ANN, "unet_metrics_v3.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"总耗时 {(time.time() - t0) / 60:.0f} 分钟")
    return 0


if __name__ == "__main__":
    sys.exit(main())
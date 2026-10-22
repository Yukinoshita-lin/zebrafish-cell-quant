#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""tools_gpu_profile.py - 分阶段定位 GPU 训练慢在哪一环

分别计时：数据增强 / 前向 / 损失 / 反向 / 优化器 / EMA / 评估
并对比 cudnn.benchmark 开关与 batch 大小的影响。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def timeit(fn, n=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def profile(base, depth, batch, amp, benchmark, h=552, w=688):
    torch.backends.cudnn.benchmark = benchmark
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import tools_train_unet_v3 as T

    model = T.build_unet(base=base, depth=depth).cuda()
    n_par = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = T.EMA(model, decay=0.995)
    x = torch.rand(batch, 1, h, w, device="cuda")
    y = (torch.rand(batch, 1, h, w, device="cuda") > 0.997).float()

    t_aug = timeit(lambda: T.augment(x, y))

    def fwd():
        with torch.amp.autocast("cuda", enabled=amp):
            outs = model(x)
        return [o.float() for o in outs]

    t_fwd = timeit(fwd)

    outs = fwd()

    def loss_step():
        return T.seg_loss(outs, y, torch.device("cuda"))

    t_loss = timeit(loss_step)

    def full():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp):
            o = model(x)
        o = [t.float() for t in o]
        l = T.seg_loss(o, y, torch.device("cuda"))
        scaler.scale(l).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        ema.update(model)

    t_full = timeit(full, n=3, warm=2)

    xe = torch.rand(40, 1, h, w, device="cuda")
    ye = (torch.rand(40, 1, h, w, device="cuda") > 0.997).float()
    t_eval = timeit(lambda: T.evaluate(model, xe, ye), n=2, warm=1)

    peak = torch.cuda.max_memory_allocated() / 1e9
    n_steps = int(np.ceil(150 / batch))
    print(f"base={base} depth={depth} batch={batch} amp={amp} "
          f"cudnn.bench={benchmark} | {n_par:.2f}M参数 {h}x{w}")
    print(f"  增强 {t_aug:7.1f}ms | 前向 {t_fwd:7.1f}ms | 损失 {t_loss:7.1f}ms"
          f" | 全步 {t_full:7.1f}ms | 评估(40图) {t_eval:7.1f}ms")
    print(f"  峰值显存 {peak:.2f}GB | 每轮({n_steps}步) 约 "
          f"{(t_full * n_steps + t_eval) / 1000:.0f}s")
    print()
    del model, opt, ema, x, y, xe, ye
    torch.cuda.empty_cache()
    return t_full, peak


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA 不可用")
        return 1
    print(f"设备: {torch.cuda.get_device_name(0)}\n")
    # 对比：batch 与 cudnn.benchmark 的影响
    profile(32, 3, 8, True, True)
    profile(32, 3, 8, True, False)
    profile(32, 4, 8, True, True)
    profile(32, 4, 16, True, True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
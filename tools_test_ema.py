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

"""tools_test_ema.py - 精确验证 EMA 递推（CPU，不干扰 GPU 训练）

做法：每步把某个参数设成已知值 p_t，逐步记录，
      与精确递推 ew_t = d*ew_{t-1} + (1-d)*p_t 逐位对照。
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from tools_train_unet_v3 import EMA

DECAY = 0.9
STEPS = 50


def main() -> int:
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 1, 1))
    ema = EMA(model, decay=DECAY)
    w0 = model[0].weight.detach().clone()
    assert torch.allclose(ema.shadow[0].weight.detach(), w0), "初始应等于模型权重"

    x = torch.randn(4, 1, 16, 16)
    model.train()
    seq = []
    for t in range(STEPS):
        # 让参数走确定值 p_t = 0.1*t，避免随机游走干扰对照
        with torch.no_grad():
            model[0].weight.fill_(0.1 * (t + 1))
        seq.append(0.1 * (t + 1))
        ema.update(model)

    # 精确递推
    expect = w0.clone()
    for p in seq:
        expect = DECAY * expect + (1 - DECAY) * p
    got = ema.shadow[0].weight.detach()
    err = (got - expect).abs().max().item()
    print(f"EMA 与精确递推最大误差 {err:.3e} -> "
          f"{'逻辑正确' if err < 1e-6 else '逻辑有误'}")

    if err >= 1e-6:
        print(f"  期望末值 {expect.flatten()[:3].tolist()}")
        print(f"  实际末值 {got.flatten()[:3].tolist()}")
        return 1

    # BN buffer 是否同步更新
    rm_changed = not torch.allclose(
        ema.shadow[1].running_mean, torch.zeros_like(ema.shadow[1].running_mean))
    print("BN running_mean 同步更新:", rm_changed)
    print("BN num_batches_tracked 类型正确:",
          ema.shadow[1].num_batches_tracked.dtype == torch.int64)

    # 前向可用
    ema.shadow.eval()
    with torch.no_grad():
        out = ema.shadow(x)
    print("EMA 前向输出:", tuple(out.shape),
          f"最大绝对值 {out.abs().max().item():.4f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
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

"""tools_norm_mem.py - 对比 BatchNorm / GroupNorm 的峰值显存与前向耗时"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def run(norm, batch=8, h=552, w=688):
    import tools_train_unet_v3 as V3
    V3.NORM = norm
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = V3.build_unet(base=32, depth=4).cuda()
    x = torch.rand(batch, 1, h, w, device="cuda")
    try:
        with torch.no_grad():
            for _ in range(2):
                model(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        dt = (time.time() - t) / 5 * 1000
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{norm:<4} batch={batch}  前向 {dt:7.1f}ms  "
              f"峰值显存 {peak:.2f}GB  OK")
    except Exception as e:
        print(f"{norm:<4} batch={batch}  失败: {type(e).__name__}: "
              f"{str(e)[:90]}")
    del model, x
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for norm in ["bn", "gn"]:
        for b in [8, 4]:
            run(norm, b)
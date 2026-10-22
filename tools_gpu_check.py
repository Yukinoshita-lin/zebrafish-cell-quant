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

import time

import torch

print("torch:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA 不可用")

print("设备:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("算力: sm_%d%d" % cap)
print("显存: %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))

# 实测一次前向+反向的耗时，用来估算每轮训练时间
dev = "cuda"
x = torch.randn(8, 1, 552, 688, device=dev)
y = (torch.rand(8, 1, 552, 688, device=dev) > 0.997).float()
model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
    torch.nn.Conv2d(32, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
    torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
    torch.nn.Conv2d(128, 1, 1),
).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
lossfn = torch.nn.BCEWithLogitsLoss()


def step():
    opt.zero_grad()
    out = model(x)
    loss = lossfn(out, y)
    loss.backward()
    opt.step()


for _ in range(5):
    step()
torch.cuda.synchronize()
t = time.time()
N = 30
for _ in range(N):
    step()
torch.cuda.synchronize()
per_step = (time.time() - t) / N
print("单步(batch8, 552x688, 32-128ch): %.1f ms" % (per_step * 1000))
print("推算每轮(150图/8=19步): %.1f s" % (per_step * 19))
print("峰值显存: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
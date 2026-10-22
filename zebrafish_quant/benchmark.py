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
benchmark.py - 后端实测与择优

策略：
    1. 枚举所有可用后端（GPU 原生 / C++ CPU 原生 / 纯 Python 兜底）
    2. 【原生后端参与速度竞争】逐个加载模型并在同一批样本切块上计时
       （含 warmup），按吞吐选最快。
       classical_python 是兜底，**不参与速度竞争**：它在近空白切块上
       可能比 GPU 还快（实测 320 vs 108 tiles/s），但精度远低于训练好
       的 YOLO（实测 26 vs 283 个检测）——按速度选它会 silent 地损失
       约 90% 检出。仅当所有原生后端都不可用（DLL 缺失/加载失败）时
       才启用兜底。想显式使用可用 --backend / prefer 强制。
    3. 结果缓存到 JSON；下次启动命中缓存，但会重新校验：
       "chosen 仍可用 + 没有更低 tier 的后端新出现"，失效则重新实测
       （后者保证 GPU DLL 被修复/恢复后能从 CPU 缓存中升级回 GPU）。

任何后端在加载或推理中出错，都会被记录并**跳过**，不会中断选择流程
——这正是"缺失加速硬件自动回退"的实现点。
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends import Backend, available_backends

CACHE_VERSION = 1


def benchmark_backend(backend: Backend, tiles: List[np.ndarray],
                      thresh: float = 0.30, nms: float = 0.20,
                      warmup: int = 1, repeat: int = 1) -> Dict:
    """对单个后端计时。返回统计字典；失败时 ok=False 并附错误信息。"""
    res = {"name": backend.name, "tier": backend.tier, "ok": False,
           "sec": 0.0, "tiles_per_sec": 0.0, "error": "", "detections": 0}
    try:
        # warmup：GPU 首次推理含 kernel 编译/显存分配，不计入
        for _ in range(warmup):
            backend.detect_tiles(tiles[:1], thresh, nms)

        t0 = time.perf_counter()
        dets_total = 0
        for _ in range(repeat):
            out = backend.detect_tiles(tiles, thresh, nms)
            dets_total = sum(len(d) for d in out)
        dt = time.perf_counter() - t0

        n = len(tiles) * repeat
        res.update(ok=True, sec=round(dt, 4),
                   tiles_per_sec=round(n / dt, 3) if dt > 0 else 0.0,
                   detections=dets_total)
    except Exception as exc:                              # noqa: BLE001
        res["error"] = f"{type(exc).__name__}: {exc}"
    return res


def _cache_valid(cache: Dict, cands: List[Tuple[Backend, str]]) -> bool:
    """缓存有效性：版本匹配 + chosen 仍可用 + 没有更低 tier 的后端新出现。

    第三条保证环境升级（如 GPU DLL 恢复/修复）后能自动重新实测，
    而不是永远粘在旧选择上。
    """
    if cache.get("version") != CACHE_VERSION:
        return False
    chosen = cache.get("chosen")
    tier_of = {b.name: b.tier for b, _ in cands}
    if chosen not in tier_of:
        return False
    return not any(t < tier_of[chosen] for t in tier_of.values())


def choose_backend(config_file: str, data_file: str, weights_file: str,
                   sample_tiles: List[np.ndarray],
                   *, thresh: float = 0.30, nms: float = 0.20,
                   cache_path: Optional[str] = None,
                   prefer: Optional[str] = None,
                   verbose: bool = True) -> Tuple[Backend, Dict]:
    """实测所有可用后端并选出最快的。

    prefer: 若指定后端名且可用，则跳过实测直接采用（便于调试/强制）。
    返回 (选中的后端, 完整报告)。
    """
    cands = available_backends()
    if not cands:
        raise RuntimeError("没有任何可用后端（连纯 Python 兜底都不可用）")

    names = [b.name for b, _ in cands]

    # 1) 缓存命中
    if cache_path and os.path.exists(cache_path) and prefer is None:
        try:
            cache = json.load(open(cache_path, encoding="utf-8"))
            if _cache_valid(cache, cands):
                for b, _ in cands:
                    if b.name == cache["chosen"]:
                        b.load(config_file, data_file, weights_file)
                        if verbose:
                            print(f"[backend] 命中缓存 -> {b.name} "
                                  f"({cache.get('tiles_per_sec','?')} tiles/s)")
                        return b, {"chosen": b.name, "from_cache": True,
                                   "results": cache.get("results", [])}
        except Exception:                                 # noqa: BLE001
            pass      # 缓存损坏则退回实测

    # 2) 强制指定：找不到就明确报错（而不是悄悄换成别的，避免误判）
    if prefer:
        for b, _ in cands:
            if b.name == prefer:
                b.load(config_file, data_file, weights_file)
                return b, {"chosen": b.name, "forced": True, "results": []}
        raise RuntimeError(
            f"指定后端 {prefer!r} 不可用（动态库缺失或硬件缺失）；"
            f"当前可用: {names}")

    # 3) 逐个实测。原生后端（GPU/CPU DLL，tier<2）参与速度竞争；
    #    classical_python 是兜底，不参与——它在近空白切块上可能比 GPU
    #    更快但精度远低于 YOLO，按速度选它会 silent 损失大量检出。
    #    仅当原生后端全部不可用时才启用兜底。
    native = [(b, why) for b, why in cands if b.tier < 2]
    fallbacks = [(b, why) for b, why in cands if b.tier >= 2]
    pool = native or cands

    def _try_pool(items: List[Tuple[Backend, str]]) -> None:
        for b, _why in items:
            try:
                b.load(config_file, data_file, weights_file)
            except Exception as exc:                      # noqa: BLE001
                r = {"name": b.name, "tier": b.tier, "ok": False,
                     "sec": 0.0, "tiles_per_sec": 0.0,
                     "error": f"加载失败: {type(exc).__name__}: {exc}"}
                results.append(r)
                if verbose:
                    print(f"[backend] {b.name}: 不可用 -> {r['error']}")
                continue
            r = benchmark_backend(b, sample_tiles, thresh, nms)
            results.append(r)
            if r["ok"]:
                loaded.append((b, r))
                if verbose:
                    print(f"[backend] {b.name}: {r['tiles_per_sec']} tiles/s "
                          f"({r['sec']}s, {r['detections']} 个检测)")
            else:
                if verbose:
                    print(f"[backend] {b.name}: 推理失败 -> {r['error']}")

    results: List[Dict] = []
    loaded: List[Tuple[Backend, Dict]] = []
    if native and fallbacks and verbose:
        print("[backend] classical_python 为兜底后端，不参与速度竞争"
              "（仅在原生后端全部不可用时启用）")
    _try_pool(pool)

    if not loaded and native and fallbacks:
        # 原生后端全部加载/推理失败 -> 启用兜底，保证流水线不中断
        if verbose:
            print("[backend] 原生后端均不可用，启用兜底后端")
        _try_pool(fallbacks)

    if not loaded:
        raise RuntimeError(
            "所有后端均不可用：\n  " +
            "\n  ".join(f"{r['name']}: {r['error']}" for r in results))

    # 4) 选最快（同速则取 tier 更低的，即更"高级"的）
    loaded.sort(key=lambda t: (-t[1]["tiles_per_sec"], t[0].tier))
    best, best_r = loaded[0]

    if cache_path:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".",
                        exist_ok=True)
            json.dump({"version": CACHE_VERSION, "chosen": best.name,
                       "tiles_per_sec": best_r["tiles_per_sec"],
                       "results": results},
                      open(cache_path, "w", encoding="utf-8"),
                      ensure_ascii=False, indent=2)
        except Exception:                                 # noqa: BLE001
            pass

    if verbose:
        print(f"[backend] 选中 -> {best.name} "
              f"({best_r['tiles_per_sec']} tiles/s)")

    return best, {"chosen": best.name, "from_cache": False, "results": results}

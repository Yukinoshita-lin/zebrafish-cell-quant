# -*- coding: utf-8 -*-
"""
后端双加速路径 / 自动回退 / 对拍 / 切块流水线 自检

无需 libdarknet、无需 GPU —— 本机当前正是"DLL 缺失"状态，
因此本测试恰好验证最重要的那条路径：DLL 缺失时能否优雅降级并照样出结果。

覆盖：
  1. 后端发现：无 DLL 时只剩 classical_python
  2. GPU/CPU 库缺失时返回 None（不抛异常）
  3. choose_backend 自动选到可用后端（DLL 缺失自动回退）
  4. 强制指定不可用的 gpu 时不崩溃、给出提示
  5. native.NativeDarknet 对缺失 DLL 抛 NativeLoadError（可被上层捕获）
  6. 全局 NMS 去除跨块重复
  7. 对拍：自比对一致率为 100%
  8. 端到端：切块流水线处理合成荧光图 -> CSV + 可视化
  9. 损坏图片不中断整批
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zebrafish_quant as zq                                   # noqa: E402
from zebrafish_quant.backends import (ClassicalBackend, CpuBackend,  # noqa: E402
                                      GpuBackend, available_backends,
                                      discover_backends)
from zebrafish_quant.benchmark import choose_backend           # noqa: E402
from zebrafish_quant.crosscheck import crosscheck, format_report  # noqa: E402
from zebrafish_quant.inference import Detection                # noqa: E402
from zebrafish_quant.tiling import nms_boxes_global            # noqa: E402
from zebrafish_quant.tiling_pipeline import TilingConfig, TilingPipeline  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  -> {detail}" if detail else ""))


def make_fluorescence(w=1100, h=900, cells=None, seed=0):
    """合成图：暗背景 + 横向肠道管 + 若干绿色亮点。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 12, np.uint8)
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, 70), 0, 0, 360, (30, 55, 28), -1)
    for (cx, cy) in (cells or []):
        cv2.circle(img, (int(cx), int(cy)), 6, (20, 235, 30), -1)
        cv2.circle(img, (int(cx), int(cy)), 3, (60, 255, 70), -1)
    return np.clip(img.astype(np.int16) + rng.integers(0, 8, (h, w, 3)),
                   0, 255).astype(np.uint8)


def gut_mask(w=1100, h=900):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, h // 2), (w // 3, 70), 0, 0, 360, 255, -1)
    return m


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="zq_backend_")
    try:
        # ---------- 1. 后端发现 ----------
        print("\n[1] 后端发现（当前无 darknet DLL）")
        cands = discover_backends()
        names = [b.name for b in cands]
        check("发现后端列表含兜底 classical_python", "classical_python" in names, str(names))
        check("无 GPU 库时 GpuBackend() 返回 None（不抛异常）", GpuBackend() is None)

        avail = available_backends()
        check("可用后端非空", len(avail) >= 1, str([(b.name, w) for b, w in avail]))
        check("classical_python 始终可用",
              any(b.name == "classical_python" for b, _ in avail))

        # ---------- 2. DLL 缺失时的加载错误可捕获 ----------
        print("\n[2] DLL 缺失 -> 抛可捕获的 NativeLoadError")
        from zebrafish_quant.native import NativeDarknet, NativeLoadError
        try:
            NativeDarknet(os.path.join(tmp, "no_such.dll"))
            check("缺失 DLL 抛 NativeLoadError", False, "未抛异常")
        except NativeLoadError:
            check("缺失 DLL 抛 NativeLoadError", True, "上层可捕获并回退")

        # ---------- 3. 自动选择（DLL 缺失自动回退）----------
        print("\n[3] choose_backend 自动回退")
        img = make_fluorescence(cells=[(400, 450), (520, 470), (600, 440)])
        tiles = []
        from zebrafish_quant.tiling import iter_tiles, tile_has_signal
        for t in iter_tiles(img, 416, 64):
            if tile_has_signal(t.image):
                tiles.append(t.image)
            if len(tiles) >= 4:
                break
        check("取到基准切块", len(tiles) >= 1, f"{len(tiles)} 块")

        cfg = os.path.join(tmp, "m.cfg")
        dat = os.path.join(tmp, "m.data")
        wts = os.path.join(tmp, "m.weights")
        for p in (cfg, dat, wts):
            open(p, "wb").write(b"x")

        backend, rep = choose_backend(cfg, dat, wts, tiles, cache_path=None,
                                      verbose=False)
        check("无 DLL 时自动回退到 classical_python",
              backend.name == "classical_python", backend.name)
        check("返回报告含实测结果", "results" in rep, str(rep.get("chosen")))

        # ---------- 4. 强制不可用后端不崩溃 ----------
        print("\n[4] 强制指定不可用 gpu -> 不崩溃")
        try:
            b2, r2 = choose_backend(cfg, dat, wts, tiles, cache_path=None,
                                    prefer="gpu", verbose=False)
            check("prefer=gpu 时抛错（gpu 不在可用列表）", False, b2.name)
        except Exception as exc:                               # noqa: BLE001
            check("prefer 不存在时抛 RuntimeError", True, type(exc).__name__)
        b3, _ = choose_backend(cfg, dat, wts, tiles, cache_path=None,
                               prefer="classical_python", verbose=False)
        check("prefer=classical_python 生效", b3.name == "classical_python")

        # ---------- 5. 全局 NMS 去重 ----------
        print("\n[5] 全局 NMS 去除跨块重复")
        dup = [
            (100, 100, 12, 12, 0.9),
            (101, 101, 12, 12, 0.8),   # 与上者几乎重合 -> 应被抑制
            (300, 300, 12, 12, 0.7),   # 独立 -> 保留
        ]
        kept = nms_boxes_global(dup, 0.30)
        check("3 个框去重后剩 2", len(kept) == 2, f"{len(kept)}")
        check("保留置信度更高的那个",
              any(abs(k[0] - 100) < 1 for k in kept), str([k[0] for k in kept]))
        check("独立框被保留",
              any(abs(k[0] - 300) < 1 for k in kept), str([k[0] for k in kept]))

        # ---------- 6. 对拍 ----------
        print("\n[6] 对拍（同后端自比对应为 100% 一致）")
        bA = ClassicalBackend()
        bB = ClassicalBackend()
        bA.load(cfg, dat, wts)
        bB.load(cfg, dat, wts)
        repc = crosscheck(bA, bB, tiles, thresh=0.0, nms=0.0)
        check("自对拍一致率 = 1.0", repc["agreement_ratio"] == 1.0,
              str(repc["agreement_ratio"]))
        check("自对拍通过", repc["passed"], format_report(repc).splitlines()[-1])
        check("中心偏移为 0", repc["max_center_offset_px"] == 0.0)

        # 构造一个"故意不同"的后端，验证对拍能发现不一致
        class ShiftedBackend(ClassicalBackend):
            name = "shifted_stub"
            tier = 1

            def detect_tiles(self, tiles_, thresh, nms):
                out = super().detect_tiles(tiles_, thresh, nms)
                for dets in out:
                    for d in dets:
                        d.cx += 40          # 整体平移 40px
                return out

        bC = ShiftedBackend()
        bC.load(cfg, dat, wts)
        repd = crosscheck(bA, bC, tiles, thresh=0.0, nms=0.0, radius=8.0)
        check("对拍能检出不一致（平移40px）", not repd["passed"],
              f"agreement={repc['agreement_ratio']} -> {repd['agreement_ratio']}")

        # ---------- 7. 端到端切块流水线 ----------
        print("\n[7] 端到端：切块流水线 -> CSV + 可视化")
        src = os.path.join(tmp, "样本_炎症组")
        os.makedirs(src, exist_ok=True)
        cells = [(400, 450), (520, 470), (600, 440), (700, 460)]
        p1 = os.path.join(src, "鱼_01.png")
        p2 = os.path.join(src, "鱼_02.png")
        for p, seed in ((p1, 0), (p2, 1)):
            im = make_fluorescence(cells=cells, seed=seed)
            cv2.imencode(".png", im)[1].tofile(p)

        out = os.path.join(tmp, "out")
        pipe = TilingPipeline(bA, None,
                              TilingConfig(tile_size=416, overlap=64,
                                           thresh=0.0, global_nms=0.30))
        rep2 = pipe.process_folder(src, out)
        check("处理成功 2 张", rep2.processed == 2, str(rep2.summary()))
        check("处理了切块", rep2.tiles_done > 0, f"{rep2.tiles_done} 块")
        csv_path = os.path.join(out, "quantification_results.csv")
        check("CSV 已生成", os.path.exists(csv_path))
        vis = os.path.join(out, "visualizations")
        check("可视化目录已生成", os.path.isdir(vis) and len(os.listdir(vis)) >= 2,
              str(os.listdir(vis)) if os.path.isdir(vis) else "无")
        rows = [r for r in rep2.rows]
        check("每行都有计数", all(r["肠道内炎症细胞总数"] >= 0 for r in rows),
              str([r["肠道内炎症细胞总数"] for r in rows]))
        check("检出了细胞（计数>0）",
              any(r["肠道内炎症细胞总数"] > 0 for r in rows),
              str([r["肠道内炎症细胞总数"] for r in rows]))

        # ---------- 8. 损坏图片不中断 ----------
        print("\n[8] 损坏图片跳过、整批继续")
        bad = os.path.join(src, "坏图.png")
        open(bad, "wb").close()
        out2 = os.path.join(tmp, "out2")
        pipe2 = TilingPipeline(bA, None,
                               TilingConfig(tile_size=416, overlap=64, thresh=0.0))
        rep3 = pipe2.process_folder(src, out2)
        check("坏图被记入失败清单", len(rep3.failures) == 1, str(len(rep3.failures)))
        check("其余图片仍处理成功", rep3.processed == 2, str(rep3.processed))
        check("失败清单 CSV 已生成",
              os.path.exists(os.path.join(out2, "failed_images.csv")))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 56)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print("   -", f)
        return 1
    print("全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

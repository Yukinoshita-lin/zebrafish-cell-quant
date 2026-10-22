# -*- coding: utf-8 -*-
"""
zebrafish_quant 冒烟自检（无需 libdarknet / 无需真实模型）

覆盖：
  1. 中文路径 + tif/png 读写（需求 4.1、Windows 兼容 5.1）
  2. letterbox 坐标还原精度（保证掩码约束计数与可视化对齐）
  3. 掩码约束计数：掩码内计数、掩码外忽略（需求 4.3）
  4. 轻微粘连拆分（需求 4.3）
  5. 损坏图片跳过而不中断（需求 5.2）
  6. CSV 必需字段与顺序（需求 4.4）
  7. 可视化落盘（需求 4.2/4.4）
  8. 验收指标口径：Dice / Precision / Recall / 逐图平均相对误差（需求 7）

运行： python tests/test_zebrafish_quant_smoke.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import zebrafish_quant as zq                      # noqa: E402
from zebrafish_quant.inference import Detection   # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(("  [OK]   " if cond else "  [FAIL] ") + name + (f"  -> {detail}" if detail else ""))


def make_fluorescence_image(w=800, h=600, cells=None, seed=0):
    """合成荧光图：暗背景 + 一条肠道管状区域 + 若干绿色亮点。"""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 12, np.uint8)

    # 肠道：横向椭圆管（自体荧光背景略亮）
    cv2.ellipse(img, (w // 2, h // 2), (w // 3, 45), 0, 0, 360, (30, 55, 28), -1)

    cells = cells or []
    for (cx, cy) in cells:
        cv2.circle(img, (int(cx), int(cy)), 6, (20, 235, 30), -1)
        cv2.circle(img, (int(cx), int(cy)), 3, (60, 255, 70), -1)
    noise = rng.integers(0, 8, (h, w, 3), dtype=np.uint8)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def det(cx, cy, w=14, h=14, conf=0.9, label="cell"):
    return Detection(label=label, confidence=conf, cx=cx, cy=cy, w=w, h=h,
                     cx_net=cx, cy_net=cy, w_net=w, h_net=h)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="zq_smoke_")
    try:
        # ---------- 1. 中文路径 + 多格式 I/O ----------
        print("\n[1] 中文路径与 tif/png 读写")
        sub = os.path.join(tmp, "斑马鱼样本_炎症组")
        os.makedirs(sub, exist_ok=True)
        img = make_fluorescence_image(cells=[(300, 300), (420, 320)])
        p_png = os.path.join(sub, "斑马鱼_01.png")
        p_tif = os.path.join(sub, "斑马鱼_02.tif")
        for p in (p_png, p_tif):
            ok, buf = cv2.imencode(os.path.splitext(p)[1], img)
            buf.tofile(p)
        try:
            a = zq.imread_robust(p_png)
            b = zq.imread_robust(p_tif)
            check("中文路径 PNG 可读", a is not None and a.shape == img.shape, str(a.shape))
            check("中文路径 TIF 可读", b is not None and b.shape == img.shape, str(b.shape))
            check("TIF/PNG 内容一致", bool(np.array_equal(a, b)))
        except Exception as exc:                       # noqa: BLE001
            check("中文路径读写", False, f"{type(exc).__name__}: {exc}")

        # 原实现的 cv2.imread 在中文路径上的表现（用于对比说明）
        legacy = cv2.imread(p_png)
        print(f"        （对比）cv2.imread 中文路径返回: {legacy is None and 'None（失败）' or 'ok'}")

        # darknet.py load_network 用 .encode("ascii")，中文模型路径会直接崩溃（缺陷 B-1）
        # 验证 stage 兜底逻辑：把模型三件套暂存到纯 ASCII 目录
        from zebrafish_quant.inference import _stage_to_ascii
        model_dir = os.path.join(tmp, "训练好的模型")
        os.makedirs(model_dir, exist_ok=True)
        m_files = []
        for nm in ("肠道细胞.cfg", "肠道细胞.data", "肠道细胞.weights"):
            fp = os.path.join(model_dir, nm)
            open(fp, "wb").write(b"x")
            m_files.append(fp)
        try:
            m_files[0].encode("ascii")
            check("（前置）中文路径确实无法通过 ascii 编码", False, "未复现 B-1")
        except UnicodeEncodeError:
            check("（前置）复现 B-1：中文路径 encode('ascii') 失败", True)
        staged, staged_dir = _stage_to_ascii(m_files)
        check("stage 后全部为纯 ASCII 路径", all(p.isascii() for p in staged),
              str([os.path.basename(p) for p in staged]))
        check("stage 后文件真实存在", all(os.path.exists(p) for p in staged))
        # 英文路径应零开销原样返回
        ascii_files = [os.path.join(tmp, "m.cfg")]
        open(ascii_files[0], "wb").write(b"x")
        s2, d2 = _stage_to_ascii(ascii_files)
        check("英文路径零开销原样返回（不复制）", s2 == ascii_files and d2 is None)
        if staged_dir:
            shutil.rmtree(staged_dir, ignore_errors=True)

        # ---------- 2. letterbox 坐标还原 ----------
        print("\n[2] letterbox 坐标还原精度")
        canvas, lb = zq.letterbox_resize(img, 416, 416)
        check("画布尺寸正确", canvas.shape == (416, 416, 3), str(canvas.shape))
        max_err = 0.0
        for (ox, oy) in [(100, 50), (400, 300), (750, 550), (0, 0)]:
            nx = ox * lb.scale + lb.pad_x
            ny = oy * lb.scale + lb.pad_y
            rx, ry = lb.to_original(nx, ny)
            max_err = max(max_err, abs(rx - ox), abs(ry - oy))
        check("坐标还原误差 < 1e-6", max_err < 1e-6, f"max_err={max_err:.2e}")
        bx = lb.box_to_original((100, 100, 28, 28))
        check("框还原保持宽高比", abs(bx[2] / bx[3] - 1.0) < 1e-6, f"{bx[2]:.2f}x{bx[3]:.2f}")

        # ---------- 3. 掩码约束计数 ----------
        print("\n[3] 掩码约束计数（掩码内计数，掩码外忽略）")
        mask = np.zeros((600, 800), np.uint8)
        cv2.ellipse(mask, (400, 300), (266, 45), 0, 0, 360, 255, -1)
        dets = [det(400, 300), det(300, 290),      # 掩码内
                det(700, 550), det(120, 60)]       # 掩码外
        res = zq.count_within_mask(dets, mask)
        check("掩码内计数=2", res.total == 2, f"total={res.total}")
        check("掩码外记录=2（供审计）", len(res.outside) == 2, f"outside={len(res.outside)}")
        empty = zq.count_within_mask(dets, np.zeros((600, 800), np.uint8))
        check("空掩码 -> 计数 0", empty.total == 0, f"total={empty.total}")
        check("噪点按 min_confidence 过滤",
              zq.count_within_mask([det(400, 300, conf=0.1)], mask, min_confidence=0.5).total == 0)

        # ---------- 4. 粘连拆分 ----------
        print("\n[4] 轻微粘连拆分")
        w = h = 120
        adh = np.zeros((h, w, 3), np.uint8)
        cv2.circle(adh, (45, 60), 7, (20, 240, 30), -1)
        cv2.circle(adh, (75, 60), 7, (20, 240, 30), -1)
        d_small = [det(20, 20, 14, 14), det(100, 20, 14, 14), det(20, 100, 14, 14)]
        d_big = [det(60, 60, 56, 56)]          # 面积远大于中位数 -> 触发拆分
        out = zq.split_adhesions(adh, d_small + d_big)
        check("粘连团被拆成 2 个（3+2=5）", len(out) == 5, f"got={len(out)}")
        single = np.zeros((h, w, 3), np.uint8)
        cv2.circle(single, (60, 60), 9, (20, 240, 30), -1)
        out2 = zq.split_adhesions(single, d_small + [det(60, 60, 40, 40)])
        check("单个大细胞不被误拆（4）", len(out2) == 4, f"got={len(out2)}")

        # ---------- 5. 损坏图片处理 ----------
        print("\n[5] 损坏/异常图片不中断批量")
        bad_dir = os.path.join(tmp, "损坏样本")
        os.makedirs(bad_dir, exist_ok=True)
        bad = os.path.join(bad_dir, "坏图.png")
        open(bad, "wb").close()                 # 0 字节
        try:
            zq.imread_robust(bad)
            check("空文件抛出 ImageReadError", False, "未抛异常")
        except zq.ImageReadError:
            check("空文件抛出 ImageReadError", True, "可被 pipeline 捕获并跳过")
        check("不存在的文件抛 ImageReadError",
              _raises(zq.ImageReadError, zq.imread_robust, os.path.join(tmp, "nope.png")))

        # ---------- 6. CSV 字段 ----------
        print("\n[6] CSV 必需字段与顺序（需求 4.4）")
        csv_path = os.path.join(tmp, "out", "results.csv")
        rows = [
            zq.make_result_row(image_path=p_png, count=12, manual_review=False,
                               status="ok", excluded=3, mask_area=15000),
            zq.make_result_row(image_path=p_tif, count=7, manual_review=True,
                               status="ok", excluded=1, mask_area=14000),
        ]
        zq.write_csv(rows, csv_path)
        import csv as _csv
        with open(csv_path, encoding="utf-8-sig") as fh:
            reader = _csv.reader(fh)
            header = next(reader)
            row1 = next(reader)
            row2 = next(reader)
        check("前 3 列=需求字段",
              header[:3] == ["图片文件名", "肠道内炎症细胞总数", "分割是否通过人工复核"],
              str(header[:3]))
        check("文件名保留（一一对应）", row1[0] == "斑马鱼_01.png", row1[0])
        check("计数值正确", row1[1] == "12", row1[1])
        check("未复核=否 / 已复核=是", row1[2] == "否" and row2[2] == "是",
              f"{row1[2]} / {row2[2]}")

        # ---------- 7. 可视化 ----------
        print("\n[7] 可视化落盘（原图分辨率）")
        vis = os.path.join(tmp, "out", "斑马鱼_01_result.png")
        zq.save_overlay(img, vis, mask=mask, counted=res.inside, excluded=res.outside)
        check("可视化图已生成", os.path.exists(vis) and os.path.getsize(vis) > 0)
        vis_img = zq.imread_robust(vis)
        check("可视化保持原图分辨率", vis_img.shape == img.shape, str(vis_img.shape))

        # ---------- 8. 验收指标口径 ----------
        print("\n[8] 验收指标（需求 7）")
        check("相同掩码 Dice=1.0", abs(zq.dice_coefficient(mask, mask) - 1.0) < 1e-6)
        d90 = zq.dice_coefficient(mask, cv2.ellipse(
            np.zeros_like(mask), (405, 300), (266, 45), 0, 0, 360, 255, -1))
        check("轻微偏移 Dice 仍 >0.9", d90 > 0.9, f"dice={d90:.4f}")

        prf = zq.match_points_to_detections(
            [(400, 300), (300, 290)], [det(402, 301), det(301, 291), det(700, 550)], radius=12)
        check("Precision=2/3", abs(prf.precision - 2 / 3) < 1e-9, f"{prf.precision:.4f}")
        check("Recall=1.0", abs(prf.recall - 1.0) < 1e-9, f"{prf.recall:.4f}")

        ce = zq.count_relative_error([10, 20], [10, 20])
        check("完全一致 -> 误差 0", ce.per_image_mean == 0.0)
        ce2 = zq.count_relative_error([10, 24], [10, 20])
        check("逐图平均相对误差=10%", abs(ce2.per_image_mean - 0.10) < 1e-9,
              f"{ce2.per_image_mean:.4f}（论文汇总口径={ce2.aggregate:.4f}）")
        ce3 = zq.count_relative_error([0, 3], [0, 0])
        check("人工=0 而 AI>0 记为 100%", abs(ce3.per_image_mean - 0.5) < 1e-9,
              f"{ce3.per_image_mean:.4f}")

        # 关键：汇总口径会抵消误差，与逐图口径可能给出相反结论
        ce4 = zq.count_relative_error([12, 18], [10, 20])
        check("误差抵消：逐图15% vs 汇总0%（口径差异实证）",
              abs(ce4.per_image_mean - 0.15) < 1e-9 and abs(ce4.aggregate - 0.0) < 1e-9,
              f"逐图={ce4.per_image_mean:.4f} 汇总={ce4.aggregate:.4f}")

        rep = zq.acceptance_report(
            {"mean": 0.90},
            zq.PRF(tp=90, fp=8, fn=7, precision=90 / 98, recall=90 / 97, f1=0.9),
            zq.CountError(0.10, 0.09, 0.08, 30, 1.2, 1.5))
        check("达标判定：Dice/P/R/误差 全 True",
              all([rep["Dice>=0.85"], rep["Precision>=0.85"],
                   rep["Recall>=0.85"], rep["平均相对误差<=0.15"]]),
              str({k: v for k, v in rep.items() if k != "明细"}))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 56)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        print("失败项：")
        for f in FAIL:
            print("   -", f)
        return 1
    print("全部通过 ✅")
    return 0


def _raises(exc_type, fn, *a, **kw) -> bool:
    try:
        fn(*a, **kw)
        return False
    except exc_type:
        return True
    except Exception:                                # noqa: BLE001
        return False


if __name__ == "__main__":
    sys.exit(main())

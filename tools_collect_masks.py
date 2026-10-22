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
tools_collect_masks.py - 汇总桌面标注包的中后肠掩码为项目训练用数据集

输入：桌面 斑马鱼待标注数据_20261019（LabelMe JSON，每图 1 个 polygon）
输出：annotations/
  masks/{name}.png    2752x2208 二值掩码（255=中后肠 ROI 内）
  manifest.csv        每图一行：批次/组/三态/split/源图路径/掩码面积等
  qc/*.png            分层抽样叠加 QC 图（原图+掩码轮廓）

去重规则：05_中后肠掩码 文件夹的版本优先（ROI 口径修正后重画的版本），
其余用 01-04 的版本；两份都记录在 manifest 的 json_src 列。
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.expanduser("~"), "Desktop",
                   "斑马鱼待标注数据_20261019")
OUT = os.path.join(ROOT, "annotations")

sys.path.insert(0, ROOT)
from tools_pack_annotation_data import BATCHES   # 批次->子目录->层 映射

FOLDERS = ["01_训练_对照基线_30张", "02_训练_造模炎症_40张",
           "03_训练_化合物处理_80张", "04_测试集_盲标_40张",
           "05_中后肠掩码_60张"]
SPLIT_OF = {"01": "train", "02": "train", "03": "train",
            "04": "test", "05": None}      # 05 是重复拷贝，split 取自主文件夹


def parse_name(name: str):
    """'260903_Ⅲ-1_Snap-40938' -> (batch, group, snap)"""
    for batch in sorted(BATCHES, key=len, reverse=True):
        prefix = batch + "_"
        if name.startswith(prefix):
            rest = name[len(prefix):]
            # group 可能含下划线以外的字符；用源目录集合匹配最长前缀
            groups = sorted(BATCHES[batch][1].keys(), key=len, reverse=True)
            for g in groups:
                if rest.startswith(g + "_"):
                    return batch, g, rest[len(g) + 1:]
    return None, None, None


def source_tif(batch: str, group: str, snap: str) -> str:
    root = BATCHES[batch][0]
    return os.path.join(root, group, snap + ".tif")


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """Unicode 安全写图：cv2.imwrite 对含 'Ⅲ'/中文的路径会静默失败
    （Windows ANSI 代码页问题），改用 imencode + Python 文件句柄。"""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE):
    """Unicode 安全读图（对应 imwrite_unicode）。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags)


def rasterize(points, w: int, h: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.float32)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    cv2.fillPoly(m, [pts.astype(np.int32)], 255)
    return m


def main() -> int:
    if not os.path.isdir(PKG):
        print(f"[错误] 标注包不存在: {PKG}")
        return 1
    os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "qc"), exist_ok=True)

    # 1) 收集所有 JSON（按文件夹优先级去重）
    picked: Dict[str, dict] = {}
    for folder in FOLDERS:
        d = os.path.join(PKG, folder)
        for jf in sorted(glob.glob(os.path.join(d, "*.json"))):
            name = os.path.splitext(os.path.basename(jf))[0]
            rec = {"json": jf, "folder": folder}
            if name not in picked:
                picked[name] = rec
            else:
                picked[name].setdefault("dup_json", []).append(jf)
    print(f"JSON 汇总: {len(picked)} 张唯一图像"
          f"（含重复标注 {sum(1 for v in picked.values() if 'dup_json' in v)} 张）")

    # 2) 逐图转换
    rows = []
    problems = []
    for name in sorted(picked):
        rec = picked[name]
        data = json.load(open(rec["json"], encoding="utf-8"))
        shapes = data.get("shapes", [])
        w, h = int(data.get("imageWidth", 2752)), int(data.get("imageHeight", 2208))
        if not shapes or shapes[0].get("shape_type") != "polygon":
            problems.append(f"{name}: shapes={len(shapes)} 类型异常，已跳过")
            continue
        if len(shapes) > 1:
            # 多 shape：保留画面最下方那条鱼的轮廓（y 中心最大），
            # 与 189 张既定约定一致；面积作次级标准。其余告警待人工重画。
            def _yc(s):
                return float(np.array(s["points"])[:, 1].mean())
            order = sorted(range(len(shapes)),
                           key=lambda i: (_yc(shapes[i]), -len(shapes[i]["points"])),
                           reverse=True)
            best = order[0]
            areas = []
            for i in range(len(shapes)):
                mm = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mm, [np.array(shapes[i]["points"],
                                           dtype=np.int32)], 255)
                areas.append(int((mm > 0).sum()))
            problems.append(f"{name}: 含 {len(shapes)} 个轮廓，保留最下方鱼的"
                            f"第{best}个（面积 {areas[best]} px，y中心 "
                            f"{_yc(shapes[best]):.0f}）；其余未采用，"
                            f"建议人工重画补全")
            pts = shapes[best]["points"]
        else:
            pts = shapes[0]["points"]
        if len(pts) < 3:
            problems.append(f"{name}: 顶点数 {len(pts)} < 3，无效多边形")
            continue
        mask = rasterize(pts, w, h)
        area_pct = float((mask > 0).mean()) * 100
        ys, xs = np.where(mask > 0)
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        # 异常检测：全量分布为中位 0.30%（P5 0.20 / P95 0.55），
        # <0.08% 或 >1.5% 视为异常（误标/圈了整个鱼）
        if area_pct < 0.08 or area_pct > 1.5:
            problems.append(f"{name}: 掩码面积 {area_pct:.2f}% 超出正常范围"
                            f"（正常 0.10~0.85%），疑似误标")
        mask_path = os.path.join(OUT, "masks", name + ".png")
        if not imwrite_unicode(mask_path, mask):
            problems.append(f"{name}: 掩码写入失败 {mask_path}")
            continue

        batch, group, snap = parse_name(name)
        split = SPLIT_OF.get(rec["folder"][:2])
        if split is None:      # 05 的重复图，split 从 01-04 记录里取
            split = "train" if rec["folder"].startswith(("01", "02", "03")) else "test"
        rows.append({
            "name": name, "batch": batch, "group": group, "snap": snap,
            "stratum": BATCHES[batch][1].get(group, "?") if batch else "?",
            "split": split,
            "src_tif": source_tif(batch, group, snap) if batch else "",
            "mask_png": mask_path,
            "mask_area_pct": f"{area_pct:.2f}",
            "mask_bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
            "n_vertices": len(pts),
            "json_src": os.path.relpath(rec["json"], PKG),
            "dup_json": "|".join(os.path.relpath(j, PKG)
                                 for j in rec.get("dup_json", [])),
        })
    print(f"掩码转换: {len(rows)} 张 -> {os.path.join(OUT, 'masks')}")

    # 3) 第二条鱼探测（上半帧荧光能量，内联计算，不依赖外部文件）
    #    实测分布：170/190 帧上半为空，20 帧存在第二条鱼（亮像素占比>=0.5%）
    sys.path.insert(0, ROOT)
    from zebrafish_quant.imaging import imread_robust
    upper_flag = {}
    for r in rows:
        src = r["src_tif"]
        if not os.path.exists(src):
            continue
        img = imread_robust(src)
        g = img[:, :, 1].astype(np.float32)
        hh = g.shape[0]
        upper = g[int(hh * 0.05):int(hh * 0.42), :]
        upper_flag[r["name"]] = bool((upper > 40).mean() >= 0.005)
    mf = os.path.join(OUT, "manifest.csv")
    fields = list(rows[0].keys()) + ["has_upper_larva"]
    for r in rows:
        r["has_upper_larva"] = "1" if upper_flag.get(r["name"]) else "0"
    with open(mf, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=fields)
        wcsv.writeheader()
        wcsv.writerows(rows)
    print(f"清单 -> {mf}"
          f"（含第二条鱼帧 {sum(1 for v in upper_flag.values() if v)} 个）")

    # 4) 问题清单
    if problems:
        print("\n[需人工处理]")
        for p in problems:
            print("  " + p)
    else:
        print("\n未发现格式/面积异常")

    # 5) QC 叠加图（分层抽样 12 张）
    sys.path.insert(0, ROOT)
    from zebrafish_quant.imaging import imread_robust
    rng = np.random.RandomState(7)
    by_key: Dict[str, List[dict]] = {}
    for r in rows:
        by_key.setdefault(f"{r['split']}_{r['stratum']}", []).append(r)
    qc_n = 0
    for key, lst in sorted(by_key.items()):
        for r in rng.choice(lst, size=min(2, len(lst)), replace=False):
            if not os.path.exists(r["src_tif"]):
                continue
            img = imread_robust(r["src_tif"])
            m = imread_unicode(r["mask_png"])
            if m is None:
                continue
            ov = img.copy()
            ov[m > 0] = (0.35 * ov[m > 0] + 0.65 * np.array([0, 0, 255])).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(ov, cnts, -1, (0, 0, 255), 3)
            sc = 1400 / ov.shape[1]
            small = cv2.resize(ov, (1400, int(ov.shape[0] * sc)))
            imwrite_unicode(os.path.join(OUT, "qc",
                                         f"{key}_{r['name']}.png"), small)
            qc_n += 1
    print(f"QC 叠加图 {qc_n} 张 -> {os.path.join(OUT, 'qc')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

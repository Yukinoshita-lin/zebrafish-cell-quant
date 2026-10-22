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
tools_pack_annotation_data.py - 生成桌面分层待标注数据包

按咨询结论（2026-10-19）把 394 张源荧光图按"对照/造模/处理"三态分层抽样，
在桌面生成独立工作文件夹，每文件夹附工作说明 txt。

分层定义（源目录 -> 层）：
    对照: MD8-20/Control、HY第7列/Control、260903/空白组、260912/空白组
    造模: MD8-20/Model、HY第7列/Model、260903/硫酸铜组、260912/硫酸铜组
    处理: MD8-20/md-*、260903|260912 的 Ⅲ-*/布洛芬组、HY第7列/7A~7H

配额（train/test 互斥，mask 从两者中抽）：
    train: 对照30 + 造模40 + 处理80 = 150
    test : 对照8  + 造模8  + 处理24 = 40（盲标，验收真值）
    mask : 60（train 40 + test 20，三态按比例）

用法：
    python tools_pack_annotation_data.py [--out 桌面目录] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from typing import Dict, List, Tuple

SRC_ROOT = r"F:\AI斑马鱼"
DESKTOP = os.path.join(os.path.expanduser("~"), "Desktop")
SEED = 20261019

# 批次标签 -> (源目录, {子目录: 层})
BATCHES: Dict[str, Tuple[str, Dict[str, str]]] = {
    "MD8-20": (os.path.join(SRC_ROOT, "MD8-20"), {
        "Control": "对照",
        "Model": "造模",
        "md-8": "处理", "md-9": "处理", "md-11": "处理", "md-13": "处理",
        "md-14": "处理", "md-18": "处理", "md-19": "处理", "md-20": "处理",
    }),
    "260903": (os.path.join(SRC_ROOT, "260903武汉大学抗炎化合物筛选"), {
        "空白组": "对照", "硫酸铜组": "造模", "布洛芬组": "处理",
        "Ⅲ-1": "处理", "Ⅲ-3": "处理", "Ⅲ-5": "处理",
        "Ⅲ-6": "处理", "Ⅲ-7": "处理", "Ⅲ-8": "处理",
    }),
    "260912": (os.path.join(SRC_ROOT, "260912武汉大学抗炎化合物二筛"), {
        "空白组": "对照", "硫酸铜组": "造模", "布洛芬组": "处理",
        "Ⅲ-1": "处理", "Ⅲ-3": "处理", "Ⅲ-7": "处理", "Ⅲ-8": "处理",
    }),
    "HY7列": (os.path.join(SRC_ROOT, "HY第7列"), {
        "Control": "对照", "Model": "造模",
        "7A（HY-107782）": "处理", "7B（HY-111299）": "处理",
        "7C（HY-N0936）": "处理", "7D（HY-N6952）": "处理",
        "7E（HY-65119）": "处理", "7G（HY-N4157）": "处理",
        "7H（HY-126940）": "处理",
    }),
}

QUOTA = {
    "train": {"对照": 30, "造模": 40, "处理": 80},
    "test": {"对照": 8, "造模": 8, "处理": 24},
}
MASK_FROM_TRAIN = {"对照": 8, "造模": 12, "处理": 20}     # 40
MASK_FROM_TEST = {"对照": 4, "造模": 4, "处理": 12}        # 20


def enumerate_sources() -> List[dict]:
    """扫描全部源图，返回 [{batch, group, stratum, src, name}, ...]"""
    items = []
    for batch, (root, groups) in BATCHES.items():
        for sub, stratum in groups.items():
            d = os.path.join(root, sub)
            if not os.path.isdir(d):
                print(f"[警告] 源目录不存在: {d}")
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.lower().endswith(".tif"):
                    continue
                items.append({"batch": batch, "group": sub,
                              "stratum": stratum,
                              "src": os.path.join(d, fn), "name": fn})
    return items


def allocate(quota: int, subgroups: List[str], pool: Dict[str, List[dict]]
             ) -> Dict[str, int]:
    """按子组大小比例分配配额（最大余数法），子组不足时取全并再分配。"""
    sizes = {g: len(pool[g]) for g in subgroups}
    total = sum(sizes.values())
    if total == 0:
        return {g: 0 for g in subgroups}
    alloc = {g: 0 for g in subgroups}
    remain = quota
    # 第一轮：比例
    for g in subgroups:
        want = int(quota * sizes[g] / total)
        take = min(want, sizes[g])
        alloc[g] = take
        remain -= take
    # 第二轮：最大余数补足
    while remain > 0:
        frac = sorted(subgroups,
                      key=lambda g: (-(quota * sizes[g] / total - alloc[g]),
                                     -sizes[g]))
        progressed = False
        for g in frac:
            if alloc[g] < sizes[g]:
                alloc[g] += 1
                remain -= 1
                progressed = True
                break
        if not progressed:
            break
    return alloc


def sample(pool: Dict[str, List[dict]], quota_by_sub: Dict[str, int],
           rng: random.Random) -> List[dict]:
    out = []
    for g, n in quota_by_sub.items():
        cand = list(pool.get(g, []))
        rng.shuffle(cand)
        out.extend(cand[:n])
    return out


def copy_as(items: List[dict], dst_dir: str) -> List[str]:
    """拷贝并加批次_组_ 前缀，返回新文件名列表。"""
    os.makedirs(dst_dir, exist_ok=True)
    names = []
    for it in items:
        new = f"{it['batch']}_{it['group']}_{it['name']}"
        shutil.copy2(it["src"], os.path.join(dst_dir, new))
        names.append(new)
    return sorted(names)


def build_plan() -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """确定性抽样（固定种子），返回 (folder -> items, 源图池三态计数)。"""
    rng = random.Random(SEED)
    items = enumerate_sources()
    src_counts = {st: sum(1 for i in items if i["stratum"] == st)
                  for st in ("对照", "造模", "处理")}

    # 按层 -> 子组 建池
    pool: Dict[str, Dict[str, List[dict]]] = {}
    for it in items:
        pool.setdefault(it["stratum"], {}).setdefault(it["group"], []).append(it)

    # train / test 互斥抽样
    train_sel: Dict[str, List[dict]] = {}
    test_sel: Dict[str, List[dict]] = {}
    for st, quota in QUOTA["train"].items():
        subs = list(pool[st].keys())
        alloc = allocate(quota, subs, pool[st])
        chosen = sample(pool[st], alloc, rng)
        train_sel[st] = chosen
        # 剩余池供 test 抽样
        left = {g: [i for i in v if i not in chosen]
                for g, v in pool[st].items()}
        t_quota = QUOTA["test"][st]
        t_alloc = allocate(t_quota, list(left.keys()), left)
        test_sel[st] = sample(left, t_alloc, rng)

    # mask 从 train/test 已选集中按层抽取
    mask_sel = []
    for st, n in MASK_FROM_TRAIN.items():
        cand = list(train_sel[st])
        rng.shuffle(cand)
        mask_sel.extend(cand[:n])
    for st, n in MASK_FROM_TEST.items():
        cand = list(test_sel[st])
        rng.shuffle(cand)
        mask_sel.extend(cand[:n])

    plan = {
        "01_训练_对照基线_30张": train_sel["对照"],
        "02_训练_造模炎症_40张": train_sel["造模"],
        "03_训练_化合物处理_80张": train_sel["处理"],
        "04_测试集_盲标_40张": sum(test_sel.values(), []),
        "05_中后肠掩码_60张": mask_sel,
    }
    return plan, src_counts


def main() -> int:
    ap = argparse.ArgumentParser(description="生成桌面分层待标注数据包")
    ap.add_argument("--out", default=None, help="输出目录（默认桌面/斑马鱼待标注数据_日期）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--notes-only", action="store_true",
                    help="只重写各文件夹的工作说明 txt，不重复拷贝图像")
    args = ap.parse_args()

    items = enumerate_sources()
    print(f"源图总数: {len(items)}")
    for st in ("对照", "造模", "处理"):
        n = sum(1 for i in items if i["stratum"] == st)
        print(f"  {st}: {n}")

    plan, src_counts = build_plan()
    print("\n=== 抽样计划 ===")
    for folder, its in plan.items():
        by_st: Dict[str, int] = {}
        for i in its:
            by_st[i["stratum"]] = by_st.get(i["stratum"], 0) + 1
        print(f"  {folder}: {len(its)} 张  {by_st}")
    # 互斥校验
    tr = {i["src"] for k, v in plan.items() if not k.startswith(("04", "05"))
          for i in v}
    te = {i["src"] for i in plan["04_测试集_盲标_40张"]}
    assert not (tr & te), "train/test 出现重叠！"
    print(f"  train∪test 唯一图像: {len(tr | te)}")

    out_root = args.out or os.path.join(
        DESKTOP, "斑马鱼待标注数据_20261019")
    if args.dry_run:
        print(f"\n[dry-run] 目标目录: {out_root}")
        return 0

    from tools_pack_notes import write_all_notes
    if args.notes_only:
        if not os.path.isdir(out_root):
            print(f"[错误] 目录不存在: {out_root}")
            return 1
        manifests = {
            folder: sorted(f"{i['batch']}_{i['group']}_{i['name']}"
                           for i in its)
            for folder, its in plan.items()
        }
        write_all_notes(out_root, plan, manifests, src_counts)
        print(f"\n已重写工作说明 -> {out_root}")
        return 0

    if os.path.exists(out_root):
        print(f"[错误] 目标已存在: {out_root}")
        return 1
    os.makedirs(out_root)
    print(f"\n输出目录: {out_root}")

    manifests = {}
    for folder, its in plan.items():
        d = os.path.join(out_root, folder)
        names = copy_as(its, d)
        manifests[folder] = names
        print(f"  {folder}: 拷贝 {len(names)} 张")

    write_all_notes(out_root, plan, manifests, src_counts)
    print("\n完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
train.py - 通过 darknet 原生库训练检测模型

关键点：darknet 内部用 fopen 打开文件（Windows 走 ANSI 接口），
**无法直接处理中文路径**。而本项目的路径（F:/AI斑马鱼/...）含中文，
因此训练前必须把 cfg/data/图片/标签/路径清单**整体暂存到纯 ASCII 目录**，
并把清单里的绝对路径改写为暂存目录中的路径。

这与 inference._stage_to_ascii（模型三件套）是同一类问题的完整版：
训练需要的不止三个文件，而是整个数据集。

用法：
    from zebrafish_quant.train import train
    train(cfg, data, dataset_dir, out_weights, max_batches=300)
"""
from __future__ import annotations

import ctypes as ct
import os
import shutil
import tempfile
from typing import Optional, Tuple

from .native import NativeDarknet, NativeLoadError


def _read_data_file(path: str) -> dict:
    d = {}
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def stage_ascii_dataset(cfg_path: str, data_path: str,
                        workdir: Optional[str] = None) -> Tuple[str, str, str]:
    """把 cfg / data / 图片 / 标签 / 路径清单整体暂存到纯 ASCII 目录。

    返回 (staged_cfg, staged_data, workdir)。
    若原路径全为 ASCII，则原样返回，不做任何复制（零开销）。
    """
    data = _read_data_file(data_path)
    train_list = data.get("train", "")
    valid_list = data.get("valid", "")
    names = data.get("names", "")

    if os.path.abspath(cfg_path).isascii() and os.path.abspath(data_path).isascii() \
            and os.path.abspath(train_list).isascii():
        return cfg_path, data_path, ""

    workdir = workdir or tempfile.mkdtemp(prefix="dn_train_")
    os.makedirs(workdir, exist_ok=True)

    # 1) cfg
    s_cfg = os.path.join(workdir, os.path.basename(cfg_path))
    shutil.copy2(cfg_path, s_cfg)

    # 2) names
    s_names = os.path.join(workdir, os.path.basename(names)) if names else ""
    if names and os.path.exists(names):
        shutil.copy2(names, s_names)

    # 3) 图片目录；**标签必须作为同级兄弟文件放在图片旁边**。
    #    darknet 的 replace_image_to_label() 只识别 COCO / PascalVOC 等固定
    #    目录模式；对自定义数据集，它走的是"没匹配到目录模式就把扩展名
    #    .png/.jpg 换成 .txt"，即期望 label 与 image 位于**同一目录**。
    #    若按 images/ + labels/ 分开存放，会直接报
    #    "Can't open label file"，训练出一个也检测不到的模型。
    root = os.path.dirname(os.path.abspath(train_list))
    img_src = os.path.join(root, "images")
    lbl_src = os.path.join(root, "labels")
    s_img = os.path.join(workdir, "images")
    os.makedirs(s_img, exist_ok=True)
    if os.path.isdir(img_src):
        shutil.copytree(img_src, s_img, dirs_exist_ok=True)
    if os.path.isdir(lbl_src):
        for f in os.listdir(lbl_src):
            if f.lower().endswith(".txt"):
                shutil.copy2(os.path.join(lbl_src, f), os.path.join(s_img, f))

    # 4) 改写路径清单：绝对路径 -> 暂存目录中的路径
    def rewrite(list_path: str, dest: str) -> str:
        if not list_path or not os.path.exists(list_path):
            return ""
        out_lines = []
        for line in open(list_path, encoding="utf-8", errors="replace"):
            p = line.strip()
            if not p:
                continue
            base = os.path.basename(p)
            out_lines.append(os.path.join(s_img, base))
        open(dest, "w", encoding="utf-8").write("\n".join(out_lines) + "\n")
        return dest

    s_train = rewrite(train_list, os.path.join(workdir, "train.txt"))
    s_valid = rewrite(valid_list, os.path.join(workdir, "val.txt"))

    # 5) 新的 .data（backup 也指向 ASCII 目录）
    backup = os.path.join(workdir, "backup")
    os.makedirs(backup, exist_ok=True)
    s_data = os.path.join(workdir, "model.data")
    with open(s_data, "w", encoding="utf-8") as fh:
        fh.write(f"classes = {data.get('classes','1')}\n")
        fh.write(f"train   = {s_train}\n")
        fh.write(f"valid   = {s_valid}\n")
        fh.write(f"names   = {s_names}\n")
        fh.write(f"backup  = {backup}/\n")

    return s_cfg, s_data, workdir


def train_detector(library_path: str, cfg_path: str, data_path: str,
                   weights_out: str = "",
                   *, max_batches: Optional[int] = None,
                   batch: int = 8, subdivisions: int = 8,
                   thresh: float = 0.30, iou_thresh: float = 0.20,
                   verbose: bool = True) -> str:
    """调用 darknet 的 train_detector 训练。

    weights_out 为空表示从头训练；否则作为继续训练的起点。
    返回最终权重文件路径（位于工作目录的 backup/ 中，已复制回原目录）。
    """
    nd = NativeDarknet(library_path)
    s_cfg, s_data, workdir = stage_ascii_dataset(cfg_path, data_path)

    if max_batches:
        _patch_training_params(s_cfg, max_batches=max_batches,
                               batch=batch, subdivisions=subdivisions)

    lib = nd.lib
    lib.train_detector.argtypes = (
        ct.c_char_p, ct.c_char_p, ct.c_char_p,
        ct.POINTER(ct.c_int), ct.c_int, ct.c_int, ct.c_int, ct.c_int,
        ct.c_float, ct.c_float, ct.c_int, ct.c_int, ct.c_int,
        ct.c_char_p, ct.c_int)

    if verbose:
        print(f"[train] cfg : {s_cfg}")
        print(f"[train] data: {s_data}")
        if workdir:
            print(f"[train] 已暂存到 ASCII 目录: {workdir}")

    # ⚠️ 不能传 ngpus=0 + gpus=NULL：
    #    darknet 内部 `network* nets = xcalloc(ngpus, sizeof(network));`
    #    当 ngpus=0 时这是 0 长度分配，后续仍会访问 nets[0]，
    #    实测直接 access violation 崩溃（且会打印出 Learning Rate: 1.8e+31
    #    之类的未初始化垃圾值）。
    #    正确做法：ngpus=1 且 gpus[0] = -1（-1 表示使用 CPU）。
    gpus = (ct.c_int * 1)(-1)
    w_arg = weights_out.encode() if weights_out else None   # NULL => 不加载权重

    lib.train_detector(
        s_data.encode(), s_cfg.encode(), w_arg,
        gpus, 1,
        0,      # clear
        1,      # dont_show
        0,      # calc_map
        thresh, iou_thresh,
        0, 0, 0, None, 0)

    # 取回最后/最好的权重
    backup = os.path.join(workdir or os.path.dirname(s_data), "backup")
    final = None
    for cand in ("cell_final.weights", "model_final.weights"):
        p = os.path.join(backup, cand)
        if os.path.exists(p):
            final = p
            break
    if final is None and os.path.isdir(backup):
        ws = [f for f in os.listdir(backup) if f.endswith(".weights")]
        if ws:
            final = os.path.join(backup, sorted(ws)[-1])
    if not final:
        raise RuntimeError("训练未产出权重文件")

    dst = os.path.join(os.path.dirname(os.path.abspath(cfg_path)),
                       os.path.basename(final))
    shutil.copy2(final, dst)
    return dst


def _patch_training_params(cfg_path: str, *, max_batches: int,
                           batch: int = 8, subdivisions: int = 8) -> None:
    """把训练参数调小，使 CPU 上能在合理时间内产出权重。

    默认 yolov4-tiny cfg 是 batch=64/subdivisions=16（每次迭代 64 张），
    在 CPU 上一次迭代就要几十秒、跑满几百次需数小时。
    这里改为 batch=8/subdivisions=8 并缩短 max_batches，
    以便在几分钟内拿到可用于**性能基准测试**的权重。

    ⚠️ 这样训出来的权重精度很低，仅供验证流水线与测速，
       正式模型必须在 GPU 上用完整数据集与充分迭代训练，
       并使用人工复核过的标注。
    """
    lines = open(cfg_path, encoding="utf-8").read().splitlines()
    out, sec = [], None
    s1 = max(1, int(max_batches * 0.8))
    s2 = max(2, int(max_batches * 0.9))
    for ln in lines:
        s = ln.strip()
        if s.startswith("["):
            sec = s.strip("[]").lower()
            out.append(ln)
            continue
        key = s.split("=")[0].strip() if "=" in s else ""
        if sec == "net" and key == "batch":
            out.append(f"batch={batch}")
        elif sec == "net" and key == "subdivisions":
            out.append(f"subdivisions={subdivisions}")
        elif key == "max_batches":
            out.append(f"max_batches={max_batches}")
        elif key == "steps":
            out.append(f"steps={s1},{s2}")
        else:
            out.append(ln)
    open(cfg_path, "w", encoding="utf-8").write("\n".join(out) + "\n")

#!/usr/bin/env python3
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
gui_app.py —— 斑马鱼中后肠荧光细胞定量工具（图形界面）

功能：
  数据   选择标注数据文件夹（含 manifest.csv + masks/），自动校验并显示统计
  训练   一键训练 U-Net 分割模型（预设 + 自定义参数 + 实时日志 + 进度条 + ETA）
  指标   实时 Dice 曲线（raw/ema）+ 训练报告摘要 + 直达报告/可视化目录
  计数   端到端「AI 分割掩码 + 经典检测」计数，输出逐图 CSV 与汇总报告

设计约定：
  * GUI 进程本身不 import torch（保持轻量），训练/计数全部以子进程运行，
    解释器用 sys.executable（源码运行为 venv python，安装包运行为随包 python）
  * 数据目录通过环境变量 UNET_ANN_DIR 传给脚本（未设时脚本行为不变）
  * 子进程 stdout 统一按 UTF-8 解码（PYTHONIOENCODING=utf-8）
  * --smoke 进入自检模式（offscreen 平台 + 2 轮迷你训练全链路验证）
  * --smoke-env 环境页自检（验证 torch 缺失识别 + 自动跳转 + 一键安装全流程）

源码运行：  python gui_app.py
自检运行：  QT_QPA_PLATFORM=offscreen python gui_app.py --smoke
环境自检：  QT_QPA_PLATFORM=offscreen python gui_app.py --smoke-env
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

from PySide6.QtCore import (Qt, QProcess, QProcessEnvironment, QTimer,
                            QSize, QUrl)
from PySide6.QtGui import (QColor, QDesktopServices, QFont, QPainter,
                           QPen)
from PySide6.QtWidgets import (QApplication, QButtonGroup, QCheckBox,
                               QComboBox, QDoubleSpinBox, QFileDialog,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QScrollArea, QSizePolicy, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

# ---------------------------------------------------------------- 常量
ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "annotations"
SMOKE = "--smoke" in sys.argv
SMOKE_ENV = "--smoke-env" in sys.argv

APP_VERSION = "1.1.1"        # 应用版本（同步 installer.iss）
TARGET_DICE = 0.85          # 需求 7 验收线
EVAL_EVERY = 5              # 训练脚本每 5 轮评估一次（进度条按轮推进）

# PyTorch 一键安装源（torch wheel 已内含 CUDA 运行库，无需单独装 Toolkit）
TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu130",
}
TORCH_SIZE_HINT = {"cpu": "约 200 MB", "cuda": "约 2.5 GB（含 CUDA 13 运行库）"}

# 依赖探测：在子进程中 import，避免拖慢 GUI 进程本身
PROBE_SCRIPT = r'''
import json, shutil
out = {}
for m in ["PySide6", "numpy", "cv2", "torch"]:
    try:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "?")
    except Exception:
        out[m] = None
if out.get("torch"):
    try:
        import torch
        out["torch_cuda_build"] = bool(torch.version.cuda)
        out["cuda_ok"] = bool(torch.cuda.is_available())
    except Exception:
        out["torch_cuda_build"] = None
        out["cuda_ok"] = None
try:
    du = shutil.disk_usage(".")
    out["disk_free_gb"] = round(du.free / 1e9, 1)
except Exception:
    out["disk_free_gb"] = None
print("PROBE:" + json.dumps(out))
'''

# 深色主题配色（扁平、低饱和）
CLR = {
    "bg": "#16171B", "sidebar": "#1B1D22", "surface": "#1F2126",
    "surface2": "#262930", "border": "#2C2F37", "text": "#E7E9EE",
    "dim": "#9398A4", "dim2": "#6B707C", "accent": "#35C99B",
    "accent_dk": "#1E7A5C", "blue": "#6E9BFF", "warn": "#E0A458",
    "danger": "#E0655F", "console": "#121317", "ok_bg": "#12312A",
}

# 训练预设（值即环境变量；÷3 微调需从 ÷4 权重初始化 + 关闭续训）
PRESETS = [
    ("快速跑通 · 60 轮 ≈12 分钟", dict(epochs=60, base=32, depth=4, norm="gn",
                                     batch=4, scale=4, lr=0.0015,
                                     aug_decay=False, init="", resume=True)),
    ("标准 · 250 轮 ≈54 分钟", dict(epochs=250, base=32, depth=4, norm="gn",
                                   batch=4, scale=4, lr=0.0015,
                                   aug_decay=False, init="", resume=True)),
    ("正式 · 400 轮 ≈84 分钟", dict(epochs=400, base=32, depth=4, norm="gn",
                                   batch=4, scale=4, lr=0.002,
                                   aug_decay=True, init="", resume=True)),
    ("÷3 微调 · 60 轮（需 ÷4 权重）", dict(epochs=60, base=32, depth=4,
                                         norm="gn", batch=2, scale=3,
                                         lr=0.0003, aug_decay=False,
                                         init="unet_model_v4_gn.pt",
                                         resume=False)),
]

STYLESHEET = f"""
QWidget {{ background: {CLR['bg']}; color: {CLR['text']};
          font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
          font-size: 13px; }}
#Sidebar {{ background: {CLR['sidebar']};
           border-right: 1px solid {CLR['border']}; }}
#NavBtn {{ text-align: left; padding: 10px 16px; border: none;
          border-left: 3px solid transparent; border-radius: 0;
          color: {CLR['dim']}; font-size: 13px; }}
#NavBtn:hover {{ background: {CLR['surface']}; color: {CLR['text']}; }}
#NavBtn:checked {{ background: {CLR['surface2']}; color: {CLR['text']};
                 border-left: 3px solid {CLR['accent']}; }}
#Card {{ background: {CLR['surface']}; border: 1px solid {CLR['border']};
        border-radius: 12px; }}
#TopBar {{ background: {CLR['bg']}; border-bottom: 1px solid {CLR['border']}; }}
#StatusBar {{ background: {CLR['bg']}; border-top: 1px solid {CLR['border']};
             color: {CLR['dim2']}; font-size: 11px; }}
#Chip {{ background: {CLR['surface']}; border: 1px solid {CLR['border']};
        border-radius: 12px; color: {CLR['dim']}; padding: 3px 12px;
        font-size: 11px; }}
#ChipOk {{ background: {CLR['ok_bg']}; border: 1px solid {CLR['accent_dk']};
          border-radius: 12px; color: {CLR['accent']}; padding: 3px 12px;
          font-size: 11px; }}
#ChipWarn {{ background: #3A2E16; border: 1px solid #7A5B22;
            border-radius: 12px; color: {CLR['warn']}; padding: 3px 12px;
            font-size: 11px; }}
QPushButton {{ background: {CLR['surface2']}; border: 1px solid {CLR['border']};
              border-radius: 8px; padding: 7px 18px; color: {CLR['text']}; }}
QPushButton:hover {{ background: #2E313A; }}
QPushButton:pressed {{ background: #23262D; }}
QPushButton:disabled {{ color: {CLR['dim2']}; border-color: #23262D; }}
#BtnPrimary {{ background: {CLR['accent']}; color: #0B2C22; border: none;
              font-weight: 500; }}
#BtnPrimary:hover {{ background: #43DCAD; }}
#BtnPrimary:disabled {{ background: #23262D; color: {CLR['dim2']}; }}
#BtnDanger {{ background: #402124; border: 1px solid #6E2F2C;
             color: {CLR['danger']}; }}
#BtnDanger:hover {{ background: #4E282B; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
  background: {CLR['console']}; border: 1px solid {CLR['border']};
  border-radius: 8px; padding: 6px 10px; color: {CLR['text']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{ background: {CLR['surface']};
  border: 1px solid {CLR['border']}; selection-background-color: {CLR['surface2']}; }}
QProgressBar {{ background: {CLR['console']};
               border: 1px solid {CLR['border']}; border-radius: 8px;
               height: 18px; text-align: center; color: {CLR['dim']};
               font-size: 11px; }}
QProgressBar::chunk {{ background: {CLR['accent']}; border-radius: 7px; }}
#LogConsole {{ font-family: 'Cascadia Mono', Consolas, monospace;
              font-size: 12px; }}
#TileValue {{ font-size: 22px; font-weight: 500; color: {CLR['text']}; }}
#TileLabel {{ font-size: 11px; color: {CLR['dim']}; }}
#PageTitle {{ font-size: 17px; font-weight: 500; }}
#PageSub {{ font-size: 12px; color: {CLR['dim']}; }}
#SectionTitle {{ font-size: 13px; font-weight: 500; color: {CLR['text']}; }}
#Hint {{ color: {CLR['dim2']}; font-size: 12px; }}
QScrollBar:vertical {{ background: {CLR['bg']}; width: 10px; }}
QScrollBar::handle:vertical {{ background: #33363E; border-radius: 5px;
                              min-height: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{ background: {CLR['bg']}; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #33363E; border-radius: 5px; }}
"""


# ---------------------------------------------------------------- 小部件
class Card(QFrame):
    """圆角表面卡片。"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self.v = QVBoxLayout(self)
        self.v.setContentsMargins(18, 16, 18, 16)
        self.v.setSpacing(10)


class NavButton(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setObjectName("NavBtn")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)


class StatTile(QFrame):
    """统计磁贴：小标签 + 大数值 + 可选副字。"""
    def __init__(self, label, value="—", sub="", accent=False, small=False):
        super().__init__()
        self.setObjectName("Card")
        v = QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 14)
        v.setSpacing(2)
        lab = QLabel(label)
        lab.setObjectName("TileLabel")
        val = QLabel(value)
        val.setObjectName("TileValue")
        if small:
            val.setStyleSheet("font-size: 14px; font-weight: 500; "
                              f"color: {CLR['text']};")
        if accent:
            val.setStyleSheet(
                val.styleSheet() + f"color: {CLR['accent']};")
        v.addWidget(lab)
        v.addWidget(val)
        if sub:
            s = QLabel(sub)
            s.setObjectName("TileLabel")
            v.addWidget(s)
        self._val = val
        self._small = small

    def set_value(self, text, color=None):
        self._val.setText(text)
        if color:
            self._val.setStyleSheet(f"color: {color};")


class CheckItem(QLabel):
    """数据校验清单行：✓/✗ + 说明。"""
    def __init__(self, ok, text):
        mark, col = ("✓", CLR["accent"]) if ok else ("✗", CLR["danger"])
        super().__init__(f'<span style="color:{col};">{mark}</span> '
                         f'<span style="color:{CLR["text"]};">{text}</span>')
        self.setObjectName("Hint")
        self.setTextFormat(Qt.RichText)


class LogConsole(QPlainTextEdit):
    """训练/计数日志：关键字着色。"""
    def __init__(self):
        super().__init__()
        self.setObjectName("LogConsole")
        self.setReadOnly(True)
        self.setMaximumBlockCount(4000)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)

    def append_line(self, text, kind=""):
        import html
        esc = html.escape(text.rstrip())
        col = {"dice": CLR["accent"], "err": CLR["danger"],
               "warn": CLR["warn"], "info": CLR["blue"]}.get(kind, CLR["text"])
        weight = " font-weight:500;" if kind in ("dice", "err") else ""
        self.appendHtml(f'<span style="color:{col};{weight}">{esc}</span>')
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())


class DiceCurveWidget(QWidget):
    """Dice 训练曲线：raw（青）/ ema（蓝）/ 验收线（琥珀虚线）。"""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.points = []          # [(epoch, raw, ema)]
        self.total = None         # 总轮数
        self.status = "等待训练开始"

    def set_data(self, points, total=None, status=None):
        self.points = list(points)
        if total:
            self.total = total
        if status:
            self.status = status
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        L, R, T, B = 46, 16, 18, 30
        cw, ch = w - L - R, h - T - B

        p.fillRect(self.rect(), QColor(CLR["console"]))
        p.setPen(QPen(QColor(CLR["border"]), 1))
        for i in range(6):                       # 横向网格 0.0~1.0
            y = T + ch - ch * i / 5
            p.drawLine(L, int(y), w - R, int(y))
            p.setPen(QColor(CLR["dim2"]))
            p.drawText(6, int(y) + 4, f"{i / 5:.1f}")
            p.setPen(QPen(QColor(CLR["border"]), 1))

        def x_of(ep, tot):
            tot = max(tot, 2)
            return L + cw * (ep - 1) / (tot - 1)

        tot = self.total or max([e for e, _, _ in self.points] + [10])
        # 验收线 0.85
        y85 = T + ch - ch * TARGET_DICE
        p.setPen(QPen(QColor(CLR["warn"]), 1.5, Qt.DashLine))
        p.drawLine(L, int(y85), w - R, int(y85))
        p.setPen(QColor(CLR["warn"]))
        p.drawText(w - R - 92, int(y85) - 6, f"验收线 {TARGET_DICE}")

        for vals, col in ((self.points, CLR["accent"]),):
            if len(vals) >= 2:
                pen = QPen(QColor(col), 2)
                p.setPen(pen)
                for (e1, r1, _), (e2, r2, _) in zip(vals, vals[1:]):
                    p.drawLine(int(x_of(e1, tot)), int(T + ch - ch * r1),
                               int(x_of(e2, tot)), int(T + ch - ch * r2))
        if len(self.points) >= 2:
            p.setPen(QPen(QColor(CLR["blue"]), 1.5, Qt.DashLine))
            for (e1, _, a1), (e2, _, a2) in zip(self.points, self.points[1:]):
                p.drawLine(int(x_of(e1, tot)), int(T + ch - ch * a1),
                           int(x_of(e2, tot)), int(T + ch - ch * a2))
        if self.points:
            e, r, a = self.points[-1]
            p.setPen(QColor(CLR["accent"]))
            p.drawText(L + 6, T + 14,
                       f"epoch {e}  raw {r:.4f}  ema {a:.4f}")
        # x 轴刻度
        p.setPen(QColor(CLR["dim2"]))
        step = max(1, tot // 8)
        ep = 1
        while ep <= tot:
            p.drawText(int(x_of(ep, tot)) - 10, h - 10, str(ep))
            ep += step
        # 图例 / 状态
        p.setPen(QColor(CLR["dim"]))
        if not self.points:
            p.drawText(L + cw / 2 - 70, T + ch / 2, self.status)


# ---------------------------------------------------------------- 页面
class DataPage(QWidget):
    """数据选择与校验。"""
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        card = Card()
        card.v.addWidget(self._lbl("标注数据文件夹"))
        row = QHBoxLayout()
        self.edit = QLineEdit(str(DEFAULT_DATA))
        self.edit.setReadOnly(True)
        btn = QPushButton("浏览…")
        btn.clicked.connect(self.pick)
        btn_def = QPushButton("恢复默认")
        btn_def.clicked.connect(lambda: self.set_dir(DEFAULT_DATA))
        row.addWidget(self.edit, 1)
        row.addWidget(btn_def)
        row.addWidget(btn)
        card.v.addLayout(row)
        card.v.addWidget(self._hint(
            "文件夹需包含 manifest.csv（标注清单）与 masks/（中后肠掩码，"
            "白色为前景）。训练只读取 manifest.csv 中 split=train/test 的行；"
            "计数另需 drafts/（检测草稿框）。"))
        v.addWidget(card)

        self.check_card = Card()
        self.check_card.v.addWidget(self._lbl("数据校验"))
        self.checks_v = QVBoxLayout()
        self.check_card.v.addLayout(self.checks_v)
        v.addWidget(self.check_card)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles = {}
        for i, (key, label, small) in enumerate([
                ("n_total", "图像总数", False), ("n_train", "训练集", False),
                ("n_test", "测试集", False),
                ("n_strata", "分层（处理/造模/对照）", True),
                ("n_masks", "掩码文件", False),
                ("n_drafts", "检测草稿框", False),
                ("v1", "计数基线 v1_counts", True)]):
            t = StatTile(label, small=small)
            self.tiles[key] = t
            grid.addWidget(t, i // 4, i % 4)
        v.addLayout(grid)

        act = QHBoxLayout()
        act.addStretch(1)
        b_open = QPushButton("打开数据文件夹")
        b_open.clicked.connect(self.open_dir)
        act.addWidget(b_open)
        v.addLayout(act)
        v.addStretch(1)

    def show_validation(self, info, checks):
        """由主窗口填充校验清单与统计磁贴。"""
        while self.checks_v.count():
            w = self.checks_v.takeAt(0).widget()
            if w:
                w.deleteLater()
        for c in checks:
            self.checks_v.addWidget(c)
        for key, t in self.tiles.items():
            t.set_value(str(info.get(key, "—")))

    def _lbl(self, t):
        l = QLabel(t)
        l.setObjectName("SectionTitle")
        return l

    def _hint(self, t):
        l = QLabel(t)
        l.setObjectName("Hint")
        l.setWordWrap(True)
        return l

    def pick(self):
        d = QFileDialog.getExistingDirectory(self, "选择标注数据文件夹",
                                             self.edit.text())
        if d:
            self.set_dir(Path(d))

    def set_dir(self, d: Path):
        self.mw.data_dir = Path(d)
        self.edit.setText(str(d))
        self.mw.validate_data()
        # 切换数据目录后刷新模型下拉，避免残留引用其他文件夹的权重
        if hasattr(self.mw, "refresh_model_lists"):
            self.mw.refresh_model_lists()

    def open_dir(self):
        open_path(self.mw.data_dir)


class TrainPage(QWidget):
    """训练控制台。"""
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        scroll.setWidget(inner)
        v = QVBoxLayout(inner)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)
        QVBoxLayout(self).addWidget(scroll)

        # ---- 预设与参数 ----
        cfg_card = Card()
        cfg_card.v.addWidget(self._lbl("训练配置"))
        top = QHBoxLayout()
        top.addWidget(QLabel("预设方案"))
        self.preset = QComboBox()
        self.preset.addItems([p[0] for p in PRESETS] + ["自定义"])
        self.preset.setCurrentIndex(2)
        self.preset.currentIndexChanged.connect(self.apply_preset)
        top.addWidget(self.preset, 1)
        top.addStretch(2)
        cfg_card.v.addLayout(top)

        g = QGridLayout()
        g.setSpacing(10)
        self.sp = {}
        rows = [
            ("epochs", "训练轮数", 1, 2000, 400),
            ("base", "基础通道 base", 8, 64, 32),
            ("depth", "网络深度 depth", 2, 5, 4),
            ("batch", "批大小 batch", 1, 16, 4),
            ("scale", "下采样倍率 ÷N", 2, 6, 4),
        ]
        for i, (key, label, lo, hi, dflt) in enumerate(rows):
            g.addWidget(QLabel(label), i // 3 * 2, i % 3)
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(dflt)
            sb.valueChanged.connect(lambda *_: self.mark_custom())
            self.sp[key] = sb
            g.addWidget(sb, i // 3 * 2 + 1, i % 3)
        for i, (key, label, lo, hi, dflt, step) in enumerate([
                ("lr", "学习率", 0.00001, 0.01, 0.002, 0.0001)]):
            g.addWidget(QLabel(label), 4, 0)
            ds = QDoubleSpinBox()
            ds.setDecimals(5)
            ds.setRange(lo, hi)
            ds.setValue(dflt)
            ds.setSingleStep(step)
            ds.valueChanged.connect(lambda *_: self.mark_custom())
            self.sp[key] = ds
            g.addWidget(ds, 5, 0)
        g.addWidget(QLabel("归一化 norm"), 4, 1)
        self.norm = QComboBox()
        self.norm.addItems(["gn（推荐，小数据集）", "bn"])
        self.norm.setCurrentIndex(0)
        self.norm.currentIndexChanged.connect(lambda *_: self.mark_custom())
        g.addWidget(self.norm, 5, 1)
        g.addWidget(QLabel("增强退火"), 4, 2)
        self.aug = QCheckBox("后期降至 0")
        self.aug.setChecked(True)
        self.aug.toggled.connect(lambda *_: self.mark_custom())
        g.addWidget(self.aug, 5, 2)
        cfg_card.v.addLayout(g)

        adv = QHBoxLayout()
        adv.addWidget(QLabel("初始化权重"))
        self.init_model = QComboBox()
        self.init_model.setMinimumWidth(220)
        self.init_model.currentIndexChanged.connect(
            lambda *_: self.mark_custom())
        adv.addWidget(self.init_model)
        self.resume = QCheckBox("允许断点续训")
        self.resume.setChecked(True)
        self.resume.toggled.connect(lambda *_: self.mark_custom())
        adv.addWidget(self.resume)
        adv.addStretch(1)
        cfg_card.v.addLayout(adv)
        cfg_card.v.addWidget(self._hint(
            "GN 批大小不得超过 4（8GB 显存会 OOM）；÷3 倍率像素量为 ÷4 的 1.78 倍，"
            "批大小需减半。停止训练后可从断点续训恢复（每 5 轮保存一次）。"))
        v.addWidget(cfg_card)

        # ---- 运行区 ----
        run_card = Card()
        h = QHBoxLayout()
        self.btn_start = QPushButton("开始训练")
        self.btn_start.setObjectName("BtnPrimary")
        self.btn_start.setMinimumWidth(120)
        self.btn_start.clicked.connect(self.mw.start_training)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("BtnDanger")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.mw.stop_proc)
        h.addWidget(self.btn_start)
        h.addWidget(self.btn_stop)
        h.addStretch(1)
        self.lbl_out = QLabel("")
        self.lbl_out.setObjectName("Hint")
        h.addWidget(self.lbl_out)
        run_card.v.addLayout(h)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("空闲")
        run_card.v.addWidget(self.progress)
        self.status = QLabel("就绪")
        self.status.setObjectName("Hint")
        run_card.v.addWidget(self.status)
        v.addWidget(run_card)

        log_card = Card()
        log_card.v.addWidget(self._lbl("运行日志"))
        self.log = LogConsole()
        self.log.setMinimumHeight(240)
        log_card.v.addWidget(self.log)
        v.addWidget(log_card, 1)

        self.apply_preset(2)

    def _lbl(self, t):
        l = QLabel(t)
        l.setObjectName("SectionTitle")
        return l

    def _hint(self, t):
        l = QLabel(t)
        l.setObjectName("Hint")
        l.setWordWrap(True)
        return l

    def params(self):
        return {
            "epochs": self.sp["epochs"].value(),
            "base": self.sp["base"].value(),
            "depth": self.sp["depth"].value(),
            "batch": self.sp["batch"].value(),
            "scale": self.sp["scale"].value(),
            "lr": round(self.sp["lr"].value(), 5),
            "norm": "gn" if self.norm.currentIndex() == 0 else "bn",
            "aug_decay": self.aug.isChecked(),
            "init": (self.init_model.currentText()
                     if self.init_model.currentIndex() > 0 else ""),
            "resume": self.resume.isChecked(),
        }

    def apply_preset(self, idx):
        if idx >= len(PRESETS):
            return
        p = PRESETS[idx][1]
        widgets = [self.sp[k] for k in self.sp] + [
            self.norm, self.aug, self.resume, self.init_model]
        for w in widgets:
            w.blockSignals(True)
        self.sp["epochs"].setValue(p["epochs"])
        self.sp["base"].setValue(p["base"])
        self.sp["depth"].setValue(p["depth"])
        self.sp["batch"].setValue(p["batch"])
        self.sp["scale"].setValue(p["scale"])
        self.sp["lr"].setValue(p["lr"])
        self.norm.setCurrentIndex(0 if p["norm"] == "gn" else 1)
        self.aug.setChecked(p["aug_decay"])
        if p["init"]:
            i = self.init_model.findText(p["init"])
            self.init_model.setCurrentIndex(i if i >= 0 else 0)
        else:
            self.init_model.setCurrentIndex(0)
        self.resume.setChecked(p["resume"])
        for w in widgets:
            w.blockSignals(False)

    def mark_custom(self):
        self.preset.blockSignals(True)
        self.preset.setCurrentIndex(self.preset.count() - 1)
        self.preset.blockSignals(False)


class MetricsPage(QWidget):
    """指标可视化。"""
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        card = Card()
        card.v.addWidget(self._lbl("Dice 训练曲线（每 5 轮评估）"))
        self.curve = DiceCurveWidget()
        card.v.addWidget(self.curve)
        v.addWidget(card, 1)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles = {
            "mean": StatTile("test Dice 均值", accent=True),
            "median": StatTile("中位数"),
            "p10": StatTile("P10（最差 10%）"),
            "best": StatTile("最佳轮次"),
        }
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, 0, i)
        v.addLayout(grid)

        self.badge = QLabel("尚无结果 —— 请先训练，或刷新已有报告")
        self.badge.setObjectName("Hint")
        v.addWidget(self.badge)

        h = QHBoxLayout()
        h.addStretch(1)
        b_ref = QPushButton("刷新报告")
        b_ref.clicked.connect(self.mw.load_metrics)
        b_txt = QPushButton("打开训练报告")
        b_txt.clicked.connect(self.open_report)
        b_vis = QPushButton("打开可视化目录")
        b_vis.clicked.connect(self.open_vis)
        h.addWidget(b_ref)
        h.addWidget(b_txt)
        h.addWidget(b_vis)
        v.addLayout(h)

    def _lbl(self, t):
        l = QLabel(t)
        l.setObjectName("SectionTitle")
        return l

    def show_metrics(self, m):
        if not m:
            for t in self.tiles.values():
                t.set_value("—")
            self.badge.setText("尚无结果 —— 请先训练，或刷新已有报告")
            return
        self.tiles["mean"].set_value(f"{m['mean']:.4f}")
        self.tiles["median"].set_value(f"{m['median']:.4f}")
        self.tiles["p10"].set_value(f"{m['p10']:.4f}")
        self.tiles["best"].set_value(
            f"{m['best_epoch']}（{m['best_dice']:.4f}）")
        if m["mean"] >= TARGET_DICE:
            self.badge.setText(
                f'<b style="color:{CLR["accent"]};">达标</b> —— test Dice '
                f'{m["mean"]:.4f} ≥ {TARGET_DICE}')
        else:
            self.badge.setText(
                f'<b style="color:{CLR["warn"]};">未达标</b> —— 距 '
                f'{TARGET_DICE} 还差 {TARGET_DICE - m["mean"]:.4f}；'
                f'训练集已过线时说明瓶颈是泛化，建议补充人工标注')

    def open_report(self):
        p = self.mw.metrics_path()
        if p and p.exists():
            open_path(p)
        else:
            QMessageBox.information(self, "提示", "尚未生成训练报告")

    def open_vis(self):
        p = self.mw.data_dir / "unet_vis_v3"
        open_path(p if p.exists() else self.mw.data_dir)


class CountPage(QWidget):
    """端到端计数。"""
    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        v = QVBoxLayout(self)
        v.setContentsMargins(28, 24, 28, 24)
        v.setSpacing(16)

        card = Card()
        card.v.addWidget(self._lbl("端到端计数（AI 分割掩码 + 经典检测过滤）"))
        h = QHBoxLayout()
        h.addWidget(QLabel("分割模型"))
        self.model = QComboBox()
        self.model.setMinimumWidth(260)
        h.addWidget(self.model)
        h.addWidget(QLabel("输出标签"))
        self.tag = QLineEdit("gui")
        self.tag.setMaximumWidth(120)
        h.addWidget(self.tag)
        self.btn = QPushButton("运行计数")
        self.btn.setObjectName("BtnPrimary")
        self.btn.clicked.connect(self.mw.start_counting)
        h.addWidget(self.btn)
        h.addStretch(1)
        card.v.addLayout(h)
        card.v.addWidget(self._hint(
            "计数口径：仅统计中心落在中后肠掩码内的检测框。若数据文件夹缺少 "
            "v1_counts.csv（人工掩码计数基线），将自动先生成再比对。"))
        v.addWidget(card)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.tiles = {
            "train": StatTile("train Dice"),
            "test": StatTile("test Dice", accent=True),
            "r": StatTile("计数相关 r"),
            "bias": StatTile("绝对偏差"),
        }
        for i, t in enumerate(self.tiles.values()):
            grid.addWidget(t, 0, i)
        v.addLayout(grid)

        log_card = Card()
        log_card.v.addWidget(self._lbl("运行日志"))
        self.log = LogConsole()
        log_card.v.addWidget(self.log, 1)
        v.addWidget(log_card, 1)

        h2 = QHBoxLayout()
        h2.addStretch(1)
        b_csv = QPushButton("打开计数 CSV")
        b_csv.clicked.connect(self.open_csv)
        b_vis = QPushButton("打开可视化目录")
        b_vis.clicked.connect(self.open_vis)
        h2.addWidget(b_csv)
        h2.addWidget(b_vis)
        v.addLayout(h2)

    def _lbl(self, t):
        l = QLabel(t)
        l.setObjectName("SectionTitle")
        return l

    def _hint(self, t):
        l = QLabel(t)
        l.setObjectName("Hint")
        l.setWordWrap(True)
        return l

    def open_csv(self):
        tag = self.tag.text().strip() or "gui"
        p = self.mw.data_dir / f"{tag}_counts.csv"
        open_path(p if p.exists() else self.mw.data_dir)

    def open_vis(self):
        tag = self.tag.text().strip() or "gui"
        p = self.mw.data_dir / f"{tag}_vis"
        open_path(p if p.exists() else self.mw.data_dir)


# ---------------------------------------------------------------- 环境页
class EnvPage(QWidget):
    """环境页：依赖自检 + 一键安装 PyTorch（CPU / CUDA 可选）。"""

    def __init__(self, mw):
        super().__init__()
        self.mw = mw
        v = QVBoxLayout(self)
        v.setContentsMargins(24, 20, 24, 20)
        v.setSpacing(12)

        # ---- 自检区放入滚动区：窗口再小也不会把行挤压成碎片 ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;}"
            "QScrollArea>QWidget>QWidget{background:transparent;}")
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.setSpacing(12)
        scroll.setWidget(box)

        card = Card()
        card.v.addWidget(self._lbl("计算环境自检"))
        card.v.addWidget(self._hint(
            "PyTorch 按需安装，安装包已内含 CUDA 13 运行库"
            "（无需单独安装 CUDA Toolkit）。有 NVIDIA GPU 请选 CUDA 版，"
            "训练快约 37 倍；否则选 CPU 版。"))
        self.rows_v = QVBoxLayout()
        self.rows_v.setSpacing(8)
        card.v.addLayout(self.rows_v)
        bv.addWidget(card)

        # 依赖行定义：(key, 名称, 说明)
        specs = [
            ("python", "Python 运行时", "随安装包内置，无需安装"),
            ("PySide6", "PySide6 界面库", "随安装包内置"),
            ("numpy", "numpy 数值库", "随安装包内置"),
            ("cv2", "opencv 图像库", "随安装包内置"),
            ("torch", "PyTorch 深度学习框架", "训练与计数必需，按需安装"),
            ("gpu", "NVIDIA GPU 驱动", "决定能否使用 CUDA 加速"),
            ("disk", "磁盘剩余空间", "模型与输出需要数 GB 空间"),
        ]
        self.status_lbl = {}
        for key, name, desc in specs:
            row = QFrame()
            row.setObjectName("Card")
            row.setToolTip(desc)          # 说明文字改为悬停提示，行保持单行
            row.setMinimumHeight(40)
            h = QHBoxLayout(row)
            h.setContentsMargins(14, 9, 14, 9)
            h.setSpacing(16)
            n = QLabel(name)
            n.setStyleSheet(f"font-size:13px; font-weight:500; "
                            f"color:{CLR['text']};")
            n.setToolTip(desc)
            h.addWidget(n)
            st = QLabel("检测中…")
            st.setWordWrap(True)           # 长文本换行，避免被裁切
            st.setMinimumWidth(300)
            st.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            st.setStyleSheet(f"font-size:13px; color:#C3C8D2;")
            st.setToolTip(desc)
            self.status_lbl[key] = st
            h.addWidget(st, 1)
            self.rows_v.addWidget(row)
        v.addWidget(scroll, 1)

        # torch 专属操作行（固定在滚动区外，始终可见）
        act = QHBoxLayout()
        act.setSpacing(10)
        self.btn_cpu = QPushButton("安装 CPU 版 PyTorch（约 200 MB）")
        self.btn_cuda = QPushButton("安装 CUDA 版 PyTorch（约 2.5 GB，推荐）")
        self.btn_cuda.setObjectName("BtnPrimary")
        self.btn_cpu.clicked.connect(lambda: self.mw.start_torch_install("cpu"))
        self.btn_cuda.clicked.connect(lambda: self.mw.start_torch_install("cuda"))
        act.addWidget(self.btn_cpu)
        act.addWidget(self.btn_cuda)
        act.addStretch(1)
        b_re = QPushButton("重新检测")
        b_re.clicked.connect(self.mw.check_env)
        act.addWidget(b_re)
        v.addLayout(act)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("空闲")
        self.progress.hide()
        v.addWidget(self.progress)

        log_card = Card()
        log_card.v.addWidget(self._lbl("安装日志"))
        self.log = LogConsole()
        self.log.setMinimumHeight(110)
        log_card.v.addWidget(self.log)
        v.addWidget(log_card)

    def _lbl(self, t):
        l = QLabel(t)
        l.setObjectName("SectionTitle")
        return l

    def _hint(self, t):
        l = QLabel(t)
        l.setObjectName("Hint")
        l.setWordWrap(True)
        return l

    def set_status(self, key, text, ok=None):
        st = self.status_lbl.get(key)
        if st is None:
            return
        st.setText(text)
        col = {True: CLR["accent"], False: CLR["warn"]}.get(ok, "#C3C8D2")
        st.setStyleSheet(f"font-size:13px; color: {col};")

    def set_torch_installed(self, installed):
        # 按钮保持可用：已安装者也应能切换 CPU / CUDA 版本
        self.btn_cpu.setEnabled(True)
        self.btn_cuda.setEnabled(True)

    def show_installing(self, kind):
        self.btn_cpu.setEnabled(False)
        self.btn_cuda.setEnabled(False)
        self.progress.show()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat(f"下载安装 {kind.upper()} 版 PyTorch… %p%")
        self.log.clear()
        self.log.append_line(
            f"开始安装 {kind.upper()} 版 PyTorch"
            f"（源：{TORCH_INDEX[kind]}，{TORCH_SIZE_HINT[kind]}）", "info")

    def install_done(self, ok):
        self.btn_cpu.setEnabled(True)
        self.btn_cuda.setEnabled(True)
        if ok:
            self.progress.setFormat("安装完成")
        else:
            self.progress.setFormat("安装失败")


# ---------------------------------------------------------------- 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("斑马鱼荧光细胞定量工具")
        self.resize(1180, 800)
        self.setMinimumSize(1060, 720)
        self.data_dir = DEFAULT_DATA
        self.proc = None
        self.proc_kind = None            # train / v1 / count / install
        self.train_total = 0
        self.epoch_times = []
        self.curve_points = []
        self.metrics = None
        self.t_train_start = 0.0
        self.smoke_done = False
        self._gpu_name = None
        self.env_torch = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- 侧栏 ----
        side = QWidget()
        side.setObjectName("Sidebar")
        side.setFixedWidth(216)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(12, 20, 12, 16)
        sv.setSpacing(4)
        logo = QLabel("斑马鱼定量")
        logo.setStyleSheet(f"font-size:16px; font-weight:500; "
                           f"color:{CLR['text']}; padding:6px 8px 2px;")
        sub = QLabel("荧光细胞分割 · 计数")
        sub.setStyleSheet(f"font-size:11px; color:{CLR['dim2']}; "
                          f"padding:0 8px 14px;")
        sv.addWidget(logo)
        sv.addWidget(sub)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.pages = {}
        for i, name in enumerate(["数据", "训练", "指标", "计数", "环境"]):
            b = NavButton(name)
            self.nav.addButton(b, i)
            sv.addWidget(b)
        sv.addStretch(1)
        ver = QLabel(f"v{APP_VERSION} · Python {sys.version.split()[0]}")
        ver.setStyleSheet(f"font-size:10px; color:{CLR['dim2']}; padding:8px;")
        sv.addWidget(ver)
        root.addWidget(side)

        # ---- 右侧 ----
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(0)
        top = QWidget()
        top.setObjectName("TopBar")
        top.setFixedHeight(64)
        th = QHBoxLayout(top)
        th.setContentsMargins(28, 12, 28, 12)
        self.page_title = QLabel("数据")
        self.page_title.setObjectName("PageTitle")
        self.page_sub = QLabel("选择标注数据文件夹")
        self.page_sub.setObjectName("PageSub")
        tv = QVBoxLayout()
        tv.setSpacing(0)
        tv.addWidget(self.page_title)
        tv.addWidget(self.page_sub)
        th.addLayout(tv)
        th.addStretch(1)
        self.gpu_chip = QLabel("检测 GPU 中…")
        self.gpu_chip.setObjectName("Chip")
        th.addWidget(self.gpu_chip)
        right.addWidget(top)

        self.stack = QStackedWidget()
        self.pages["data"] = DataPage(self)
        self.pages["train"] = TrainPage(self)
        self.pages["metrics"] = MetricsPage(self)
        self.pages["count"] = CountPage(self)
        self.pages["env"] = EnvPage(self)
        for p in self.pages.values():
            self.stack.addWidget(p)
        right.addWidget(self.stack, 1)

        status = QWidget()
        status.setObjectName("StatusBar")
        status.setFixedHeight(30)
        sh = QHBoxLayout(status)
        sh.setContentsMargins(28, 0, 28, 0)
        self.lbl_data = QLabel(f"数据：{self.data_dir}")
        self.lbl_right = QLabel("")
        sh.addWidget(self.lbl_data)
        sh.addStretch(1)
        sh.addWidget(self.lbl_right)
        right.addWidget(status)
        root.addLayout(right, 1)

        self._subtitles = {"data": "选择标注数据文件夹",
                           "train": "配置并训练中后肠分割模型",
                           "metrics": "Dice 曲线与训练报告",
                           "count": "掩码内荧光细胞计数",
                           "env": "依赖自检与 PyTorch 安装"}
        self.nav.idClicked.connect(self.switch_page)
        self.switch_page(0)

        # 页面就绪后再触发首次数据校验（避免访问尚未注册的页面）
        self.pages["data"].set_dir(DEFAULT_DATA)
        self.refresh_model_lists()
        self.check_gpu()
        self.check_env(_auto=True)
        QTimer.singleShot(600, self._after_init)

    # ---------------- 页面切换 ----------------
    def switch_page(self, idx):
        self.stack.setCurrentIndex(idx)
        names = ["data", "train", "metrics", "count", "env"]
        btn = self.nav.button(idx)
        if btn:
            btn.setChecked(True)
        self.page_title.setText(["数据", "训练", "指标", "计数", "环境"][idx])
        self.page_sub.setText(self._subtitles[names[idx]])
        if idx == 3:
            self.refresh_model_lists()

    def _after_init(self):
        if SMOKE_ENV:
            QTimer.singleShot(300, self.run_smoke_env)
        elif SMOKE:
            QTimer.singleShot(300, self.run_smoke)

    # ---------------- GPU 状态 ----------------
    def check_gpu(self):
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.finished.connect(lambda code, st, pp=p: self._gpu_done(pp))
        p.errorOccurred.connect(lambda err, pp=p: self._gpu_missing(pp))
        p.start("nvidia-smi",
                ["--query-gpu=name,driver_version", "--format=csv,noheader"])
        self._gpu_proc = p

    def _gpu_missing(self, p):
        try:
            p.kill()
        except Exception:
            pass
        self.gpu_chip.setText("未检测到 NVIDIA GPU（训练将极慢）")
        self.gpu_chip.setObjectName("ChipWarn")
        self.gpu_chip.style().polish(self.gpu_chip)
        self.pages["env"].set_status("gpu", "未检测到 NVIDIA GPU", False)

    def _gpu_done(self, p):
        out = bytes(p.readAllStandardOutput().data()).decode(
            "utf-8", "replace").strip()
        if out:
            # "NVIDIA GeForce RTX 4060 Laptop GPU, 591.96" → 短名 + 驱动版本
            parts = [s.strip() for s in out.split(",")]
            short = parts[0].replace("NVIDIA GeForce ", "").replace(" GPU", "")
            driver = parts[-1] if len(parts) > 1 else ""
            self._gpu_name = short
            self.gpu_chip.setText(f"GPU · {short}"
                                  + (f" · 驱动 {driver}" if driver else ""))
            self.gpu_chip.setObjectName("ChipOk")
            self.gpu_chip.style().polish(self.gpu_chip)
            self.pages["env"].set_status(
                "gpu", f"{short}" + (f"（驱动 {driver}）" if driver else ""), True)
        else:
            self._gpu_missing(p)

    # ---------------- 环境自检 ----------------
    def ensure_env_probed(self, timeout=90):
        """启动时的异步探测尚未返回时，同步补测一次（阻塞数秒）。"""
        if self.env_torch is not None:
            return True
        try:
            import subprocess
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            r = subprocess.run([sys.executable, "-c", PROBE_SCRIPT],
                               capture_output=True, text=True,
                               timeout=timeout, env=env)
            for line in r.stdout.splitlines():
                if line.startswith("PROBE:"):
                    import json
                    info = json.loads(line[6:])
                    self.env_torch = info.get("torch")
                    return True
        except Exception:
            pass
        return False

    def probe_python(self, callback):
        """在子进程中探测已安装库，避免 GUI 进程直接 import 重库。"""
        p = QProcess(self)
        p.setProcessChannelMode(QProcess.MergedChannels)
        p.finished.connect(lambda code, st, pp=p, cb=callback:
                           self._probe_done(pp, cb))
        p.start(sys.executable, ["-c", PROBE_SCRIPT])
        self._probe_proc = p

    def _probe_done(self, p, callback):
        out = bytes(p.readAllStandardOutput().data()).decode(
            "utf-8", "replace")
        info = None
        for line in out.splitlines():
            if line.startswith("PROBE:"):
                try:
                    import json
                    info = json.loads(line[6:])
                except Exception:
                    info = None
        callback(info)

    def check_env(self, _auto=False):
        """刷新环境页所有依赖状态；torch 缺失时首次自动跳到本页。"""
        ep = self.pages["env"]
        ep.set_status("python", f"Python {sys.version.split()[0]} · 内置", True)
        for key in ("PySide6", "numpy", "cv2"):
            ep.set_status(key, "检测中…")

        def done(info):
            if info is None:
                for key in ("PySide6", "numpy", "cv2", "torch", "disk"):
                    ep.set_status(key, "检测失败（子进程无输出）", False)
                return
            for key, mod in [("PySide6", "PySide6"), ("numpy", "numpy"),
                             ("cv2", "cv2")]:
                ver = info.get(mod)
                if ver:
                    ep.set_status(key, f"{mod} {ver} · 内置", True)
                else:
                    ep.set_status(key, f"{mod} 缺失（不应发生）", False)
            # PyTorch
            tv = info.get("torch")
            self.env_torch = tv
            if tv:
                if info.get("torch_cuda_build"):
                    if info.get("cuda_ok"):
                        ep.set_status("torch", f"{tv} · GPU 加速可用", True)
                    else:
                        ep.set_status(
                            "torch", "CUDA 版不可用（驱动过旧或无 GPU）", False)
                else:
                    ep.set_status("torch", f"{tv} · CPU 版（训练慢约 37 倍）",
                                  None)
                ep.set_torch_installed(True)
            else:
                ep.set_status("torch", "未安装 —— 训练/计数前必须安装", False)
                ep.set_torch_installed(False)
                if _auto:
                    # 首次启动发现 torch 缺失：自动引导到环境页
                    self.switch_page(4)
            # 磁盘
            free = info.get("disk_free_gb")
            if free is not None:
                ok = free >= 5
                ep.set_status("disk", f"{free} GB 可用"
                              + ("" if ok else "（不足 5 GB，建议清理）"), ok)
            # GPU 与 torch 的组合结论
            if info.get("cuda_ok"):
                ep.set_status("gpu", ep.status_lbl["gpu"].text()
                              + " · torch 已识别", True)

        self.probe_python(done)

    # ---------------- PyTorch 一键安装 ----------------
    def start_torch_install(self, kind):
        if self.proc:
            return
        has_gpu = self._gpu_name is not None
        if kind == "cuda" and not has_gpu:
            QMessageBox.warning(
                self, "未检测到 NVIDIA GPU",
                "本机未检测到 NVIDIA GPU，安装 CUDA 版 PyTorch 也无法加速。\n"
                "建议安装 CPU 版；如确认有 GPU 请检查驱动。")
            return
        if SMOKE or SMOKE_ENV:
            ret = QMessageBox.Yes        # 自检模式自动确认
        else:
            ret = QMessageBox.question(
                self, f"安装 {kind.upper()} 版 PyTorch",
                f"将从 PyTorch 官方源下载并安装 {kind.upper()} 版"
                f"（{TORCH_SIZE_HINT[kind]}），耗时取决于网络。\n\n"
                f"安装目标：{sys.executable}\n\n是否开始？")
        if ret != QMessageBox.Yes:
            return
        ep = self.pages["env"]
        ep.show_installing(kind)
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        # pip 临时目录指到应用所在盘（系统盘空间可能不足，CUDA wheel 2.5GB）
        tmp = ROOT / "tmp_install"
        try:
            tmp.mkdir(exist_ok=True)
        except Exception:
            tmp = None
        if tmp:
            env["TMPDIR"] = str(tmp)
            env["TMP"] = str(tmp)
            env["TEMP"] = str(tmp)
        self.launch("install", None, env, extra_args=[
            "-m", "pip", "install", "--no-cache-dir", "torch",
            "--index-url", TORCH_INDEX[kind]])

    # ---------------- 数据校验 ----------------
    def validate_data(self):
        import csv as _csv
        d = self.data_dir
        self.lbl_data.setText(f"数据：{d}")
        info = {"n_total": 0, "n_train": 0, "n_test": 0, "n_strata": "—",
                "n_masks": 0, "n_drafts": 0, "v1": "—"}
        checks = []
        man = d / "manifest.csv"
        strata = {}
        if man.exists():
            try:
                with open(man, encoding="utf-8-sig") as f:
                    rows = list(_csv.DictReader(f))
                info["n_total"] = len(rows)
                info["n_train"] = sum(1 for r in rows
                                      if r.get("split") == "train")
                info["n_test"] = sum(1 for r in rows
                                     if r.get("split") == "test")
                for r in rows:
                    strata[r.get("stratum", "?")] = \
                        strata.get(r.get("stratum", "?"), 0) + 1
                info["n_strata"] = " / ".join(
                    f"{k} {v}" for k, v in sorted(strata.items()))
                checks.append(CheckItem(True,
                    f"manifest.csv · {len(rows)} 张"
                    f"（train {info['n_train']} / test {info['n_test']}）"))
            except Exception as e:
                checks.append(CheckItem(False, f"manifest.csv 解析失败：{e}"))
        else:
            checks.append(CheckItem(False,
                "未找到 manifest.csv —— 该文件夹不是有效的数据集"))
        masks = d / "masks"
        info["n_masks"] = len(list(masks.glob("*.png"))) if masks.exists() else 0
        checks.append(CheckItem(info["n_masks"] > 0,
                                f"masks/ · {info['n_masks']} 个掩码"))
        drafts = d / "drafts"
        info["n_drafts"] = (len(list(drafts.glob("*.txt")))
                            if drafts.exists() else 0)
        checks.append(CheckItem(info["n_drafts"] > 0,
                                f"drafts/ · {info['n_drafts']} 个草稿框"
                                f"（计数需要；训练不需要）"))
        v1 = d / "v1_counts.csv"
        info["v1"] = "已有" if v1.exists() else "计数时自动生成"
        checks.append(CheckItem(v1.exists(), f"v1_counts.csv · {info['v1']}"))

        if "data" in self.pages:
            self.pages["data"].show_validation(info, checks)
        return info

    # ---------------- 模型列表 ----------------
    def refresh_model_lists(self):
        d = self.data_dir
        files = sorted(p.name for p in d.glob("unet_model_*.pt")) \
            if d.exists() else []
        for combo in (self.pages["train"].init_model,
                      self.pages["count"].model):
            cur = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("（无）")
            combo.addItems(files)
            combo.blockSignals(False)
            i = combo.findText(cur)
            if i >= 0:
                combo.setCurrentIndex(i)
            elif files:
                best = next((f for f in files if "v4_gn" in f), files[0])
                combo.setCurrentIndex(max(0, combo.findText(best)))

    def metrics_path(self):
        return self.data_dir / "unet_metrics_v3.txt"

    # ---------------- 训练 ----------------
    def start_training(self):
        if self.proc:
            return
        if self.env_torch is None:
            self.ensure_env_probed()
        if not self.env_torch:
            ret = QMessageBox.question(
                self, "尚未安装 PyTorch",
                "训练需要 PyTorch 深度学习框架，当前尚未安装。\n"
                "是否前往「环境」页一键安装（CPU 版约 200MB / CUDA 版约 2.5GB）？")
            if ret == QMessageBox.Yes:
                self.switch_page(4)
            return
        info = self.validate_data()
        if not (self.data_dir / "manifest.csv").exists():
            QMessageBox.warning(self, "无法训练",
                                "请先选择包含 manifest.csv 的数据文件夹。")
            self.switch_page(0)
            return
        if info["n_masks"] == 0:
            QMessageBox.warning(self, "无法训练", "数据文件夹中缺少 masks/ 掩码。")
            return
        self.refresh_model_lists()      # 先刷新再读参数，防止引用旧目录的权重
        p = self.pages["train"].params()
        self.pages["train"].log.clear()
        self.pages["train"].log.append_line(
            f"数据：{self.data_dir}", "info")
        self.pages["train"].log.append_line(
            f"配置：÷{p['scale']} base{p['base']} depth{p['depth']} "
            f"{p['epochs']} 轮 batch{p['batch']} lr{p['lr']} "
            f"norm={p['norm']} aug_decay={int(p['aug_decay'])} "
            f"init={p['init'] or '无'} resume={int(p['resume'])}", "info")

        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["UNET_ANN_DIR"] = str(self.data_dir)
        env["UNET_EPOCHS"] = str(p["epochs"])
        env["UNET_BASE"] = str(p["base"])
        env["UNET_DEPTH"] = str(p["depth"])
        env["UNET_BATCH"] = str(p["batch"])
        env["UNET_SCALE"] = str(p["scale"])
        env["UNET_LR"] = str(p["lr"])
        env["UNET_NORM"] = p["norm"]
        env["UNET_AUG_DECAY"] = "1" if p["aug_decay"] else "0"
        env["UNET_RESUME"] = "1" if p["resume"] else "0"
        if p["init"]:
            env["UNET_INIT"] = p["init"]
        else:
            env.pop("UNET_INIT", None)

        self.train_total = p["epochs"]
        self.epoch_times = []
        self.curve_points = []
        self.metrics = None
        self.pages["train"].progress.setRange(0, p["epochs"])
        self.pages["train"].progress.setValue(0)
        self.pages["train"].progress.setFormat("加载数据…")
        self.pages["metrics"].curve.set_data([], p["epochs"],
                                             "训练进行中…")
        self.pages["metrics"].show_metrics(None)
        self.pages["metrics"].badge.setText("训练进行中…")
        self.t_train_start = time.time()
        self.launch("train", ROOT / "tools_train_unet_v3.py", env)

    # ---------------- 计数 ----------------
    def start_counting(self):
        if self.proc:
            return
        if self.env_torch is None:
            self.ensure_env_probed()
        if not self.env_torch:
            ret = QMessageBox.question(
                self, "尚未安装 PyTorch",
                "计数需要加载 U-Net 模型（PyTorch），当前尚未安装。\n"
                "是否前往「环境」页一键安装（CPU 版约 200MB / CUDA 版约 2.5GB）？")
            if ret == QMessageBox.Yes:
                self.switch_page(4)
            return
        d = self.data_dir
        if not (d / "manifest.csv").exists():
            QMessageBox.warning(self, "无法计数",
                                "请先选择包含 manifest.csv 的数据文件夹。")
            self.switch_page(0)
            return
        self.refresh_model_lists()
        model = self.pages["count"].model.currentText()
        if model in ("", "（无）"):
            QMessageBox.warning(self, "无法计数",
                                "数据文件夹中没有 unet_model_*.pt 模型权重，"
                                "请先训练。")
            return
        tag = self.pages["count"].tag.text().strip() or "gui"
        self.pages["count"].log.clear()
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        env["UNET_ANN_DIR"] = str(d)
        env["V2_MODEL_PATH"] = str(d / model)
        env["V2_TAG"] = tag
        self.count_tag = tag
        self.count_env = env
        self.pages["count"].log.append_line(
            f"模型：{model} · 数据：{d}", "info")
        if (d / "v1_counts.csv").exists():
            self.launch("count", ROOT / "tools_v2_count.py", env)
        else:
            self.pages["count"].log.append_line(
                "未发现 v1_counts.csv，先生成人工掩码计数基线…", "warn")
            self.launch("v1", ROOT / "tools_v1_count.py", env)

    # ---------------- 子进程管理 ----------------
    def launch(self, kind, script, env, extra_args=None):
        self.proc_kind = kind
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        penv = QProcessEnvironment.systemEnvironment()
        for k, v in env.items():
            penv.insert(k, str(v))
        self.proc.setProcessEnvironment(penv)
        self.proc.readyReadStandardOutput.connect(self.on_stdout)
        self.proc.finished.connect(self.on_finished)
        self.proc.errorOccurred.connect(self.on_proc_error)
        # 子进程用 pythonw.exe（GUI 子系统），避免控制台黑窗闪烁；
        # 输出经管道捕获，功能与 python.exe 完全一致
        exe = sys.executable
        if exe.lower().endswith("python.exe"):
            cand = Path(exe).with_name("pythonw.exe")
            if cand.exists():
                exe = str(cand)
        args = extra_args if extra_args else ["-u", str(script)]
        self.proc.start(exe, args)
        tp = self.pages["train"]
        if kind == "install":
            tp.btn_start.setEnabled(False)
            tp.btn_stop.setEnabled(True)
            tp.lbl_out.setText("运行中：pip install torch")
        else:
            tp.btn_start.setEnabled(False)
            tp.btn_stop.setEnabled(True)
            tp.lbl_out.setText(f"运行中：{script.name}")
            self.pages["count"].btn.setEnabled(False)

    def stop_proc(self):
        if self.proc:
            self.log_any("用户请求停止（已保存的断点仍可续训）…", "warn")
            self.proc.terminate()
            QTimer.singleShot(3000, self._kill_if_alive)

    def _kill_if_alive(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()

    def on_proc_error(self, err):
        self.log_any(f"进程错误：{err}", "err")

    def on_stdout(self):
        data = bytes(self.proc.readAllStandardOutput().data()).decode(
            "utf-8", "replace")
        if self.proc_kind == "install":
            self.handle_install_output(data)
            return
        for line in data.splitlines():
            if line.strip():
                self.handle_line(line)

    def handle_install_output(self, data):
        """pip 下载输出：进度条用 \\r 刷新，取末段解析百分比，过滤噪声行。"""
        ep = self.pages["env"]
        seg = data.split("\r")[-1].strip()
        m = re.search(r"(\d{1,3})%", seg)
        if m:
            pct = max(0, min(100, int(m.group(1))))
            ep.progress.setValue(pct)
            ep.progress.setFormat(f"下载安装 PyTorch… {pct}%")
        for line in data.replace("\r", "\n").split("\n"):
            s = line.strip()
            if not s:
                continue
            if re.match(r"\s*\d{1,3}%", s) or set(s) <= set("-=|> "):
                continue                      # 纯进度刷新行，不进日志
            kind = ("err" if ("ERROR" in s or "错误" in s)
                    else "dice" if "Successfully installed" in s else "")
            ep.log.append_line(s, kind)

    def handle_line(self, line):
        kind = ""
        if "Dice raw" in line:
            kind = "dice"
        elif "Traceback" in line or "Error" in line or "错误" in line:
            kind = "err"
        elif "警告" in line or "跳过" in line:
            kind = "warn"
        self.log_any(line, kind)

        # 训练进度
        m = re.search(r"epoch\s+(\d+)\s*/\s*(\d+)", line)
        if m and self.proc_kind == "train":
            cur, tot = int(m.group(1)), int(m.group(2))
            self.pages["train"].progress.setRange(0, tot)
            self.pages["train"].progress.setValue(cur)
            self.pages["train"].progress.setFormat(f"epoch {cur}/{tot}  %p%")
            mt = re.search(r"\((\d+)s\)", line)
            if mt:
                self.epoch_times.append(int(mt.group(1)))
            md = re.search(r"Dice raw ([\d.]+) / ema ([\d.]+)", line)
            status = f"epoch {cur}/{tot}"
            if md:
                raw, ema = float(md.group(1)), float(md.group(2))
                self.curve_points.append((cur, raw, ema))
                self.pages["metrics"].curve.set_data(
                    self.curve_points, tot,
                    f"epoch {cur}  raw {raw:.4f}  ema {ema:.4f}")
                status += f" · Dice raw {raw:.4f} / ema {ema:.4f}"
            if len(self.epoch_times) >= 2:
                avg = sum(self.epoch_times[-10:]) / len(self.epoch_times[-10:])
                left = (tot - cur) * avg / 60
                status += f" · 预计剩余 {left:.0f} 分钟"
            self.pages["train"].status.setText(status)
        elif self.proc_kind == "train":
            m2 = re.search(r"预处理 (\d+)/(\d+)", line)
            if m2:
                self.pages["train"].progress.setRange(0, int(m2.group(2)))
                self.pages["train"].progress.setValue(int(m2.group(1)))
                self.pages["train"].progress.setFormat("预处理数据 %v/%m")

    def log_any(self, text, kind=""):
        if self.proc_kind == "install":
            self.pages["env"].log.append_line(text, kind)
        elif self.proc_kind in ("v1", "count"):
            self.pages["count"].log.append_line(text, kind)
        else:
            self.pages["train"].log.append_line(text, kind)

    def on_finished(self, code, status):
        kind = self.proc_kind
        self.proc = None
        self.proc_kind = None
        tp = self.pages["train"]
        tp.btn_start.setEnabled(True)
        tp.btn_stop.setEnabled(False)
        self.pages["count"].btn.setEnabled(True)
        crashed = status != QProcess.NormalExit
        if kind == "train":
            tp.lbl_out.setText("")
            if code == 0 and not crashed:
                tp.progress.setFormat("完成")
                tp.status.setText(
                    f"训练完成 · 总耗时 {(time.time() - self.t_train_start) / 60:.0f} 分钟")
                tp.log.append_line("训练完成，正在解析报告…", "dice")
                self.load_metrics()
                self.switch_page(2)
            else:
                tp.progress.setFormat("已停止")
                tp.status.setText("训练被停止或异常退出"
                                  "（断点已保留，可重新开始续训）")
        elif kind == "v1":
            self.pages["count"].log.append_line("基线已生成，开始计数…", "info")
            self.launch("count", ROOT / "tools_v2_count.py", self.count_env)
        elif kind == "count":
            self.load_count_report()
        elif kind == "install":
            ok = code == 0 and not crashed
            self.pages["env"].install_done(ok)
            self.pages["train"].lbl_out.setText("")
            self.pages["env"].log.append_line(
                "安装结束，正在重新检测环境…", "info")
            self.check_env()
            if SMOKE_ENV:
                # 等重检返回后做断言
                t = QTimer(self)
                t.timeout.connect(lambda: self._smoke_env_after_install(t))
                t.start(500)

    # ---------------- 报告解析 ----------------
    def load_metrics(self):
        p = self.metrics_path()
        if not p.exists():
            self.pages["metrics"].badge.setText(
                "未找到 unet_metrics_v3.txt（尚未训练或输出在别处）")
            return
        text = p.read_text(encoding="utf-8")
        m = re.search(r"测试集 Dice @0\.5: 均值 ([\d.]+)\s+中位 ([\d.]+)\s+"
                      r"P10 ([\d.]+)", text)
        mb = re.search(r"最佳 epoch: (\d+)（Dice ([\d.]+)）", text)
        if m:
            self.metrics = {
                "mean": float(m.group(1)), "median": float(m.group(2)),
                "p10": float(m.group(3)),
                "best_epoch": int(mb.group(1)) if mb else 0,
                "best_dice": float(mb.group(2)) if mb else 0.0,
            }
            self.pages["metrics"].show_metrics(self.metrics)
        pts = []
        for line in text.splitlines():
            mm = re.match(r"\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
            if mm:
                pts.append((int(mm.group(1)), float(mm.group(2)),
                            float(mm.group(3))))
        tot = max((e for e, _, _ in pts), default=None)
        self.pages["metrics"].curve.set_data(pts, tot)
        self.pages["metrics"].curve.update()

    def load_count_report(self):
        tag = getattr(self, "count_tag", "gui")
        p = self.data_dir / f"{tag}_report.txt"
        tiles = self.pages["count"].tiles
        if not p.exists():
            self.pages["count"].log.append_line(
                f"未找到 {p.name}", "err")
            return
        text = p.read_text(encoding="utf-8")
        mt = re.search(r"train \d+ 图: 均值 ([\d.]+)", text)
        me = re.search(r"test\s+\d+ 图: 均值 ([\d.]+)", text)
        mr = re.search(r"计数相关系数 r = ([\d.]+)", text)
        mb = re.search(r"绝对偏差均值 ([\d.]+)", text)
        if mt:
            tiles["train"].set_value(mt.group(1))
        if me:
            tiles["test"].set_value(me.group(1))
        if mr:
            tiles["r"].set_value(mr.group(1))
        if mb:
            tiles["bias"].set_value(mb.group(1))
        self.pages["count"].log.append_line(
            "计数完成，汇总见上方磁贴；CSV 与可视化已写入数据文件夹。", "dice")

    # ---------------- 自检模式 ----------------
    def run_smoke(self):
        smoke_dir = os.environ.get("SMOKE_DATA_DIR")
        if smoke_dir:
            self.data_dir = Path(smoke_dir)
        else:
            self.data_dir = DEFAULT_DATA
        self.switch_page(0)
        self.validate_data()
        self.refresh_model_lists()
        self.switch_page(1)
        tp = self.pages["train"]
        tp.sp["epochs"].setValue(2)
        tp.sp["batch"].setValue(4)
        tp.preset.setCurrentIndex(tp.preset.count() - 1)
        tp.resume.setChecked(False)
        tp.init_model.setCurrentIndex(0)   # 不自带初始化权重
        self.pages["train"].log.append_line(
            f"[自检] 数据 {self.data_dir} · 2 轮迷你训练", "info")
        self.start_training()

    def smoke_check(self):
        ok = True
        msgs = []
        log_text = self.pages["train"].log.toPlainText()
        if "epoch 2/2" not in log_text:
            ok, msgs = False, msgs + ["日志缺少 epoch 2/2"]
        if len(self.curve_points) < 2:
            ok, msgs = False, msgs + [f"曲线点不足: {len(self.curve_points)}"]
        if not self.metrics or self.metrics.get("mean", 0) <= 0:
            ok, msgs = False, msgs + ["报告解析失败"]
        else:
            msgs.append(f"test Dice 均值 {self.metrics['mean']:.4f}")
        msgs.append(f"曲线 {len(self.curve_points)} 点")
        print("SMOKE_RESULT: " + ("PASS " if ok else "FAIL ")
              + " | ".join(msgs), flush=True)
        if not ok:
            print("----- 训练日志尾部（诊断用） -----", flush=True)
            for line in log_text.splitlines()[-30:]:
                print("  | " + line, flush=True)
            print("----- proc 状态 -----", flush=True)
            print(f"  proc_kind={self.proc_kind} train_total={self.train_total}",
                  flush=True)
        QApplication.exit(0 if ok else 1)

    # ---------------- 环境页自检（验证瘦身安装包）----------------
    def run_smoke_env(self):
        """等启动探测返回 → 校验 torch 缺失被识别且自动跳到环境页 →
        点击真实的「安装 CPU 版」按钮走完整安装流程。"""
        self._env_tries = 0
        t = QTimer(self)
        t.timeout.connect(lambda: self._smoke_env_poll(t))
        t.start(500)

    def _smoke_env_poll(self, timer):
        self._env_tries += 1
        ep = self.pages["env"]
        txt = ep.status_lbl["torch"].text()
        if "检测中" in txt and self._env_tries < 120:
            return                          # 等待探测子进程返回（最多 60s）
        timer.stop()
        auto = self.stack.currentIndex() == 4
        print(f"SMOKE_ENV: torch 状态='{txt}' 自动跳转环境页={auto}",
              flush=True)
        if "未安装" in txt and auto:
            ep.btn_cpu.click()              # 真实一键安装（SMOKE 自动确认）
        elif self.env_torch:
            print("SMOKE_ENV_RESULT: PASS torch 已存在（开发环境）",
                  flush=True)
            QApplication.exit(0)
        else:
            print("SMOKE_ENV_RESULT: FAIL torch 状态不符合预期", flush=True)
            QApplication.exit(1)

    def smoke_env_check(self):
        ok = self.env_torch is not None
        print("SMOKE_ENV_RESULT: " + ("PASS " if ok else "FAIL ")
              + f"安装后 torch={self.env_torch}", flush=True)
        QApplication.exit(0 if ok else 1)

    def _smoke_env_after_install(self, timer):
        if self.env_torch is None:
            self._env_tries += 1
            if self._env_tries < 240:       # 最多等 120s
                return
        timer.stop()
        self.smoke_env_check()

    def closeEvent(self, ev):
        if self.proc:
            self.proc.terminate()
        super().closeEvent(ev)


# ---------------------------------------------------------------- 工具函数
def open_path(p: Path):
    try:
        os.startfile(str(p))     # Windows
    except Exception:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p)))


# ---------------------------------------------------------------- 入口
def main():
    if SMOKE or SMOKE_ENV:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    f = QFont("Microsoft YaHei UI", 9)
    f.setStyleHint(QFont.SansSerif)
    app.setFont(f)
    win = MainWindow()
    win.show()
    if SMOKE:
        # 包装 on_finished：训练结束后做全链路断言。
        # launch() 里 connect(self.on_finished) 会解析到实例属性，
        # 因此先打补丁再启动训练即可生效。
        orig_finished = win.on_finished

        def wrapped(code, status):
            orig_finished(code, status)
            if win.proc_kind is None and win.train_total:
                QTimer.singleShot(800, win.smoke_check)

        win.on_finished = wrapped
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

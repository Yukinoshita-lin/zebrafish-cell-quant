#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui_screenshot.py —— 离屏渲染 GUI 并截图，用于视觉校验（无需显示器交互）。"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

import gui_app


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(gui_app.STYLESHEET)
    win = gui_app.MainWindow()
    win.resize(1180, 800)
    win.show()
    win.switch_page(4)          # 环境页

    def shot():
        # 等探测返回后再截，状态文字才是最终值
        if "检测中" in win.pages["env"].status_lbl["torch"].text():
            QTimer.singleShot(800, shot)
            return
        out = ROOT / "gui_env_page.png"
        win.grab().save(str(out))
        print(f"已保存: {out}")
        # 顺带导出各状态文字，便于核对
        for key, lbl in win.pages["env"].status_lbl.items():
            print(f"  [{key}] {lbl.text()}")
        app.quit()

    QTimer.singleShot(1500, shot)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

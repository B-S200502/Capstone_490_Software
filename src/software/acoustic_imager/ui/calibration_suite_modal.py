"""
Calibration Suite launcher: single "Start Calibration" button that closes the main app,
runs the standalone calibration suite app (with silence timer and log), then restarts the main app.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from ..state import button_state
from ..config import MENU_ACTIVE_BLUE

MODAL_W = 560
MODAL_H = 380
PAD = 20
HEADER_H = 50
START_BTN_H = 48
TEXT_LH = 22


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _launch_standalone_and_exit() -> None:
    """Spawn standalone calibration app (with --restart-main) and exit this process."""
    root = _repo_root()
    standalone_py = root / "utilities" / "calibration" / "calibration_standalone_app.py"
    if not standalone_py.exists():
        return
    gain = (button_state.gain_mode or "HIGH").lower()
    if gain not in ("high", "low"):
        gain = "high"
    cmd = [sys.executable, str(standalone_py), "--restart-main", "--gain", gain]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(root),
            env={**os.environ},
            start_new_session=True,
        )
    except Exception:
        pass
    sys.exit(0)


def draw_calibration_suite_modal(frame: np.ndarray) -> None:
    """Draw launcher: title, description, Start Calibration, Close."""
    if not button_state.calibration_suite_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)
    dim_color = (200, 200, 200)

    ui_cache.apply_modal_dim(frame, 0.15)

    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), border_color, 3, cv2.LINE_AA)

    cv2.putText(frame, "Calibration Suite", (modal_x + PAD, modal_y + 34), font, 0.68, text_color, 1, cv2.LINE_AA)

    # Description
    lines = [
        "Start Calibration will close this app and open",
        "the calibration suite in a separate window.",
        "When calibration is done, this app will restart.",
    ]
    y_text = modal_y + HEADER_H + PAD
    for line in lines:
        cv2.putText(frame, line, (modal_x + PAD, y_text), font, 0.44, dim_color, 1, cv2.LINE_AA)
        y_text += TEXT_LH

    # Start Calibration button
    btn_y = y_text + 20
    row_w = MODAL_W - 2 * PAD
    cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), (120, 160, 255), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("Start Calibration", font, 0.54, 1)
    cv2.putText(frame, "Start Calibration", (modal_x + (MODAL_W - tw) // 2, btn_y + START_BTN_H // 2 + 6), font, 0.54, text_color, 1, cv2.LINE_AA)

    if "cal_suite_start" not in menu_buttons:
        menu_buttons["cal_suite_start"] = Button(0, 0, row_w, START_BTN_H, "")
    menu_buttons["cal_suite_start"].x = modal_x + PAD
    menu_buttons["cal_suite_start"].y = btn_y
    menu_buttons["cal_suite_start"].w = row_w
    menu_buttons["cal_suite_start"].h = START_BTN_H

    # Close button (bottom right)
    close_w, close_h = 72, 28
    close_x = modal_x + MODAL_W - close_w - PAD
    close_y = modal_y + MODAL_H - close_h - PAD
    if "cal_suite_close" not in menu_buttons:
        menu_buttons["cal_suite_close"] = Button(close_x, close_y, close_w, close_h, "Close")
    else:
        menu_buttons["cal_suite_close"].x, menu_buttons["cal_suite_close"].y = close_x, close_y
        menu_buttons["cal_suite_close"].w, menu_buttons["cal_suite_close"].h = close_w, close_h
    menu_buttons["cal_suite_close"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    if "cal_suite_modal_panel" not in menu_buttons:
        menu_buttons["cal_suite_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["cal_suite_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def handle_calibration_suite_modal_click(x: int, y: int) -> bool:
    if not button_state.calibration_suite_modal_open:
        return False

    if "cal_suite_close" in menu_buttons and menu_buttons["cal_suite_close"].contains(x, y):
        button_state.calibration_suite_modal_open = False
        return True

    if "cal_suite_start" in menu_buttons and menu_buttons["cal_suite_start"].contains(x, y):
        _launch_standalone_and_exit()
        return True

    if "cal_suite_modal_panel" in menu_buttons and menu_buttons["cal_suite_modal_panel"].contains(x, y):
        return True

    button_state.calibration_suite_modal_open = False
    return True


def handle_calibration_suite_modal_scroll(delta: int) -> bool:
    if not button_state.calibration_suite_modal_open:
        return False
    return False


def handle_calibration_suite_modal_mouse(event: int, x: int, y: int, fw: int, fh: int) -> bool:
    if not button_state.calibration_suite_modal_open:
        return False
    return False

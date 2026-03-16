"""
Screen calibration modal: instructions and Start button.
After Start, the user collects 3 tap points in the main view (overlay + tap handling in main.py).
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from ..state import button_state
from ..config import MENU_ACTIVE_BLUE

MODAL_W = 560
MODAL_H = 460
PAD = 20
HEADER_H = 50
START_BTN_H = 48
RESET_BTN_H = 36
TEXT_LH = 20


def draw_screen_calibration_modal(frame: np.ndarray) -> None:
    """Draw modal: title, instructions, Start, Close."""
    if not button_state.screen_calibration_modal_open:
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

    cv2.putText(frame, "Screen calibration", (modal_x + PAD, modal_y + 34), font, 0.68, text_color, 1, cv2.LINE_AA)

    # Instructions (like projector/whiteboard calibration)
    lines = [
        "Align the display with where you see the sound.",
        "Keep the imager in a fixed spot.",
        "Place a sound source (e.g. 20 kHz) at 3 different",
        "positions, well apart. For each position, tap on",
        "the screen where you see the sound.",
    ]
    y_text = modal_y + HEADER_H + PAD
    for line in lines:
        cv2.putText(frame, line, (modal_x + PAD, y_text), font, 0.44, dim_color, 1, cv2.LINE_AA)
        y_text += TEXT_LH

    # Start button
    btn_y = y_text + 20
    row_w = MODAL_W - 2 * PAD
    cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (modal_x + PAD, btn_y), (modal_x + PAD + row_w, btn_y + START_BTN_H), (120, 160, 255), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("Start", font, 0.54, 1)
    cv2.putText(frame, "Start", (modal_x + (MODAL_W - tw) // 2, btn_y + START_BTN_H // 2 + 6), font, 0.54, text_color, 1, cv2.LINE_AA)

    if "screen_cal_start" not in menu_buttons:
        menu_buttons["screen_cal_start"] = Button(0, 0, row_w, START_BTN_H, "")
    menu_buttons["screen_cal_start"].x = modal_x + PAD
    menu_buttons["screen_cal_start"].y = btn_y
    menu_buttons["screen_cal_start"].w = row_w
    menu_buttons["screen_cal_start"].h = START_BTN_H

    # Reset settings button (grey outline; clears saved screen calibration)
    reset_y = btn_y + START_BTN_H + 12
    cv2.rectangle(frame, (modal_x + PAD, reset_y), (modal_x + PAD + row_w, reset_y + RESET_BTN_H), (60, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (modal_x + PAD, reset_y), (modal_x + PAD + row_w, reset_y + RESET_BTN_H), (120, 120, 120), 1, cv2.LINE_AA)
    (tw_r, _), _ = cv2.getTextSize("Reset settings", font, 0.46, 1)
    cv2.putText(frame, "Reset settings", (modal_x + (MODAL_W - tw_r) // 2, reset_y + RESET_BTN_H // 2 + 5), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)
    if "screen_cal_reset" not in menu_buttons:
        menu_buttons["screen_cal_reset"] = Button(0, 0, row_w, RESET_BTN_H, "")
    menu_buttons["screen_cal_reset"].x = modal_x + PAD
    menu_buttons["screen_cal_reset"].y = reset_y
    menu_buttons["screen_cal_reset"].w = row_w
    menu_buttons["screen_cal_reset"].h = RESET_BTN_H

    # Close button (bottom right)
    close_w, close_h = 72, 28
    close_x = modal_x + MODAL_W - close_w - PAD
    close_y = modal_y + MODAL_H - close_h - PAD
    if "screen_cal_close" not in menu_buttons:
        menu_buttons["screen_cal_close"] = Button(close_x, close_y, close_w, close_h, "Close")
    else:
        menu_buttons["screen_cal_close"].x, menu_buttons["screen_cal_close"].y = close_x, close_y
        menu_buttons["screen_cal_close"].w, menu_buttons["screen_cal_close"].h = close_w, close_h
    menu_buttons["screen_cal_close"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    if "screen_cal_modal_panel" not in menu_buttons:
        menu_buttons["screen_cal_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["screen_cal_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def handle_screen_calibration_modal_click(x: int, y: int) -> bool:
    if not button_state.screen_calibration_modal_open:
        return False

    if "screen_cal_close" in menu_buttons and menu_buttons["screen_cal_close"].contains(x, y):
        button_state.screen_calibration_modal_open = False
        return True

    if "screen_cal_start" in menu_buttons and menu_buttons["screen_cal_start"].contains(x, y):
        button_state.screen_calibration_modal_open = False
        button_state.screen_calibration_active = True
        button_state.screen_calibration_step = 1
        button_state.screen_calibration_points = []
        button_state.screen_calibration_stable_ready = False
        button_state.screen_calibration_stable_since_t = 0.0
        button_state.screen_calibration_candidate_x = 0.0
        button_state.screen_calibration_candidate_y = 0.0
        button_state.screen_calibration_message = ""
        return True

    if "screen_cal_reset" in menu_buttons and menu_buttons["screen_cal_reset"].contains(x, y):
        button_state.screen_calibration_reset_requested = True
        button_state.screen_calibration_modal_open = False
        return True

    if "screen_cal_modal_panel" in menu_buttons and menu_buttons["screen_cal_modal_panel"].contains(x, y):
        return True

    button_state.screen_calibration_modal_open = False
    return True


def handle_screen_calibration_modal_scroll(delta: int) -> bool:
    if not button_state.screen_calibration_modal_open:
        return False
    return False


def handle_screen_calibration_modal_mouse(event: int, x: int, y: int, fw: int, fh: int) -> bool:
    if not button_state.screen_calibration_modal_open:
        return False
    return False

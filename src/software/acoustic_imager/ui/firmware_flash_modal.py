"""
Flash Firmware modal: version display, progress bar placeholder, success message, close.
"""

from __future__ import annotations

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from ..state import button_state
from ..config import MENU_ACTIVE_BLUE

MODAL_W = 400
MODAL_H = 200
PAD = 24
PROGRESS_BAR_H = 24
PROGRESS_BAR_W = MODAL_W - 2 * PAD


def draw_firmware_flash_modal(frame: np.ndarray) -> None:
    """Draw Flash Firmware modal with version label and progress bar placeholder."""
    if not button_state.firmware_flash_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)
    dim_color = (180, 180, 180)

    ui_cache.apply_modal_dim(frame, 0.5)

    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), border_color, 3, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "Flash Firmware", (modal_x + PAD, modal_y + 36), font, 0.65, text_color, 1, cv2.LINE_AA)

    # Version placeholder
    version_text = f"Version: {button_state.firmware_flash_version}"
    cv2.putText(frame, version_text, (modal_x + PAD, modal_y + 72), font, 0.50, dim_color, 1, cv2.LINE_AA)

    status = button_state.firmware_flash_status
    if status == "success":
        # Success message (green)
        status_text = "Firmware Flash Successful"
        status_color = (0, 255, 100)  # BGR green
        (tw, th), _ = cv2.getTextSize(status_text, font, 0.55, 1)
        text_x = modal_x + (MODAL_W - tw) // 2
        text_y = modal_y + 100 + PROGRESS_BAR_H // 2 + th // 2
        cv2.putText(frame, status_text, (text_x, text_y), font, 0.55, status_color, 1, cv2.LINE_AA)
    elif status == "error":
        # Error message (red)
        status_text = "Firmware Flash Failed"
        status_color = (0, 0, 255)  # BGR red
        (tw, th), _ = cv2.getTextSize(status_text, font, 0.55, 1)
        text_x = modal_x + (MODAL_W - tw) // 2
        text_y = modal_y + 100 + PROGRESS_BAR_H // 2 + th // 2
        cv2.putText(frame, status_text, (text_x, text_y), font, 0.55, status_color, 1, cv2.LINE_AA)
    else:
        # Progress bar placeholder (empty track; fill will be added when flashing is implemented)
        bar_x = modal_x + PAD
        bar_y = modal_y + 100
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + PROGRESS_BAR_W, bar_y + PROGRESS_BAR_H), (50, 50, 50), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + PROGRESS_BAR_W, bar_y + PROGRESS_BAR_H), (80, 80, 80), 1, cv2.LINE_AA)
        # TODO: fill progress based on flash progress (0.0 to 1.0)

    # Close button
    close_w, close_h = 72, 28
    close_x = modal_x + MODAL_W - close_w - PAD
    close_y = modal_y + MODAL_H - close_h - PAD
    if "firmware_flash_close" not in menu_buttons:
        menu_buttons["firmware_flash_close"] = Button(close_x, close_y, close_w, close_h, "Close")
    else:
        menu_buttons["firmware_flash_close"].x, menu_buttons["firmware_flash_close"].y = close_x, close_y
        menu_buttons["firmware_flash_close"].w, menu_buttons["firmware_flash_close"].h = close_w, close_h
    menu_buttons["firmware_flash_close"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    # Panel for click-outside-to-close
    if "firmware_flash_modal_panel" not in menu_buttons:
        menu_buttons["firmware_flash_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["firmware_flash_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def handle_firmware_flash_modal_click(x: int, y: int) -> bool:
    """Handle click. Returns True if handled."""
    if not button_state.firmware_flash_modal_open:
        return False

    if "firmware_flash_close" in menu_buttons and menu_buttons["firmware_flash_close"].contains(x, y):
        button_state.firmware_flash_modal_open = False
        button_state.firmware_flash_status = ""
        return True

    if "firmware_flash_modal_panel" in menu_buttons and menu_buttons["firmware_flash_modal_panel"].contains(x, y):
        return True

    # Click outside: close
    button_state.firmware_flash_modal_open = False
    button_state.firmware_flash_status = ""
    return True

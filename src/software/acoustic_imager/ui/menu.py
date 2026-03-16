"""
MENU dropdown and recording timestamp HUD.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons
from ..state import button_state, HUD
from .video_recorder import VideoRecorder
from ..config import HUD_MENU_OPACITY, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT


def _blue_gradient_overlay(h: int, w: int, top_bgr: Tuple[int, int, int], bot_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top_bgr[c], bot_bgr[c], h, dtype=np.uint8).reshape(-1, 1)
    return out


# Keys for menu dropdown (positions are shifted by offset_x when drawing)
_MENU_DROPDOWN_KEYS = ("wifi", "main_menu_settings", "fps30", "fpsmax", "gain", "source", "spectrum_analyzer")


def draw_menu(frame: np.ndarray, offset_x: float = 0.0, offset_y: float = 0.0) -> None:
    """Draw menu button and dropdown. offset_x = right, offset_y = down (retract with bottom HUD on swipe down)."""
    if "menu" not in menu_buttons:
        return

    menu_buttons["menu"].is_active = button_state.menu_open

    menu_btn = menu_buttons["menu"]
    actual_frame_height = frame.shape[0]
    menu_btn.y = actual_frame_height - menu_btn.h - 5

    x, y, w, h = menu_btn.x, menu_btn.y, menu_btn.w, menu_btn.h
    x_off = x  # menu button stays fixed; horizontal offset does not move the button
    y_off = y + int(offset_y)

    roi = frame[y_off:y_off+h, x_off:x_off+w]
    if menu_btn.is_active:
        overlay = _blue_gradient_overlay(h, w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
    else:
        overlay = np.empty_like(roi)
        overlay[:] = (0, 0, 0)  # same as bottom HUD pills so opacity matches
    alpha = HUD_MENU_OPACITY
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    border_color = (255, 255, 255)  # white border always
    cv2.rectangle(frame, (x_off, y_off), (x_off + w, y_off + h), border_color, 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thick = 1
    tw, th = cv2.getTextSize(menu_btn.text, font, scale, thick)[0]
    tx = x_off + (w - tw) // 2
    ty = y_off + (h + th) // 2
    # cv2.putText(frame, menu_btn.text, (tx, ty), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, menu_btn.text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    if not button_state.menu_open:
        return

    # SHOT and Gallery live in bottom HUD only; menu dropdown has 5 rows (wifi|settings, fps, gain, source, spectrum_analyzer)
    item_h, gap, menu_w = 40, 8, menu_btn.w
    dropdown_h = 5 * (item_h + gap) + gap
    dropdown_y = menu_btn.y - dropdown_h - gap
    oy = int(offset_y)

    menu_buttons["main_menu_settings"].is_active = HUD.settings_modal_open
    menu_buttons["wifi"].is_active = HUD.wifi_modal_open
    menu_buttons["fps30"].is_active = (button_state.fps_mode == "30")
    # menu_buttons["fps60"].is_active = (button_state.fps_mode == "60")  # 60FPS button commented out
    menu_buttons["fpsmax"].is_active = (button_state.fps_mode == "MAX")

    menu_buttons["gain"].is_active = True
    menu_buttons["source"].is_active = True
    menu_buttons["source"].text = f"SRC: {button_state.source_mode}"
    menu_buttons["spectrum_analyzer"].is_active = True
    menu_buttons["spectrum_analyzer"].text = f"SPECTRUM: {button_state.spectrum_analyzer_mode}"
    menu_buttons["gallery"].is_active = button_state.gallery_open

    white_border = (255, 255, 255)
    ox = 0  # dropdown stays fixed horizontally; no horizontal offset applied
    for k in _MENU_DROPDOWN_KEYS:
        if k in menu_buttons:
            menu_buttons[k].y += oy
    hud_bg = (0, 0, 0)  # same as menu button and bottom HUD pills for matching opacity
    for k in _MENU_DROPDOWN_KEYS:
        if k in menu_buttons:
            icon_type = "wifi" if k == "wifi" else ("settings" if k == "main_menu_settings" else None)
            menu_buttons[k].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE, active_border_color=white_border, fill_alpha=HUD_MENU_OPACITY, inactive_bg=hud_bg, icon_type=icon_type)
    for k in _MENU_DROPDOWN_KEYS:
        if k in menu_buttons:
            menu_buttons[k].y -= oy


def get_recording_timestamp_rect() -> Optional[Tuple[int, int, int, int]]:
    """Recording state (flash + time) is now drawn inside the REC pill in bottom_hud; no separate bar."""
    return None


def draw_recording_timestamp(frame: np.ndarray, video_recorder: Optional[VideoRecorder]) -> None:
    if video_recorder is None or not video_recorder.is_recording:
        return

    rect = get_recording_timestamp_rect()
    if rect is None:
        return

    menu_x, timestamp_y, menu_w, timestamp_h = rect

    elapsed = video_recorder.get_elapsed_time()
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    if timestamp_y < 0 or timestamp_y + timestamp_h > frame.shape[0]:
        return

    roi = frame[timestamp_y:timestamp_y+timestamp_h, menu_x:menu_x+menu_w]
    if roi.shape[1] != menu_w or roi.shape[0] != timestamp_h:
        return

    paused = video_recorder.is_paused
    key = (menu_w, timestamp_h, paused)

    hud = ui_cache._REC_HUD_CACHE.get(key)
    if hud is None:
        hud = np.zeros((timestamp_h, menu_w, 3), dtype=np.uint8)
        bg_color = (30, 30, 30) if not paused else (40, 40, 60)
        hud[:] = bg_color

        band_h = 10
        band = hud.copy()
        band[:band_h] = np.clip(band[:band_h].astype(np.int16) + 15, 0, 255).astype(np.uint8)
        alpha = HUD_MENU_OPACITY
        cv2.addWeighted(band, alpha, hud, 1- alpha, 0.0, hud)

        border_color = (200, 200, 200) if not paused else (100, 100, 200)
        cv2.rectangle(hud, (0, 0), (menu_w - 1, timestamp_h - 1), border_color, 1, cv2.LINE_8)

        ui_cache._REC_HUD_CACHE[key] = hud

    overlay = np.empty_like(roi)
    bg_color = (100, 0, 0) if not paused else (0, 100, 150)
    overlay[:] = bg_color
    alpha = HUD_MENU_OPACITY
    cv2.addWeighted(overlay, alpha, roi, 1-alpha, 0.0, dst=roi)

    border_color = (255, 50, 50) if not paused else (0, 165, 255)
    cv2.rectangle(frame, (menu_x, timestamp_y), (menu_x + menu_w, timestamp_y + timestamp_h),
              border_color, 2, cv2.LINE_8)

    indicator_x = menu_x + 10
    indicator_y = timestamp_y + timestamp_h // 2

    if paused:
        bar_w = 3
        bar_h = 10
        bar_gap = 4
        cv2.rectangle(frame,
                    (indicator_x, indicator_y - bar_h//2),
                    (indicator_x + bar_w, indicator_y + bar_h//2),
                    (0, 200, 255), -1, cv2.LINE_8)
        cv2.rectangle(frame,
                    (indicator_x + bar_w + bar_gap, indicator_y - bar_h//2),
                    (indicator_x + 2*bar_w + bar_gap, indicator_y + bar_h//2),
                    (0, 200, 255), -1, cv2.LINE_8)
    else:
        pulse = int((time.time() * 2) % 2)
        if pulse:
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (0, 0, 255), -1, cv2.LINE_8)
            cv2.circle(frame, (indicator_x + 5, indicator_y), 6, (255, 255, 255), 1, cv2.LINE_8)

    timestamp_text = f"{minutes:02d}:{seconds:02d}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thick = 1

    if timestamp_text in ui_cache._REC_TEXT_SIZE_CACHE:
        text_w, text_h = ui_cache._REC_TEXT_SIZE_CACHE[timestamp_text]
    else:
        (text_w, text_h), _ = cv2.getTextSize(timestamp_text, font, font_scale, font_thick)
        ui_cache._REC_TEXT_SIZE_CACHE[timestamp_text] = (text_w, text_h)

    text_color = (255, 255, 255) if not paused else (200, 230, 255)

    text_x = menu_x + (menu_w - text_w) // 2 + 10
    text_y = timestamp_y + (timestamp_h + text_h) // 2

    cv2.putText(frame, timestamp_text, (text_x, text_y),
            font, font_scale, text_color, font_thick, cv2.LINE_8)

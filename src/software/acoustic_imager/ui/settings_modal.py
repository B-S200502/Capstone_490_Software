"""
System Settings modal: Display (Camera, Light/Dark, Crosshairs, Heatmap Color),
Advanced (Debug), Email Settings. Touch-scrollable like gallery.
"""

from __future__ import annotations

import time
import cv2
import numpy as np

from .button import menu_buttons, Button
from . import ui_cache
from .. import state
from ..state import HUD, button_state
from ..config import MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT

MODAL_W = 580
MODAL_H = 520
HEADER_H = 64   # more room for title; line moved lower
CONTENT_PAD = 20
ROW_H = 42
ITEM_GAP = 14
SECTION_GAP = 24
TOGGLE_W, TOGGLE_H = 52, 28
# Hit area: only the switch plus ~0.5" to the left (debug-friendly, avoid full-row taps)
TOGGLE_HIT_EXTRA_LEFT = 48
# Extra hit padding for Email Settings button (easier touch)
EMAIL_BTN_HIT_PAD_X = 8
EMAIL_BTN_HIT_PAD_Y = 8
SCROLLBAR_W = 14   # slightly wider for touch
SCROLL_BTN_H = 36

COLORMAPS = ["MAGMA", "JET", "TURBO", "INFERNO"]

# Inertia (like gallery)
SETTINGS_FLING_GAIN = 2.0
SETTINGS_EMA_ALPHA = 0.6
SETTINGS_FRICTION = 1.2
SETTINGS_STOP_VELOCITY = 15.0


def _draw_toggle(frame: np.ndarray, x: int, y: int, w: int, h: int, on: bool) -> None:
    """Draw a toggle switch (rounded rect track + circle knob)."""
    track_h = max(14, h - 8)
    track_y = y + (h - track_h) // 2
    track_w = w
    color = MENU_ACTIVE_BLUE if on else (80, 80, 80)
    cv2.rectangle(frame, (x, track_y), (x + track_w, track_y + track_h), color, -1, cv2.LINE_AA)
    r = min(4, track_h // 2)
    cv2.rectangle(frame, (x + r, track_y), (x + track_w - r, track_y + track_h), color, -1)
    knob_r = (track_h - 4) // 2
    knob_y = track_y + track_h // 2
    knob_x = x + track_w - knob_r - 4 if on else x + knob_r + 4
    cv2.circle(frame, (knob_x, knob_y), knob_r, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (knob_x, knob_y), knob_r, (160, 160, 160), 1, cv2.LINE_AA)


CONTENT_BOTTOM_PAD = 50  # extra space below last button so it scrolls fully into view
# Flash Firmware button (red, below Email Settings)
FLASH_FIRMWARE_BTN_H = 44


def _compute_content_height() -> int:
    """Total height of scrollable content (must match actual drawn layout)."""
    CAL_SUITE_BTN_H = 44
    # Display: label + gap + 3 toggles + gap + heatmap label + gap + color row + section_gap
    display_h = 20 + ITEM_GAP + 3 * (ROW_H + ITEM_GAP) + 6 + ROW_H + ITEM_GAP + (ROW_H - 4) + SECTION_GAP  # 6 = reduced gap before heatmap
    # Divider + Advanced: section_gap + label + gap + 6 toggles + Calibration Suite button + section_gap
    advanced_h = SECTION_GAP + 20 + ITEM_GAP + 6 * (ROW_H + ITEM_GAP) + (CAL_SUITE_BTN_H + ITEM_GAP) + SECTION_GAP
    # Divider + Share: section_gap + label + gap + email button + gap + flash label + gap + flash button + bottom padding
    share_h = SECTION_GAP + 20 + ITEM_GAP + 44 + ITEM_GAP + 20 + ITEM_GAP + FLASH_FIRMWARE_BTN_H + CONTENT_BOTTOM_PAD
    return display_h + advanced_h + share_h


# Content cache: avoid redrawing scrollable content every frame when state unchanged
_settings_content_cache: np.ndarray | None = None
_settings_content_cache_key: tuple | None = None
_settings_content_buffer: np.ndarray | None = None


def _update_settings_button_positions(
    content_top: int,
    scroll_offset: int,
    row_x: int,
    row_w: int,
) -> None:
    """Update hit rects for all settings content buttons (same layout as draw, no redraw)."""
    hit_x = row_x + row_w - TOGGLE_W - TOGGLE_HIT_EXTRA_LEFT
    color_btn_w = (row_w - 3 * ITEM_GAP) // 4
    email_h = 44
    # Content y positions (must match _build_settings_content; Calibration Suite after Record)
    cal_suite_btn_h = 44
    y_cam, y_cross, y_color, y_debug, y_radar_dbg, y_radar, y_mapstyle, y_possvc, y_record = 34, 90, 208, 314, 370, 426, 482, 538, 594
    y_calibration_suite = y_record + ROW_H + ITEM_GAP  # after Record Compass History
    y_email = y_calibration_suite + cal_suite_btn_h + ITEM_GAP + SECTION_GAP + SECTION_GAP + 20 + ITEM_GAP  # after divider + "Setup Email" label
    y_flash = y_email + email_h + ITEM_GAP + 20 + ITEM_GAP  # after "Update to latest firmware" label
    if "settings_cam" in menu_buttons:
        b = menu_buttons["settings_cam"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_cam - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_crosshairs" in menu_buttons:
        b = menu_buttons["settings_crosshairs"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_cross - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    for i in range(len(COLORMAPS)):
        k = f"settings_colormap_{i}"
        if k in menu_buttons:
            cx = i * (color_btn_w + ITEM_GAP)
            menu_buttons[k].x = row_x + cx
            menu_buttons[k].y = content_top + y_color - scroll_offset
            menu_buttons[k].w, menu_buttons[k].h = color_btn_w, ROW_H - 4
    if "settings_debug" in menu_buttons:
        b = menu_buttons["settings_debug"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_debug - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_show_radar_debug" in menu_buttons:
        b = menu_buttons["settings_show_radar_debug"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_radar_dbg - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_radar_ui" in menu_buttons:
        b = menu_buttons["settings_radar_ui"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_radar - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_position_services" in menu_buttons:
        b = menu_buttons["settings_position_services"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_possvc - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_map_style" in menu_buttons:
        b = menu_buttons["settings_map_style"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_mapstyle - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_record_compass_history" in menu_buttons:
        b = menu_buttons["settings_record_compass_history"]
        b.x, b.y, b.w, b.h = hit_x, content_top + y_record - scroll_offset, TOGGLE_W + TOGGLE_HIT_EXTRA_LEFT, ROW_H
    if "settings_calibration_suite" in menu_buttons:
        b = menu_buttons["settings_calibration_suite"]
        b.x, b.y = row_x, content_top + y_calibration_suite - scroll_offset
        b.w, b.h = row_w, cal_suite_btn_h
    if "settings_email" in menu_buttons:
        b = menu_buttons["settings_email"]
        b.x = row_x - EMAIL_BTN_HIT_PAD_X
        b.y = (content_top + y_email - scroll_offset) - EMAIL_BTN_HIT_PAD_Y
        b.w = row_w + 2 * EMAIL_BTN_HIT_PAD_X
        b.h = email_h + 2 * EMAIL_BTN_HIT_PAD_Y
    if "settings_flash_firmware" in menu_buttons:
        b = menu_buttons["settings_flash_firmware"]
        b.x, b.y = row_x, content_top + y_flash - scroll_offset
        b.w, b.h = row_w, FLASH_FIRMWARE_BTN_H


def _build_settings_content(
    content_canvas: np.ndarray,
    row_w: int,
    font: int,
    text_color: tuple,
    section_color: tuple,
) -> None:
    """Draw full scrollable content onto content_canvas (for cache fill)."""
    content_canvas[:] = (40, 40, 40)
    y = 0
    pad = CONTENT_PAD

    def _toggle_row(
        canvas: np.ndarray, label: str, on: bool, key: str, y_pos: int, disabled: bool = False
    ) -> int:
        color = (120, 120, 120) if disabled else text_color
        cv2.putText(canvas, label, (0, y_pos + ROW_H // 2 + 6), font, 0.50, color, 1, cv2.LINE_AA)
        toggle_x = row_w - TOGGLE_W
        toggle_y = y_pos + (ROW_H - TOGGLE_H) // 2
        _draw_toggle(canvas, toggle_x, toggle_y, TOGGLE_W, TOGGLE_H, on and not disabled)
        if key not in menu_buttons:
            menu_buttons[key] = Button(0, 0, row_w, ROW_H, "")
        return y_pos + ROW_H + ITEM_GAP

    cv2.putText(content_canvas, "Display", (0, y + 25), font, 0.56, section_color, 1, cv2.LINE_AA)
    y += 20 + ITEM_GAP
    y = _toggle_row(content_canvas, "Camera", button_state.camera_enabled, "settings_cam", y)
    y = _toggle_row(content_canvas, "Crosshairs", button_state.crosshairs_enabled, "settings_crosshairs", y)
    y += 6
    cv2.putText(content_canvas, "Heatmap Theme", (0, y + ROW_H // 2 + 2), font, 0.50, text_color, 1, cv2.LINE_AA)
    y += ROW_H + ITEM_GAP
    color_btn_w = (row_w - 3 * ITEM_GAP) // 4
    for i, cm in enumerate(COLORMAPS):
        cx = i * (color_btn_w + ITEM_GAP)
        is_selected = button_state.colormap_mode == cm
        if is_selected:
            cv2.rectangle(content_canvas, (cx, y), (cx + color_btn_w, y + ROW_H - 4), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
            cv2.rectangle(content_canvas, (cx, y), (cx + color_btn_w, y + ROW_H - 4), MENU_ACTIVE_BLUE_LIGHT, 1, cv2.LINE_AA)
        else:
            cv2.rectangle(content_canvas, (cx, y), (cx + color_btn_w, y + ROW_H - 4), (50, 50, 50), -1, cv2.LINE_AA)
            cv2.rectangle(content_canvas, (cx, y), (cx + color_btn_w, y + ROW_H - 4), (80, 80, 80), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(cm, font, 0.42, 1)
        text_x = cx + (color_btn_w - tw) // 2
        text_y = y + (ROW_H - 4) // 2 + th // 2 + 2
        cv2.putText(content_canvas, cm, (text_x, text_y), font, 0.42, text_color, 1, cv2.LINE_AA)
        k = f"settings_colormap_{i}"
        if k not in menu_buttons:
            menu_buttons[k] = Button(0, 0, color_btn_w, ROW_H - 4, "")
    y += ROW_H - 4 + SECTION_GAP
    cv2.line(content_canvas, (0, y), (row_w, y), section_color, 1, cv2.LINE_AA)
    y += SECTION_GAP
    cv2.putText(content_canvas, "Advanced", (0, y + 16), font, 0.56, section_color, 1, cv2.LINE_AA)
    y += 20 + ITEM_GAP
    y = _toggle_row(content_canvas, "SPI debug", button_state.debug_enabled, "settings_debug", y)
    y = _toggle_row(
        content_canvas,
        "Show Radar Debug",
        button_state.show_radar_debug,
        "settings_show_radar_debug",
        y,
    )
    y = _toggle_row(
        content_canvas,
        "Radar UI",
        button_state.radar_ui_enabled,
        "settings_radar_ui",
        y,
        disabled=not state.MAGNETOMETER_AVAILABLE,
    )
    y = _toggle_row(
        content_canvas,
        "Map Tiles (Dark/Light)",
        button_state.map_tile_style == "dark",
        "settings_map_style",
        y,
    )
    y = _toggle_row(content_canvas, "Position Services", button_state.position_services_enabled, "settings_position_services", y)
    y = _toggle_row(
        content_canvas,
        "Record Compass History",
        button_state.record_compass_history,
        "settings_record_compass_history",
        y,
    )
    # Calibration Suite button (under Advanced)
    cal_suite_h = 44
    cv2.rectangle(content_canvas, (0, y), (row_w, y + cal_suite_h), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(content_canvas, (0, y), (row_w, y + cal_suite_h), MENU_ACTIVE_BLUE_LIGHT, 1, cv2.LINE_AA)
    (tw_cs, _), _ = cv2.getTextSize("Calibration Suite", font, 0.52, 1)
    cv2.putText(content_canvas, "Calibration Suite", ((row_w - tw_cs) // 2, y + cal_suite_h // 2 + 6), font, 0.52, text_color, 1, cv2.LINE_AA)
    if "settings_calibration_suite" not in menu_buttons:
        menu_buttons["settings_calibration_suite"] = Button(0, 0, row_w, cal_suite_h, "")
    y += cal_suite_h + ITEM_GAP
    y += SECTION_GAP
    cv2.line(content_canvas, (0, y), (row_w, y), section_color, 1, cv2.LINE_AA)
    y += SECTION_GAP
    cv2.putText(content_canvas, "Setup Email & Send Test", (0, y + 16), font, 0.56, section_color, 1, cv2.LINE_AA)
    y += 20 + ITEM_GAP
    email_h = 44
    cv2.rectangle(content_canvas, (0, y), (row_w, y + email_h), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(content_canvas, (0, y), (row_w, y + email_h), MENU_ACTIVE_BLUE_LIGHT, 1, cv2.LINE_AA)
    cv2.putText(content_canvas, "Email Settings", ((row_w - 110) // 2, y + email_h // 2 + 6), font, 0.54, text_color, 1, cv2.LINE_AA)
    if "settings_email" not in menu_buttons:
        menu_buttons["settings_email"] = Button(0, 0, row_w, email_h, "")
    y += email_h + ITEM_GAP

    # Flash Firmware: grey label then red button
    cv2.putText(content_canvas, "Update to latest firmware", (0, y + 14), font, 0.50, section_color, 1, cv2.LINE_AA)
    y += 20 + ITEM_GAP

    # Flash Firmware button (red, below Email Settings)
    FLASH_BTN_RED = (0, 0, 255)  # BGR red
    FLASH_BTN_RED_LIGHT = (100, 100, 255)
    cv2.rectangle(content_canvas, (0, y), (row_w, y + FLASH_FIRMWARE_BTN_H), FLASH_BTN_RED, -1, cv2.LINE_AA)
    cv2.rectangle(content_canvas, (0, y), (row_w, y + FLASH_FIRMWARE_BTN_H), FLASH_BTN_RED_LIGHT, 1, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize("Flash Firmware", font, 0.54, 1)
    text_x = (row_w - tw) // 2
    text_y = y + FLASH_FIRMWARE_BTN_H // 2 + th // 2 + 2
    cv2.putText(content_canvas, "Flash Firmware", (text_x, text_y), font, 0.54, text_color, 1, cv2.LINE_AA)
    if "settings_flash_firmware" not in menu_buttons:
        menu_buttons["settings_flash_firmware"] = Button(0, 0, row_w, FLASH_FIRMWARE_BTN_H, "")


def draw_settings_modal(frame: np.ndarray) -> None:
    """Draw System Settings modal with scrollable content and improved spacing."""
    if not HUD.settings_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)
    section_color = (180, 180, 180)

    # Dark overlay (cached black buffer, no frame copy)
    ui_cache.apply_modal_dim(frame, 0.5)

    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + MODAL_W, modal_y + MODAL_H), border_color, 3, cv2.LINE_AA)

    # Fixed header: title + close (more room for title, line lower)
    pad = CONTENT_PAD
    row_x = modal_x + pad
    row_w = MODAL_W - 2 * pad - SCROLLBAR_W

    title_y = modal_y + 32
    cv2.putText(frame, "System Settings", (row_x, title_y), font, 0.72, text_color, 1, cv2.LINE_AA)

    line_y = modal_y + 54
    cv2.line(frame, (row_x, line_y), (modal_x + MODAL_W - pad - SCROLLBAR_W, line_y), (255, 255, 255), 3, cv2.LINE_AA)

    close_w, close_h = 72, 28
    close_x = modal_x + MODAL_W - close_w - pad - SCROLLBAR_W
    close_y = modal_y + 14
    if "settings_close" not in menu_buttons:
        menu_buttons["settings_close"] = Button(close_x, close_y, close_w, close_h, "Close")
    else:
        menu_buttons["settings_close"].x, menu_buttons["settings_close"].y = close_x, close_y
        menu_buttons["settings_close"].w, menu_buttons["settings_close"].h = close_w, close_h
    menu_buttons["settings_close"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    # Scrollable content area
    content_top = modal_y + HEADER_H
    content_h = MODAL_H - HEADER_H
    content_bottom = content_top + content_h

    total_content_h = _compute_content_height()
    max_scroll = max(0, total_content_h - content_h)

    # Inertia (touch fling)
    if HUD.settings_modal_inertia_active and not HUD.settings_modal_content_dragging and not HUD.settings_modal_scroll_dragging:
        now = time.perf_counter()
        dt = max(1e-6, now - HUD.settings_modal_last_drag_t)
        HUD.settings_modal_last_drag_t = now
        new_scroll = float(HUD.settings_modal_scroll_offset) + HUD.settings_modal_scroll_velocity * dt
        decay = max(0.0, 1.0 - SETTINGS_FRICTION * dt)
        HUD.settings_modal_scroll_velocity *= decay
        max_scroll_f = float(max_scroll)
        if new_scroll < 0.0:
            new_scroll = 0.0
            HUD.settings_modal_scroll_velocity = 0.0
            HUD.settings_modal_inertia_active = False
        elif new_scroll > max_scroll_f:
            new_scroll = max_scroll_f
            HUD.settings_modal_scroll_velocity = 0.0
            HUD.settings_modal_inertia_active = False
        if abs(HUD.settings_modal_scroll_velocity) < SETTINGS_STOP_VELOCITY:
            HUD.settings_modal_scroll_velocity = 0.0
            HUD.settings_modal_inertia_active = False
        HUD.settings_modal_scroll_offset = int(new_scroll)

    scroll_offset = HUD.settings_modal_scroll_offset
    scroll_offset = min(scroll_offset, max_scroll)
    HUD.settings_modal_scroll_offset = scroll_offset

    # Content cache: redraw only when toggles/colormap change
    content_w = MODAL_W - SCROLLBAR_W - 2 * pad
    state_key = (
        button_state.camera_enabled,
        button_state.crosshairs_enabled,
        button_state.colormap_mode,
        button_state.debug_enabled,
        button_state.radar_ui_enabled,
        button_state.map_tile_style,
        button_state.position_services_enabled,
        button_state.record_compass_history,
        button_state.show_radar_debug,
    )
    global _settings_content_cache, _settings_content_cache_key, _settings_content_buffer
    cache_hit = _settings_content_cache_key == state_key and _settings_content_cache is not None
    if _settings_content_cache_key != state_key or _settings_content_cache is None:
        if _settings_content_buffer is None or _settings_content_buffer.shape[0] != total_content_h or _settings_content_buffer.shape[1] != content_w:
            _settings_content_buffer = np.zeros((total_content_h, content_w, 3), dtype=np.uint8)
        _build_settings_content(_settings_content_buffer, row_w, font, text_color, section_color)
        _settings_content_cache = _settings_content_buffer.copy()
        _settings_content_cache_key = state_key

    # Blit visible portion of cached content
    src_y1 = scroll_offset
    src_y2 = min(scroll_offset + content_h, total_content_h)
    dst_y1 = content_top
    dst_y2 = content_top + (src_y2 - src_y1)
    visible = _settings_content_cache[src_y1:src_y2, :]
    if visible.size > 0:
        frame[dst_y1:dst_y2, row_x:row_x + visible.shape[1]] = visible

    _update_settings_button_positions(content_top, scroll_offset, row_x, row_w)

    # Scrollbar
    sb_x = modal_x + MODAL_W - SCROLLBAR_W - 4
    sb_y = content_top
    sb_h = content_h
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + SCROLLBAR_W, sb_y + sb_h), (60, 60, 60), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (sb_x, sb_y), (sb_x + SCROLLBAR_W, sb_y + sb_h), (90, 90, 90), 1, cv2.LINE_AA)

    if max_scroll > 0:
        thumb_ratio = content_h / total_content_h
        thumb_h = max(40, int(sb_h * thumb_ratio))
        thumb_range = sb_h - thumb_h
        thumb_y = sb_y + int(thumb_range * scroll_offset / max_scroll) if max_scroll > 0 else sb_y
        cv2.rectangle(frame, (sb_x + 2, thumb_y), (sb_x + SCROLLBAR_W - 2, thumb_y + thumb_h), (120, 120, 120), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (sb_x + 2, thumb_y), (sb_x + SCROLLBAR_W - 2, thumb_y + thumb_h), (160, 160, 160), 1, cv2.LINE_AA)

        if "settings_scroll_up" not in menu_buttons:
            menu_buttons["settings_scroll_up"] = Button(sb_x, sb_y, SCROLLBAR_W, SCROLL_BTN_H, "")
        else:
            menu_buttons["settings_scroll_up"].x, menu_buttons["settings_scroll_up"].y = sb_x, sb_y
            menu_buttons["settings_scroll_up"].w, menu_buttons["settings_scroll_up"].h = SCROLLBAR_W, SCROLL_BTN_H
        cv2.putText(frame, "^", (sb_x + 2, sb_y + SCROLL_BTN_H - 8), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        if "settings_scroll_down" not in menu_buttons:
            menu_buttons["settings_scroll_down"] = Button(sb_x, sb_y + sb_h - SCROLL_BTN_H, SCROLLBAR_W, SCROLL_BTN_H, "")
        else:
            menu_buttons["settings_scroll_down"].x = sb_x
            menu_buttons["settings_scroll_down"].y = sb_y + sb_h - SCROLL_BTN_H
            menu_buttons["settings_scroll_down"].w, menu_buttons["settings_scroll_down"].h = SCROLLBAR_W, SCROLL_BTN_H
        cv2.putText(frame, "v", (sb_x + 4, sb_y + sb_h - 10), font, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

        if "settings_scrollbar" not in menu_buttons:
            menu_buttons["settings_scrollbar"] = Button(sb_x, sb_y, SCROLLBAR_W, sb_h, "")
        else:
            menu_buttons["settings_scrollbar"].x, menu_buttons["settings_scrollbar"].y = sb_x, sb_y
            menu_buttons["settings_scrollbar"].w, menu_buttons["settings_scrollbar"].h = SCROLLBAR_W, sb_h
    else:
        for k in ("settings_scroll_up", "settings_scroll_down", "settings_scrollbar"):
            if k in menu_buttons:
                menu_buttons[k].w = 0
                menu_buttons[k].h = 0

    # Panel for click-outside-to-close (full modal area)
    if "settings_modal_panel" not in menu_buttons:
        menu_buttons["settings_modal_panel"] = Button(modal_x, modal_y, MODAL_W, MODAL_H, "")
    else:
        b = menu_buttons["settings_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, MODAL_W, MODAL_H


def _content_area_bounds(fw: int, fh: int) -> tuple:
    """Return (x, y, w, h) of scrollable content area in screen coords."""
    modal_x = (fw - MODAL_W) // 2
    modal_y = (fh - MODAL_H) // 2
    row_x = modal_x + CONTENT_PAD
    content_top = modal_y + HEADER_H
    row_w = MODAL_W - 2 * CONTENT_PAD - SCROLLBAR_W
    content_h = MODAL_H - HEADER_H
    return (row_x, content_top, row_w, content_h)


def _is_in_content_area(x: int, y: int, fw: int, fh: int) -> bool:
    """True if (x,y) is in the scrollable content area (not scrollbar, not header)."""
    cx, cy, cw, ch = _content_area_bounds(fw, fh)
    return cx <= x < cx + cw and cy <= y < cy + ch


def handle_settings_modal_mouse(event: int, x: int, y: int, fw: int, fh: int) -> bool:
    """Handle mouse events for touch/drag scroll. Returns True if consumed."""
    if not HUD.settings_modal_open:
        return False

    now = time.perf_counter()
    content_top = (fh - MODAL_H) // 2 + HEADER_H
    content_h = MODAL_H - HEADER_H
    total_content_h = _compute_content_height()
    max_scroll = max(0, total_content_h - content_h)

    if event == cv2.EVENT_LBUTTONDOWN:
        # Content area touch-drag start (only if not on a button)
        if _is_in_content_area(x, y, fw, fh):
            # Check we didn't hit a button - those are handled by click
            if not _hit_any_settings_button(x, y):
                HUD.settings_modal_content_dragging = True
                HUD.settings_modal_content_drag_start_y = y
                HUD.settings_modal_content_drag_start_scroll = HUD.settings_modal_scroll_offset
                HUD.settings_modal_content_drag_moved = False
                HUD.settings_modal_scroll_velocity = 0.0
                HUD.settings_modal_last_drag_t = now
                HUD.settings_modal_last_drag_y = y
                HUD.settings_modal_inertia_active = False
                return True
        return False

    if event == cv2.EVENT_MOUSEMOVE:
        if HUD.settings_modal_content_dragging:
            dy = y - HUD.settings_modal_content_drag_start_y
            if abs(dy) > ui_cache.DRAG_PX:
                HUD.settings_modal_content_drag_moved = True
            if HUD.settings_modal_content_drag_moved:
                new_scroll = HUD.settings_modal_content_drag_start_scroll - dy
                HUD.settings_modal_scroll_offset = max(0, min(int(new_scroll), max_scroll))
                dt = max(1e-6, now - HUD.settings_modal_last_drag_t)
                inst_v = (HUD.settings_modal_last_drag_y - y) / dt * SETTINGS_FLING_GAIN
                HUD.settings_modal_scroll_velocity = (
                    (1.0 - SETTINGS_EMA_ALPHA) * HUD.settings_modal_scroll_velocity
                    + SETTINGS_EMA_ALPHA * inst_v
                )
                HUD.settings_modal_last_drag_t = now
                HUD.settings_modal_last_drag_y = y
            return True
        if HUD.settings_modal_scroll_dragging:
            dy = y - HUD.settings_modal_drag_start_y
            new_scroll = HUD.settings_modal_drag_start_scroll + dy
            HUD.settings_modal_scroll_offset = max(0, min(max_scroll, int(new_scroll)))
            return True
        return False

    if event == cv2.EVENT_LBUTTONUP:
        if HUD.settings_modal_content_dragging:
            HUD.settings_modal_content_dragging = False
            if HUD.settings_modal_content_drag_moved and abs(HUD.settings_modal_scroll_velocity) > 50.0:
                HUD.settings_modal_inertia_active = True
                HUD.settings_modal_last_drag_t = now
            else:
                HUD.settings_modal_inertia_active = False
                HUD.settings_modal_scroll_velocity = 0.0
            return True
        if HUD.settings_modal_scroll_dragging:
            HUD.settings_modal_scroll_dragging = False
            return True
        return False

    return False


def _hit_any_settings_button(x: int, y: int) -> bool:
    """True if (x,y) hits any interactive button in the modal."""
    for k in (
        "settings_close", "settings_scroll_up", "settings_scroll_down", "settings_scrollbar",
        "settings_flash_firmware", "settings_email", "settings_calibration_suite", "settings_cam", "settings_theme", "settings_crosshairs", "settings_debug",
        "settings_radar_ui", "settings_map_style", "settings_position_services", "settings_record_compass_history",
        "settings_show_radar_debug",
    ):
        if k in menu_buttons:
            b = menu_buttons[k]
            if b.w > 0 and b.h > 0 and b.contains(x, y):
                return True
    for i in range(len(COLORMAPS)):
        k = f"settings_colormap_{i}"
        if k in menu_buttons and menu_buttons[k].contains(x, y):
            return True
    return False


def handle_settings_modal_click(x: int, y: int) -> bool:
    """Handle click. Returns True if handled."""
    if not HUD.settings_modal_open:
        return False

    # Don't handle if this was a content drag start (handled by mouse)
    if HUD.settings_modal_content_dragging:
        return True

    # Close
    if "settings_close" in menu_buttons and menu_buttons["settings_close"].contains(x, y):
        _reset_settings_modal_scroll_state()
        HUD.settings_modal_open = False
        return True

    # Scroll buttons
    if "settings_scroll_up" in menu_buttons and menu_buttons["settings_scroll_up"].w > 0 and menu_buttons["settings_scroll_up"].contains(x, y):
        HUD.settings_modal_scroll_offset = max(0, HUD.settings_modal_scroll_offset - 60)
        return True
    if "settings_scroll_down" in menu_buttons and menu_buttons["settings_scroll_down"].w > 0 and menu_buttons["settings_scroll_down"].contains(x, y):
        total_content_h = _compute_content_height()
        content_h = MODAL_H - HEADER_H
        max_scroll = max(0, total_content_h - content_h)
        HUD.settings_modal_scroll_offset = min(max_scroll, HUD.settings_modal_scroll_offset + 60)
        return True

    # Scrollbar drag start
    if "settings_scrollbar" in menu_buttons and menu_buttons["settings_scrollbar"].w > 0 and menu_buttons["settings_scrollbar"].contains(x, y):
        HUD.settings_modal_scroll_dragging = True
        HUD.settings_modal_drag_start_y = y
        HUD.settings_modal_drag_start_scroll = HUD.settings_modal_scroll_offset
        return True

    # Calibration Suite
    if "settings_calibration_suite" in menu_buttons and menu_buttons["settings_calibration_suite"].contains(x, y):
        _reset_settings_modal_scroll_state()
        HUD.settings_modal_open = False
        button_state.calibration_suite_modal_open = True
        return True

    # Flash Firmware
    if "settings_flash_firmware" in menu_buttons and menu_buttons["settings_flash_firmware"].contains(x, y):
        _reset_settings_modal_scroll_state()
        HUD.settings_modal_open = False
        button_state.firmware_flash_modal_open = True
        return True

    # Email Settings
    if "settings_email" in menu_buttons and menu_buttons["settings_email"].contains(x, y):
        _reset_settings_modal_scroll_state()
        HUD.settings_modal_open = False
        button_state.email_settings_modal_open = True
        button_state.email_modal_screen = "provider"
        button_state.email_modal_provider = ""
        button_state.menu_open = False
        return True

    # Toggles
    if "settings_cam" in menu_buttons and menu_buttons["settings_cam"].contains(x, y):
        button_state.camera_enabled = not button_state.camera_enabled
        return True
    if "settings_theme" in menu_buttons and menu_buttons["settings_theme"].contains(x, y):
        pass
        return True
    if "settings_crosshairs" in menu_buttons and menu_buttons["settings_crosshairs"].contains(x, y):
        button_state.crosshairs_enabled = not button_state.crosshairs_enabled
        return True
    if "settings_debug" in menu_buttons and menu_buttons["settings_debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        return True
    if "settings_radar_ui" in menu_buttons and menu_buttons["settings_radar_ui"].contains(x, y):
        if state.MAGNETOMETER_AVAILABLE:
            button_state.radar_ui_enabled = not button_state.radar_ui_enabled
        return True
    if "settings_map_style" in menu_buttons and menu_buttons["settings_map_style"].contains(x, y):
        button_state.map_tile_style = "light" if button_state.map_tile_style == "dark" else "dark"
        return True
    if "settings_position_services" in menu_buttons and menu_buttons["settings_position_services"].contains(x, y):
        button_state.position_services_enabled = not button_state.position_services_enabled
        return True
    if "settings_record_compass_history" in menu_buttons and menu_buttons["settings_record_compass_history"].contains(x, y):
        button_state.record_compass_history = not button_state.record_compass_history
        return True
    if "settings_show_radar_debug" in menu_buttons and menu_buttons["settings_show_radar_debug"].contains(x, y):
        button_state.show_radar_debug = not button_state.show_radar_debug
        return True

    # Heatmap color selector
    for i, cm in enumerate(COLORMAPS):
        k = f"settings_colormap_{i}"
        if k in menu_buttons and menu_buttons[k].contains(x, y):
            button_state.colormap_mode = cm
            return True

    # Click on panel: keep open
    if "settings_modal_panel" in menu_buttons and menu_buttons["settings_modal_panel"].contains(x, y):
        return True

    _reset_settings_modal_scroll_state()
    HUD.settings_modal_open = False
    return True


def _reset_settings_modal_scroll_state() -> None:
    """Reset all scroll-related state when modal closes."""
    HUD.settings_modal_scroll_offset = 0
    HUD.settings_modal_content_dragging = False
    HUD.settings_modal_scroll_dragging = False
    HUD.settings_modal_inertia_active = False
    HUD.settings_modal_scroll_velocity = 0.0


def handle_settings_modal_scroll(delta: int) -> bool:
    """Handle mouse wheel or scroll gesture. delta > 0 = scroll down. Returns True if handled."""
    if not HUD.settings_modal_open:
        return False
    total_content_h = _compute_content_height()
    content_h = MODAL_H - HEADER_H
    max_scroll = max(0, total_content_h - content_h)
    if max_scroll <= 0:
        return False
    new_offset = HUD.settings_modal_scroll_offset + delta
    HUD.settings_modal_scroll_offset = max(0, min(max_scroll, new_offset))
    return True



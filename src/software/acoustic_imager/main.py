#!/usr/bin/env python3
"""
Acoustic Imager - Main Application

A real-time acoustic beamforming visualization system that supports:
- SIM (simulated) and SPI (hardware) data sources
- Interactive bandpass filtering
- MUSIC beamforming algorithm
- Camera background overlay
- Video recording and screenshots
- Configurable FPS and gain modes

This is the main entry point that orchestrates all modules.
"""

from __future__ import annotations

import logging
import sys
import time

log = logging.getLogger(__name__)
import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import os

# ===============================================================
# Display Configuration for SSH/Headless Operation
# ===============================================================
# Uncomment ONE of these if running over SSH:
# os.environ['DISPLAY'] = ':0'              # Use local display on Pi (monitor attached)
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Headless mode (no display, recording only)

# ===============================================================
# Import modularized components
# ===============================================================
from acoustic_imager import config
from acoustic_imager import state
from acoustic_imager.custom_types import LatestFrame, SourceStats

# Data sources
from acoustic_imager.sources.sim_source import SimSource
from acoustic_imager.sources.spi_source import SPISource
from acoustic_imager.sources.spi_loopback_source import SPILoopbackSource
from acoustic_imager.io.spi_manager import SPIManager
from acoustic_imager.io.gain_control import GAIN_CONTROL


# DSP modules
from acoustic_imager.dsp.beamforming import (
    directivity_ratio,
    music_spectrum,
    music_spectrum_2d,
    music_spectrum_2d_refined,
    music_2d_peak_angles,
)
from acoustic_imager.dsp.heatmap import (
    spectra_to_heatmap_absolute,
    build_w_lut_u8,
    blend_heatmap_left,
    draw_crosshairs,
    find_local_max,
    percentile_uint8_fast,
    CROSSHAIR_TRACK_RADIUS,
    CROSSHAIR_DISMISS_RADIUS_PX,
)
from acoustic_imager.dsp.bars import (
    draw_db_colorbar,
    freq_to_y,
    y_to_freq,
)
from acoustic_imager.dsp.spectrum_analyzer import draw_spectrum_analyzer, spectrum_closest_curve_point

from acoustic_imager.system_info import get_system_network_info
from acoustic_imager.ui.top_hud import draw_hud, handle_hud_click
from acoustic_imager.ui.bottom_hud import draw_bottom_hud, BOTTOM_HUD_HEIGHT
from acoustic_imager.state import HUD

# I/O managers
from acoustic_imager.io.camera_manager import CameraManager
from acoustic_imager.io.gallery_metadata import load_metadata

# UI components (flat modules alongside hud and video_recorder)
from acoustic_imager.state import button_state
from acoustic_imager.ui.button import (
    buttons,
    menu_buttons,
    init_buttons,
    init_menu_buttons,
    update_button_states,
    draw_buttons,
    FPS_MODE_TO_TARGET,
)
from acoustic_imager.ui.screenshot import save_screenshot, draw_screenshot_flash
from acoustic_imager.ui.menu import draw_menu, get_recording_timestamp_rect, _MENU_DROPDOWN_KEYS
from acoustic_imager.ui.gallery import draw_gallery_view, get_gallery_items
from acoustic_imager.ui.handlers import (
    handle_button_click,
    handle_gallery_click,
    handle_gallery_mouse,
    handle_gallery_viewer_mouse,
    handle_email_modal_click,
)
from acoustic_imager.ui.wifi_modal import draw_wifi_modal, handle_wifi_modal_click, handle_wifi_modal_touch_drag
from acoustic_imager.io.magnetometer import MagnetometerReader, probe_i2c_magnetometer
from acoustic_imager.io.gps_reader import GPSReader
from acoustic_imager.io.position_manager import PositionManager
from acoustic_imager.io.directional_history_store import DirectionalHistoryStore
from acoustic_imager.ui.settings_modal import (
    draw_settings_modal,
    handle_settings_modal_click,
    handle_settings_modal_mouse,
    handle_settings_modal_scroll,
)
from acoustic_imager.ui.firmware_flash_modal import (
    draw_firmware_flash_modal,
    handle_firmware_flash_modal_click,
)
from acoustic_imager.ui.calibration_suite_modal import (
    draw_calibration_suite_modal,
    handle_calibration_suite_modal_click,
    handle_calibration_suite_modal_mouse,
    handle_calibration_suite_modal_scroll,
)
from acoustic_imager.ui.acoustic_radar_map import draw_radar_map_widget, update_detection_history
from acoustic_imager.ui.video_recorder import VideoRecorder
from acoustic_imager.ui.battery_icon import draw_battery_icon_for_view

# region agent log
_AGENT_DEBUG_LOG_PATH = "/home/acousticlord/Capstone_490_Software/.cursor/debug-a9e491.log"
_AGENT_DEBUG_SESSION = "a9e491"


def _agent_debug_log(
    run_id: str,
    hypothesis_id: str,
    location: str,
    message: str,
    data: dict,
) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
# endregion

# ===============================================================
# External dependencies (from parent directories)
# Optional - provide fallbacks if not available
# ===============================================================
try:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
    from fftframe import FFTFrame  # type: ignore
except ImportError:
    # Fallback: minimal FFTFrame stub
    class FFTFrame:
        def __init__(self):
            self.channel_count = 0
            self.sampling_rate = 0
            self.fft_size = 0
            self.frame_id = 0
            self.fft_data = None

try:
    from heatmap_pipeline_test import create_background_frame  # type: ignore
except ImportError:
    # Fallback: create simple gradient background
    def create_background_frame(width: int, height: int) -> np.ndarray:
        """Create a simple gradient background."""
        bg = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            val = int(20 + (y / height) * 30)
            bg[y, :] = (val, val, val)
        return bg

try:
    sys.path.append(str(Path(__file__).resolve().parents[3] / "utilities"))
    from stage_profiler import StageProfiler  # type: ignore
except ImportError:
    # Fallback: minimal profiler stub
    class StageProfiler:
        def __init__(self, keep=120):
            self.stages = {}

        def start_frame(self):
            pass

        def mark(self, stage: str):
            pass

        def end_frame(self):
            pass

        def ms(self, stage: str) -> float:
            return 0.0


# ===============================================================
# Mouse callback for interactive controls
# ===============================================================
def mouse_callback(event, x: int, y: int, flags, param) -> None:
    """
    Handle mouse events for:
    - Button clicks (camera, source, debug, menu, etc.)
    - Bandpass filter dragging (frequency bar handles)
    - Gallery view navigation
    """
    global video_recorder

    content_width, content_height, content_offset_x, content_right, h = param
    mx, my = x, y

    def _bottom_hide_target() -> float:
        base = float(getattr(config, "UI_BOTTOM_HUD_HIDE_OFFSET", 60))
        extra = float(getattr(config, "UI_BOTTOM_HUD_DEBUG_EXTRA_HIDE_OFFSET", 20)) if button_state.debug_enabled else 0.0
        return base + extra

    # Update hover states
    state.CURSOR_POS = (mx, my)
    update_button_states(mx, my)

    # Always give gallery first dibs on all mouse events (down/move/up)
    if button_state.gallery_open:
        if button_state.gallery_viewer_mode == "grid":
            if handle_gallery_mouse(event, mx, my, flags, state.OUTPUT_DIR):
                return
        else:
            # Viewer (image/video): horizontal swipe with inertia + clicks
            if handle_gallery_viewer_mouse(event, mx, my, flags, state.OUTPUT_DIR):
                return

    # Handle left button down: UI (top HUD, menu, bottom HUD) and buttons take priority over gestures
    if event == cv2.EVENT_LBUTTONDOWN:
        # Check gallery view first (if open)
        if handle_gallery_mouse(event,mx, my, flags, state.OUTPUT_DIR):
            return

        # WiFi modal (when open, handle first). Touch-drag scroll before click.
        if HUD.wifi_modal_open:
            if handle_wifi_modal_touch_drag(event, mx, my, config.WIDTH, config.HEIGHT):
                state.ui_click_was_on_ui = True
                return
            if handle_wifi_modal_click(mx, my, config.WIDTH, config.HEIGHT):
                state.ui_click_was_on_ui = True
                return

        # Settings modal (when open, handle first)
        if HUD.settings_modal_open:
            if handle_settings_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                state.ui_click_was_on_ui = True
                return
            if handle_settings_modal_click(mx, my):
                state.ui_click_was_on_ui = True
                return

        # Email Settings modal (when open, handle first)
        if button_state.email_settings_modal_open:
            if handle_email_modal_click(mx, my, state.OUTPUT_DIR):
                state.ui_click_was_on_ui = True
                return

        # Firmware Flash modal (when open, handle first)
        if button_state.firmware_flash_modal_open:
            if handle_firmware_flash_modal_click(mx, my):
                state.ui_click_was_on_ui = True
                return

        # Calibration Suite modal (mouse first for drag-to-scroll, then click)
        if button_state.calibration_suite_modal_open:
            if handle_calibration_suite_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                state.ui_click_was_on_ui = True
                return
            if handle_calibration_suite_modal_click(mx, my):
                state.ui_click_was_on_ui = True
                return

        # 1) Top HUD pill click
        try:
            from acoustic_imager.ui.top_hud import handle_hud_click
            if hasattr(state, "HUD_RECTS") and state.HUD_RECTS is not None:
                new_panel = handle_hud_click(mx, my, state.HUD_RECTS, HUD.open_panel)
                if new_panel != HUD.open_panel:
                    HUD.open_panel = new_panel
                    state.ui_click_was_on_ui = True
                    return
        except Exception:
            pass

        # 2) Bottom HUD pills first (so they work when menu is closed; only when bar visible)
        if state.ui_bottom_hud_offset <= 15:
            hit_pad = getattr(config, "UI_BOTTOM_HUD_HIT_PAD", 8)
            # Gallery: trigger on release (set pressed state here; actual open on LBUTTONUP)
            if "gallery" in menu_buttons:
                b = menu_buttons["gallery"]
                if b.w > 0 and b.h > 0 and (b.x - hit_pad <= mx < b.x + b.w + hit_pad and
                        b.y - hit_pad <= my < b.y + b.h + hit_pad):
                    button_state.gallery_pill_pressed = True
                    state.ui_click_was_on_ui = True
                    return
            for k in ("shot", "rec", "rec_resume", "rec_stop"):
                if k in menu_buttons:
                    b = menu_buttons[k]
                    if b.w > 0 and b.h > 0:
                        if (b.x - hit_pad <= mx < b.x + b.w + hit_pad and
                                b.y - hit_pad <= my < b.y + b.h + hit_pad):
                            video_recorder = handle_button_click(
                                mx, my,
                                current_frame=state.CURRENT_FRAME,
                                output_dir=state.OUTPUT_DIR,
                                camera_available=state.CAMERA_AVAILABLE,
                                video_recorder=video_recorder,
                                width=config.WIDTH,
                                height=config.HEIGHT,
                            )
                            state.ui_click_was_on_ui = True
                            return

        # 3) Menu button and dropdown (only when menu visible; hit-test accounts for offset_x and offset_y)
        menu_mx = mx - int(state.ui_menu_offset)
        menu_my = my - int(state.ui_menu_offset_y)
        if state.ui_menu_offset <= 15 and state.ui_menu_offset_y <= 10:
            if "menu" in menu_buttons and menu_buttons["menu"].contains(menu_mx, menu_my):
                video_recorder = handle_button_click(
                    mx, my,
                    current_frame=state.CURRENT_FRAME,
                    output_dir=state.OUTPUT_DIR,
                    camera_available=state.CAMERA_AVAILABLE,
                    video_recorder=video_recorder,
                    width=config.WIDTH,
                    height=config.HEIGHT,
                )
                state.ui_click_was_on_ui = True
                return
            if button_state.menu_open:
                for k in _MENU_DROPDOWN_KEYS:
                    if k not in menu_buttons:
                        continue
                    b = menu_buttons[k]
                    if b.contains(menu_mx, menu_my):
                        video_recorder = handle_button_click(
                            mx, my,
                            current_frame=state.CURRENT_FRAME,
                            output_dir=state.OUTPUT_DIR,
                            camera_available=state.CAMERA_AVAILABLE,
                            video_recorder=video_recorder,
                            width=config.WIDTH,
                            height=config.HEIGHT,
                        )
                        state.ui_click_was_on_ui = True
                        return

        # 4) Recording timestamp bar (if used)
        if button_state.is_recording and video_recorder is not None:
            rect = get_recording_timestamp_rect()
            if rect is not None:
                rx, ry, rw, rh = rect
                if rx <= mx <= rx + rw and ry <= my <= ry + rh:
                    button_state.is_paused = not button_state.is_paused
                    if button_state.is_paused:
                        video_recorder.pause_recording()
                    else:
                        video_recorder.resume_recording()
                    state.ui_click_was_on_ui = True
                    return

        # 5) Bandpass drag: frequency bar
        bar_left = content_right
        if mx >= bar_left and mx < config.WIDTH:
            y_min = freq_to_y(state.F_MIN_HZ, h, config.F_DISPLAY_MAX)
            y_max = freq_to_y(state.F_MAX_HZ, h, config.F_DISPLAY_MAX)
            dmin = abs(my - y_min)
            dmax = abs(my - y_max)
            if dmin <= config.DRAG_MARGIN_PX or dmax <= config.DRAG_MARGIN_PX:
                state.DRAG_TARGET = "min" if dmin <= dmax else "max"
                state.DRAG_ACTIVE = True
                f = y_to_freq(my, h, config.F_DISPLAY_MAX)
                if state.DRAG_TARGET == "min":
                    state.F_MIN_HZ = min(f, state.F_MAX_HZ)
                else:
                    state.F_MAX_HZ = max(f, state.F_MIN_HZ)
                state.ui_click_was_on_ui = True
                return
            if min(y_min, y_max) <= my <= max(y_min, y_max):
                state.DRAG_TARGET = "box"
                state.DRAG_ACTIVE = True
                state.DRAG_START_Y = my
                state.DRAG_START_F_MIN = state.F_MIN_HZ
                state.DRAG_START_F_MAX = state.F_MAX_HZ
                state.ui_click_was_on_ui = True
                return
            # Spectrum cursor: double-tap toggles cursor; single tap on line places/moves dot; tap in graph moves line
            bar_w = config.FREQ_BAR_WIDTH
            graph_left = 8
            graph_right = bar_w - 5
            cursor_x_bar = mx - bar_left
            if graph_left <= cursor_x_bar <= graph_right:
                now = time.time()
                dt_ms = (now - state.SPECTRUM_CURSOR_LAST_TAP_TIME) * 1000
                tap_dist = ((mx - state.SPECTRUM_CURSOR_LAST_TAP_X) ** 2 + (my - state.SPECTRUM_CURSOR_LAST_TAP_Y) ** 2) ** 0.5
                is_double_tap = dt_ms < config.UI_DOUBLE_TAP_MS and tap_dist < config.UI_DOUBLE_TAP_RADIUS_PX
                if is_double_tap:
                    if state.SPECTRUM_CURSOR_X is not None:
                        state.SPECTRUM_CURSOR_X = None
                        state.SPECTRUM_CURSOR_DOT_ACTIVE = False
                        state.SPECTRUM_CURSOR_DOT_FREQ = None
                        state.SPECTRUM_CURSOR_PENDING_TAP_X = None
                        state.SPECTRUM_CURSOR_PENDING_TAP_Y = None
                        state.SPECTRUM_CURSOR_DOT_DRAG_ACTIVE = False
                        state.SPECTRUM_CURSOR_DOT_BAR_X = None
                        state.SPECTRUM_CURSOR_DOT_BAR_Y = None
                    else:
                        state.SPECTRUM_CURSOR_X = float(np.clip(cursor_x_bar, graph_left, graph_right))
                        state.SPECTRUM_CURSOR_DOT_ACTIVE = False
                    state.SPECTRUM_CURSOR_LAST_TAP_TIME = 0.0
                    state.ui_click_was_on_ui = True
                    return
                # Hit-test dot first: press on dot starts drag (dot grows, draggable along curve)
                dot_hit_radius = 24
                on_dot = (
                    state.SPECTRUM_CURSOR_DOT_ACTIVE
                    and state.SPECTRUM_CURSOR_DOT_BAR_X is not None
                    and state.SPECTRUM_CURSOR_DOT_BAR_Y is not None
                )
                if on_dot:
                    dot_bx = state.SPECTRUM_CURSOR_DOT_BAR_X
                    dot_by = state.SPECTRUM_CURSOR_DOT_BAR_Y
                    if dot_bx is not None and dot_by is not None:
                        dot_scr_x = bar_left + dot_bx
                        dist_sq = (mx - dot_scr_x) ** 2 + (my - dot_by) ** 2
                        if dist_sq <= dot_hit_radius ** 2:
                            state.SPECTRUM_CURSOR_DOT_DRAG_ACTIVE = True
                            state.SPECTRUM_CURSOR_PENDING_TAP_X = float(np.clip(cursor_x_bar, graph_left, graph_right))
                            state.SPECTRUM_CURSOR_PENDING_TAP_Y = int(np.clip(my, 0, h - 1))
                            state.ui_click_was_on_ui = True
                            return
                cursor_line_margin = 16
                on_red_line = (
                    state.SPECTRUM_CURSOR_X is not None
                    and abs(cursor_x_bar - state.SPECTRUM_CURSOR_X) <= cursor_line_margin
                )
                if on_red_line:
                    state.SPECTRUM_CURSOR_DOT_ACTIVE = True
                    state.SPECTRUM_CURSOR_PENDING_TAP_Y = int(np.clip(my, 0, h - 1))
                    state.SPECTRUM_CURSOR_LAST_TAP_TIME = now
                    state.SPECTRUM_CURSOR_LAST_TAP_X = mx
                    state.SPECTRUM_CURSOR_LAST_TAP_Y = my
                    state.ui_click_was_on_ui = True
                    return
                # Cursor visible: single tap elsewhere in graph moves line and starts drag
                if state.SPECTRUM_CURSOR_X is not None:
                    state.SPECTRUM_CURSOR_X = float(np.clip(cursor_x_bar, graph_left, graph_right))
                    state.DRAG_TARGET = "cursor"
                    state.DRAG_ACTIVE = True
                state.SPECTRUM_CURSOR_LAST_TAP_TIME = now
                state.SPECTRUM_CURSOR_LAST_TAP_X = mx
                state.SPECTRUM_CURSOR_LAST_TAP_Y = my
                state.ui_click_was_on_ui = True
                return
        # 6) Other UI buttons (camera, source, debug)
        for b in buttons.values():
            if b.contains(mx, my):
                video_recorder = handle_button_click(
                    mx, my,
                    current_frame=state.CURRENT_FRAME,
                    output_dir=state.OUTPUT_DIR,
                    camera_available=state.CAMERA_AVAILABLE,
                    video_recorder=video_recorder,
                    width=config.WIDTH,
                    height=config.HEIGHT,
                )
                state.ui_click_was_on_ui = True
                return

        # No UI hit: record drag start for possible swipe/double-tap (only in content area)
        state.ui_click_was_on_ui = False
        state.ui_drag_start_x = mx
        state.ui_drag_start_y = my
        state.ui_drag_start_time = time.time()

    # Handle left button up: Gallery pill (trigger on release) then swipe/double-tap
    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_pill_pressed:
        button_state.gallery_pill_pressed = False
        if state.ui_bottom_hud_offset <= 15 and "gallery" in menu_buttons:
            b = menu_buttons["gallery"]
            if b.w > 0 and b.h > 0:
                hit_pad = getattr(config, "UI_BOTTOM_HUD_HIT_PAD", 8)
                if (b.x - hit_pad <= mx < b.x + b.w + hit_pad and
                        b.y - hit_pad <= my < b.y + b.h + hit_pad):
                    from acoustic_imager.ui.archive_panel import load_archive_folders
                    if button_state.is_recording and video_recorder is not None:
                        video_recorder.stop_recording()
                        button_state.is_recording = False
                        button_state.is_paused = False
                        button_state.gallery_storage_dirty = True
                    button_state.gallery_open = True
                    button_state.menu_open = False
                    button_state.gallery_storage_dirty = True
                    if state.OUTPUT_DIR:
                        button_state.gallery_archive_folders = load_archive_folders(state.OUTPUT_DIR)
                    return
        return

    if event == cv2.EVENT_LBUTTONUP and not button_state.gallery_open and not state.ui_click_was_on_ui:
        content_left = content_offset_x
        content_top = getattr(config, "UI_CONTENT_TOP_MARGIN", 58)
        content_bottom = h - getattr(config, "UI_CONTENT_BOTTOM_MARGIN", 62)
        dx = mx - state.ui_drag_start_x
        dy = my - state.ui_drag_start_y
        dist = (dx * dx + dy * dy) ** 0.5
        now = time.time()
        in_content = (
            content_left <= mx < content_right and content_top <= my < content_bottom
            and content_left <= state.ui_drag_start_x < content_right
            and content_top <= state.ui_drag_start_y < content_bottom
        )
        # Drag started in content (raw camera/heatmap area) – use for swipe even if release is outside
        in_content_start = (
            content_left <= state.ui_drag_start_x < content_right
            and content_top <= state.ui_drag_start_y < content_bottom
        )
        # Swipe-down from bottom strip (e.g. from bottom HUD area) to hide bottom HUD + menu
        swipe_down_from_bottom = (
            not in_content
            and dist >= config.UI_SWIPE_THRESHOLD_PX
            and abs(dy) > abs(dx)
            and dy > 0
            and state.ui_drag_start_y >= h - 120
        )
        # Swipe-down from top strip when top HUD is hidden: bring top HUD back
        swipe_down_from_top_hidden = (
            not in_content
            and dist >= config.UI_SWIPE_THRESHOLD_PX
            and abs(dy) > abs(dx)
            and dy > 0
            and state.ui_drag_start_y <= getattr(config, "UI_CONTENT_TOP_MARGIN", 58) + 20
            and state.ui_top_hud_offset_target <= float(config.UI_TOP_HUD_HIDE_OFFSET) * 0.5
        )
        # Swipe-up from bottom strip when HUD/menu are hidden: bring them back
        swipe_up_from_bottom_hidden = (
            not in_content
            and dist >= config.UI_SWIPE_THRESHOLD_PX
            and abs(dy) > abs(dx)
            and dy < 0
            and state.ui_drag_start_y >= h - 120
            and (state.ui_bottom_hud_offset_target > 1 or state.ui_menu_offset_y_target > 1)
        )
        # Allow gestures when: full content tap/swipe, or swipe that started in content, or swipe in top/bottom strips
        swipe_started_in_content = in_content_start and dist >= config.UI_SWIPE_THRESHOLD_PX
        if (
            not in_content
            and not swipe_down_from_bottom
            and not swipe_down_from_top_hidden
            and not swipe_up_from_bottom_hidden
            and not swipe_started_in_content
        ):
            pass  # do not run gesture
        else:
            if dist >= config.UI_SWIPE_THRESHOLD_PX:
                # Swipe: when menu is open, first swipe down only closes the menu (takes 2 swipes to slide out).
                if abs(dy) > abs(dx):
                    if dy > 0:  # swipe down
                        if button_state.menu_open:
                            button_state.menu_open = False
                            return
                        # Downward swipe from upper/mid area: toggle top HUD; from lower half: hide bottom HUD + menu
                        mid_y = h // 2
                        if state.ui_drag_start_y <= mid_y:
                            if state.ui_top_hud_offset_target < 0:
                                state.ui_top_hud_offset_target = 0.0
                            else:
                                state.ui_top_hud_offset_target = float(config.UI_TOP_HUD_HIDE_OFFSET)
                        else:
                            state.ui_bottom_hud_offset_target = _bottom_hide_target()
                            state.ui_menu_offset_y_target = float(getattr(config, "UI_MENU_HIDE_OFFSET_Y", 80))
                            # region agent log
                            _agent_debug_log(
                                "run1",
                                "H3",
                                "main.py:swipe_down",
                                "swipe_down_hide_bottom_targets",
                                {
                                    "bottom_target": float(state.ui_bottom_hud_offset_target),
                                    "menu_y_target": float(state.ui_menu_offset_y_target),
                                    "hide_offset_cfg": float(config.UI_BOTTOM_HUD_HIDE_OFFSET),
                                    "hide_offset_extra_cfg": float(
                                        getattr(config, "UI_BOTTOM_HUD_DEBUG_EXTRA_HIDE_OFFSET", 20)
                                    ),
                                    "debug_enabled": bool(button_state.debug_enabled),
                                },
                            )
                            # endregion
                    else:  # swipe up
                        if state.ui_bottom_hud_offset_target > 0 or state.ui_menu_offset_y_target > 0:
                            state.ui_bottom_hud_offset_target = 0.0
                            state.ui_menu_offset_y_target = 0.0
                            # region agent log
                            _agent_debug_log(
                                "run1",
                                "H3",
                                "main.py:swipe_up",
                                "swipe_up_show_bottom_targets",
                                {
                                    "bottom_target": float(state.ui_bottom_hud_offset_target),
                                    "menu_y_target": float(state.ui_menu_offset_y_target),
                                },
                            )
                            # endregion
                        elif (
                            not button_state.menu_open
                            and state.ui_bottom_hud_offset_target <= 1
                            and state.ui_menu_offset_y_target <= 1
                            and state.ui_drag_start_y >= h - 140
                        ):
                            # Swipe up from bottom: open the menu (bottom HUD and menu button visible, menu closed)
                            button_state.menu_open = True
                        else:
                            state.ui_top_hud_offset_target = float(config.UI_TOP_HUD_HIDE_OFFSET)
                else:
                    if dx < 0:  # swipe left -> show menu (x)
                        state.ui_menu_offset_target = 0.0
                    # swipe right does nothing (menu button stays in place)
                return
            if dist <= config.UI_TAP_MAX_MOVE_PX and in_content:
                # Single tap in content area when menu is open: close menu, keep menu button visible
                if button_state.menu_open:
                    button_state.menu_open = False
                    return
                dt_ms = (now - state.ui_last_tap_time) * 1000
                tap_dist = ((mx - state.ui_last_tap_x) ** 2 + (my - state.ui_last_tap_y) ** 2) ** 0.5
                if dt_ms < config.UI_DOUBLE_TAP_MS and tap_dist < config.UI_DOUBLE_TAP_RADIUS_PX:
                    # Double-tap has priority: hide/show HUD
                    menu_hide_y = getattr(config, "UI_MENU_HIDE_OFFSET_Y", 80)
                    menu_hidden = (
                        state.ui_menu_offset_target >= config.UI_MENU_HIDE_OFFSET * 0.5
                        or state.ui_menu_offset_y_target >= menu_hide_y * 0.5
                    )
                    all_hidden = (
                        state.ui_top_hud_offset_target <= config.UI_TOP_HUD_HIDE_OFFSET * 0.5
                        and state.ui_bottom_hud_offset_target >= _bottom_hide_target() * 0.5
                        and menu_hidden
                    )
                    if all_hidden:
                        state.ui_top_hud_offset_target = 0.0
                        state.ui_bottom_hud_offset_target = 0.0
                        state.ui_menu_offset_target = 0.0
                        state.ui_menu_offset_y_target = 0.0
                        # region agent log
                        _agent_debug_log(
                            "run1",
                            "H2",
                            "main.py:double_tap",
                            "double_tap_show_hud_targets",
                            {
                                "top_target": state.ui_top_hud_offset_target,
                                "bottom_target": state.ui_bottom_hud_offset_target,
                                "menu_target": state.ui_menu_offset_target,
                                "menu_y_target": state.ui_menu_offset_y_target,
                            },
                        )
                        # endregion
                    else:
                        state.ui_top_hud_offset_target = float(config.UI_TOP_HUD_HIDE_OFFSET)
                        state.ui_bottom_hud_offset_target = _bottom_hide_target()
                        state.ui_menu_offset_target = float(config.UI_MENU_HIDE_OFFSET)
                        state.ui_menu_offset_y_target = float(menu_hide_y)
                        button_state.menu_open = False
                        # region agent log
                        _agent_debug_log(
                            "run1",
                            "H2",
                            "main.py:double_tap",
                            "double_tap_hide_hud_targets",
                            {
                                "top_target": state.ui_top_hud_offset_target,
                                "bottom_target": state.ui_bottom_hud_offset_target,
                                "menu_target": state.ui_menu_offset_target,
                                "menu_y_target": state.ui_menu_offset_y_target,
                                "hide_offset_cfg": float(config.UI_BOTTOM_HUD_HIDE_OFFSET),
                            },
                        )
                        # endregion
                    state.ui_last_tap_time = 0.0
                    return
                # Single tap: crosshair toggle in heatmap (if crosshairs enabled)
                if button_state.crosshairs_enabled and content_offset_x <= mx < content_right and 0 <= my < content_height:
                    dx = (mx - content_offset_x) - int(button_state.crosshair_x)
                    dy = my - int(button_state.crosshair_y)
                    dist_sq = dx * dx + dy * dy
                    if button_state.crosshair_visible and dist_sq <= CROSSHAIR_DISMISS_RADIUS_PX * CROSSHAIR_DISMISS_RADIUS_PX:
                        button_state.crosshair_visible = False
                    else:
                        button_state.crosshair_visible = True
                        button_state.crosshair_x = float(mx - content_offset_x)
                        button_state.crosshair_y = float(my)
                        # Reset trend/acceleration state for new track
                        button_state.crosshair_level_history = []
                        button_state.crosshair_trend_history = []
                        button_state.crosshair_prev_baseline_db = None
                        button_state.crosshair_next_boundary_time = 0.0
                    state.ui_last_tap_time = now
                    state.ui_last_tap_x = mx
                    state.ui_last_tap_y = my
                    return
                state.ui_last_tap_time = now
                state.ui_last_tap_x = mx
                state.ui_last_tap_y = my

    # Handle mouse move (dragging)
    elif event == cv2.EVENT_MOUSEMOVE:

        # WiFi modal list touch-drag scroll
        if HUD.wifi_modal_open:
            if handle_wifi_modal_touch_drag(event, mx, my, config.WIDTH, config.HEIGHT):
                return

        # Settings modal touch/drag scroll
        if HUD.settings_modal_open:
            if handle_settings_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                return

        # Calibration Suite modal touch/drag scroll (log area)
        if button_state.calibration_suite_modal_open:
            if handle_calibration_suite_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                return

        bar_left = content_right

        # Dot drag: follow finger along curve (checked first; we don't set DRAG_ACTIVE for dot drag)
        if state.SPECTRUM_CURSOR_DOT_DRAG_ACTIVE and mx >= bar_left and mx < config.WIDTH:
            bar_w = config.FREQ_BAR_WIDTH
            graph_left = 8
            graph_right = bar_w - 5
            cursor_x_bar = float(np.clip(mx - bar_left, graph_left, graph_right))
            state.SPECTRUM_CURSOR_PENDING_TAP_X = cursor_x_bar
            state.SPECTRUM_CURSOR_PENDING_TAP_Y = int(np.clip(my, 0, h - 1))

        # Handle frequency bar dragging
        elif state.DRAG_ACTIVE and mx >= bar_left and mx < config.WIDTH:
            if state.DRAG_TARGET == "box":
                # Drag the entire box - maintain the frequency range
                freq_range = state.DRAG_START_F_MAX - state.DRAG_START_F_MIN

                # Convert y offset to frequency offset (swap order to fix inverse scrolling)
                freq_offset = y_to_freq(my, h, config.F_DISPLAY_MAX) - y_to_freq(state.DRAG_START_Y, h, config.F_DISPLAY_MAX)

                # Move both frequencies
                new_f_min = state.DRAG_START_F_MIN + freq_offset
                new_f_max = state.DRAG_START_F_MAX + freq_offset

                # Clamp to valid range while maintaining the box size
                if new_f_min < 0:
                    new_f_min = 0
                    new_f_max = freq_range
                elif new_f_max > config.F_DISPLAY_MAX:
                    new_f_max = config.F_DISPLAY_MAX
                    new_f_min = config.F_DISPLAY_MAX - freq_range

                state.F_MIN_HZ = new_f_min
                state.F_MAX_HZ = new_f_max
            elif state.DRAG_TARGET == "cursor":
                bar_w = config.FREQ_BAR_WIDTH
                graph_left = 8
                graph_right = bar_w - 5
                cursor_x_bar = float(np.clip(mx - bar_left, graph_left, graph_right))
                state.SPECTRUM_CURSOR_X = cursor_x_bar
            else:
                # Drag individual handle
                f = y_to_freq(my, h, config.F_DISPLAY_MAX)
                if state.DRAG_TARGET == "min":
                    state.F_MIN_HZ = min(f, state.F_MAX_HZ)
                elif state.DRAG_TARGET == "max":
                    state.F_MAX_HZ = max(f, state.F_MIN_HZ)

    # Handle mouse wheel (scroll)
    elif event == getattr(cv2, "EVENT_MOUSEWHEEL", 10):
        if HUD.settings_modal_open:
            # flags: positive = scroll up, negative = scroll down (platform-dependent)
            delta = -80 if flags > 0 else 80
            if button_state.calibration_suite_modal_open and handle_calibration_suite_modal_scroll(delta):
                return
            if handle_settings_modal_scroll(delta):
                return

    # Handle left button up
    elif event == cv2.EVENT_LBUTTONUP:

        # WiFi modal list touch-drag scroll end
        if HUD.wifi_modal_open:
            if handle_wifi_modal_touch_drag(event, mx, my, config.WIDTH, config.HEIGHT):
                pass  # consumed

        # Settings modal touch/drag scroll end
        if HUD.settings_modal_open:
            if handle_settings_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                pass  # consumed

        # Calibration Suite modal touch/drag scroll end
        if button_state.calibration_suite_modal_open:
            if handle_calibration_suite_modal_mouse(event, mx, my, config.WIDTH, config.HEIGHT):
                pass  # consumed

        # End dot drag (dot stays at last snapped position)
        state.SPECTRUM_CURSOR_DOT_DRAG_ACTIVE = False

        # End frequency bar drag
        state.DRAG_ACTIVE = False
        state.DRAG_TARGET = None

    # Clamp frequency values
    state.F_MIN_HZ = float(np.clip(state.F_MIN_HZ, 0.0, config.F_DISPLAY_MAX))
    state.F_MAX_HZ = float(np.clip(state.F_MAX_HZ, 0.0, config.F_DISPLAY_MAX))
    if state.F_MIN_HZ > state.F_MAX_HZ:
        state.F_MIN_HZ, state.F_MAX_HZ = state.F_MAX_HZ, state.F_MIN_HZ


# ===============================================================
# Main application
# ===============================================================
def main() -> None:
    """Main application loop."""
    global video_recorder

    prev_mode = button_state.source_mode

    # ---- Profiler (for performance monitoring) ----
    prof = StageProfiler(keep=120)
    PRINT_EVERY = 60  # frames

    print("=" * 70)
    print("Acoustic Imager - Real-Time Beamforming Visualization")
    print("=" * 70)
    print(f"Resolution: {config.WIDTH}x{config.HEIGHT}")
    print(f"Mics: {config.N_MICS} | FFT bins: {config.N_BINS}")
    print(f"Sample rate: {config.SAMPLE_RATE_HZ} Hz")
    print(f"FPS modes: 30 / 60 / MAX (default: {button_state.fps_mode})")
    print("=" * 70)

    # ---- Setup output directory ----
    # Use data folder at repository root
    repo_root = Path(__file__).resolve().parents[3]  # Go up to Capstone_490_Software/
    state.OUTPUT_DIR = repo_root / "data" / "heatmap_captures"
    state.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {state.OUTPUT_DIR}")

    # Radar map tile cache: under main data/ folder (same as heatmap_captures, directional_history)
    config.RADAR_MAP_CACHE_DIR = repo_root / "data" / "map_tiles"
    config.RADAR_MAP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Logging: file + stderr (journalctl captures stderr) ----
    log_file = state.OUTPUT_DIR / "acoustic_imager.log"
    log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(log_fmt))
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.INFO)
    stderr_handler.setFormatter(logging.Formatter(log_fmt))
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(stderr_handler)

    # Load persisted gallery metadata (tags, priority, tag_data)
    prios, file_tags, tag_data = load_metadata(state.OUTPUT_DIR)
    button_state.gallery_file_priorities = prios
    button_state.gallery_file_tags = file_tags
    button_state.gallery_tag_data = tag_data
    print()

    # ---- Initialize data sources ----
    global sim_source, spi_hw, spi_loopback

    sim_source = SimSource()

    spi_loopback = SPILoopbackSource()

    # HW = real SPI from STM32 (16 mics, MCU_STATUS on physical pin 26 / BCM7)
    spi_hw = SPISource(SPIManager(use_frame_ready=True))

    # Force initial mode from config if UI didn't set it yet
    if button_state.source_mode not in config.SOURCE_MODES:
        button_state.source_mode = config.SOURCE_DEFAULT

    GAIN_CONTROL.set_mode(button_state.gain_mode)

    prev_mode = button_state.source_mode

    # Start the selected SPI provider (if any)
    if button_state.source_mode == "LOOP":
        spi_loopback.start()
    elif button_state.source_mode == "HW":
        spi_hw.start()

    # ---- Initialize camera ----
    camera_mgr = CameraManager()
    camera_mgr.detect_and_open()
    if state.CAMERA_AVAILABLE and button_state.camera_enabled:
        camera_mgr.start()

    detector = None
    if state.CAMERA_AVAILABLE and button_state.camera_enabled:
        #detector = DetectorClient(target_fps=8.0)

        def frame_provider():
            f = camera_mgr.get_latest_frame()
            if f is None or getattr(f, "size", 0) == 0:
                return None
            return f   # always BGR now

        #detector.start(frame_provider)

    # ---- Create static background ----
    background_full = create_background_frame(config.WIDTH, config.HEIGHT)
    # Content strip between DB bar and freq bar: heatmap lives here, blobs not under the bars
    content_offset_x = config.DB_BAR_WIDTH
    content_width = config.HEATMAP_WIDTH
    content_height = config.HEATMAP_HEIGHT
    content_right = content_offset_x + content_width  # x where freq bar starts

    # ---- Preallocated buffers (optimization) ----
    base_frame = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    cam_bgr = np.empty((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)

    # ---- Build blend LUT ----
    w_lut_u8 = build_w_lut_u8(config.ALPHA, config.BLEND_GAMMA)

    # ---- Initialize video recorder ----
    global video_recorder
    video_recorder = VideoRecorder(state.OUTPUT_DIR, config.WIDTH, config.HEIGHT, fps=30)

    # ---- Setup OpenCV window ----
    cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(config.WINDOW_NAME, mouse_callback, param=(content_width, content_height, content_offset_x, content_right, config.HEIGHT))

    # ---- Initialize UI ----
    button_state.debug_enabled = False
    init_buttons(content_right, state.CAMERA_AVAILABLE)
    init_menu_buttons(content_right, config.HEIGHT)

    # ---- Magnetometer (compass) reader ----
    mag_reader = MagnetometerReader(
        config.MAG_UART_DEVICE,
        config.MAG_UART_BAUD,
        demo=getattr(config, "MAG_COMPASS_DEMO", False),
        use_i2c=getattr(config, "MAG_USE_I2C", True),
        i2c_bus=getattr(config, "MAG_I2C_BUS", 1),
        i2c_addr=getattr(config, "MAG_I2C_ADDR", 0x1E),
        i2c_gain_reg=getattr(config, "MAG_I2C_GAIN_REG", 0xA0),
    )
    mag_reader.start()
    if getattr(config, "MAG_USE_I2C", True):
        state.MAGNETOMETER_AVAILABLE = probe_i2c_magnetometer(
            getattr(config, "MAG_I2C_BUS", 1),
            getattr(config, "MAG_I2C_ADDR", 0x1E),
        )
        if not state.MAGNETOMETER_AVAILABLE:
            button_state.radar_ui_enabled = False

    # ---- GPS (BN-880) reader ----
    gps_reader = GPSReader(
        getattr(config, "GPS_UART_DEVICE", "/dev/ttyAMA0"),
        getattr(config, "GPS_UART_BAUD", 9600),
        enabled=getattr(config, "GPS_USE_UART", True),
    )
    position_manager = PositionManager()
    gps_running = False
    pos_running = False
    if button_state.position_services_enabled:
        gps_reader.start()
        position_manager.start()
        gps_running = True
        pos_running = True

    directional_store = DirectionalHistoryStore(
        base_dir=str(
            (
                Path(getattr(config, "DIRECTIONAL_HISTORY_DIR", "data/directional_history"))
                if Path(getattr(config, "DIRECTIONAL_HISTORY_DIR", "data/directional_history")).is_absolute()
                else (repo_root / Path(getattr(config, "DIRECTIONAL_HISTORY_DIR", "data/directional_history")))
            )
        ),
        retention_days=int(getattr(config, "DIRECTIONAL_HISTORY_RETENTION_DAYS", 7)),
        flush_interval_s=float(getattr(config, "DIRECTIONAL_HISTORY_FLUSH_SEC", 1.0)),
        enabled=bool(getattr(config, "DIRECTIONAL_HISTORY_ENABLED", True)),
    )

    # ---- Loop state ----
    frame_count = 0
    start_time = time.time()
    last_spi_fft_data: Optional[np.ndarray] = None
    heatmap_prev: Optional[np.ndarray] = None
    last_spi_bins: Optional[list] = None
    last_spi_peak_angles: Optional[list] = None
    cov_avg: dict[int, np.ndarray] = {}  # bin_idx -> averaged covariance (N_MICS, N_MICS) for MUSIC
    # Reusable buffers for HW/LOOP heatmap (avoid per-frame allocs)
    _heatmap_max_bins = 32
    _heatmap_max_n_ang = max(
        len(config.ANGLES),
        len(getattr(config, "ANGLES_2D_X", config.ANGLES)),
        max(0, int(getattr(config, "SPI_MUSIC_2D_COARSE_RESOLUTION", 0))),
    )
    heatmap_spec_buf: Optional[np.ndarray] = None
    heatmap_angle_x_buf: Optional[np.ndarray] = None
    heatmap_angle_y_buf: Optional[np.ndarray] = None
    heatmap_power_buf: Optional[np.ndarray] = None
    heatmap_band_freqs_buf: Optional[np.ndarray] = None
    heatmap_heat_buf: Optional[np.ndarray] = None  # (content_height, content_width) float32 for heat_out
    # Cache for "MUSIC every N frames": reuse angles/spec on skip frames; frozen_bins keeps bin set stable so cache is valid
    last_music_bins_list: Optional[list] = None
    last_music_angle_x_cache: Optional[np.ndarray] = None
    last_music_angle_y_cache: Optional[np.ndarray] = None
    last_music_spec_cache: Optional[np.ndarray] = None
    frozen_bins: Optional[list] = None

    # FPS tracking
    fps_ema = 0.0
    last_t = time.perf_counter()
    next_tick = time.perf_counter()

    try:
        while True:
            prof.start_frame()
            elapsed = time.time() - start_time

            # ---- FPS estimation ----
            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            inst_fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_ema = (0.92 * fps_ema + 0.08 * inst_fps) if fps_ema > 0 else inst_fps
            prof.mark("fps_est")

            # ---- FPS throttling ----
            if button_state.fps_mode in ("30", "60"):
                fps_target = FPS_MODE_TO_TARGET[button_state.fps_mode]
                period = 1.0 / max(1, fps_target)
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
            else:
                # MAX mode = unthrottled
                next_tick = time.perf_counter()
            prof.mark("throttle")

            # ---- Position services runtime toggle ----
            if button_state.position_services_enabled and not gps_running:
                gps_reader.start()
                gps_running = True
            elif (not button_state.position_services_enabled) and gps_running:
                gps_reader.stop()
                gps_running = False
            if button_state.position_services_enabled and not pos_running:
                position_manager.start()
                pos_running = True
            elif (not button_state.position_services_enabled) and pos_running:
                position_manager.stop()
                pos_running = False

            mode = button_state.source_mode
            if mode != prev_mode:
                # stop both (safe)
                spi_loopback.stop()
                spi_hw.stop()

                last_spi_fft_data = None
                heatmap_prev = None
                last_spi_bins = None
                last_spi_peak_angles = None
                cov_avg.clear()

                # start whichever is selected
                if mode == "LOOP":
                    spi_loopback.start()
                elif mode == "HW":
                    spi_hw.start()
                    log.info("HW source active, SPI: /dev/spidev%d.%d", config.SPI_BUS, config.SPI_DEV)

                prev_mode = mode

            # ---- Get current bandpass range ----
            f_min = state.F_MIN_HZ
            f_max = state.F_MAX_HZ

            # ---- Read FFT data from source ----
            mode = button_state.source_mode

            if mode == "SIM":
                latest_frame = sim_source.read_frame()
                source_label = "SIM"
            elif mode == "SIM_2":
                latest_frame = sim_source.read_frame_sim2()
                source_label = "SIM_2"
            elif mode == "REF":
                # 0 dB reference: flat FFT so all spectrum bars sit at peak (baseline test)
                ref_fft = np.ones((config.N_MICS, config.N_BINS), dtype=np.complex64)
                latest_frame = LatestFrame(
                    fft_data=ref_fft, frame_id=0, ok=True,
                    stats=SourceStats(frames_ok=0, last_err="", sclk_hz_rep=0),
                )
                source_label = "REF"
            elif mode == "LOOP":
                latest_frame = spi_loopback.get_latest()
                source_label = "LOOP"
            else:  # "HW"
                latest_frame = spi_hw.get_latest()
                source_label = "HW"

            source_stats = latest_frame.stats
            fft_data = latest_frame.fft_data if latest_frame.ok else None

            # Fallback to last known data if current read failed
            if fft_data is None:
                if source_label in ("HW", "LOOP") and last_spi_fft_data is not None:
                    fft_data = last_spi_fft_data
                else:
                    fft_data = np.zeros((config.N_MICS, config.N_BINS), dtype=np.complex64)
            elif source_label in ("HW", "LOOP"):
                last_spi_fft_data = fft_data

            prof.mark("read_source")

            # ---- Beamforming + Heatmap generation ----
            spec_matrix = None
            band_freqs = np.array([], dtype=np.float64)
            if source_label == "REF":
                # 0 dB reference: uniform heatmap (spread across whole view), no MUSIC blobs
                heatmap_left = np.full((content_height, content_width), 255, dtype=np.uint8)
            elif source_label in ("SIM", "SIM_2"):
                if source_label == "SIM_2":
                    sim_freqs = list(getattr(sim_source, "last_sim2_freqs", []))
                else:
                    sim_freqs = list(config.SIM_SOURCE_FREQS)
                # Filter sources by bandpass
                selected_indices = [
                    i for i, f in enumerate(sim_freqs)
                    if f_min <= f <= f_max
                ]

                if not selected_indices:
                    heatmap_left = np.zeros((content_height, content_width), dtype=np.uint8)
                else:
                    n_sel = len(selected_indices)
                    sim_proj_mode = getattr(config, "HEATMAP_PROJECTION_MODE", "linear")
                    sim_dual_angle = sim_proj_mode == "dual_angle"
                    if sim_dual_angle:
                        n_ang_sim = len(config.ANGLES_2D_X)
                    else:
                        n_ang_sim = len(config.ANGLES)
                    spec_matrix = np.zeros((n_sel, n_ang_sim), dtype=np.float32)
                    angle_x_deg_sim = np.empty(n_sel, dtype=np.float64) if sim_dual_angle else None
                    angle_y_deg_sim = np.empty(n_sel, dtype=np.float64) if sim_dual_angle else None
                    band_freqs = np.array([float(sim_freqs[i]) for i in selected_indices], dtype=np.float64)
                    power = np.zeros(n_sel, dtype=np.float32)

                    running_max_power = 1e-12
                    for row_idx, src_idx in enumerate(selected_indices):
                        f_sig = float(sim_freqs[src_idx])
                        f_idx = int(np.argmin(np.abs(config.f_axis - f_sig)))
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T

                        if sim_dual_angle:
                            spec_2d = music_spectrum_2d(
                                R, config.ANGLES_2D_X, config.ANGLES_2D_Y, f_sig, n_sel,
                                config.x_coords, config.y_coords, config.SPEED_SOUND,
                            )
                            angle_x_deg_sim[row_idx], angle_y_deg_sim[row_idx] = music_2d_peak_angles(
                                spec_2d, config.ANGLES_2D_X, config.ANGLES_2D_Y,
                            )
                            iy = int(np.argmax(spec_2d) % spec_2d.shape[1])
                            spec_matrix[row_idx, :] = spec_2d[:, iy]
                        else:
                            spec_matrix[row_idx, :] = music_spectrum(
                                R, config.ANGLES, f_sig, n_sel,
                                config.x_coords, config.y_coords, config.SPEED_SOUND
                            )
                        p = float(np.sum(np.abs(Xf) ** 2).real)
                        power[row_idx] = p
                        running_max_power = max(running_max_power, p)

                    power_rel = power / (running_max_power + 1e-12)
                    sim_heatmap_kw = dict(
                        spec_matrix=spec_matrix,
                        power_rel=power_rel,
                        out_width=content_width,
                        out_height=content_height,
                        db_min=config.REL_DB_MIN,
                        db_max=config.REL_DB_MAX,
                        x_offset_px=getattr(config, "HEATMAP_X_OFFSET_PX", 0),
                        angle_min_deg=getattr(config, "HEATMAP_ANGLE_MIN_DEG", -90.0),
                        angle_max_deg=getattr(config, "HEATMAP_ANGLE_MAX_DEG", 90.0),
                        band_freqs_hz=band_freqs,
                        f_min_hz=f_min,
                        f_max_hz=f_max,
                        projection_mode=sim_proj_mode,
                        circle_radius_px=getattr(config, "HEATMAP_CIRCLE_RADIUS_PX", 0),
                        assumed_distance_m=getattr(config, "HEATMAP_ASSUMED_DISTANCE_M", 1.0),
                        camera_hfov_deg=getattr(config, "HEATMAP_CAMERA_HFOV_DEG", 53.0),
                        camera_vfov_deg=getattr(config, "HEATMAP_CAMERA_VFOV_DEG", 0.0),
                    )
                    if sim_dual_angle and angle_x_deg_sim is not None and angle_y_deg_sim is not None:
                        sim_heatmap_kw["angle_x_deg"] = angle_x_deg_sim
                        sim_heatmap_kw["angle_y_deg"] = angle_y_deg_sim
                    heatmap_left = spectra_to_heatmap_absolute(**sim_heatmap_kw)

            else:  # SPI mode (HW + LOOP): top-K bins by power within bandpass, above noise floor
                # Per-mic gain correction and whole-array boost, then optional per-mic normalize
                gain = np.asarray(config.SPI_MIC_GAIN, dtype=np.float32).flatten()
                if gain.size < config.N_MICS:
                    gain = np.pad(gain, (0, config.N_MICS - gain.size), constant_values=1.0)
                if gain.size > config.N_MICS:
                    gain = gain[: config.N_MICS]
                mic_gain = gain.reshape(config.N_MICS, 1)
                array_gain = float(config.SPI_ARRAY_GAIN)
                fft_corrected = (fft_data * mic_gain * array_gain).astype(np.complex64)
                if config.SPI_PER_MIC_NORMALIZE:
                    norms = np.sqrt(np.sum(np.abs(fft_corrected) ** 2, axis=1)) + 1e-12
                    fft_for_heatmap = (fft_corrected / norms[:, np.newaxis]).astype(np.complex64)
                else:
                    fft_for_heatmap = fft_corrected
                candidate_bins = np.array([
                    b for b in range(config.N_BINS)
                    if f_min <= float(config.f_axis[b]) <= f_max
                ], dtype=np.intp)
                total_bandpass_power = 0.0
                if len(candidate_bins) == 0:
                    bins = []
                else:
                    power_per_bin = np.sum(np.abs(fft_for_heatmap[:, candidate_bins]) ** 2, axis=0)
                    total_bandpass_power = float(power_per_bin.sum()) + 1e-12
                    p_max = float(power_per_bin.max()) + 1e-12
                    power_db = 10.0 * np.log10((power_per_bin.astype(np.float64) + 1e-12) / p_max)
                    above_floor = power_db >= (-config.SPI_NOISE_FLOOR_DB)
                    candidate_bins = candidate_bins[above_floor]
                    power_per_bin = power_per_bin[above_floor]
                    K = min(config.SPI_TOP_K_BINS, len(candidate_bins))
                    if K <= 0:
                        bins = []
                    else:
                        top_idx = np.argsort(power_per_bin)[::-1][:K]
                        bins = candidate_bins[top_idx].tolist()

                if not bins:
                    frozen_bins = None
                    heatmap_left = np.zeros((content_height, content_width), dtype=np.uint8)
                else:
                    proj_mode = getattr(config, "HEATMAP_PROJECTION_MODE", "linear")
                    use_dual_angle = proj_mode == "dual_angle"
                    coarse_res = max(0, int(getattr(config, "SPI_MUSIC_2D_COARSE_RESOLUTION", 0)))
                    if use_dual_angle and coarse_res > 0:
                        n_ang = coarse_res
                    elif use_dual_angle:
                        n_ang = len(config.ANGLES_2D_X)
                    else:
                        n_ang = len(config.ANGLES)
                    n = len(bins)
                    every_n = max(1, int(getattr(config, "SPI_MUSIC_EVERY_N_FRAMES", 1)))
                    if use_dual_angle and every_n > 1 and (frame_count % every_n) != 0 and frozen_bins is not None and len(frozen_bins) > 0:
                        bins = frozen_bins
                        n = len(bins)
                    if heatmap_spec_buf is None or heatmap_spec_buf.shape[0] < n or heatmap_spec_buf.shape[1] < n_ang:
                        heatmap_spec_buf = np.zeros((_heatmap_max_bins, _heatmap_max_n_ang), dtype=np.float32)
                        heatmap_angle_x_buf = np.empty(_heatmap_max_bins, dtype=np.float64)
                        heatmap_angle_y_buf = np.empty(_heatmap_max_bins, dtype=np.float64)
                        heatmap_power_buf = np.zeros(_heatmap_max_bins, dtype=np.float32)
                        heatmap_band_freqs_buf = np.zeros(_heatmap_max_bins, dtype=np.float64)
                    spec_matrix = heatmap_spec_buf[:n, :n_ang]
                    spec_matrix.fill(0)
                    angle_x_deg = heatmap_angle_x_buf[:n] if use_dual_angle else None
                    angle_y_deg = heatmap_angle_y_buf[:n] if use_dual_angle else None
                    for j, b in enumerate(bins):
                        heatmap_band_freqs_buf[j] = float(config.f_axis[b])
                    band_freqs = heatmap_band_freqs_buf[:n]
                    power = heatmap_power_buf[:n]
                    power.fill(0)

                    skip_music = (
                        use_dual_angle
                        and every_n > 1
                        and (frame_count % every_n) != 0
                        and last_music_bins_list is not None
                        and len(bins) == len(last_music_bins_list)
                        and all(bins[j] == last_music_bins_list[j] for j in range(len(bins)))
                        and last_music_spec_cache is not None
                        and last_music_angle_x_cache is not None
                    )

                    n_avg = max(1, int(getattr(config, "SPI_COV_AVG_FRAMES", 1)))
                    alpha = 1.0 / n_avg
                    if skip_music:
                        heatmap_angle_x_buf[:n] = last_music_angle_x_cache[:n]
                        heatmap_angle_y_buf[:n] = last_music_angle_y_cache[:n]
                        heatmap_spec_buf[:n, :n_ang] = last_music_spec_cache[:n, :n_ang]
                        for i, f_idx in enumerate(bins):
                            Xf = fft_for_heatmap[:, f_idx][:, np.newaxis]
                            power[i] = float(np.sum(np.abs(Xf) ** 2).real)
                    else:
                        for i, f_idx in enumerate(bins):
                            f_sig = float(config.f_axis[f_idx])
                            Xf = fft_for_heatmap[:, f_idx][:, np.newaxis]
                            R = Xf @ Xf.conj().T
                            if n_avg > 1:
                                if f_idx not in cov_avg:
                                    cov_avg[f_idx] = R.copy()
                                else:
                                    cov_avg[f_idx] = (1.0 - alpha) * cov_avg[f_idx] + alpha * R
                                R_use = cov_avg[f_idx]
                            else:
                                R_use = R

                            if use_dual_angle:
                                if coarse_res > 0:
                                    spec_coarse, angle_x_deg[i], angle_y_deg[i] = music_spectrum_2d_refined(
                                        R_use, config.ANGLES_2D_X, config.ANGLES_2D_Y, f_sig,
                                        config.SPI_MUSIC_N_SOURCES, config.x_coords_hw, config.y_coords_hw,
                                        config.SPEED_SOUND,
                                        coarse_resolution=coarse_res,
                                        refine_half_width=int(getattr(config, "SPI_MUSIC_2D_REFINE_HALF_WIDTH", 2)),
                                    )
                                    iy = int(np.argmax(spec_coarse) % spec_coarse.shape[1])
                                    spec_matrix[i, :] = spec_coarse[:, iy]
                                else:
                                    spec_2d = music_spectrum_2d(
                                        R_use, config.ANGLES_2D_X, config.ANGLES_2D_Y, f_sig,
                                        config.SPI_MUSIC_N_SOURCES, config.x_coords_hw, config.y_coords_hw,
                                        config.SPEED_SOUND,
                                    )
                                    angle_x_deg[i], angle_y_deg[i] = music_2d_peak_angles(
                                        spec_2d, config.ANGLES_2D_X, config.ANGLES_2D_Y,
                                    )
                                    iy = int(np.argmax(spec_2d) % spec_2d.shape[1])
                                    spec_matrix[i, :] = spec_2d[:, iy]
                            else:
                                spec_matrix[i, :] = music_spectrum(
                                    R_use, config.ANGLES, f_sig, config.SPI_MUSIC_N_SOURCES,
                                    config.x_coords_hw, config.y_coords_hw, config.SPEED_SOUND
                                )
                            power[i] = float(np.sum(np.abs(Xf) ** 2).real)
                            if config.SPI_DIRECTIVITY_MIN > 0:
                                dr = directivity_ratio(R_use)
                                if dr < config.SPI_DIRECTIVITY_MIN:
                                    power[i] = 0.0
                        if use_dual_angle:
                            if last_music_angle_x_cache is None or last_music_spec_cache is None:
                                last_music_angle_x_cache = np.empty(_heatmap_max_bins, dtype=np.float64)
                                last_music_angle_y_cache = np.empty(_heatmap_max_bins, dtype=np.float64)
                                last_music_spec_cache = np.zeros((_heatmap_max_bins, _heatmap_max_n_ang), dtype=np.float32)
                            last_music_angle_x_cache[:n] = heatmap_angle_x_buf[:n]
                            last_music_angle_y_cache[:n] = heatmap_angle_y_buf[:n]
                            last_music_spec_cache[:n, :n_ang] = heatmap_spec_buf[:n, :n_ang]
                            last_music_bins_list = list(bins)
                            frozen_bins = list(bins)

                    prof.mark("heat_music")
                    # Peak-angle stability: use angle_x_deg when dual_angle, else 1D peak from spec_matrix
                    if config.SPI_ANGLE_STABILITY_DEG > 0 and len(bins) > 0:
                        if use_dual_angle and angle_x_deg is not None:
                            current_angles = [float(angle_x_deg[j]) for j in range(len(bins))]
                        else:
                            peak_idx_per_bin = np.argmax(spec_matrix, axis=1)
                            ang_arr = config.ANGLES_2D_X if use_dual_angle else config.ANGLES
                            current_angles = [float(ang_arr[int(peak_idx_per_bin[j])]) for j in range(len(bins))]
                        if last_spi_bins is not None and last_spi_peak_angles is not None:
                            last_bin_to_angle = dict(zip(last_spi_bins, last_spi_peak_angles))
                            for j in range(len(bins)):
                                prev_angle = last_bin_to_angle.get(bins[j])
                                if prev_angle is not None and abs(current_angles[j] - prev_angle) > config.SPI_ANGLE_STABILITY_DEG:
                                    power[j] = 0.0
                        last_spi_bins = list(bins)
                        last_spi_peak_angles = list(current_angles)
                    elif len(bins) > 0:
                        if use_dual_angle and angle_x_deg is not None:
                            last_spi_peak_angles = [float(angle_x_deg[j]) for j in range(len(bins))]
                        else:
                            peak_idx_per_bin = np.argmax(spec_matrix, axis=1)
                            ang_arr = config.ANGLES_2D_X if use_dual_angle else config.ANGLES
                            last_spi_peak_angles = [float(ang_arr[int(peak_idx_per_bin[j])]) for j in range(len(bins))]
                        last_spi_bins = list(bins)
                    else:
                        last_spi_bins = None
                        last_spi_peak_angles = None

                    prof.mark("heat_stability")
                    # Per-frame normalization; gamma > 1 makes strongest bins stand out more
                    power_rel = power / (power.max() + 1e-12)
                    power_rel = np.power(power_rel, config.SPI_HEATMAP_POWER_GAMMA)
                    heatmap_kw = dict(
                        spec_matrix=spec_matrix,
                        power_rel=power_rel,
                        out_width=content_width,
                        out_height=content_height,
                        db_min=config.REL_DB_MIN,
                        db_max=config.REL_DB_MAX,
                        x_offset_px=getattr(config, "HEATMAP_X_OFFSET_PX", 0),
                        angle_min_deg=getattr(config, "HEATMAP_ANGLE_MIN_DEG", -90.0),
                        angle_max_deg=getattr(config, "HEATMAP_ANGLE_MAX_DEG", 90.0),
                        band_freqs_hz=band_freqs,
                        f_min_hz=f_min,
                        f_max_hz=f_max,
                        projection_mode=getattr(config, "HEATMAP_PROJECTION_MODE", "linear"),
                        circle_radius_px=getattr(config, "HEATMAP_CIRCLE_RADIUS_PX", 0),
                        assumed_distance_m=getattr(config, "HEATMAP_ASSUMED_DISTANCE_M", 1.0),
                        camera_hfov_deg=getattr(config, "HEATMAP_CAMERA_HFOV_DEG", 53.0),
                        camera_vfov_deg=getattr(config, "HEATMAP_CAMERA_VFOV_DEG", 0.0),
                    )
                    if use_dual_angle and angle_x_deg is not None and angle_y_deg is not None:
                        heatmap_kw["angle_x_deg"] = angle_x_deg
                        heatmap_kw["angle_y_deg"] = angle_y_deg
                    if heatmap_heat_buf is None or heatmap_heat_buf.shape != (content_height, content_width):
                        heatmap_heat_buf = np.zeros((content_height, content_width), dtype=np.float32)
                    heatmap_kw["heat_out"] = heatmap_heat_buf
                    heatmap_left = spectra_to_heatmap_absolute(**heatmap_kw)
                    prof.mark("heat_draw")
                # Scale heatmap by bandpass level so it brightens with sound, dims when quiet (floor keeps blobs visible)
                level = max(
                    config.HEATMAP_LEVEL_FLOOR,
                    min(1.0, total_bandpass_power / config.HEATMAP_LEVEL_REFERENCE),
                )
                # Per-frame contrast stretch: percentile from level-scaled subsample so stretch matches original visibility
                pct = config.HEATMAP_CONTRAST_STRETCH_PERCENTILE
                p_val = 0.0
                if pct > 0 and heatmap_left.size > 0:
                    k = max(1, int(getattr(config, "HEATMAP_STRETCH_SUBSAMPLE", 1)))
                    sub = heatmap_left[::k, ::k]
                    sub_scaled = (sub.astype(np.float32) * level).clip(0, 255).astype(np.uint8)
                    p_val = percentile_uint8_fast(sub_scaled, pct)
                if pct > 0 and p_val > 1e-6:
                    heatmap_left = (heatmap_left.astype(np.float32) * level * (255.0 / p_val)).clip(0, 255).astype(np.uint8)
                else:
                    heatmap_left = (heatmap_left.astype(np.float32) * level).astype(np.uint8)
                prof.mark("heat_scale")
                # Temporal smoothing for SPI: blend with previous frame
                if source_label in ("HW", "LOOP") and heatmap_prev is not None and heatmap_prev.shape == heatmap_left.shape:
                    heatmap_left = (
                        config.HEATMAP_SMOOTH_ALPHA * heatmap_prev.astype(np.float32)
                        + (1.0 - config.HEATMAP_SMOOTH_ALPHA) * heatmap_left.astype(np.float32)
                    ).astype(np.uint8)

            prof.mark("beamform+heatmap")

            # ---- Append one acoustic detection sample for radar history ----
            if spec_matrix is not None and spec_matrix.size > 0:
                try:
                    t_now = time.time()
                    _n_rows, n_ang = spec_matrix.shape
                    detections: list[tuple[float, float]] = []  # (rel_angle_deg, db_value)
                    row_max = np.max(spec_matrix, axis=1).astype(np.float64)
                    row_ref = float(np.max(row_max)) if row_max.size > 0 else 0.0
                    if row_ref <= 1e-12:
                        row_ref = 1e-12

                    if str(source_label) == "SIM_2":
                        min_rel = float(getattr(config, "SIM2_RADAR_MIN_ROW_REL", 0.30))
                        sep_min = float(getattr(config, "SIM2_RADAR_MIN_SEPARATION_DEG", 12.0))
                        order = np.argsort(-row_max)  # descending by row strength
                        for ri in order.tolist():
                            rel_strength = float(row_max[ri]) / row_ref
                            if rel_strength < min_rel:
                                continue
                            ci = int(np.argmax(spec_matrix[ri, :]))
                            if len(config.ANGLES) == n_ang:
                                rel_angle = float(config.ANGLES[ci])
                            else:
                                rel_angle = float(-90.0 + (180.0 * ci / max(1, n_ang - 1)))
                            # Skip near-duplicate detections in same frame.
                            if any(abs(rel_angle - det[0]) < sep_min for det in detections):
                                continue
                            db_row = float(config.REL_DB_MIN + rel_strength * (config.REL_DB_MAX - config.REL_DB_MIN))
                            detections.append((rel_angle, db_row))
                    else:
                        _ri, _ci = np.unravel_index(int(np.argmax(spec_matrix)), spec_matrix.shape)
                        if len(config.ANGLES) == n_ang:
                            rel_angle = float(config.ANGLES[_ci])
                        else:
                            rel_angle = float(-90.0 + (180.0 * _ci / max(1, n_ang - 1)))
                        peak_u8 = int(np.max(heatmap_left))
                        peak_db = float(
                            config.REL_DB_MIN + (peak_u8 / 255.0) * (config.REL_DB_MAX - config.REL_DB_MIN)
                        )
                        detections.append((rel_angle, peak_db))

                    for rel_angle, peak_db in detections:
                        world_bearing = (float(HUD.compass_heading_deg) + float(rel_angle)) % 360.0
                        if bool(getattr(button_state, "record_compass_history", False)):
                            directional_store.add_event(
                                {
                                    "timestamp": t_now,
                                    "source_mode": str(button_state.source_mode),
                                    "bearing_world_deg": float(world_bearing),
                                    "bearing_rel_deg": float(rel_angle),
                                    "db_value": float(peak_db),
                                    "heading_deg": float(HUD.compass_heading_deg),
                                    "position_source": str(HUD.position.source),
                                    "lat": HUD.position.lat,
                                    "lon": HUD.position.lon,
                                    "accuracy_m": HUD.position.accuracy_m,
                                }
                            )
                            HUD.directional_log_error = ""
                            HUD.directional_log_last_write_s = t_now
                            HUD.directional_log_date = str(getattr(directional_store, "_current_date", ""))
                            _cur_file = getattr(directional_store, "_current_file", None)
                            HUD.directional_log_file = str(_cur_file) if _cur_file else ""
                        if peak_db >= float(getattr(config, "RADAR_MIN_DB", -45.0)):
                            update_detection_history(
                                rel_angle_deg=rel_angle,
                                db_value=peak_db,
                                heading_deg=HUD.compass_heading_deg,
                                now_s=t_now,
                            )
                except Exception as exc:
                    HUD.directional_log_error = str(exc)

            # ---- Background (camera or static) ----
            if button_state.camera_enabled and state.CAMERA_AVAILABLE:
                cam_frame = camera_mgr.get_latest_frame()
                if cam_frame is not None and getattr(cam_frame, "size", 0) > 0:
                    try:
                        if camera_mgr.camera_type == "picamera2":
                            # already BGR
                            if cam_frame.shape[1] == config.WIDTH and cam_frame.shape[0] == config.HEIGHT:
                                base_frame[:] = cam_frame
                            else:
                                cv2.resize(cam_frame, (config.WIDTH, config.HEIGHT),
                                        dst=base_frame, interpolation=cv2.INTER_LINEAR)
                        else:
                            # opencv backend already BGR
                            if cam_frame.shape[:2] == (config.HEIGHT, config.WIDTH):
                                base_frame[:] = cam_frame
                            else:
                                cv2.resize(cam_frame, (config.WIDTH, config.HEIGHT), dst=base_frame, interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        base_frame[:] = background_full
                else:
                    base_frame[:] = background_full
            else:
                base_frame[:] = background_full

            prof.mark("background")

            # ---- Blend heatmap onto background ----
            output_frame = blend_heatmap_left(base_frame, heatmap_left, content_width, content_height, content_offset_x, w_lut_u8, button_state.colormap_mode)
            prof.mark("blend")
            if source_label in ("HW", "LOOP"):
                heatmap_prev = heatmap_left.copy()
            else:
                heatmap_prev = None

            # ---- Process pending spectrum cursor (tap or dot drag): snap to closest point on blue curve ----
            if state.SPECTRUM_CURSOR_PENDING_TAP_Y is not None and fft_data is not None:
                use_db = button_state.spectrum_analyzer_mode in ("dB", "dBA")
                cursor_x_for_snap = (
                    state.SPECTRUM_CURSOR_PENDING_TAP_X
                    if state.SPECTRUM_CURSOR_PENDING_TAP_X is not None
                    else (state.SPECTRUM_CURSOR_X or (config.FREQ_BAR_WIDTH // 2))
                )
                curve_x, dot_freq = spectrum_closest_curve_point(
                    cursor_x_for_snap,
                    state.SPECTRUM_CURSOR_PENDING_TAP_Y,
                    config.HEIGHT,
                    config.F_DISPLAY_MAX,
                    fft_data,
                    config.f_axis,
                    config.FREQ_BAR_WIDTH,
                    use_db=use_db,
                    mode=button_state.spectrum_analyzer_mode,
                )
                state.SPECTRUM_CURSOR_X = curve_x
                state.SPECTRUM_CURSOR_DOT_FREQ = dot_freq
                state.SPECTRUM_CURSOR_PENDING_TAP_X = None
                state.SPECTRUM_CURSOR_PENDING_TAP_Y = None

            # ---- Draw spectrum analyzer (dB scale + curve) and dB colorbar ----
            spectrum_cursor_dot_bar_pos = []
            draw_spectrum_analyzer(
                output_frame, fft_data, config.f_axis, f_min, f_max,
                config.FREQ_BAR_WIDTH, config.F_DISPLAY_MAX,
                mode=button_state.spectrum_analyzer_mode,
                spectrum_cursor_x=state.SPECTRUM_CURSOR_X,
                spectrum_cursor_dot_active=state.SPECTRUM_CURSOR_DOT_ACTIVE,
                spectrum_cursor_dot_freq=state.SPECTRUM_CURSOR_DOT_FREQ,
                spectrum_cursor_dot_dragging=state.SPECTRUM_CURSOR_DOT_DRAG_ACTIVE,
                spectrum_cursor_dot_bar_pos=spectrum_cursor_dot_bar_pos,
            )
            if len(spectrum_cursor_dot_bar_pos) == 2:
                state.SPECTRUM_CURSOR_DOT_BAR_X = spectrum_cursor_dot_bar_pos[0]
                state.SPECTRUM_CURSOR_DOT_BAR_Y = spectrum_cursor_dot_bar_pos[1]
            else:
                state.SPECTRUM_CURSOR_DOT_BAR_X = None
                state.SPECTRUM_CURSOR_DOT_BAR_Y = None
            draw_db_colorbar(output_frame, config.REL_DB_MIN, config.REL_DB_MAX, config.DB_BAR_WIDTH, colormap=button_state.colormap_mode)
            prof.mark("bars")

            # ---- Crosshairs on heatmap (5mm cross, local freq at this blob, click to show/dismiss, auto-tracking) ----
            if button_state.crosshairs_enabled and button_state.crosshair_visible:
                # Auto-tracking: stick to local max in heatmap
                cx = int(button_state.crosshair_x)
                cy = int(button_state.crosshair_y)
                nx, ny = find_local_max(heatmap_left, cx, cy, CROSSHAIR_TRACK_RADIUS)
                # If blob left the screen (edge) or faded (very low intensity), hide crosshair until next click
                edge_margin = 3
                blob_gone = (
                    nx < edge_margin or nx >= content_width - edge_margin
                    or ny < edge_margin or ny >= content_height - edge_margin
                    or int(heatmap_left[ny, nx]) < 12
                )
                if blob_gone:
                    button_state.crosshair_visible = False
                else:
                    button_state.crosshair_x = float(nx)
                    button_state.crosshair_y = float(ny)
                    # Current dB at tracked pixel for trend
                    val_at = int(heatmap_left[ny, nx])
                    db_at = config.REL_DB_MIN + (val_at / 255.0) * (config.REL_DB_MAX - config.REL_DB_MIN)
                    now_t = time.time()
                    # Append to level history (keep 60 s)
                    button_state.crosshair_level_history.append((now_t, db_at))
                    max_history_t = 60.0
                    button_state.crosshair_level_history = [
                        (t, d) for t, d in button_state.crosshair_level_history if t >= now_t - max_history_t
                    ]
                    # 3 s trend: baseline = mean of [boundary-3, boundary], current = mean of last 3 s
                    trend_db = None
                    if button_state.crosshair_next_boundary_time <= 0:
                        button_state.crosshair_next_boundary_time = now_t + 3.0
                    elif now_t >= button_state.crosshair_next_boundary_time:
                        boundary = button_state.crosshair_next_boundary_time
                        prev_3 = [(t, d) for t, d in button_state.crosshair_level_history if boundary - 3 <= t < boundary]
                        if prev_3:
                            baseline = sum(d for _, d in prev_3) / len(prev_3)
                            if button_state.crosshair_prev_baseline_db is not None:
                                trend_db = baseline - button_state.crosshair_prev_baseline_db
                                button_state.crosshair_trend_history.append(trend_db)
                                if len(button_state.crosshair_trend_history) > 4:
                                    button_state.crosshair_trend_history = button_state.crosshair_trend_history[-4:]
                            button_state.crosshair_prev_baseline_db = baseline
                        button_state.crosshair_next_boundary_time = boundary + 3.0
                    # Live trend: current 3 s mean vs previous 3 s mean
                    if trend_db is None and button_state.crosshair_level_history:
                        recent = [(t, d) for t, d in button_state.crosshair_level_history if t >= now_t - 3]
                        if recent and button_state.crosshair_prev_baseline_db is not None:
                            cur_mean = sum(d for _, d in recent) / len(recent)
                            trend_db = cur_mean - button_state.crosshair_prev_baseline_db
                    # 12 s acceleration: slope of trend over last 4 samples (4 × 3 s)
                    accel_db = None
                    if len(button_state.crosshair_trend_history) >= 2:
                        th = button_state.crosshair_trend_history
                        accel_db = (th[-1] - th[0]) / max(1, len(th) - 1)  # dB per 3 s period
                    # Frequency dominant at this location; angle for protractor; SIM: distance from closest source
                    distance_to_source_m = None
                    angle_deg = None
                    if spec_matrix is not None and band_freqs.size > 0:
                        n_ang = spec_matrix.shape[1]
                        proj_mode = getattr(config, "HEATMAP_PROJECTION_MODE", "linear")
                        if proj_mode == "camera_circle":
                            center_x = (content_width - 1) / 2.0
                            center_y = (content_height - 1) / 2.0
                            # Reverse rotation: display angle = DOA + 90°, so DOA = display - 90°
                            angle_rad = np.arctan2(center_y - ny, nx - center_x)
                            angle_deg = float(np.degrees(angle_rad - np.pi / 2.0))
                            angle_idx = int(np.clip(round((angle_deg + 90.0) / 180.0 * (n_ang - 1)), 0, n_ang - 1))
                        elif proj_mode == "camera_plane":
                            cx = (content_width - 1) / 2.0
                            hfov_deg = getattr(config, "HEATMAP_CAMERA_HFOV_DEG", 53.0)
                            hfov_rad = np.deg2rad(max(1e-6, float(hfov_deg)))
                            fx = (content_width / 2.0) / np.tan(hfov_rad / 2.0)
                            sin_theta = np.clip((nx - cx) / max(1e-12, fx), -1.0, 1.0)
                            cos_theta = np.sqrt(1.0 - sin_theta * sin_theta)
                            angle_rad = np.arctan2(sin_theta, cos_theta)
                            angle_deg = float(np.degrees(angle_rad))
                            angle_idx = int(np.clip(round((angle_deg + 90.0) / 180.0 * (n_ang - 1)), 0, n_ang - 1))
                            # y = frequency: t_y = 1 - ny/(h-1), freq = f_min + t_y*(f_max - f_min)
                            t_y = 1.0 - ny / max(1, content_height - 1)
                            t_y = np.clip(t_y, 0.0, 1.0)
                            f_peak_hz = float(f_min + t_y * (f_max - f_min))
                            row = int(np.clip(np.argmin(np.abs(band_freqs - f_peak_hz)), 0, spec_matrix.shape[0] - 1))
                        elif proj_mode == "dual_angle":
                            x_off = getattr(config, "HEATMAP_X_OFFSET_PX", 0)
                            ang_min = getattr(config, "HEATMAP_ANGLE_MIN_DEG", -90.0)
                            ang_max = getattr(config, "HEATMAP_ANGLE_MAX_DEG", 90.0)
                            span = max(1e-6, ang_max - ang_min)
                            t_x = np.clip((nx - x_off) / max(1, content_width - 1), 0.0, 1.0)
                            t_y = np.clip(ny / max(1, content_height - 1), 0.0, 1.0)  # matches flipped y: top=angle_min, bottom=angle_max
                            angle_x_deg = ang_min + t_x * span
                            angle_y_deg = ang_min + t_y * span
                            angle_deg = float(angle_x_deg)  # primary for protractor / tooltip
                            angle_idx = int(np.clip(round(t_x * (n_ang - 1)), 0, n_ang - 1))
                            row = int(np.argmax(spec_matrix[:, angle_idx]))
                            f_peak_hz = float(band_freqs[row])
                        else:
                            x_off = getattr(config, "HEATMAP_X_OFFSET_PX", 0)
                            ang_min = getattr(config, "HEATMAP_ANGLE_MIN_DEG", -90.0)
                            ang_max = getattr(config, "HEATMAP_ANGLE_MAX_DEG", 90.0)
                            t = (nx - x_off) / max(1, content_width - 1)
                            t = np.clip(t, 0.0, 1.0)
                            angle_deg = ang_min + t * (ang_max - ang_min)
                            angle_idx = int(np.clip(round((angle_deg + 90.0) / 180.0 * (n_ang - 1)), 0, n_ang - 1))
                        if proj_mode != "camera_plane":
                            row = int(np.argmax(spec_matrix[:, angle_idx]))
                            f_peak_hz = float(band_freqs[row])
                        if source_label == "SIM":
                            sim_dists = getattr(config, "SIM_SOURCE_DISTANCES_M", None)
                            if sim_dists and len(sim_dists) == len(config.SIM_SOURCE_ANGLES):
                                closest = int(np.argmin(np.abs(np.array(config.SIM_SOURCE_ANGLES) - angle_deg)))
                                distance_to_source_m = float(sim_dists[closest])
                    else:
                        f_peak_hz = (f_min + f_max) / 2.0
                    draw_crosshairs(
                        output_frame, nx, ny, content_width, content_height,
                        heatmap_left, config.REL_DB_MIN, config.REL_DB_MAX, f_peak_hz,
                        trend_db=trend_db, accel_db=accel_db,
                        distance_to_source_m=distance_to_source_m,
                        angle_deg=angle_deg,
                        content_offset_x=content_offset_x,
                    )

            # ---- Draw debug info ----
            if button_state.debug_enabled:
                # Collect debug text lines (abbreviated to save space)
                debug_lines = [
                    f"Frame: {frame_count}  t={elapsed:.2f}s",
                    f"Source: {source_label}",
                ]

                if source_label in ("HW", "LOOP"):
                    mhz = (source_stats.sclk_hz_rep / 1e6) if source_stats.sclk_hz_rep else 0
                    # One frame = full 16-mic payload (header + payload + trailer)
                    bytes_per_s = config.FRAME_BYTES * fps_ema
                    mbps_bytes = bytes_per_s / 1e6
                    mbps_bits = (bytes_per_s * 8) / 1e6

                    debug_lines.extend([
                        f"SPI {mhz:.0f}MHz  FPS: {fps_ema:5.1f}",
                        f"ok:{source_stats.frames_ok} badParse:{source_stats.bad_parse} badCRC:{source_stats.bad_crc}",
                        f"Throughput: {mbps_bytes:.2f}MB/s ({mbps_bits:.1f}Mb/s)",
                    ])
                    if source_stats.last_err:
                        debug_lines.append(f"Err: {source_stats.last_err[:40]}")

                # Calculate box dimensions
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45
                font_thickness = 1
                line_height = 18
                padding = 12

                # Calculate max available width (don't extend beyond menu button)
                # Menu is at: content_right - 100 - 15
                max_available_width = content_width - 135

                # Measure max text width
                max_text_width = 0
                for line in debug_lines:
                    (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
                    max_text_width = max(max_text_width, text_w)

                # Constrain to max available width
                box_w = min(max_text_width + 2 * padding, max_available_width)
                box_h = len(debug_lines) * line_height + 2 * padding

                # Position above bottom HUD pills with clear gap (no overlap)
                box_x = config.DB_BAR_WIDTH + 10
                debug_above_bottom_gap = 18  # space between debug box bottom and top of bottom HUD pills
                box_y = (
                    config.HEIGHT
                    - box_h
                    - BOTTOM_HUD_HEIGHT
                    - debug_above_bottom_gap
                    + int(state.ui_bottom_hud_offset)
                )
                # If bottom HUD is in hide mode, add proportional extra shift so
                # the full debug panel exits the frame at full hide progress.
                if state.ui_bottom_hud_offset_target > 0:
                    base_hide = float(getattr(config, "UI_BOTTOM_HUD_HIDE_OFFSET", 60))
                    needed_hide = float(box_h + BOTTOM_HUD_HEIGHT + debug_above_bottom_gap)
                    extra_hide = max(0.0, needed_hide - base_hide)
                    progress = float(state.ui_bottom_hud_offset) / max(1e-6, float(state.ui_bottom_hud_offset_target))
                    progress = float(np.clip(progress, 0.0, 1.0))
                    box_y += int(round(extra_hide * progress))
                # region agent log
                if (frame_count % 20) == 0:
                    _agent_debug_log(
                        "run1",
                        "H1",
                        "main.py:debug_box",
                        "debug_box_position_sample",
                        {
                            "frame": int(frame_count),
                            "debug_enabled": bool(button_state.debug_enabled),
                            "box_y": int(box_y),
                            "box_h": int(box_h),
                            "box_bottom": int(box_y + box_h),
                            "screen_h": int(config.HEIGHT),
                            "bottom_offset": float(state.ui_bottom_hud_offset),
                            "bottom_target": float(state.ui_bottom_hud_offset_target),
                            "hide_offset_cfg": float(config.UI_BOTTOM_HUD_HIDE_OFFSET),
                            "hide_progress": float(
                                np.clip(
                                    float(state.ui_bottom_hud_offset)
                                    / max(1e-6, float(state.ui_bottom_hud_offset_target))
                                    if state.ui_bottom_hud_offset_target > 0
                                    else 0.0,
                                    0.0,
                                    1.0,
                                )
                            ),
                        },
                    )
                # endregion

                # Draw semi-transparent grey background box
                overlay = output_frame.copy()
                cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
                cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)

                # Draw border
                cv2.rectangle(output_frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 2, cv2.LINE_AA)

                # Draw text lines
                text_x = box_x + padding
                text_y = box_y + padding + 13
                for line in debug_lines:
                    cv2.putText(output_frame, line, (text_x, text_y),
                               font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    text_y += line_height

            # ---- UI visibility animation (lerp offsets toward targets) ----
            speed = config.UI_VISIBILITY_ANIM_SPEED
            state.ui_top_hud_offset += (state.ui_top_hud_offset_target - state.ui_top_hud_offset) * speed
            state.ui_bottom_hud_offset += (state.ui_bottom_hud_offset_target - state.ui_bottom_hud_offset) * speed
            state.ui_menu_offset += (state.ui_menu_offset_target - state.ui_menu_offset) * speed
            state.ui_menu_offset_y += (state.ui_menu_offset_y_target - state.ui_menu_offset_y) * speed

            # ---- Draw UI buttons ----
            draw_buttons(output_frame)
            if not button_state.gallery_open:
                draw_bottom_hud(output_frame, video_recorder, offset_y=state.ui_bottom_hud_offset)
            draw_menu(output_frame, offset_x=state.ui_menu_offset, offset_y=state.ui_menu_offset_y)

            wifi_ssid, ip_addr, device_name = get_system_network_info(frame_count)
            HUD.connected_ssid = (wifi_ssid or "").strip()
            hud_rects = draw_hud(
                output_frame,
                details_level=HUD.details_level,
                open_panel=HUD.open_panel,
                fps_ema=fps_ema,
                elapsed_s=elapsed,
                frame_count=frame_count,
                source_label=source_label,
                source_stats=source_stats,
                fps_mode=button_state.fps_mode,
                frame_bytes=config.FRAME_BYTES,
                offset_y=state.ui_top_hud_offset,
                battery_percent=None,  # placeholder until live data
                time_remaining_sec=None,  # from battery hardware when available
                wifi_connection_name=wifi_ssid or None,
                ip_address=ip_addr or None,
                device_name=device_name or None,
            )

            state.HUD_RECTS = hud_rects

            # Radar mini-map under Mb/s pill (main view only, before modal overlays).
            if not button_state.gallery_open and button_state.radar_ui_enabled:
                draw_radar_map_widget(
                    output_frame,
                    hud_rects.net,
                    heading_deg=HUD.compass_heading_deg,
                    colormap_mode=button_state.colormap_mode,
                    tile_style=button_state.map_tile_style,
                    show_debug=button_state.show_radar_debug,
                    hud_offset_y=state.ui_bottom_hud_offset,
                    now_s=time.time(),
                )

            # WiFi modal drawn after top HUD
            if HUD.wifi_modal_open:
                draw_wifi_modal(output_frame)

            # Settings modal drawn after top HUD
            if HUD.settings_modal_open:
                draw_settings_modal(output_frame)

            # Email modal drawn after top HUD so its dim overlay covers the HUD (higher z-order)
            if button_state.email_settings_modal_open:
                from acoustic_imager.ui.email_modal import draw_email_modal
                draw_email_modal(output_frame, state.OUTPUT_DIR)

            # Firmware Flash modal
            if button_state.firmware_flash_modal_open:
                draw_firmware_flash_modal(output_frame)

            # Calibration Suite modal
            if button_state.calibration_suite_modal_open:
                draw_calibration_suite_modal(output_frame)

            # ---- Draw gallery view if open ----
            if button_state.gallery_open:
                draw_gallery_view(output_frame, state.OUTPUT_DIR)

            # ---- Draw screenshot flash effect ----
            draw_screenshot_flash(output_frame)

            # ---- Battery icon (in time HUD pill when main view; gallery draws its own) ----
            if button_state.gallery_open:
                draw_battery_icon_for_view(output_frame, percent=None)  # None = placeholder until live data

            prof.mark("ui")

            # ---- Store current frame for screenshots ----
            state.CURRENT_FRAME = output_frame.copy()
            # Keep calibration-suite background cache updated when modal is closed
            if not button_state.calibration_suite_modal_open and output_frame is not None and hasattr(output_frame, "shape") and output_frame.shape == (config.HEIGHT, config.WIDTH, 3):
                state.CALIBRATION_SUITE_BACKGROUND_FRAME = output_frame.copy()
            prof.mark("copy_frame")

            # ---- Record video if active ----
            if button_state.is_recording and video_recorder:
                video_recorder.write_frame(output_frame)
            prof.mark("record")

            # ---- Display frame ----
            cv2.imshow(config.WINDOW_NAME, output_frame)
            prof.mark("imshow")

            # ---- Handle keyboard input ----
            key = cv2.waitKey(1) & 0xFF
            prof.mark("waitKey")

            prof.end_frame()

            # ---- Print profiler stats periodically ----
            if (frame_count % PRINT_EVERY) == 0 and frame_count > 0:
                print(
                    "ms avg | "
                    f"read={prof.ms('read_source'):.2f} "
                    f"heat_music={prof.ms('heat_music'):.2f} "
                    f"heat_stability={prof.ms('heat_stability'):.2f} "
                    f"heat_draw={prof.ms('heat_draw'):.2f} "
                    f"heat_scale={prof.ms('heat_scale'):.2f} "
                    f"heat={prof.ms('beamform+heatmap'):.2f} "
                    f"bg={prof.ms('background'):.2f} "
                    f"blend={prof.ms('blend'):.2f} "
                    f"bars={prof.ms('bars'):.2f} "
                    f"ui={prof.ms('ui'):.2f} "
                    f"imshow={prof.ms('imshow'):.2f} "
                    f"waitKey={prof.ms('waitKey'):.2f} "
                    f"total={prof.ms('frame_total'):.2f}"
                )

            # ---- Check for exit ----
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27 and HUD.wifi_modal_open:
                HUD.wifi_modal_open = False
            elif key == 27 and HUD.settings_modal_open:
                HUD.settings_modal_open = False
            elif key == 27 and button_state.email_settings_modal_open:
                # ESC with email modal open: return to System Settings
                button_state.email_settings_modal_open = False
                HUD.settings_modal_open = True
                button_state.email_modal_screen = "provider"
                button_state.email_modal_provider = ""
                button_state.email_test_status = ""
                button_state.email_test_message = ""
            elif key == 27 and button_state.firmware_flash_modal_open:
                button_state.firmware_flash_modal_open = False
                button_state.firmware_flash_status = ""
            elif key == 27 and button_state.calibration_suite_modal_open:
                button_state.calibration_suite_modal_open = False
            elif key == ord("c"):
                # Reset magnetometer calibration extrema (use during heading tests)
                HUD.mag_x_min = HUD.mag_x_max = None
                HUD.mag_y_min = HUD.mag_y_max = None
                HUD.mag_z_min = HUD.mag_z_max = None
                HUD.mag_span_x = HUD.mag_span_y = HUD.mag_span_z = 0
                HUD.mag_cal_active = False
                print("[compass] calibration extrema reset")
            elif key == ord("q") or key == 27:  # 'q' or ESC
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up...")

        # Stop and cleanup resources
        if video_recorder:
            video_recorder.cleanup()

        try:
            mag_reader.stop()
        except Exception:
            pass
        try:
            gps_reader.stop()
        except Exception:
            pass
        try:
            position_manager.stop()
        except Exception:
            pass
        try:
            directional_store.close()
        except Exception:
            pass

        spi_loopback.stop()
        spi_hw.stop()

        camera_mgr.close()

        cv2.destroyAllWindows()
        print("Shutdown complete.")


if __name__ == "__main__":
    main()

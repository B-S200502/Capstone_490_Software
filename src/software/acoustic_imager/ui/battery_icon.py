"""
Battery (charge) icon displayed in the UI.

Placeholder implementation; will eventually show live battery charge.
Position varies by view:
- Main heatmap/camera: top-left
- Gallery grid: under STORAGE section in side dock
- Single media viewer (image/video): top-right
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np



# Icon dimensions (body + tip)
BATTERY_BODY_W = 28
BATTERY_BODY_H = 14
BATTERY_TIP_W = 4
BATTERY_TIP_H = 6
BATTERY_PAD = 2  # inner padding for fill
BORDER_COLOR = (255, 255, 255)
BORDER_THICKNESS = 1

# Fill colors (BGR) by charge level
FILL_HIGH = (0, 200, 80)   # green
FILL_MED = (0, 200, 255)   # yellow
FILL_LOW = (0, 80, 255)    # red


def _fill_color(percent: int) -> tuple:
    """Return BGR color for given charge percentage."""
    if percent > 50:
        return FILL_HIGH
    if percent > 20:
        return FILL_MED
    return FILL_LOW


def _battery_position_for_view(frame: np.ndarray) -> Tuple[int, int]:
    """Return (x, y) top-left for battery based on current view."""
    from ..state import button_state

    h, w = frame.shape[:2]
    bw = BATTERY_BODY_W + BATTERY_TIP_W
    bh = BATTERY_BODY_H
    pad = 12

    if not button_state.gallery_open:
        # Main heatmap: top-left of camera feed segment (right of dB bar)
        from ..config import DB_BAR_WIDTH
        return (DB_BAR_WIDTH + pad, pad)

    if button_state.gallery_viewer_mode in ("image", "video"):
        return (w - pad - bw, pad)  # Single media viewer: top-right

    # Gallery grid: under STORAGE in side dock
    GRID_SIDE_DOCK_WIDTH = 113
    dock_x = w - GRID_SIDE_DOCK_WIDTH
    dock_w = GRID_SIDE_DOCK_WIDTH
    # Center horizontally in dock; place at bottom below storage circle + Free/Used text
    bx = dock_x + (dock_w - bw) // 2
    by = h - 10 - bh  # Just above bottom edge, below storage section
    return (bx, by)


def draw_battery_icon(
    frame: np.ndarray,
    x: int = 12,
    y: int = 12,
    percent: Optional[int] = None,
) -> None:
    """
    Draw a battery/charge icon at the given position.

    Args:
        frame: BGR image to draw on (modified in place).
        x, y: Top-left position of the icon.
        percent: Charge 0-100. None = placeholder (shows 100% for now).
    """
    if percent is None:
        percent = 100  # placeholder until live data is connected

    percent = max(0, min(100, percent))
    h, w = frame.shape[:2]
    if x < 0 or y < 0 or x + BATTERY_BODY_W + BATTERY_TIP_W > w or y + BATTERY_BODY_H > h:
        return

    # Battery body outline (rectangle)
    body_x1, body_y1 = x, y
    body_x2 = x + BATTERY_BODY_W
    body_y2 = y + BATTERY_BODY_H
    cv2.rectangle(
        frame, (body_x1, body_y1), (body_x2, body_y2),
        BORDER_COLOR, BORDER_THICKNESS, cv2.LINE_AA
    )

    # Battery tip (positive terminal) on the right
    tip_x1 = body_x2
    tip_y1 = y + (BATTERY_BODY_H - BATTERY_TIP_H) // 2
    tip_x2 = tip_x1 + BATTERY_TIP_W
    tip_y2 = tip_y1 + BATTERY_TIP_H
    cv2.rectangle(
        frame, (tip_x1, tip_y1), (tip_x2, tip_y2),
        BORDER_COLOR, BORDER_THICKNESS, cv2.LINE_AA
    )

    # Fill level (inner rectangle)
    fill_w = max(0, int((BATTERY_BODY_W - 2 * BATTERY_PAD) * percent / 100))
    if fill_w > 0:
        fill_x1 = body_x1 + BATTERY_PAD
        fill_y1 = body_y1 + BATTERY_PAD
        fill_x2 = fill_x1 + fill_w
        fill_y2 = body_y2 - BATTERY_PAD
        cv2.rectangle(
            frame, (fill_x1, fill_y1), (fill_x2, fill_y2),
            _fill_color(percent), -1, cv2.LINE_AA
        )


def draw_battery_icon_for_view(
    frame: np.ndarray,
    percent: Optional[int] = None,
) -> None:
    """
    Draw battery icon at the appropriate position for the current view:
    - Main heatmap: top-left
    - Gallery grid: under STORAGE in side dock
    - Single media viewer: top-right
    """
    x, y = _battery_position_for_view(frame)
    draw_battery_icon(frame, x=x, y=y, percent=percent)

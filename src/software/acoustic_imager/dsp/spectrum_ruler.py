"""
Mode-aware spectrum ruler: draws the bottom strip scale for NORM (0-100%), dB (-60..0 dB), or dBA (-60..0 dBA).
"""

from __future__ import annotations

import numpy as np
import cv2

from ..config import REL_DB_MIN, REL_DB_MAX


RULER_HEIGHT = 58
BAR_MARGIN_LEFT = 8
BAR_MARGIN_RIGHT = 5
RULER_TICK_COLOR = (180, 180, 180)
RULER_LABEL_GAP_FROM_TICK = 5
DB_LABEL_COLOR = (200, 200, 200)

_RULER_FONT = cv2.FONT_HERSHEY_DUPLEX
_RULER_FONT_SCALE = 0.40
_RULER_THICKNESS = 1
_RULER_LABEL_CACHE: dict[str, np.ndarray] = {}

DB_MIN = REL_DB_MIN
DB_MAX = REL_DB_MAX


def _get_rotated_label_image(label: str) -> np.ndarray:
    """Return rotated single-channel canvas (255 where text) for the given label. Cached."""
    if label in _RULER_LABEL_CACHE:
        return _RULER_LABEL_CACHE[label]
    (tw, th), _ = cv2.getTextSize(label, _RULER_FONT, _RULER_FONT_SCALE, _RULER_THICKNESS)
    pad = 2
    canvas = np.zeros((th + 2 * pad, tw + 2 * pad), dtype=np.uint8)
    cv2.putText(canvas, label, (pad, pad + th), _RULER_FONT, _RULER_FONT_SCALE, 255, _RULER_THICKNESS, cv2.LINE_AA)
    rotated = cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _RULER_LABEL_CACHE[label] = rotated
    return rotated


def _draw_tick_and_label(
    roi: np.ndarray,
    bar_w: int,
    x: int,
    label: str,
    strip_h: int,
    y_baseline: int,
    y_tick_top: int,
    y_tick_bottom: int,
) -> None:
    cv2.line(roi, (x, y_tick_top), (x, y_tick_bottom), RULER_TICK_COLOR, 1, cv2.LINE_AA)
    rotated = _get_rotated_label_image(label)
    r_h, r_w = rotated.shape
    x0 = x - r_w // 2 - RULER_LABEL_GAP_FROM_TICK
    x0 = max(0, min(bar_w - r_w, x0))
    y0 = max(0, min(strip_h - r_h, y_baseline - r_h - 10))
    r_h_clip = min(r_h, strip_h - y0)
    r_w_clip = min(r_w, bar_w - x0)
    if r_h_clip > 0 and r_w_clip > 0:
        label_roi = roi[y0 : y0 + r_h_clip, x0 : x0 + r_w_clip]
        mask = rotated[:r_h_clip, :r_w_clip] > 128
        if mask.any():
            for c in range(3):
                label_roi[:, :, c][mask] = DB_LABEL_COLOR[c]


def draw_spectrum_ruler(roi: np.ndarray, bar_w: int, mode: str) -> None:
    """Draw ruler graphics (baseline, ticks, labels) onto roi. No background - graph shows through.
    mode: "NORM" (0-100% linear), "dB" (-60..0 dB), "dBA" (-60..0 dBA).
    """
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    strip_h = roi.shape[0]
    y_baseline = strip_h - 1
    y_tick_top = y_baseline - 8
    y_tick_bottom = y_baseline

    cv2.line(roi, (graph_left, y_baseline), (graph_right, y_baseline), RULER_TICK_COLOR, 1, cv2.LINE_AA)

    if mode == "NORM":
        # Linear: 100% left (loud), 0% right (quiet) — matches dB/dBA and bar/curve direction
        for pct in [0, 25, 50, 75, 100]:
            frac = (100 - pct) / 100.0
            x = int(np.clip(graph_left + frac * graph_width, graph_left, graph_right))
            label = f"{pct}%"
            _draw_tick_and_label(roi, bar_w, x, label, strip_h, y_baseline, y_tick_top, y_tick_bottom)
    else:
        # dB or dBA: left = 0, right = -60; same scale
        db_span = DB_MAX - DB_MIN
        suffix = " dBA" if mode == "dBA" else " dB"
        for db in range(int(DB_MIN), int(DB_MAX) + 1, 10):
            frac = (DB_MAX - db) / db_span
            x = int(np.clip(graph_left + frac * graph_width, graph_left, graph_right))
            label = f"{db}{suffix}" if db != 0 else f"0{suffix}"
            _draw_tick_and_label(roi, bar_w, x, label, strip_h, y_baseline, y_tick_top, y_tick_bottom)

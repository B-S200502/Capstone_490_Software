"""
Mode-aware spectrum ruler: draws the bottom strip scale for NORM (0-100%), dB (-60..0 dB), or dBA (-60..0 dBA).
Also draws the left vertical frequency scale ruler (cached).
"""

from __future__ import annotations

import numpy as np
import cv2

from ..config import REL_DB_MIN, REL_DB_MAX
from .bars import freq_to_y


RULER_HEIGHT = 58
BAR_MARGIN_LEFT = 8
BAR_MARGIN_RIGHT = 5
# Left vertical strip width for frequency scale (graph starts to the right of this)
FREQ_RULER_WIDTH = 36
RULER_TICK_COLOR = (180, 180, 180)
RULER_LABEL_GAP_FROM_TICK = 5
DB_LABEL_COLOR = (200, 200, 200)

_RULER_FONT = cv2.FONT_HERSHEY_DUPLEX
_RULER_FONT_SCALE = 0.40
_RULER_THICKNESS = 1
_RULER_LABEL_CACHE: dict[str, np.ndarray] = {}
# Pre-rendered ruler strip (bottom RULER_HEIGHT rows) by (h, bar_w, mode)
_RULER_STRIP_CACHE: dict[tuple[int, int, str], np.ndarray] = {}
# Pre-rendered left vertical frequency ruler by (graph_h, f_display_max)
_FREQ_RULER_STRIP_CACHE: dict[tuple[int, float], np.ndarray] = {}

_FREQ_RULER_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FREQ_RULER_FONT_SCALE = 0.35
_FREQ_RULER_MINOR_FONT_SCALE = 0.28  # smaller numbers for minor ticks
_FREQ_RULER_THICKNESS = 1

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
    Baseline aligns with graph (starts at FREQ_RULER_WIDTH).
    """
    graph_left = FREQ_RULER_WIDTH
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


def get_cached_ruler_strip(
    panel_bg: np.ndarray,
    h: int,
    bar_w: int,
    mode: str,
) -> np.ndarray:
    """Return a (RULER_HEIGHT, bar_w, 3) strip with panel bottom + ruler. Cached by (h, bar_w, mode)."""
    key = (h, bar_w, mode)
    if key in _RULER_STRIP_CACHE:
        return _RULER_STRIP_CACHE[key]
    graph_h = h - RULER_HEIGHT
    strip = panel_bg[graph_h:h, :].copy()
    draw_spectrum_ruler(strip, bar_w, mode)
    _RULER_STRIP_CACHE[key] = strip
    return strip


def _freq_ruler_tick_values(f_display_max: float) -> list[float]:
    """Return list of frequencies (Hz) for vertical ruler ticks (0 to f_display_max)."""
    f_max = float(f_display_max)
    if f_max <= 0:
        return [0.0]
    # Nice steps: ~5–8 ticks
    if f_max <= 6e3:
        step = 1000.0
    elif f_max <= 15e3:
        step = 2000.0
    elif f_max <= 30e3:
        step = 5000.0
    else:
        step = 10000.0
    ticks = [0.0]
    f = step
    while f < f_max - 0.01 * step:
        ticks.append(f)
        f += step
    if f_max > 0 and (not ticks or abs(ticks[-1] - f_max) > 0.01 * step):
        ticks.append(f_max)
    return ticks


# Number of minor ticks between each pair of major ticks (e.g. 4 → ticks at 1/5, 2/5, 3/5, 4/5 between 10k and 20k)
_FREQ_RULER_MINOR_COUNT = 4


def _freq_ruler_minor_tick_values(f_display_max: float) -> list[float]:
    """Return frequencies for small/minor ticks (evenly spaced between each pair of majors)."""
    major = _freq_ruler_tick_values(f_display_max)
    if len(major) < 2:
        return []
    minor = []
    n = _FREQ_RULER_MINOR_COUNT
    for i in range(len(major) - 1):
        lo, hi = major[i], major[i + 1]
        step = (hi - lo) / (n + 1)
        for j in range(1, n + 1):
            f = lo + step * j
            minor.append(f)
    return minor


# Minor ticks: shorter (2 px), same color
_FREQ_RULER_MINOR_TICK_LEN = 2

def draw_freq_ruler_vertical(roi: np.ndarray, graph_h: int, f_display_max: float) -> None:
    """Draw frequency scale (major + minor ticks, labels). Line on LEFT; numbers on RIGHT.
    Top label (e.g. 45k) is pushed down so it doesn't go off the top of the screen.
    """
    strip_h, strip_w = roi.shape[0], roi.shape[1]
    if strip_h <= 0 or strip_w <= 0:
        return
    cv2.line(roi, (0, 0), (0, strip_h - 1), RULER_TICK_COLOR, 1, cv2.LINE_AA)
    # Minor ticks (smaller, several between each major) + small numeric labels
    minor_ticks = _freq_ruler_minor_tick_values(f_display_max)
    scale_minor = _FREQ_RULER_MINOR_FONT_SCALE
    for freq_hz in minor_ticks:
        y = freq_to_y(freq_hz, graph_h, f_display_max)
        y = int(np.clip(y, 0, strip_h - 1))
        cv2.line(roi, (0, y), (_FREQ_RULER_MINOR_TICK_LEN, y), RULER_TICK_COLOR, 1, cv2.LINE_AA)
        if freq_hz >= 1000:
            label = f"{freq_hz / 1000:.0f}k"
        else:
            label = "0"
        (tw, th), _ = cv2.getTextSize(label, _FREQ_RULER_FONT, scale_minor, _FREQ_RULER_THICKNESS)
        lx = _FREQ_RULER_MINOR_TICK_LEN + 2  # closer to the left, next to the minor tick
        ly_centered = y + th // 2
        ly = max(th, min(strip_h - 1, ly_centered))
        cv2.putText(roi, label, (lx, ly), _FREQ_RULER_FONT, scale_minor, DB_LABEL_COLOR, _FREQ_RULER_THICKNESS, cv2.LINE_AA)
    # Major ticks and labels
    ticks = _freq_ruler_tick_values(f_display_max)
    for freq_hz in ticks:
        y = freq_to_y(freq_hz, graph_h, f_display_max)
        y = int(np.clip(y, 0, strip_h - 1))
        cv2.line(roi, (0, y), (5, y), RULER_TICK_COLOR, 1, cv2.LINE_AA)
        if freq_hz >= 1000:
            label = f"{freq_hz / 1000:.0f}k"
        else:
            label = "0"
        (tw, th), _ = cv2.getTextSize(label, _FREQ_RULER_FONT, _FREQ_RULER_FONT_SCALE, _FREQ_RULER_THICKNESS)
        lx = strip_w - 2 - tw
        # cv2.putText uses (lx, ly) as bottom-left of text; text extends up to ly - th
        # Center label on tick: baseline = y + th//2; clamp so top (ly - th) >= 0 and bottom ly <= strip_h - 1
        ly_centered = y + th // 2
        ly = max(th, min(strip_h - 1, ly_centered))  # push top label south so it doesn't go off screen
        cv2.putText(roi, label, (lx, ly), _FREQ_RULER_FONT, _FREQ_RULER_FONT_SCALE, DB_LABEL_COLOR, _FREQ_RULER_THICKNESS, cv2.LINE_AA)


def get_cached_freq_ruler_strip(
    panel_bg: np.ndarray,
    graph_h: int,
    f_display_max: float,
) -> np.ndarray:
    """Return (graph_h, FREQ_RULER_WIDTH, 3) strip with panel bg + frequency scale. Cached by (graph_h, f_display_max).
    Grey ruler line is on the left edge of the strip (bar column 0); numbers to the right.
    """
    key = (graph_h, float(f_display_max))
    if key in _FREQ_RULER_STRIP_CACHE:
        return _FREQ_RULER_STRIP_CACHE[key]
    strip = panel_bg[0:graph_h, 0:FREQ_RULER_WIDTH].copy()
    draw_freq_ruler_vertical(strip, graph_h, f_display_max)
    _FREQ_RULER_STRIP_CACHE[key] = strip
    return strip

"""
Spectrum analyzer overlay: frequency bar with smooth spectrum curve.

Three modes (MENU: SPECTRUM: dB / NORM / dBA):
- dB: amplitude in dB (relative to peak), bottom ruler, spectrum curve, bandpass overlay.
- NORM: normalized amplitude (power^0.4), linear 0-100% ruler, spectrum curve, bandpass overlay.
- dBA: A-weighted dB, same scale as dB with "dBA" labels on ruler and cursor.

Uses the same FFT data as the beamforming pipeline.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from ..config import (
    FREQ_BAR_BLUE,
    REL_DB_MIN,
    REL_DB_MAX,
)
from .bars import _get_panel_bg, freq_to_y, y_to_freq
from .spectrum_ruler import RULER_HEIGHT, BAR_MARGIN_LEFT, BAR_MARGIN_RIGHT, draw_spectrum_ruler


SPECTRUM_DB_MIN = REL_DB_MIN
SPECTRUM_DB_MAX = REL_DB_MAX

GRAPH_COLOR_TOP = (0, 220, 255)
GRAPH_COLOR_BOT = (0, 120, 255)
GRAPH_DIM_TOP = (140, 160, 180)
GRAPH_DIM_BOT = (90, 120, 150)
SPECTRUM_CURVE_BGR = (255, 255, 200)
SPECTRUM_CURVE_THICKNESS = 2

# Subsample curve to reduce polylines cost (step 2 = every 2nd bin)
CURVE_SUBSAMPLE = 2


def _a_weighting_db(f_hz: np.ndarray) -> np.ndarray:
    """A-weighting in dB per frequency (IEC 61672 / ISO 226 style). f_hz can be a scalar or array."""
    f = np.asarray(f_hz, dtype=np.float64)
    f2 = np.maximum(f * f, 1e-30)
    # Standard poles/zeros (approx): 20.6, 107.7, 737.9, 12194 Hz
    c1, c2, c3, c4 = 20.6**2, 107.7**2, 737.9**2, 12194**2
    num = (12194**2) * (f2 * f2)
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194**2)
    den = np.maximum(den, 1e-30)
    a_db = 20.0 * np.log10(num / den)
    return a_db


# Red vertical cursor for measuring freq/dB (drawn on top of bandpass)
SPECTRUM_CURSOR_COLOR_BGR = (0, 0, 255)


def spectrum_closest_curve_point(
    cursor_x_bar: float,
    tap_y: int,
    h: int,
    f_display_max: float,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    bar_w: int,
    use_db: bool = True,
    mode: str = "dB",
) -> tuple[float, float]:
    """
    Find the point on the blue spectrum curve closest to (cursor_x_bar, tap_y).
    Returns (curve_x_bar, dot_freq_hz) so the dot always lands on the curve.
    mode: "dB" | "NORM" | "dBA" (use_db True for dB and dBA).
    """
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(1, graph_right - graph_left)
    graph_h_safe = max(1, h)
    if mag_valid.size == 0:
        return (float((graph_left + graph_right) // 2), 0.0)
    if use_db:
        ref = float(np.max(mag_valid)) + 1e-20
        mag_db = 10.0 * np.log10((mag_valid.astype(np.float64) + 1e-20) / ref)
        if mode == "dBA":
            mag_db = mag_db + _a_weighting_db(f_valid)
        db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
        frac = np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0)
    else:
        mag_norm = mag_valid / (float(np.max(mag_valid)) + 1e-12)
        frac = (mag_norm ** 0.4).astype(np.float64)
    x_curve = np.clip(graph_right - frac * graph_width, graph_left, graph_right).astype(np.float64)
    y_curve = (graph_h_safe - 1) - (f_valid.astype(np.float64) / f_display_max) * (graph_h_safe - 1)
    y_curve = np.clip(y_curve, 0, graph_h_safe - 1)
    dist_sq = (x_curve - cursor_x_bar) ** 2 + (y_curve - float(tap_y)) ** 2
    idx = int(np.argmin(dist_sq))
    return (float(np.clip(x_curve[idx], graph_left, graph_right)), float(f_valid[idx]))


def spectrum_curve_x_at_y(
    tap_y: int,
    h: int,
    f_display_max: float,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    bar_w: int,
    mode: str = "dB",
) -> float:
    """Return the curve x (bar coords) at the frequency corresponding to tap_y. mode: dB | dBA | NORM."""
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]
    if mag_valid.size == 0:
        return float(bar_w // 2)
    freq = y_to_freq(tap_y, h, f_display_max)
    idx = int(np.argmin(np.abs(f_valid - freq)))
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    if mode in ("dB", "dBA"):
        ref = float(np.max(mag_valid)) + 1e-20
        mag_db = 10.0 * np.log10((float(mag_valid[idx]) + 1e-20) / ref)
        if mode == "dBA":
            mag_db = mag_db + float(_a_weighting_db(np.array([f_valid[idx]]))[0])
        db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
        frac = float(np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0))
    else:
        mag_norm = float(mag_valid[idx]) / (float(np.max(mag_valid)) + 1e-12)
        frac = (mag_norm ** 0.4)
    curve_x = graph_right - frac * max(1, graph_right - graph_left)
    return float(np.clip(curve_x, graph_left, graph_right))


def draw_spectrum_analyzer(
    frame: np.ndarray,
    fft_data: np.ndarray,
    f_axis: np.ndarray,
    f_min: float,
    f_max: float,
    freq_bar_width: int,
    f_display_max: float,
    mode: str = "dB",
    spectrum_cursor_x: Optional[float] = None,
    spectrum_cursor_dot_active: bool = False,
    spectrum_cursor_dot_freq: Optional[float] = None,
    spectrum_cursor_dot_dragging: bool = False,
    spectrum_cursor_dot_bar_pos: Optional[list] = None,
) -> None:
    """
    Draw the spectrum analyzer bar on the RIGHT side of the frame.

    mode="dB"   -> dB-scaled bars + curve + bottom ruler + bandpass overlay.
    mode="NORM" -> normalized (power^0.4) bars + linear 0-100% ruler + curve + bandpass overlay.
    mode="dBA"  -> A-weighted dB bars + curve + dBA ruler + bandpass overlay.
    """
    h, w, _ = frame.shape
    bar_w = int(max(1, freq_bar_width))
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    bar_left = w - bar_w
    bar_right = w

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
    valid = f_axis <= f_display_max
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    panel_bg = _get_panel_bg(h, bar_w)
    bar = panel_bg.copy()

    use_db = mode in ("dB", "dBA")
    draw_curve = True
    draw_bandpass = True
    graph_h = h - RULER_HEIGHT
    draw_h = h
    graph_h_safe = max(1, draw_h)
    graph_left = BAR_MARGIN_LEFT
    graph_right = bar_w - BAR_MARGIN_RIGHT
    graph_width = max(2, graph_right - graph_left)

    # Cursor readout (freq/dB at vertical line); set inside mag_valid block, drawn after bandpass
    cursor_draw_x: Optional[int] = None
    dot_draw_x: Optional[int] = None  # when set, dot is drawn here (on curve) instead of on the line
    if spectrum_cursor_x is not None:
        cursor_draw_x = int(round(np.clip(spectrum_cursor_x, graph_left, graph_right)))
    cursor_draw_y: Optional[int] = None
    cursor_freq_hz: Optional[float] = None
    cursor_db_val: Optional[float] = None

    if mag_valid.size > 0:
        if use_db:
            ref = float(np.max(mag_valid)) + 1e-20
            mag_db = 10.0 * np.log10((mag_valid.astype(np.float64) + 1e-20) / ref)
            if mode == "dBA":
                mag_db = mag_db + _a_weighting_db(f_valid)
            db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
            frac = np.clip((mag_db - SPECTRUM_DB_MIN) / db_span, 0.0, 1.0)
        else:
            mag_norm = mag_valid / (float(np.max(mag_valid)) + 1e-12)
            frac = (mag_norm ** 0.4).astype(np.float64)

        # Vectorized bar drawing: one segment per row (max length per row), slice assign
        y_float = (graph_h_safe - 1) - (f_valid.astype(np.float64) / f_display_max) * (graph_h_safe - 1)
        y_int = np.clip(y_float.astype(np.int32), 0, graph_h_safe - 1)
        length_all = (frac * graph_width).astype(np.int32)
        length_all = np.clip(length_all, 0, graph_width)
        in_band = (f_valid >= float(f_min)) & (f_valid <= float(f_max))
        t_row = np.arange(graph_h_safe, dtype=np.float64) / max(1, graph_h_safe - 1)
        color_top = np.array(GRAPH_COLOR_TOP, dtype=np.float64)
        color_bot = np.array(GRAPH_COLOR_BOT, dtype=np.float64)
        dim_top = np.array(GRAPH_DIM_TOP, dtype=np.float64)
        dim_bot = np.array(GRAPH_DIM_BOT, dtype=np.float64)
        length_per_row = np.zeros(graph_h_safe, dtype=np.int32)
        in_band_per_row = np.zeros(graph_h_safe, dtype=bool)
        for i in range(len(y_int)):
            r = y_int[i]
            if length_all[i] >= length_per_row[r]:
                length_per_row[r] = length_all[i]
                in_band_per_row[r] = in_band[i]
        for row in range(graph_h_safe):
            L = length_per_row[row]
            if L <= 0:
                continue
            x0 = int(np.clip(graph_right - L, graph_left, graph_right))
            x1 = graph_right
            t = t_row[row]
            if in_band_per_row[row]:
                color = ((1 - t) * color_top + t * color_bot).astype(np.uint8)
            else:
                color = ((1 - t) * dim_top + t * dim_bot).astype(np.uint8)
            bar[row, x0:x1, :] = color

        if draw_curve:
            step = max(1, CURVE_SUBSAMPLE)
            idx = np.arange(0, len(f_valid), step)
            if len(idx) < 2:
                idx = np.arange(len(f_valid))
            f_sub = f_valid[idx]
            frac_sub = frac[idx]
            y_curve = np.clip(
                (graph_h_safe - 1) - (f_sub.astype(np.float64) / f_display_max) * (graph_h_safe - 1),
                0, graph_h_safe - 1
            ).astype(np.int32)
            length_curve = (frac_sub * graph_width).astype(np.int32)
            x_curve = np.clip(graph_right - length_curve, graph_left, graph_right).astype(np.int32)
            pts_arr = np.column_stack((x_curve, y_curve)).astype(np.int32)
            if len(pts_arr) >= 2:
                cv2.polylines(
                    bar, [pts_arr], isClosed=False,
                    color=SPECTRUM_CURVE_BGR, thickness=SPECTRUM_CURVE_THICKNESS, lineType=cv2.LINE_AA
                )

        # Compute cursor: line position; dot position/readout use pinned dot_freq when set (stops bouncing)
        if spectrum_cursor_x is not None:
            cx = float(np.clip(spectrum_cursor_x, graph_left, graph_right))
            cursor_draw_x = int(round(cx))
            if spectrum_cursor_dot_freq is not None and spectrum_cursor_dot_active:
                # Dot pinned to closest curve point: draw at (curve_x, curve_y) for this freq so it stays on the blue curve
                cursor_freq_hz = float(spectrum_cursor_dot_freq)
                idx = int(np.argmin(np.abs(f_valid - spectrum_cursor_dot_freq)))
                cursor_draw_y = int(np.clip((graph_h_safe - 1) - (cursor_freq_hz / f_display_max) * (graph_h_safe - 1), 0, graph_h_safe - 1))
                curve_x_at_dot = graph_right - frac[idx] * graph_width
                dot_draw_x = int(np.clip(round(curve_x_at_dot), graph_left, graph_right))
                cursor_mag = float(mag_valid[idx])
                ref_cursor = float(np.max(mag_valid)) + 1e-20
                cursor_db_val = 10.0 * np.log10((cursor_mag + 1e-20) / ref_cursor)
            else:
                # No pinned dot: dot follows curve at cursor x (for drag before first tap)
                # frac 0 = right (0 dB), frac 1 = left (-60 dB), so target_frac = (graph_right - cx) / width
                target_frac = (graph_right - cx) / max(1, graph_width)
                target_frac = float(np.clip(target_frac, 0.0, 1.0))
                idx = int(np.argmin(np.abs(frac - target_frac)))
                cursor_freq_hz = float(f_valid[idx])
                cursor_mag = float(mag_valid[idx])
                ref_cursor = float(np.max(mag_valid)) + 1e-20
                cursor_db_val = 10.0 * np.log10((cursor_mag + 1e-20) / ref_cursor)
                cursor_draw_y = int(np.clip((graph_h_safe - 1) - (cursor_freq_hz / f_display_max) * (graph_h_safe - 1), 0, graph_h_safe - 1))

    # Ruler for all modes (NORM = linear 0-100%, dB/dBA = -60..0)
    draw_spectrum_ruler(bar[graph_h : h, :], bar_w, mode)

    # ---- Bandpass overlay (sliding window; all modes) ----
    if draw_bandpass:
        # Use draw_h so overlay aligns with full-height graph (ruler is drawn on top of bottom strip)
        y_min = int(np.clip(freq_to_y(f_min, draw_h, f_display_max), 0, draw_h - 1))
        y_max = int(np.clip(freq_to_y(f_max, draw_h, f_display_max), 0, draw_h - 1))
        label_x = BAR_MARGIN_LEFT
        fmin_khz = float(f_min) / 1000.0
        fmax_khz = float(f_max) / 1000.0
        y_min_txt = int(np.clip(y_min - 6, 12, h - 6))
        y_max_txt = int(np.clip(y_max - 6, 12, h - 6))

        y_top = min(y_min, y_max)
        y_bottom = max(y_min, y_max)
        if y_bottom > y_top:
            tint_blue = (240, 180, 70)
            roi = bar[y_top : y_bottom + 1, 1 : bar_w - 1]
            overlay_roi = np.full((roi.shape[0], roi.shape[1], 3), tint_blue, dtype=roi.dtype)
            blended = np.empty_like(roi)
            cv2.addWeighted(overlay_roi, 0.12, roi, 0.88, 0, blended)
            np.copyto(roi, blended)

        font = cv2.FONT_HERSHEY_SIMPLEX
        for (txt, y_pos) in [(f"{fmin_khz:5.1f} kHz", y_min_txt), (f"{fmax_khz:5.1f} kHz", y_max_txt)]:
            cv2.putText(bar, txt, (label_x, y_pos), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.line(bar, (0, y_min), (bar_w - 1, y_min), FREQ_BAR_BLUE, 2)
        cv2.line(bar, (0, y_max), (bar_w - 1, y_max), FREQ_BAR_BLUE, 2)
        cv2.line(bar, (0, y_min), (0, y_max), FREQ_BAR_BLUE, 2)
        cv2.line(bar, (bar_w - 1, y_min), (bar_w - 1, y_max), FREQ_BAR_BLUE, 2)

        handle_x = bar_w // 2
        cv2.circle(bar, (handle_x, y_min), 7, FREQ_BAR_BLUE, -1, cv2.LINE_AA)
        cv2.circle(bar, (handle_x, y_max), 7, FREQ_BAR_BLUE, -1, cv2.LINE_AA)
        cv2.circle(bar, (handle_x, y_min), 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(bar, (handle_x, y_max), 7, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- Spectrum cursor (red vertical line; dot + label only when user has tapped on line) ----
    if spectrum_cursor_dot_bar_pos is not None:
        spectrum_cursor_dot_bar_pos.clear()
    if cursor_draw_x is not None:
        cx = cursor_draw_x
        cv2.line(bar, (cx, 0), (cx, h - 1), SPECTRUM_CURSOR_COLOR_BGR, 2, cv2.LINE_AA)
    if (
        spectrum_cursor_dot_active
        and cursor_freq_hz is not None
        and cursor_db_val is not None
        and (dot_draw_x is not None or cursor_draw_x is not None)
    ):
        dx = dot_draw_x if dot_draw_x is not None else cursor_draw_x
        dot_r = 9 if spectrum_cursor_dot_dragging else 6
        cv2.circle(bar, (dx, cursor_draw_y), dot_r, SPECTRUM_CURSOR_COLOR_BGR, -1, cv2.LINE_AA)
        cv2.circle(bar, (dx, cursor_draw_y), dot_r, (255, 255, 255), 1, cv2.LINE_AA)
        if spectrum_cursor_dot_bar_pos is not None:
            spectrum_cursor_dot_bar_pos[:] = [float(dx), float(cursor_draw_y)]
        freq_khz = cursor_freq_hz / 1000.0
        if mode == "dBA":
            a_at_cursor = float(_a_weighting_db(np.array([cursor_freq_hz]))[0])
            label_val = cursor_db_val + a_at_cursor
            label = f"{freq_khz:.1f} kHz  {label_val:.1f} dBA"
        else:
            # Show dB in same direction as ruler: left = 0 dB, right = -60 dB
            db_span = SPECTRUM_DB_MAX - SPECTRUM_DB_MIN
            label_db = SPECTRUM_DB_MAX - (dx - graph_left) / max(1, graph_width) * db_span
            label_db = float(np.clip(label_db, SPECTRUM_DB_MIN, SPECTRUM_DB_MAX))
            label = f"{freq_khz:.1f} kHz  {label_db:.0f} dB"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        # Place label above the dot, avoid clipping
        ly = max(th + 2, cursor_draw_y - 8)
        lx = dx + 6
        if lx + tw > bar_w - 2:
            lx = dx - tw - 6
        lx = max(2, min(lx, bar_w - tw - 2))
        cv2.putText(bar, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar

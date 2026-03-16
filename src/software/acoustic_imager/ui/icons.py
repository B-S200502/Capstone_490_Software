"""
Shared icon drawing (WiFi, Settings) used by HUD and menu buttons.
Icons are cached by (type, size, color, bg_key) for performance.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Dict

import cv2
import numpy as np

# Cache: (icon_type, size, color, bg_key) -> pre-rendered icon (h, w, 3) BGR
_ICON_CACHE: Dict[Tuple[str, int, Tuple[int, int, int], str], np.ndarray] = {}


def _bg_key(bg: Optional[Tuple[int, int, int]]) -> str:
    return "none" if bg is None else f"{bg[0]},{bg[1]},{bg[2]}"


def _get_cached_icon(
    icon_type: str,
    size: int,
    color: Tuple[int, int, int],
    bg_color: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """Return cached or freshly rendered icon."""
    key = (icon_type, size, color, _bg_key(bg_color))
    if key in _ICON_CACHE:
        return _ICON_CACHE[key]
    canvas_size = size * 2 + 10
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if bg_color is not None:
        canvas[:] = bg_color
    cx = cy = canvas_size // 2

    if icon_type == "wifi":
        _render_wifi(canvas, cx, cy, size, color)
    elif icon_type == "settings":
        _render_gear(canvas, cx, cy, size, color)

    _ICON_CACHE[key] = canvas
    return canvas


def _render_wifi(
    canvas: np.ndarray, cx: int, cy: int, size: int, color: Tuple[int, int, int]
) -> None:
    """WiFi: 3 stacked arcs + center dot."""
    scale = size / 12.0
    r1, r2, r3 = int(10 * scale), int(6.5 * scale), int(3 * scale)
    y_off = int(2 * scale)
    thickness = max(1, int(1.5 * scale))
    dot_r = max(1, int(2 * scale))
    for r in (r1, r2, r3):
        cv2.ellipse(canvas, (cx, cy + y_off), (r, r), 0, 180, 360, color, thickness, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy + y_off), dot_r, color, -1, cv2.LINE_AA)


def _render_gear(
    canvas: np.ndarray, cx: int, cy: int, size: int, color: Tuple[int, int, int]
) -> None:
    """Gear icon: cog with 8 teeth (trapezoidal) and hollow center."""
    scale = size / 12.0
    r_outer = int(10 * scale)
    r_inner = max(2, int(4 * scale))
    n_teeth = 8
    tooth_angle_deg = 22
    gap_angle_deg = 360.0 / n_teeth - tooth_angle_deg
    thickness = max(1, int(1.5 * scale))

    for i in range(n_teeth):
        base_ang = math.radians(i * (360.0 / n_teeth) + gap_angle_deg / 2)
        ang1 = base_ang
        ang2 = base_ang + math.radians(tooth_angle_deg)
        pts = [
            (int(cx + r_inner * math.cos(ang1)), int(cy + r_inner * math.sin(ang1))),
            (int(cx + r_outer * math.cos(ang1)), int(cy + r_outer * math.sin(ang1))),
            (int(cx + r_outer * math.cos(ang2)), int(cy + r_outer * math.sin(ang2))),
            (int(cx + r_inner * math.cos(ang2)), int(cy + r_inner * math.sin(ang2))),
        ]
        cv2.fillPoly(canvas, [np.array(pts)], color, lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [np.array(pts)], True, color, thickness, cv2.LINE_AA)

    cv2.circle(canvas, (cx, cy), r_inner, color, thickness, cv2.LINE_AA)


def _blit_icon(
    frame: np.ndarray,
    icon: np.ndarray,
    cx: int,
    cy: int,
    bg_color: Optional[Tuple[int, int, int]],
    circular: bool = False,
    clip_radius: Optional[int] = None,
) -> None:
    """Blit cached icon to frame centered at (cx, cy). If circular=True, only copy pixels inside circle.
    If clip_radius is set, use that as max radius (so border can match icon outline)."""
    h, w = icon.shape[:2]
    x0, y0 = cx - w // 2, cy - h // 2
    x1, y1 = x0 + w, y0 + h
    fh, fw = frame.shape[:2]
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(fw, x1), min(fh, y1)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    src_x0, src_y0 = sx0 - x0, sy0 - y0
    src_x1, src_y1 = src_x0 + (sx1 - sx0), src_y0 + (sy1 - sy0)
    roi = frame[sy0:sy1, sx0:sx1]
    src = icon[src_y0:src_y1, src_x0:src_x1].copy()
    if circular:
        # Mask to circle: center of src, radius = min dimension / 2 (or clip_radius if set)
        sh, sw = src.shape[:2]
        icx, icy = sw // 2, sh // 2
        r = min(icx, icy)
        if clip_radius is not None:
            r = min(r, clip_radius)
        yy, xx = np.ogrid[:sh, :sw]
        circle_mask = ((xx - icx) ** 2 + (yy - icy) ** 2) <= r * r
        inv_mask = ~circle_mask
        src[inv_mask] = [0, 0, 0]  # don't copy outside circle; use bg_color logic below
        if bg_color is not None:
            roi[circle_mask] = src[circle_mask]
        else:
            mask = np.any(src != [0, 0, 0], axis=2)
            roi[mask] = src[mask]
    elif bg_color is not None:
        roi[:] = src
    else:
        mask = np.any(src != [0, 0, 0], axis=2)
        roi[mask] = src[mask]


def draw_wifi_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    *,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple[int, int, int]] = None,
    size: int = 12,
    circular: bool = False,
    clip_radius: Optional[int] = None,
) -> None:
    """Draw WiFi icon. Cached. If circular=True, clip to circle (for HUD). clip_radius keeps bg inside outline."""
    icon = _get_cached_icon("wifi", size, color, bg_color)
    _blit_icon(frame, icon, cx, cy, bg_color, circular=circular, clip_radius=clip_radius)


def draw_settings_icon(
    frame: np.ndarray,
    cx: int,
    cy: int,
    *,
    color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 12,
) -> None:
    """Draw gear/settings icon. Cached."""
    icon = _get_cached_icon("settings", size, color, None)
    _blit_icon(frame, icon, cx, cy, None)

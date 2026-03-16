"""
Shared priority circle drawing.
Neon style (gradient + glow) for set priorities; plain flat circle for unset.
Used in gallery grid labels and in the priority modal.
"""

from typing import Tuple

import cv2
import numpy as np


def draw_priority_circle_neon(
    frame: np.ndarray, cx: int, cy: int, radius: int, color_bgr: Tuple[int, int, int],
    neon: bool = True,
) -> None:
    """Draw a priority circle. When neon=True uses radial gradient + glow; when False draws a plain flat circle."""
    if not neon:
        cv2.circle(frame, (cx, cy), radius, color_bgr, -1, cv2.LINE_AA)
        return
    glow_radius = radius + 6
    glow_extra = 10  # extra padding so blur has room to fall off
    size = 2 * (glow_radius + glow_extra) + 1
    h, w = frame.shape[:2]
    x0 = cx - glow_radius - glow_extra
    y0 = cy - glow_radius - glow_extra
    x1 = x0 + size
    y1 = y0 + size
    px0 = max(0, x0)
    py0 = max(0, y0)
    px1 = min(w, x1)
    py1 = min(h, y1)
    if px1 <= px0 or py1 <= py0:
        return
    roi = frame[py0:py1, px0:px1]
    lcx = cx - px0
    lcy = cy - py0
    # Glow: white circle, blur, tint
    glow_canvas = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    cv2.circle(glow_canvas, (lcx, lcy), glow_radius, (255, 255, 255), -1, cv2.LINE_AA)
    glow_canvas = cv2.GaussianBlur(glow_canvas, (11, 11), 3.0)
    glow_canvas = glow_canvas.astype(np.float32)
    # Radial mask: smooth falloff to 0 so no rectangular clip (circular glow only)
    yy, xx = np.ogrid[:roi.shape[0], :roi.shape[1]]
    dist = np.sqrt((xx - lcx) ** 2 + (yy - lcy) ** 2)
    inner = glow_radius - 2
    outer = glow_radius + glow_extra
    radial = np.clip((outer - dist) / (outer - inner), 0, 1).astype(np.float32)
    for c in range(3):
        glow_canvas[:, :, c] = glow_canvas[:, :, c] * radial * (color_bgr[c] / 255.0)
    add_strength = 0.48
    roi[:] = np.minimum(255, (roi.astype(np.float32) + glow_canvas * add_strength)).astype(np.uint8)
    # Gradient: softer edge so it blends into glow, bright center (neon tube)
    for r_frac, intensity in [(1.0, 0.55), (0.78, 0.72), (0.55, 0.88), (0.3, 0.98), (0.12, 1.0)]:
        r = max(1, int(radius * r_frac))
        shade = tuple(min(255, int(color_bgr[c] * intensity)) for c in range(3))
        cv2.circle(frame, (cx, cy), r, shade, -1, cv2.LINE_AA)

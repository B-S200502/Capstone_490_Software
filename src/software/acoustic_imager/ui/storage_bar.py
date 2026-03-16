"""
Storage bar: circular neon progress ring for disk usage.

Draws a gradient ring with a "snake" filled portion for used %, percentage
in the center, and optional GB used below. Styled to match the reference
neon circle (blue -> purple -> pink gradient, glow).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Tuple, Optional, List
from datetime import datetime

import cv2
import numpy as np

from ..state import button_state

try:
    from ..config import STORAGE_BAR_GLOW, STORAGE_BAR_BRIGHTNESS, BG_GRADIENT_TOP
except Exception:
    STORAGE_BAR_GLOW = 0.35
    STORAGE_BAR_BRIGHTNESS = 1.0
    BG_GRADIENT_TOP = (42, 42, 42)


def _brighten(bgr: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Apply global brightness multiplier to a BGR color."""
    b, g, r = bgr
    return (min(255, int(b * STORAGE_BAR_BRIGHTNESS)), min(255, int(g * STORAGE_BAR_BRIGHTNESS)), min(255, int(r * STORAGE_BAR_BRIGHTNESS)))


# Neon gradient stops (angle 0 = right; we use 270 = top so gradient runs top->right->bottom->left)
# BGR: cyan/blue, purple, magenta/pink
NEON_BLUE_BGR: Tuple[int, int, int] = (255, 180, 80)    # cyan-blue
NEON_PURPLE_BGR: Tuple[int, int, int] = (255, 0, 180)  # purple
NEON_PINK_BGR: Tuple[int, int, int] = (220, 0, 255)    # pink/magenta
# Unfilled track: dim neon blue/teal mixed with gray (matches "Free" text)
TRACK_BGR: Tuple[int, int, int] = (145, 130, 100)      # blue–teal + gray (BGR)
TEXT_BGR: Tuple[int, int, int] = (255, 120, 220)      # neon purple for % text and "Used" label
# Lighter variants for Used/Free text (readable on bright screens)
USED_TEXT_LIGHT_BGR: Tuple[int, int, int] = (255, 200, 255)   # lighter neon purple
FREE_TEXT_LIGHT_BGR: Tuple[int, int, int] = (220, 210, 190)  # lighter blue-gray
# Storage label: light, legible, with subtle glow
STORAGE_LABEL_BGR: Tuple[int, int, int] = (235, 235, 240)


def _angle_to_bgr(angle_deg: float) -> Tuple[int, int, int]:
    """Map angle [0, 360) to gradient BGR (blue -> purple -> pink -> blue)."""
    # 0–120: blue -> purple, 120–240: purple -> pink, 240–360: pink -> blue
    angle_deg = angle_deg % 360.0
    if angle_deg < 120:
        t = angle_deg / 120.0
        return (int(NEON_BLUE_BGR[0] * (1 - t) + NEON_PURPLE_BGR[0] * t),
                int(NEON_BLUE_BGR[1] * (1 - t) + NEON_PURPLE_BGR[1] * t),
                int(NEON_BLUE_BGR[2] * (1 - t) + NEON_PURPLE_BGR[2] * t))
    if angle_deg < 240:
        t = (angle_deg - 120) / 120.0
        return (int(NEON_PURPLE_BGR[0] * (1 - t) + NEON_PINK_BGR[0] * t),
                int(NEON_PURPLE_BGR[1] * (1 - t) + NEON_PINK_BGR[1] * t),
                int(NEON_PURPLE_BGR[2] * (1 - t) + NEON_PINK_BGR[2] * t))
    t = (angle_deg - 240) / 120.0
    return (int(NEON_PINK_BGR[0] * (1 - t) + NEON_BLUE_BGR[0] * t),
            int(NEON_PINK_BGR[1] * (1 - t) + NEON_BLUE_BGR[1] * t),
            int(NEON_PINK_BGR[2] * (1 - t) + NEON_BLUE_BGR[2] * t))


def _feathered_composite(
    frame: np.ndarray,
    y0: int, y1: int, x0: int, x1: int,
    patch: np.ndarray,
    blend_patch: float,
    feather_px: int = 28,
) -> None:
    """Composite patch onto frame with soft edges so no visible boundary artifact."""
    crop = frame[y0:y1, x0:x1]
    ph, pw = patch.shape[:2]
    # Distance from each pixel to nearest patch edge (in pixels)
    i = np.arange(ph, dtype=np.float32).reshape(-1, 1)
    j = np.arange(pw, dtype=np.float32).reshape(1, -1)
    dist = np.minimum(np.minimum(i, ph - 1 - i), np.minimum(j, pw - 1 - j))
    mask = np.clip(dist / max(1, feather_px), 0.0, 1.0)
    if patch.ndim == 3:
        mask = mask[:, :, np.newaxis]
    blended = (patch.astype(np.float32) * blend_patch + crop.astype(np.float32) * (1 - blend_patch))
    out = (blended * mask + crop.astype(np.float32) * (1 - mask)).clip(0, 255).astype(np.uint8)
    frame[y0:y1, x0:x1] = out


def feathered_composite(
    frame: np.ndarray,
    y0: int, y1: int, x0: int, x1: int,
    patch: np.ndarray,
    blend_patch: float,
    feather_px: int = 28,
) -> None:
    """Composite patch onto frame with soft edges (public for use by dock/buttons)."""
    _feathered_composite(frame, y0, y1, x0, x1, patch, blend_patch, feather_px)


def _draw_arc_segment(
    frame: np.ndarray,
    center: Tuple[int, int],
    radius: int,
    thickness: int,
    start_deg: float,
    end_deg: float,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Draw an arc from start_deg to end_deg (OpenCV: 0 = right, 90 = bottom)."""
    # OpenCV ellipse: startAngle and endAngle in degrees; arc drawn counter-clockwise from start to end.
    # We want clockwise from top: so from 270 (top) going clockwise = 270 -> 270-angle (CCW) or 270 -> 270+angle.
    # In OpenCV, angle parameter rotates the axes; startAngle/endAngle are relative to that. So with angle=0,
    # startAngle=270, endAngle=270+arc_len gives arc from top going right (clockwise).
    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        0.0,
        start_deg,
        end_deg,
        color_bgr,
        thickness,
        cv2.LINE_AA,
    )


def draw_storage_circle(
    frame: np.ndarray,
    cx: int,
    cy: int,
    used_percent: float,
    radius: int = 42,
    ring_thickness: int = 10,
    used_gb_str: Optional[str] = None,
    free_gb_str: Optional[str] = None,
    show_glow: bool = True,
) -> None:
    """
    Draw a circular neon storage progress ring.

    - Full ring in dim neon blue–gray (track); filled "snake" from top clockwise by used_percent.
    - Filled arc uses blue->purple->pink gradient. Glow hugs the snake (same radius, thicker + blur).
    - Center: percentage text (e.g. "14%").
    - Below circle: two lines "Used: <number>" and "Free: <number>" when provided.
    """
    h, w = frame.shape[:2]
    used_percent = max(0.0, min(100.0, used_percent))
    arc_angle = 360.0 * (used_percent / 100.0)

    # 1) Track: full circle (muted)
    _draw_arc_segment(frame, (cx, cy), radius, ring_thickness, 0, 360, _brighten(TRACK_BGR))

    # 1b) "Storage" label above the ring (same gap as below); ~20% larger, lighter color, subtle glow
    font = cv2.FONT_HERSHEY_SIMPLEX
    buffer_ring_text = 14
    label_text = "Storage"
    label_scale = 0.62  # ~20% larger than 0.52
    (lw, lh), _ = cv2.getTextSize(label_text, font, label_scale, 1)
    label_x = cx - lw // 2
    label_y = cy - radius - ring_thickness - buffer_ring_text - int(lh)
    if label_y >= 2:
        pad_s = 18
        sx0 = max(0, label_x - pad_s)
        sy0 = max(0, label_y - int(lh) - pad_s)
        sx1 = min(w, label_x + lw + pad_s)
        sy1 = min(h, label_y + pad_s)
        if sx1 > sx0 and sy1 > sy0:
            spatch = frame[sy0:sy1, sx0:sx1].copy()
            slx, sly = label_x - sx0, label_y - sy0
            for thickness in (3, 2):
                cv2.putText(spatch, label_text, (slx, sly), font, label_scale, _brighten(STORAGE_LABEL_BGR), thickness, cv2.LINE_AA)
            spatch = cv2.GaussianBlur(spatch, (0, 0), 2.5)
            _feathered_composite(frame, sy0, sy1, sx0, sx1, spatch, STORAGE_BAR_GLOW * 0.55, feather_px=18)  # lighter glow on label
        cv2.putText(frame, label_text, (label_x, label_y), font, label_scale, _brighten(STORAGE_LABEL_BGR), 1, cv2.LINE_AA)

    # 2) Filled arc (snake): from top (270°) clockwise by arc_angle, with gradient
    # Draw in small segments so we can vary color along the arc
    segment_deg = 4.0
    start_deg = 270.0  # top
    end_deg = 270.0 + arc_angle
    if arc_angle >= 0.5:
        d = start_deg
        while d < end_deg:
            seg_end = min(d + segment_deg, end_deg)
            # Gradient by position along circle (0 = top, 90 = right, ...)
            angle_for_color = (d - 270.0) % 360.0
            color = _brighten(_angle_to_bgr(angle_for_color))
            _draw_arc_segment(frame, (cx, cy), radius, ring_thickness, d, seg_end, color)
            d = seg_end

    # 3) Glow surrounding the snake: same arc, thicker stroke, blur; ~15% stronger glow
    if show_glow and arc_angle >= 1.0:
        pad = 55
        x0 = max(0, cx - radius - ring_thickness - pad)
        y0 = max(0, cy - radius - ring_thickness - pad)
        x1 = min(w, cx + radius + ring_thickness + pad)
        y1 = min(h, cy + radius + ring_thickness + pad)
        if x1 > x0 and y1 > y0:
            patch = frame[y0:y1, x0:x1].copy()
            lcx, lcy = cx - x0, cy - y0
            glow_thickness = ring_thickness + 15  # +15% vs 6
            for d in np.arange(270.0, 270.0 + arc_angle, 6.0):
                seg_end = min(float(d + 6.0), 270.0 + arc_angle)
                angle_for_color = float((d - 270.0) % 360.0)
                color = _brighten(_angle_to_bgr(angle_for_color))
                cv2.ellipse(patch, (lcx, lcy), (radius, radius), 0.0, d, seg_end, color, glow_thickness, cv2.LINE_AA)
            patch = cv2.GaussianBlur(patch, (0, 0), 4.5)
            _feathered_composite(frame, y0, y1, x0, x1, patch, STORAGE_BAR_GLOW)

    # 4) Redraw filled arc on top of glow so it stays sharp
    if arc_angle >= 0.5:
        d = start_deg
        while d < end_deg:
            seg_end = min(d + segment_deg, end_deg)
            angle_for_color = (d - 270.0) % 360.0
            color = _brighten(_angle_to_bgr(angle_for_color))
            _draw_arc_segment(frame, (cx, cy), radius, ring_thickness, d, seg_end, color)
            d = seg_end

    # 5) Center text: percentage with glow (same idea as snake glow)
    # Circle behind percentage (gallery background color) to avoid purple bleed from text glow
    inner_r = max(0, radius - ring_thickness - 2)
    if inner_r > 4:
        cv2.circle(frame, (cx, cy), inner_r, BG_GRADIENT_TOP, -1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pct_str = f"{used_percent:.1f}%" if used_percent >= 0.1 else f"{used_percent:.2f}%"
    scale = 0.52
    (tw, th), bl = cv2.getTextSize(pct_str, font, scale, 1)
    tx = cx - tw // 2
    ty = cy + th // 2
    # Glow: draw text on a patch, blur, composite; ~15% stronger glow
    pad = 35
    gx0 = max(0, tx - pad)
    gy0 = max(0, ty - th - pad)
    gx1 = min(w, tx + tw + pad)
    gy1 = min(h, ty + pad)
    if gx1 > gx0 and gy1 > gy0:
        glow_patch = frame[gy0:gy1, gx0:gx1].copy()
        ltx, lty = tx - gx0, ty - gy0
        for thickness in (5, 4, 3, 2):
            cv2.putText(glow_patch, pct_str, (ltx, lty), font, scale, _brighten(TEXT_BGR), thickness, cv2.LINE_AA)
        glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 3.5)
        _feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, STORAGE_BAR_GLOW)
    cv2.putText(frame, pct_str, (tx, ty), font, scale, _brighten(TEXT_BGR), 1, cv2.LINE_AA)

    # 6) Optional: "Used: X" and "Free: Y" on two lines, aligned; colors and glows match bar (Used=neon purple, Free=track blue–gray)
    if used_gb_str is not None or free_gb_str is not None:
        label_font_scale = 0.46
        (lw_used_label, lh), _ = cv2.getTextSize("Used: ", font, label_font_scale, 1)
        (lw_free_label, _), _ = cv2.getTextSize("Free: ", font, label_font_scale, 1)
        label_w = max(lw_used_label, lw_free_label)
        num1_w = cv2.getTextSize(used_gb_str or "", font, label_font_scale, 1)[0][0]
        num2_w = cv2.getTextSize(free_gb_str or "", font, label_font_scale, 1)[0][0]
        total_w = label_w + max(num1_w, num2_w)
        start_x = (cx - total_w // 2) + 3
        buffer_below_ring = 14  # same as buffer_ring_text above
        base_ly = cy + radius + ring_thickness + buffer_below_ring + int(lh)
        line_h = int(lh) + 2
        pad = 10
        # Used line: neon purple + glow (match snake)
        if used_gb_str and base_ly <= h - 2:
            gx0 = max(0, start_x - pad)
            gy0 = max(0, base_ly - int(lh) - pad)
            gx1 = min(w, start_x + total_w + pad)
            gy1 = min(h, base_ly + pad)
            if gx1 > gx0 and gy1 > gy0:
                patch = frame[gy0:gy1, gx0:gx1].copy()
                lx1, ly1 = start_x - gx0, base_ly - gy0
                for thickness in (4, 3, 2):
                    ucolor = _brighten(USED_TEXT_LIGHT_BGR)
                    cv2.putText(patch, "Used: ", (lx1, ly1), font, label_font_scale, ucolor, thickness, cv2.LINE_AA)
                    cv2.putText(patch, used_gb_str, (lx1 + label_w, ly1), font, label_font_scale, ucolor, thickness, cv2.LINE_AA)
                patch = cv2.GaussianBlur(patch, (0, 0), 2.5)
                _feathered_composite(frame, gy0, gy1, gx0, gx1, patch, STORAGE_BAR_GLOW, feather_px=14)
            ucolor = _brighten(USED_TEXT_LIGHT_BGR)
            cv2.putText(frame, "Used: ", (start_x, base_ly), font, label_font_scale, ucolor, 1, cv2.LINE_AA)
            cv2.putText(frame, used_gb_str, (start_x + (label_w - 3), base_ly), font, label_font_scale, ucolor, 1, cv2.LINE_AA)
        # Free line: track color + glow (match unfilled ring)
        if free_gb_str:
            ly2 = base_ly + (line_h if used_gb_str else 0)
            if ly2 <= h - 2:
                gx0 = max(0, start_x - pad)
                gy0 = max(0, ly2 - int(lh) - pad)
                gx1 = min(w, start_x + total_w + pad)
                gy1 = min(h, ly2 + pad)
                if gx1 > gx0 and gy1 > gy0:
                    patch = frame[gy0:gy1, gx0:gx1].copy()
                    lx1, ly1 = start_x - gx0, ly2 - gy0
                    fcolor = _brighten(FREE_TEXT_LIGHT_BGR)
                    for thickness in (4, 3, 2):
                        cv2.putText(patch, "Free: ", (lx1, ly1), font, label_font_scale, fcolor, thickness, cv2.LINE_AA)
                        cv2.putText(patch, free_gb_str, (lx1 + label_w, ly1), font, label_font_scale, fcolor, thickness, cv2.LINE_AA)
                    patch = cv2.GaussianBlur(patch, (0, 0), 2.5)
                    _feathered_composite(frame, gy0, gy1, gx0, gx1, patch, STORAGE_BAR_GLOW, feather_px=14)
                fcolor = _brighten(FREE_TEXT_LIGHT_BGR)
                cv2.putText(frame, "Free: ", (start_x, ly2), font, label_font_scale, fcolor, 1, cv2.LINE_AA)
                cv2.putText(frame, free_gb_str, (start_x + (label_w - 3), ly2), font, label_font_scale, fcolor, 1, cv2.LINE_AA)


def _format_size(size_bytes: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


# Layout: match grid side dock top rows so circle is centered in the dock below them
DOCK_ROW_HEIGHT = 75
DOCK_DIVIDER_THICKNESS = 2
STORAGE_BLOCK_TOP_OFFSET = 3 + 3 * DOCK_ROW_HEIGHT + 2 * DOCK_DIVIDER_THICKNESS  # sort_bottom from dock_y


def draw_storage_bar(
    frame: np.ndarray,
    dock_x: int,
    dock_w: int,
    header_h: int,
    items: List[Tuple[Path, str, datetime]],
    output_dir: Optional[Path],
) -> None:
    """
    Draw the circular neon storage bar inside the dock. Position is viewport-fixed
    so it stays visible as the grid scrolls. Uses live disk usage so the bar
    updates immediately after deletes.
    """
    total_media_size = 0
    if items:
        try:
            total_media_size = sum(p.stat().st_size for p, _, _ in items)
        except OSError:
            pass

    if output_dir and output_dir.exists():
        try:
            path_str = os.path.realpath(str(output_dir))
            if button_state.gallery_storage_dirty:
                os.sync()
                button_state.gallery_storage_dirty = False
            disk_usage = shutil.disk_usage(path_str)
            total_space = disk_usage.total
            used_space = disk_usage.used
        except OSError:
            total_space = 128 * 1024 * 1024 * 1024
            used_space = total_media_size
    else:
        if not items:
            return
        total_space = 128 * 1024 * 1024 * 1024
        used_space = total_media_size

    used_percent = (used_space / total_space * 100) if total_space > 0 else 0.0
    used_gb_str = _format_size(used_space)
    free_gb_str = _format_size(max(0, total_space - used_space))

    frame_h = frame.shape[0]
    block_top = header_h + STORAGE_BLOCK_TOP_OFFSET
    available_h = frame_h - block_top
    if available_h < 100:
        return
    cy = block_top + available_h // 2
    cx = dock_x + dock_w // 2

    draw_storage_circle(
        frame,
        cx,
        cy,
        used_percent,
        radius=42,
        ring_thickness=10,
        used_gb_str=used_gb_str,
        free_gb_str=free_gb_str,
        show_glow=True,
    )

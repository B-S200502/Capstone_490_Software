"""
Bottom HUD: SHOT, REC, and Gallery pills in the bottom-left next to the dB scaler.
Taller pills for better clickability. When recording is paused, REC splits into Resume (yellow) and Stop (red).
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import DB_BAR_WIDTH, FREQ_BAR_WIDTH, UI_PILL_H, UI_PILL_W, HUD_MENU_OPACITY, ACTION_BTN_GLOW
from ..state import button_state, HUD
from .button import (
    menu_buttons,
    Button,
    _draw_camera_icon,
    _draw_gallery_icon,
    _draw_rec_icon,
    _draw_pause_icon,
)
from .storage_bar import feathered_composite
from .top_hud import _draw_pill

# Same size as top HUD (UI_PILL_H, UI_PILL_W) for consistency and clickability
PILL_H = UI_PILL_H
PILL_W = UI_PILL_W
GAP = 8
BOTTOM_MARGIN = 5
LEFT_MARGIN = 10
# When paused: Resume gets this fraction of REC pill width, Stop the rest
REC_RESUME_FRAC = 0.60

BOTTOM_HUD_HEIGHT = PILL_H

# Icon size for pills
ICON_SIZE = 14

# Same opacity as menu and top HUD (single knob: HUD_MENU_OPACITY)
BOTTOM_PILL_ALPHA = HUD_MENU_OPACITY

# Cache layout so we only update button rects when frame size or state changes (avoids churn + timing issues)
_last_layout: Optional[Tuple[int, int, int, int, int, bool]] = None

# Paused split colors (BGR): yellow for Resume, red for Stop
RESUME_PILL_BG: Tuple[int, int, int] = (0, 180, 255)   # yellow
STOP_PILL_BG: Tuple[int, int, int] = (0, 0, 200)       # red
RESUME_PILL_BORDER: Tuple[int, int, int] = (0, 220, 255)
STOP_PILL_BORDER: Tuple[int, int, int] = (0, 0, 255)

# Compass (BN-880 magnetometer) between gallery and MENU; size matches MENU button height (50)
COMPASS_BUFFER_PX = 1
MENU_BUTTON_HEIGHT = 50
COMPASS_RADIUS = MENU_BUTTON_HEIGHT // 2  # 25
# Pocket-compass style: golden-orange casing, cream face, N/E/S/W, red/blue needle
COMPASS_CASING_BGR: Tuple[int, int, int] = (0, 165, 255)   # golden-orange
COMPASS_FACE_BGR: Tuple[int, int, int] = (220, 230, 230)   # cream
COMPASS_NEEDLE_N_BGR: Tuple[int, int, int] = (0, 0, 255)   # red (north)
COMPASS_NEEDLE_S_BGR: Tuple[int, int, int] = (255, 200, 100)  # light blue (south)
COMPASS_CARDINAL_BGR: Tuple[int, int, int] = (50, 50, 50)
COMPASS_CROSSHAIR_BGR: Tuple[int, int, int] = (0, 200, 255)  # light orange
COMPASS_PIVOT_BGR: Tuple[int, int, int] = (80, 80, 80)


def _draw_compass(
    frame: np.ndarray, cx: int, cy: int, radius: int, heading_deg: float = 0.0
) -> None:
    """Draw pocket-style compass: casing, cream face, N/E/S/W, crosshairs, red/blue needle. heading_deg 0 = north (needle up)."""
    fh, fw = frame.shape[:2]
    pad = radius + 4
    x0 = max(0, cx - pad)
    y0 = max(0, cy - pad)
    x1 = min(fw, cx + pad)
    y1 = min(fh, cy + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1].copy()
    lcx, lcy = cx - x0, cy - y0
    r = min(radius, lcx, lcy, roi.shape[1] - lcx, roi.shape[0] - lcy)
    if r <= 0:
        return
    # Casing: thick golden-orange ring (outer), then cream face (inner)
    ring_thick = max(2, r // 6)
    inner_r = r - ring_thick
    cv2.circle(roi, (lcx, lcy), r, COMPASS_CASING_BGR, ring_thick, cv2.LINE_AA)
    mask = np.zeros((roi.shape[0], roi.shape[1]), dtype=np.uint8)
    cv2.circle(mask, (lcx, lcy), max(0, inner_r), 255, -1)
    overlay = np.empty_like(roi)
    overlay[:] = COMPASS_FACE_BGR
    blended = cv2.addWeighted(overlay, BOTTOM_PILL_ALPHA, roi, 1.0 - BOTTOM_PILL_ALPHA, 0.0)
    for c in range(3):
        roi[:, :, c] = np.where(mask > 0, blended[:, :, c], roi[:, :, c])
    # Faint crosshairs (N-S, E-W) from center toward cardinals
    for angle_deg in (0, 90, 180, 270):
        rad = np.radians(angle_deg)
        ex = int(lcx + np.sin(rad) * inner_r)
        ey = int(lcy - np.cos(rad) * inner_r)
        cv2.line(roi, (lcx, lcy), (ex, ey), COMPASS_CROSSHAIR_BGR, 1, cv2.LINE_AA)
    frame[y0:y1, x0:x1] = roi
    # Cardinal labels N, E, S, W (fixed on face; 0 = N up)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    for label, angle_deg in (("N", 0), ("E", 90), ("S", 180), ("W", 270)):
        rad = np.radians(angle_deg)
        tx = int(cx + np.sin(rad) * (inner_r - 10))
        ty = int(cy - np.cos(rad) * (inner_r - 10))
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx -= tw // 2
        ty += th // 2
        cv2.putText(frame, label, (tx, ty), font, scale, COMPASS_CARDINAL_BGR, thick, cv2.LINE_AA)
    # Needle: north half red, south half light blue, rotated by heading_deg
    rad = np.radians(heading_deg)
    dx, dy = np.sin(rad), -np.cos(rad)
    needle_len = inner_r - 6
    tip_n = (int(cx + dx * needle_len), int(cy + dy * needle_len))
    tip_s = (int(cx - dx * needle_len), int(cy - dy * needle_len))
    cv2.line(frame, (cx, cy), tip_n, COMPASS_NEEDLE_N_BGR, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy), tip_s, COMPASS_NEEDLE_S_BGR, 2, cv2.LINE_AA)
    cv2.circle(frame, tip_n, 3, COMPASS_NEEDLE_N_BGR, -1, cv2.LINE_AA)
    cv2.circle(frame, tip_s, 2, COMPASS_NEEDLE_S_BGR, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 3, COMPASS_PIVOT_BGR, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, COMPASS_CASING_BGR, 1, cv2.LINE_AA)


def _draw_pill_tint(
    frame: np.ndarray, x: int, y: int, w: int, h: int,
    tint_bgr: Tuple[int, int, int], border_bgr: Tuple[int, int, int], alpha: float = 0.5,
) -> None:
    """Draw a pill with a solid tint (e.g. yellow for Resume, red for Stop)."""
    H, W = frame.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W, x + w)
    y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = tint_bgr
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), border_bgr, 2, cv2.LINE_AA)


def _rec_label() -> str:
    """Single REC button: REC (idle) → PAUSE (recording) → STOP (paused)."""
    if not button_state.is_recording:
        return "REC"
    if button_state.is_paused:
        return "STOP"
    return "PAUSE"


def draw_bottom_hud(
    frame: np.ndarray,
    video_recorder: Optional[object] = None,
    offset_y: float = 0.0,
) -> None:
    """Draw SHOT, REC, and Gallery pills. offset_y moves the bar (0=visible, positive=retracted down)."""
    global _last_layout
    h, _ = frame.shape[:2]
    y = h - PILL_H - BOTTOM_MARGIN + int(offset_y)
    x_start = DB_BAR_WIDTH + LEFT_MARGIN

    x_shot = x_start
    x_rec = x_start + PILL_W + GAP
    x_gallery = x_start + 2 * (PILL_W + GAP)

    rec_text = _rec_label()
    is_rec = button_state.is_recording
    is_paused = getattr(button_state, "is_paused", False)

    # When paused: two buttons rec_resume (60%) and rec_stop (40%); rec is zero-size so it doesn't hit
    resume_w = int(PILL_W * REC_RESUME_FRAC)
    stop_w = PILL_W - resume_w

    layout_key = (h, y, x_shot, x_rec, x_gallery, is_paused)
    _last_layout = layout_key

    # Always update pill rects every frame when drawing so hit-test works whether menu is open or closed
    if "shot" in menu_buttons:
        menu_buttons["shot"].x, menu_buttons["shot"].y, menu_buttons["shot"].w, menu_buttons["shot"].h = x_shot, y, PILL_W, PILL_H
    if "rec" in menu_buttons:
        menu_buttons["rec"].x, menu_buttons["rec"].y = x_rec, y
        menu_buttons["rec"].w = 0 if is_paused else PILL_W
        menu_buttons["rec"].h = PILL_H
        menu_buttons["rec"].text = rec_text
    if is_paused:
        if "rec_resume" not in menu_buttons:
            menu_buttons["rec_resume"] = Button(x_rec, y, resume_w, PILL_H, "Resume")
        else:
            menu_buttons["rec_resume"].x, menu_buttons["rec_resume"].y = x_rec, y
            menu_buttons["rec_resume"].w, menu_buttons["rec_resume"].h = resume_w, PILL_H
        if "rec_stop" not in menu_buttons:
            menu_buttons["rec_stop"] = Button(x_rec + resume_w, y, stop_w, PILL_H, "Stop")
        else:
            menu_buttons["rec_stop"].x, menu_buttons["rec_stop"].y = x_rec + resume_w, y
            menu_buttons["rec_stop"].w, menu_buttons["rec_stop"].h = stop_w, PILL_H
    else:
        if "rec_resume" in menu_buttons:
            menu_buttons["rec_resume"].w = 0
        if "rec_stop" in menu_buttons:
            menu_buttons["rec_stop"].w = 0
    if "gallery" not in menu_buttons:
        menu_buttons["gallery"] = Button(x_gallery, y, PILL_W, PILL_H, "GALLERY")
    else:
        menu_buttons["gallery"].x, menu_buttons["gallery"].y, menu_buttons["gallery"].w, menu_buttons["gallery"].h = x_gallery, y, PILL_W, PILL_H

    font = cv2.FONT_HERSHEY_SIMPLEX
    cy = y + PILL_H // 2
    text_baseline = cy + 6
    icon_w = ICON_SIZE * 2
    gap_icon_text = 6

    def _center_text(pill_x: int, pill_w: int, label: str, scale: float = 0.52):
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        return pill_x + (pill_w - tw) // 2, tw

    def _centered_icon_text_start(pill_x: int, pill_w: int, label: str, scale: float = 0.52):
        (tw, _), _ = cv2.getTextSize(label, font, scale, 1)
        total = icon_w + gap_icon_text + tw
        return pill_x + (pill_w - total) // 2

    # ── SHOT pill ──
    _draw_pill(frame, x_shot, y, PILL_W, PILL_H, is_active=False, alpha=BOTTOM_PILL_ALPHA)
    shot_start = _centered_icon_text_start(x_shot, PILL_W, "SHOT")
    _draw_camera_icon(frame, shot_start + icon_w // 2, cy, size=ICON_SIZE)
    cv2.putText(frame, "SHOT", (shot_start + icon_w + gap_icon_text, text_baseline), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ── REC pill(s): one pill or split into Resume (yellow, 60%) + Stop (red, 40%) when paused ──
    rec_cy = cy
    if is_paused:
        _draw_pill_tint(frame, x_rec, y, resume_w, PILL_H, RESUME_PILL_BG, RESUME_PILL_BORDER, alpha=BOTTOM_PILL_ALPHA)
        _draw_pill_tint(frame, x_rec + resume_w, y, stop_w, PILL_H, STOP_PILL_BG, STOP_PILL_BORDER, alpha=BOTTOM_PILL_ALPHA)
        # Resume pill: pause icon (||) + "Resume" on first line; time tracker below, both shifted up ~15% for better balance
        pause_size = 10
        (resume_tw, resume_th), _ = cv2.getTextSize("Resume", font, 0.48, 1)
        content_w = pause_size * 2 + 4 + gap_icon_text + resume_tw
        start_x = x_rec + (resume_w - content_w) // 2
        pause_cx = start_x + pause_size + 2
        y_offset = int(PILL_H * 0.15)
        line1_y = y + PILL_H // 2 - 4 - y_offset
        line2_y = y + PILL_H // 2 + resume_th + 4 - y_offset
        _draw_pause_icon(frame, pause_cx, line1_y, size=pause_size, color=(0, 0, 0))
        cv2.putText(frame, "Resume", (start_x + pause_size * 2 + 4 + gap_icon_text, line1_y + 6), font, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
        if video_recorder is not None and getattr(video_recorder, "get_elapsed_time", None):
            elapsed = video_recorder.get_elapsed_time()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_txt = f"{minutes:02d}:{seconds:02d}"
            (tw, th), _ = cv2.getTextSize(time_txt, font, 0.5, 1)
            # Center tracker directly under the center of the "Resume" text
            resume_text_x = start_x + pause_size * 2 + 4 + gap_icon_text
            resume_center_x = resume_text_x + resume_tw // 2
            time_x = resume_center_x - tw // 2
            cv2.putText(frame, time_txt, (time_x, line2_y + 6), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        # Stop pill: centered "Stop"
        (stop_tx, _) = _center_text(x_rec + resume_w, stop_w, "Stop")
        cv2.putText(frame, "Stop", (stop_tx, text_baseline), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        _draw_pill(frame, x_rec, y, PILL_W, PILL_H, is_active=is_rec, alpha=BOTTOM_PILL_ALPHA)
        if is_rec and video_recorder is not None and getattr(video_recorder, "get_elapsed_time", None):
            rec_cx = x_rec + 22
            elapsed = video_recorder.get_elapsed_time()
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            time_txt = f"{minutes:02d}:{seconds:02d}"
            (tw, th), _ = cv2.getTextSize(time_txt, font, 0.55, 1)
            time_x = x_rec + PILL_W - tw - 14
            cv2.putText(frame, time_txt, (time_x, text_baseline), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            pulse = int((time.time() * 2) % 2)
            if pulse:
                cv2.circle(frame, (rec_cx, rec_cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (rec_cx, rec_cy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                _draw_rec_icon(frame, rec_cx, rec_cy, size=6, is_active=True)
        else:
            rec_start = _centered_icon_text_start(x_rec, PILL_W, rec_text)
            rec_cx = rec_start + icon_w // 2
            _draw_rec_icon(frame, rec_cx, rec_cy, size=8, is_active=False)
            cv2.putText(frame, rec_text, (rec_start + icon_w + gap_icon_text, text_baseline), font, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Gallery pill (same blue as top HUD when active or pressed) ──
    gallery_active = button_state.gallery_open or getattr(button_state, "gallery_pill_pressed", False)
    _draw_pill(frame, x_gallery, y, PILL_W, PILL_H, is_active=gallery_active, alpha=BOTTOM_PILL_ALPHA)
    (gal_tw, gal_th), _ = cv2.getTextSize("GALLERY", font, 0.48, 1)
    gal_start_x = x_gallery + (PILL_W - (icon_w + gap_icon_text + gal_tw)) // 2
    _draw_gallery_icon(frame, gal_start_x + icon_w // 2, cy, size=ICON_SIZE)
    gal_tx = gal_start_x + icon_w + gap_icon_text
    gal_ty = text_baseline
    # Very slight neon glow for "GALLERY" text
    pad_g = 8
    fh, fw = frame.shape[:2]
    gx0 = max(0, gal_tx - pad_g)
    gy0 = max(0, gal_ty - gal_th - pad_g)
    gx1 = min(fw, gal_tx + gal_tw + pad_g)
    gy1 = min(fh, gal_ty + pad_g)
    if gx1 > gx0 and gy1 > gy0:
        gpatch = frame[gy0:gy1, gx0:gx1].copy()
        ltx, lty = gal_tx - gx0, gal_ty - gy0
        cv2.putText(gpatch, "GALLERY", (ltx, lty), font, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
        gpatch = cv2.GaussianBlur(gpatch, (0, 0), 2.0)
        feathered_composite(frame, gy0, gy1, gx0, gx1, gpatch, ACTION_BTN_GLOW * 0.6, feather_px=8)
    cv2.putText(frame, "GALLERY", (gal_tx, gal_ty), font, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    # Compass between gallery and MENU: match MENU height, equidistant with 1px buffer each side
    if "menu" in menu_buttons:
        menu_x = menu_buttons["menu"].x
    else:
        menu_x = fw - FREQ_BAR_WIDTH - 180 - 15
    gallery_right = x_gallery + PILL_W
    compass_cx = (gallery_right + COMPASS_BUFFER_PX + menu_x - COMPASS_BUFFER_PX) // 2
    compass_cy = y + PILL_H // 2
    _draw_compass(frame, compass_cx, compass_cy, COMPASS_RADIUS, HUD.compass_heading_deg)

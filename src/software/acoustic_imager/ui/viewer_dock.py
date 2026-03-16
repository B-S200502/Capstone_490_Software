"""
Viewer dock: shared bottom chrome for image/video gallery viewer.

Back button, dock with title/metadata/trash, and (for video) play + progress.
Layout: Line 1 = type | filename | priority | metadata | trash; Line 2 = tag icon + tag text; Line 3 (video) = player.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

from .button import menu_buttons, Button
from .priority_circle import draw_priority_circle_neon
from .grid_side_dock import PRIORITY_COLORS
from ..config import DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT
from ..state import button_state


def _vertical_gradient(h: int, w: int, top_bgr: tuple, bot_bgr: tuple) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top_bgr[c], bot_bgr[c], h, dtype=np.uint8).reshape(-1, 1)
    return out

# Taller dock for 2–3 rows: line1 (type/filename/priority/meta), line2 (tags), line3 (video player)
VIEWER_DOCK_HEIGHT = 118

# Green feedback (match main/HUD active green) for back/play etc.
FEEDBACK_GREEN_BG = (40, 200, 60)
FEEDBACK_GREEN_BORDER = (80, 255, 100)
# Yellow glow feedback for prev/next chevron buttons (no square)
FEEDBACK_YELLOW_GLOW_COLOR = (0, 220, 255)  # BGR light yellow
FEEDBACK_GLOW_RADIUS = 32
FEEDBACK_DURATION_S = 0.28


def _format_mtime(mtime: Optional[datetime]) -> str:
    """Short date/time for dock metadata. Returns '' if mtime is None."""
    if mtime is None:
        return ""
    try:
        return mtime.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _format_filesize(size_bytes: int) -> str:
    """Human-readable file size for dock metadata."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def _valid_tag(s) -> bool:
    """Return True only if s is non-empty, not a placeholder, and has real content (for tag display)."""
    if s is None:
        return False
    t = str(s).strip()
    if not t or t.lower() == "null":
        return False
    if t in ("?", "??", "???"):
        return False
    if not any(c.isalnum() for c in t):
        return False
    cleaned = t.replace("?", "").replace(" ", "").replace("\u00a0", "").replace("\u200b", "").replace("\ufffd", "").strip()
    if not cleaned:
        return False
    return True


def _draw_viewer_tag_icon(frame: np.ndarray, x: int, y: int, color: tuple = (160, 200, 160)) -> None:
    """Draw tag/label icon at (x, y) for dock line 2 (pentagon + circle), 30% larger."""
    cx, cy = x + 12, y + 13
    pts = np.array([
        [cx - 9, cy - 6], [cx + 3, cy - 6],
        [cx + 9, cy],
        [cx + 3, cy + 6], [cx - 9, cy + 6],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 4, cy), 3, color, -1, cv2.LINE_AA)


def trigger_viewer_button_feedback(button_key: str) -> None:
    """Call when user clicks back, prev, next, or play so dock can draw green flash."""
    button_state.viewer_dock_feedback_button = button_key
    button_state.viewer_dock_feedback_time = time.time()


# Back button: same size in grid and viewer, circular
BACK_BTN_SIZE = 50

def draw_viewer_back_button_on_top(frame: np.ndarray) -> None:
    """Draw the back button on top of viewer content so it is never covered. Call after drawing cards/arrows."""
    back_btn_x, back_btn_y = 10, 15
    back_btn_w = back_btn_h = BACK_BTN_SIZE
    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "")
    else:
        menu_buttons["gallery_back"].x = back_btn_x
        menu_buttons["gallery_back"].y = back_btn_y
        menu_buttons["gallery_back"].w = back_btn_w
        menu_buttons["gallery_back"].h = back_btn_h
    menu_buttons["gallery_back"].draw(frame, transparent=True, icon_type="back")
    draw_viewer_button_feedback(
        frame, "gallery_back",
        back_btn_x, back_btn_y, back_btn_w, back_btn_h,
    )


def draw_viewer_button_feedback(
    frame: np.ndarray,
    button_key: str,
    x: int,
    y: int,
    w: int,
    h: int,
    glow_center: Optional[tuple] = None,
) -> None:
    """If this button was recently clicked, draw feedback: yellow glow for prev/next, white glow for back, green for others."""
    if getattr(button_state, "viewer_dock_feedback_button", "") != button_key:
        return
    elapsed = time.time() - getattr(button_state, "viewer_dock_feedback_time", 0.0)
    if elapsed > FEEDBACK_DURATION_S:
        return
    fade = 1.0 - (elapsed / FEEDBACK_DURATION_S)
    frame_h, frame_w = frame.shape[:2]
    cx, cy = x + w // 2, y + h // 2

    if button_key in ("gallery_prev", "gallery_next"):
        # Light yellow glow centered on the double triangles (use glow_center when provided)
        gcx, gcy = glow_center if glow_center is not None else (cx, cy)
        r = FEEDBACK_GLOW_RADIUS
        pad = r + 15
        x0 = max(0, gcx - pad)
        y0 = max(0, gcy - pad)
        x1 = min(frame_w, gcx + pad)
        y1 = min(frame_h, gcy + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        lcx, lcy = gcx - x0, gcy - y0
        glow_canvas = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
        cv2.circle(glow_canvas, (lcx, lcy), r, (255, 255, 255), -1, cv2.LINE_AA)
        glow_canvas = cv2.GaussianBlur(glow_canvas, (21, 21), 5.0)
        glow_canvas = glow_canvas.astype(np.float32)
        g = FEEDBACK_YELLOW_GLOW_COLOR
        for c in range(3):
            glow_canvas[:, :, c] = glow_canvas[:, :, c] * (g[c] / 255.0)
        strength = 0.5 * fade
        roi[:] = np.minimum(255, (roi.astype(np.float32) + glow_canvas * strength)).astype(np.uint8)
        return

    if button_key == "gallery_back":
        # Slight white glow when pressing back (no green square)
        r = 28
        pad = r + 12
        gcx, gcy = cx, cy
        x0 = max(0, gcx - pad)
        y0 = max(0, gcy - pad)
        x1 = min(frame_w, gcx + pad)
        y1 = min(frame_h, gcy + pad)
        if x1 > x0 and y1 > y0:
            roi = frame[y0:y1, x0:x1]
            lcx, lcy = gcx - x0, gcy - y0
            glow_canvas = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
            cv2.circle(glow_canvas, (lcx, lcy), r, (255, 255, 255), -1, cv2.LINE_AA)
            glow_canvas = cv2.GaussianBlur(glow_canvas, (17, 17), 4.0)
            strength = 0.4 * fade
            roi[:] = np.minimum(255, (roi.astype(np.float32) + glow_canvas.astype(np.float32) * strength)).astype(np.uint8)
        return

    # Green flash for play, etc.
    alpha = 0.35 * fade
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(frame_w, x + w)
    y1 = min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    overlay = np.empty_like(roi)
    overlay[:] = FEEDBACK_GREEN_BG
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
    border_alpha = 0.7 * fade
    if border_alpha > 0.05:
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), FEEDBACK_GREEN_BORDER, 2, cv2.LINE_AA)


def draw_viewer_chrome(
    frame: np.ndarray,
    filepath: Path,
    item_type: str,
    mtime: datetime,
    is_video: bool = False,
    total_frames: int = 0,
    fps: float = 0,
    current_idx: int = 0,
    play_text: str = "PLAY",
) -> int:
    """
    Draw shared viewer chrome: Line 1 = type | filename | priority | metadata | trash;
    Line 2 = tag icon + tag text; Line 3 (video only) = play + progress.
    Returns controls_y (top of dock, content area bottom).
    """
    frame_h, frame_w = frame.shape[:2]
    controls_y = frame_h - VIEWER_DOCK_HEIGHT
    font = cv2.FONT_HERSHEY_SIMPLEX

    dock_h = frame_h - controls_y
    dock_roi = frame[controls_y:frame_h, 0:frame_w]
    grad = _vertical_gradient(dock_h, frame_w, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT)
    dock_roi[:] = grad
    cv2.line(frame, (0, controls_y), (frame_w, controls_y), (70, 70, 70), 1)

    pad = 14
    row0_y = controls_y + 22
    row1_y = controls_y + 46
    font_scale_title = 0.58
    font_scale_meta = 0.48
    priority_radius = 7
    priority_gap = 9
    space_for_priority = priority_gap + 2 * priority_radius + 8

    # File size for metadata
    try:
        size_str = _format_filesize(filepath.stat().st_size)
    except Exception:
        size_str = ""

    # Fixed left-to-right: [VID]/[IMG] | filename | metadata | delete
    type_label = "[VID]" if item_type == "video" else "[IMG]"
    type_color = (100, 150, 255) if item_type == "video" else (100, 200, 100)
    (type_w, type_th), _ = cv2.getTextSize(type_label, font, 0.6, 1)
    type_x = pad

    trash_w, trash_h = 58, 32
    trash_x = frame_w - pad - trash_w
    trash_y = row0_y - trash_h // 2
    if "gallery_delete" not in menu_buttons:
        menu_buttons["gallery_delete"] = Button(trash_x, trash_y, trash_w, trash_h, "")
    else:
        menu_buttons["gallery_delete"].x = trash_x
        menu_buttons["gallery_delete"].y = trash_y
        menu_buttons["gallery_delete"].w = trash_w
        menu_buttons["gallery_delete"].h = trash_h
    menu_buttons["gallery_delete"].is_active = True
    menu_buttons["gallery_delete"].draw(frame, transparent=True, active_color=(0, 0, 200), icon_type="trash")

    # Build metadata string (mtime, size, video time) – use ASCII separator to avoid "??" from · in font
    mtime_str = _format_mtime(mtime) if mtime is not None else ""
    meta_parts = []
    if mtime_str:
        meta_parts.append(mtime_str)
    if size_str:
        meta_parts.append(size_str)
    if is_video and total_frames > 0 and fps > 0:
        cur_s = current_idx / fps
        total_s = total_frames / fps
        meta_parts.append(f"{int(cur_s // 60):02d}:{int(cur_s % 60):02d} / {int(total_s // 60):02d}:{int(total_s % 60):02d}")
    meta_str = "  |  ".join(meta_parts)

    fn_x = type_x + type_w + 10
    gap = 10
    (meta_w, _), _ = cv2.getTextSize(meta_str, font, font_scale_meta, 1)
    meta_x = trash_x - gap - meta_w
    filename_max_w = meta_x - space_for_priority - fn_x
    min_filename_w = 60
    if filename_max_w < min_filename_w:
        max_meta_w = trash_x - gap - fn_x - space_for_priority - min_filename_w
        if max_meta_w > 0 and meta_w > max_meta_w:
            for n in range(len(meta_str) - 1, 0, -1):
                candidate = meta_str[:n] + "..."
                (w, _), _ = cv2.getTextSize(candidate, font, font_scale_meta, 1)
                if w <= max_meta_w:
                    meta_str = candidate
                    meta_w = w
                    break
            meta_x = trash_x - gap - meta_w
            filename_max_w = meta_x - space_for_priority - fn_x

    cv2.putText(frame, type_label, (type_x, row0_y), font, 0.6, type_color, 1, cv2.LINE_AA)

    filename = filepath.name
    (fn_w, _), _ = cv2.getTextSize(filename, font, font_scale_title, 1)
    if fn_w > filename_max_w and filename_max_w > 20:
        for n in range(len(filename), 0, -1):
            candidate = filename[:n] + "..."
            (w, _), _ = cv2.getTextSize(candidate, font, font_scale_title, 1)
            if w <= filename_max_w:
                filename = candidate
                fn_w = w
                break
        else:
            filename = "..."
            (fn_w, _), _ = cv2.getTextSize(filename, font, font_scale_title, 1)
    cv2.putText(frame, filename, (fn_x, row0_y), font, font_scale_title, (240, 240, 240), 1, cv2.LINE_AA)

    dot_cx = fn_x + fn_w + priority_gap + priority_radius
    dot_cy = row0_y - type_th // 2
    file_priorities = getattr(button_state, "gallery_file_priorities", {})
    priority = file_priorities.get(filepath.name, "")
    if priority and priority in PRIORITY_COLORS:
        draw_priority_circle_neon(frame, dot_cx, dot_cy, priority_radius, PRIORITY_COLORS[priority])
    else:
        draw_priority_circle_neon(frame, dot_cx, dot_cy, priority_radius, (120, 120, 120), neon=False)

    cv2.putText(frame, meta_str, (meta_x, row0_y), font, font_scale_meta, (240, 240, 240), 1, cv2.LINE_AA)

    tag_data_map = getattr(button_state, "gallery_tag_data", {})
    file_tags_map = getattr(button_state, "gallery_file_tags", {})
    tag_data = tag_data_map.get(filepath.name, {})
    preset_tags = file_tags_map.get(filepath.name, [])
    tag_parts = []
    for key in ("asset_type", "leak_type"):
        val = tag_data.get(key)
        if _valid_tag(val):
            tag_parts.append(str(val).strip())
    for t in preset_tags or []:
        if _valid_tag(t):
            tag_parts.append(str(t).strip())
    tag_text = " | ".join(tag_parts) if tag_parts else ""
    tag_icon_w = 30
    if tag_text:
        _draw_viewer_tag_icon(frame, pad, row1_y - 12, (160, 200, 160))
        max_tag_chars = 40
        if len(tag_text) > max_tag_chars:
            tag_text = tag_text[: max_tag_chars - 3] + "..."
        tag_scale = 0.624
        cv2.putText(frame, tag_text, (pad + tag_icon_w, row1_y + 4), font, tag_scale, (180, 200, 180), 1, cv2.LINE_AA)

    if is_video:
        play_btn_w, play_btn_h = 80, 34
        play_btn_x = pad
        play_btn_y = controls_y + 70
        if "gallery_play" not in menu_buttons:
            menu_buttons["gallery_play"] = Button(play_btn_x, play_btn_y, play_btn_w, play_btn_h, play_text)
        else:
            menu_buttons["gallery_play"].x = play_btn_x
            menu_buttons["gallery_play"].y = play_btn_y
            menu_buttons["gallery_play"].w = play_btn_w
            menu_buttons["gallery_play"].h = play_btn_h
            menu_buttons["gallery_play"].text = play_text
        menu_buttons["gallery_play"].draw(frame, transparent=True)
        draw_viewer_button_feedback(
            frame, "gallery_play",
            play_btn_x, play_btn_y, play_btn_w, play_btn_h,
        )

        progress_x = play_btn_x + play_btn_w + 12
        progress_h_visual = 8
        play_center_y = play_btn_y + play_btn_h // 2
        progress_y = play_center_y - progress_h_visual // 2
        progress_w = frame_w - progress_x - pad
        progress_h_hit = 40
        progress_y_hit = progress_y - (progress_h_hit - progress_h_visual) // 2
        cv2.rectangle(frame, (progress_x, progress_y), (progress_x + progress_w, progress_y + progress_h_visual),
                     (60, 60, 60), -1)
        if total_frames > 0:
            fill = int(progress_w * (current_idx / total_frames))
            cv2.rectangle(frame, (progress_x, progress_y), (progress_x + fill, progress_y + progress_h_visual),
                         (100, 200, 255), -1)
        if "gallery_progress" not in menu_buttons:
            menu_buttons["gallery_progress"] = Button(progress_x, progress_y_hit, progress_w, progress_h_hit, "")
        else:
            menu_buttons["gallery_progress"].x = progress_x
            menu_buttons["gallery_progress"].y = progress_y_hit
            menu_buttons["gallery_progress"].w = progress_w
            menu_buttons["gallery_progress"].h = progress_h_hit

    return controls_y

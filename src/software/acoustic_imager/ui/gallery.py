"""
Gallery view: grid, image/video viewers, delete modal.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button

# Cache formatted date strings by (year, month, day) to avoid repeated strftime per frame
_DATE_STR_CACHE: Dict[Tuple[int, int, int], str] = {}
# Cache type badge [IMG]/[VID] text dimensions (scale 0.45)
_TYPE_BADGE_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}
from .viewer_dock import (
    draw_viewer_chrome,
    draw_viewer_back_button_on_top,
    draw_viewer_button_feedback,
    VIEWER_DOCK_HEIGHT as VIEWER_DOCK_H,
    BACK_BTN_SIZE,
    _vertical_gradient,
)
from .grid_side_dock import (
    draw_grid_side_dock,
    GRID_SIDE_DOCK_WIDTH,
    DOCK_ROW_HEIGHT,
    DOCK_TOP_INSET_X,
    MODAL_EDGE_MARGIN,
    MODAL_GAP_WEST,
    PRIORITY_COLORS,
    DOCK_ROW_BORDER,
    DOCK_ROW_TOP_HIGHLIGHT,
    DOCK_ROW_WHITE_BORDER,
    SEARCH_BAR_TEXT_COLOR,
)
from .storage_bar import _format_size, feathered_composite
from .keyboard import (
    KEY_WIDTH_MULT_COMPACT,
    COMPACT_KEY_SCALE,
    dimensions_for_scale,
    draw_key_bg_clipped,
    KEY_BORDER_BGR as _TK_KEY_BORDER,
    KEY_TEXT_BGR as _TK_KEY_TEXT,
)
from .standard_keyboard import (
    draw_standard_alpha_keyboard,
    draw_standard_symbol_keyboard,
    compute_standard_keyboard_dimensions,
)
from .priority_circle import draw_priority_circle_neon
from .archive_panel import (
    load_archive_folders,
    save_archive_folders,
    add_folder,
    rename_folder as archive_rename_folder,
    move_files_to_folder,
    remove_files_from_all_folders,
    item_idx_to_grid_pos,
    archive_panel_grid_pos,
    MAX_FOLDERS,
)
from ..state import button_state, HUD
from .. import state as app_state
from ..io.email_config import get_email_verified
from ..config import (
    MENU_ACTIVE_BLUE,
    MENU_ACTIVE_BLUE_LIGHT,
    GALLERY_ACTION_STYLE,
    ACTION_BTN_FILL_DARK_TOP,
    ACTION_BTN_FILL_DARK_BOT,
    ACTION_BTN_NEON_BORDER_BGR,
    ACTION_BTN_NEON_GLOW,
    ACTION_BTN_FILL_ALPHA,
    ACTION_BTN_BORDER_THICKNESS,
    CLASSIC_ACTION_FILL_TOP,
    CLASSIC_ACTION_FILL_BOT,
    CLASSIC_ACTION_TEXT_BGR,
    CLASSIC_ACTION_BORDER_BGR,
    BG_GRADIENT_TOP,
    BG_GRADIENT_BOT,
    DOCK_GRADIENT_TOP,
    DOCK_GRADIENT_BOT,
)

# Viewer nav: double-chevron icon size and faint neon glow
VIEWER_CHEVRON_SIZE = 14
VIEWER_CHEVRON_GLOW_COLOR = (255, 220, 180)  # BGR faint warm white / cyan tint
VIEWER_CHEVRON_GLOW_STRENGTH = 0.35


def _draw_double_chevron_right(frame: np.ndarray, cx: int, cy: int) -> None:
    """Draw double right-pointing chevrons with faint neon glow (no square)."""
    s = VIEWER_CHEVRON_SIZE
    # Two adjacent right-pointing triangles (thick/rounded feel via size)
    pts1 = np.array([[cx + s, cy], [cx - s // 2, cy - s], [cx - s // 2, cy + s]], dtype=np.int32)
    pts2 = np.array([[cx + 2 * s, cy], [cx + s // 2, cy - s], [cx + s // 2, cy + s]], dtype=np.int32)
    _draw_chevron_pair_glow(frame, cx, cy, [pts1, pts2])


def _draw_double_chevron_left(frame: np.ndarray, cx: int, cy: int) -> None:
    """Draw double left-pointing chevrons with faint neon glow (no square)."""
    s = VIEWER_CHEVRON_SIZE
    pts1 = np.array([[cx - s, cy], [cx + s // 2, cy - s], [cx + s // 2, cy + s]], dtype=np.int32)
    pts2 = np.array([[cx - 2 * s, cy], [cx - s // 2, cy - s], [cx - s // 2, cy + s]], dtype=np.int32)
    _draw_chevron_pair_glow(frame, cx, cy, [pts1, pts2])


def _draw_chevron_pair_glow(frame: np.ndarray, cx: int, cy: int, poly_list: list) -> None:
    """Draw a list of polygon (chevron) shapes with faint neon glow, then solid white fill."""
    h, w = frame.shape[:2]
    pad = 25
    x0 = max(0, cx - pad)
    y0 = max(0, cy - pad)
    x1 = min(w, cx + pad)
    y1 = min(h, cy + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    lcx, lcy = cx - x0, cy - y0
    glow_canvas = np.zeros((roi.shape[0], roi.shape[1], 3), dtype=np.uint8)
    for pts in poly_list:
        local_pts = pts - np.array([x0, y0])
        cv2.fillPoly(glow_canvas, [local_pts], (255, 255, 255), cv2.LINE_AA)
    glow_canvas = cv2.GaussianBlur(glow_canvas, (15, 15), 4.0)
    glow_canvas = glow_canvas.astype(np.float32)
    g = VIEWER_CHEVRON_GLOW_COLOR
    for c in range(3):
        glow_canvas[:, :, c] = glow_canvas[:, :, c] * (g[c] / 255.0)
    roi[:] = np.minimum(255, (roi.astype(np.float32) + glow_canvas * VIEWER_CHEVRON_GLOW_STRENGTH)).astype(np.uint8)
    for pts in poly_list:
        cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, (240, 240, 240), 1, cv2.LINE_AA)


# Rubber band at first/last item: stiff so no ghost card appears beyond the end
VIEWER_RUBBER_BAND_FACTOR = 0.35

# Carousel: card width and gap so next/prev are always visible (continuous peek)
VIEWER_CARD_WIDTH_RATIO = 0.88   # card 88% of viewport → ~6% peek each side, slight step-back, more fluid
VIEWER_CARD_GAP_RATIO = 0.02     # 2% gap between cards
VIEWER_SWIPE_SNAP_RATIO = 0.42   # snap to next/prev after dragging this fraction of slot width
VIEWER_AT_REST_EPSILON = 1.0     # when |offset| < this and not dragging/inertia, show only current (black borders)
# At first/last: show neighbor only when swiping toward it past this (px). During inertia never show neighbor (no ghost).
VIEWER_EDGE_PEEK_THRESHOLD_PX = 18

# Cache first frame of videos for carousel peek (path -> BGR frame)
_video_first_frame_cache: Dict[Path, np.ndarray] = {}

# Tag keyboard uses .keyboard module (COMPACT_KEY_SCALE); _TK_* aliases set after import
_TK_DIMS = dimensions_for_scale(COMPACT_KEY_SCALE, width_mult=KEY_WIDTH_MULT_COMPACT)
_TK_W = _TK_DIMS["key_w"]
_TK_H = _TK_DIMS["key_h"]
_TK_GAP = _TK_DIMS["key_gap"]
_TK_BAR_H = _TK_DIMS["bar_h"]
_TK_SP_W = _TK_W * _TK_DIMS["special_key_w_mult"]
_TK_FONT_KEY = _TK_DIMS["font_key"]
_TK_FONT_SPECIAL = _TK_DIMS["font_special"]

_TAG_FIELDS = [
    ("asset_name", "Asset Name", ""),
    ("asset_type", "Asset Type", "Ex. Pipe, Bolt, etc..."),
    ("leak_type",  "Leak Type",  "Ex. Gas, Air, etc..."),
]


def _tk_vgrad(h: int, w: int, top: tuple, bot: tuple) -> np.ndarray:
    """Vertical gradient helper for tag keyboard keys."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top[c], bot[c], h, dtype=np.uint8).reshape(-1, 1)
    return out


def _draw_tag_icon_small(frame: np.ndarray, cx: int, cy: int,
                          color: tuple = (200, 200, 200)) -> None:
    """Draw a small tag/label pentagon icon centered at (cx, cy)."""
    pts = np.array([
        [cx - 7, cy - 5], [cx + 2, cy - 5],
        [cx + 7, cy],
        [cx + 2, cy + 5], [cx - 7, cy + 5],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 3, cy), 2, color, -1, cv2.LINE_AA)


def _draw_folder_icon(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    name: str,
    n_files: int,
) -> None:
    """Draw a folder-shaped icon (gallery button blue, tabbed, glossy) at (x,y) with size w x h."""
    # Match gallery action button blue (CLASSIC_ACTION_FILL_TOP / MENU_ACTIVE_BLUE)
    base_color = CLASSIC_ACTION_FILL_TOP  # (130, 82, 32) BGR
    highlight_color = (255, 255, 255)
    shadow_color = (90, 60, 20)  # darker blue for shadow

    tab_h = max(6, h // 4)
    tab_w = max(14, w // 2)

    # Folder shape: body + tab. Tab at top-left, body below.
    # Main body rectangle
    cv2.rectangle(frame, (x + 2, y + tab_h), (x + w - 2, y + h - 2), base_color, -1, cv2.LINE_AA)
    # Tab (raised rectangle on top-left)
    cv2.rectangle(frame, (x + 2, y + 2), (x + tab_w, y + tab_h + 2), base_color, -1, cv2.LINE_AA)

    # Glossy highlight on tab top edge
    cv2.line(frame, (x + 3, y + tab_h), (x + tab_w - 2, y + 3), highlight_color, 1, cv2.LINE_AA)
    # Glossy highlight on body top edge
    cv2.line(frame, (x + 3, y + tab_h + 2), (x + w - 3, y + tab_h + 2), (245, 245, 250), 1, cv2.LINE_AA)
    # Shadow on right and bottom
    cv2.line(frame, (x + w - 2, y + tab_h), (x + w - 2, y + h - 2), shadow_color, 1, cv2.LINE_AA)
    cv2.line(frame, (x + 2, y + h - 2), (x + w - 2, y + h - 2), shadow_color, 1, cv2.LINE_AA)
    # Border (slightly lighter blue for definition)
    border_color = CLASSIC_ACTION_FILL_BOT  # (165, 110, 48)
    cv2.rectangle(frame, (x + 2, y + tab_h), (x + w - 2, y + h - 2), border_color, 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x + 2, y + 2), (x + tab_w, y + tab_h + 2), border_color, 1, cv2.LINE_AA)

    # Label below icon
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = (name or "Folder")[:10]
    cv2.putText(frame, label, (x + 2, y + h + 14), font, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{n_files} files", (x + 2, y + h + 28), font, 0.32, (180, 180, 180), 1, cv2.LINE_AA)


def _draw_archive_panel(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    output_dir: Optional[Path],
    header_h: int,
) -> None:
    """Draw the archive panel: title, + to add folder, or folder list (max 4) with folder icons."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    vis_top = header_h
    vis_bot = frame.shape[0]
    cx0 = max(x, 0)
    cx1 = min(x + w, frame.shape[1])
    cy0 = max(y, vis_top)
    cy1 = min(y + h, vis_bot)
    if cx1 <= cx0 or cy1 <= cy0:
        return

    cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (40, 40, 40), -1)

    # Title bar at top - "Archive" centered
    title_h = 24
    cv2.rectangle(frame, (cx0, cy0), (cx1, cy0 + title_h), (50, 50, 50), -1)
    cv2.line(frame, (cx0, cy0 + title_h), (cx1, cy0 + title_h), (80, 80, 80), 1, cv2.LINE_AA)
    title_text = "Archive"
    (tw, _), _ = cv2.getTextSize(title_text, font, 0.5, 1)
    cv2.putText(frame, title_text, (cx0 + ((cx1 - cx0) - tw) // 2, cy0 + title_h - 6),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    content_y = y + title_h
    content_h = h - title_h

    folders = getattr(button_state, "gallery_archive_folders", [])
    if not folders:
        # Empty: show + icon to add folder
        plus_cx = x + w // 2
        plus_cy = content_y + content_h // 2 - 20
        plus_sz = 40
        cv2.rectangle(frame, (plus_cx - plus_sz, plus_cy - plus_sz), (plus_cx + plus_sz, plus_cy + plus_sz),
                      (0, 0, 0), -1)
        cv2.rectangle(frame, (plus_cx - plus_sz, plus_cy - plus_sz), (plus_cx + plus_sz, plus_cy + plus_sz),
                      CLASSIC_ACTION_BORDER_BGR, 2, cv2.LINE_AA)
        cv2.line(frame, (plus_cx - 20, plus_cy), (plus_cx + 20, plus_cy), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.line(frame, (plus_cx, plus_cy - 20), (plus_cx, plus_cy + 20), (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, "Add folder", (x + w // 2 - 45, y + h - 20), font, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        if "archive_add_btn" not in menu_buttons:
            menu_buttons["archive_add_btn"] = Button(plus_cx - plus_sz, plus_cy - plus_sz, plus_sz * 2, plus_sz * 2, "")
        else:
            b = menu_buttons["archive_add_btn"]
            b.x, b.y, b.w, b.h = plus_cx - plus_sz, plus_cy - plus_sz, plus_sz * 2, plus_sz * 2
    else:
        # 4 equal segments (2x2), each folder centered in its segment
        n_cols, n_rows = 2, 2
        seg_w = w // n_cols
        seg_h = content_h // n_rows
        icon_w = min(seg_w - 16, 70)
        icon_h = min(seg_h - 48, 50)
        icon_w = max(icon_w, 40)
        icon_h = max(icon_h, 36)

        for i, folder in enumerate(folders[:4]):
            r, c = i // 2, i % 2
            seg_x = x + c * seg_w
            seg_y = content_y + r * seg_h
            # Center folder icon in segment
            sx = seg_x + (seg_w - icon_w) // 2
            sy = seg_y + (seg_h - icon_h - 32) // 2
            name = (folder.get("name", "Folder") or "Folder")[:10]
            n_files = len(folder.get("files", []))
            _draw_folder_icon(frame, sx, sy, icon_w, icon_h, name, n_files)
            # Button covers icon + label
            btn_h = icon_h + 32
            key = f"archive_folder_{folder.get('id', i)}"
            if key not in menu_buttons:
                menu_buttons[key] = Button(sx, sy, icon_w, btn_h, "")
            else:
                menu_buttons[key].x, menu_buttons[key].y = sx, sy
                menu_buttons[key].w, menu_buttons[key].h = icon_w, btn_h

        if len(folders) < MAX_FOLDERS:
            add_sz = 28
            add_slot_c = len(folders) % 2
            add_slot_r = len(folders) // 2
            add_seg_x = x + add_slot_c * seg_w
            add_seg_y = content_y + add_slot_r * seg_h
            add_x = add_seg_x + (seg_w - add_sz) // 2
            add_y = add_seg_y + (seg_h - add_sz - 32) // 2
            cv2.rectangle(frame, (add_x, add_y), (add_x + add_sz, add_y + add_sz), (50, 50, 50), -1)
            cv2.rectangle(frame, (add_x, add_y), (add_x + add_sz, add_y + add_sz), CLASSIC_ACTION_BORDER_BGR, 1, cv2.LINE_AA)
            cv2.line(frame, (add_x + 6, add_y + add_sz // 2), (add_x + add_sz - 6, add_y + add_sz // 2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (add_x + add_sz // 2, add_y + 6), (add_x + add_sz // 2, add_y + add_sz - 6), (255, 255, 255), 2, cv2.LINE_AA)
            if "archive_add_btn" not in menu_buttons:
                menu_buttons["archive_add_btn"] = Button(add_x, add_y, add_sz, add_sz, "")
            else:
                menu_buttons["archive_add_btn"].x, menu_buttons["archive_add_btn"].y = add_x, add_y
                menu_buttons["archive_add_btn"].w, menu_buttons["archive_add_btn"].h = add_sz, add_sz

    # Panel hit area
    if "archive_panel" not in menu_buttons:
        menu_buttons["archive_panel"] = Button(x, y, w, h, "")
    else:
        menu_buttons["archive_panel"].x, menu_buttons["archive_panel"].y = x, y
        menu_buttons["archive_panel"].w, menu_buttons["archive_panel"].h = w, h

    # Uniform white border drawn last so nothing overwrites it
    cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (255, 255, 255), 2, cv2.LINE_AA)


def _draw_tag_icon_grid(frame: np.ndarray, cx: int, cy: int,
                         color: tuple = (160, 200, 160)) -> None:
    """Draw tag icon at 10% larger scale for grid labels."""
    pts = np.array([
        [cx - 8, cy - 6], [cx + 2, cy - 6],
        [cx + 8, cy],
        [cx + 2, cy + 6], [cx - 8, cy + 6],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 4, cy), 2, color, -1, cv2.LINE_AA)


def _draw_viewer_tag_button(frame: np.ndarray, filepath: Path) -> None:
    """Tag-icon button in the top-right of the viewer content area."""
    fw = frame.shape[1]
    btn_x, btn_y, btn_w, btn_h = fw - 60, 15, 44, 36
    tag_data = getattr(button_state, 'gallery_tag_data', {})
    has_tags = bool(tag_data.get(filepath.name))
    bg = (50, 85, 50) if has_tags else (35, 35, 42)
    border = (85, 165, 85) if has_tags else (70, 70, 82)
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), bg, -1)
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), border, 1, cv2.LINE_AA)
    _draw_tag_icon_small(frame, btn_x + btn_w // 2, btn_y + btn_h // 2)
    if "viewer_tag_btn" not in menu_buttons:
        menu_buttons["viewer_tag_btn"] = Button(btn_x, btn_y, btn_w, btn_h, "")
    else:
        b = menu_buttons["viewer_tag_btn"]
        b.x, b.y, b.w, b.h = btn_x, btn_y, btn_w, btn_h


def draw_tag_info_panel(frame: np.ndarray, filepath: Path) -> None:
    """Read-only info panel (top-right) showing tag data for the current viewer file."""
    fw = frame.shape[1]
    panel_w, panel_h = 265, 115
    panel_x = fw - panel_w - 8
    panel_y = 58
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (28, 28, 34), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (90, 90, 102), 2, cv2.LINE_AA)
    cv2.line(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y), (130, 110, 80), 2, cv2.LINE_AA)

    tag_data = getattr(button_state, 'gallery_tag_data', {})
    data = tag_data.get(filepath.name, {})

    lines = [
        ("Asset Name", filepath.stem),
        ("Asset Type", data.get("asset_type", "") or "—"),
        ("Leak Type",  data.get("leak_type",  "") or "—"),
    ]
    y = panel_y + 24
    for label, value in lines:
        cv2.putText(frame, label + ":", (panel_x + 10, y),
                    font, 0.42, (160, 160, 175), 1, cv2.LINE_AA)
        val_disp = value[:26] if len(value) > 26 else value
        cv2.putText(frame, val_disp, (panel_x + 110, y),
                    font, 0.42, (230, 230, 235), 1, cv2.LINE_AA)
        y += 28

    # Close [x] button
    cx_btn = panel_x + panel_w - 20
    cy_btn = panel_y + 6
    cv2.putText(frame, "x", (cx_btn, cy_btn + 13), font, 0.48, (170, 170, 180), 1, cv2.LINE_AA)
    if "tag_info_close" not in menu_buttons:
        menu_buttons["tag_info_close"] = Button(cx_btn - 4, cy_btn, 22, 20, "x")
    else:
        b = menu_buttons["tag_info_close"]
        b.x, b.y, b.w, b.h = cx_btn - 4, cy_btn, 22, 20

    if "tag_info_panel" not in menu_buttons:
        menu_buttons["tag_info_panel"] = Button(panel_x, panel_y, panel_w, panel_h, "")
    else:
        b = menu_buttons["tag_info_panel"]
        b.x, b.y, b.w, b.h = panel_x, panel_y, panel_w, panel_h


def _draw_tag_keyboard(frame: np.ndarray, y_top: int, form_x: int, form_w: int, available_height: int = 0, fused: bool = False) -> None:
    """Reuse standard keyboard from standard_keyboard module; when fused=True no separate panel (keyboard is part of Edit Tags modal)."""
    fh, _ = frame.shape[:2]
    n_rows = 5
    # Fused: tighter gap and margins to maximize key surface; standalone: original spacing
    if fused and available_height > 0:
        key_gap = 5
        keyboard_region_h = max(120, available_height - _TK_BAR_H - 2)
        panel_bottom_margin = 4
    else:
        key_gap = 6
        keyboard_region_h = max(120, available_height - _TK_BAR_H - 4) if available_height > 0 else 200
        panel_bottom_margin = 6
    dims = compute_standard_keyboard_dimensions(form_w, keyboard_region_h, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    keyboard_h = n_rows * (key_h + key_gap) + key_gap
    panel_w = form_w
    panel_h = _TK_BAR_H + keyboard_h + panel_bottom_margin
    px = form_x
    py = y_top

    if not fused:
        roi = frame[py: py + panel_h, px: px + panel_w]
        if GALLERY_ACTION_STYLE == "classic":
            fill_top, fill_bot = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
            panel_border = CLASSIC_ACTION_BORDER_BGR
        else:
            fill_top, fill_bot = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
            panel_border = ACTION_BTN_NEON_BORDER_BGR
        grad = _vertical_gradient(panel_h, panel_w, fill_top, fill_bot)
        cv2.addWeighted(grad, ACTION_BTN_FILL_ALPHA, roi, 1.0 - ACTION_BTN_FILL_ALPHA, 0.0, dst=roi)
        if GALLERY_ACTION_STYLE != "classic" and ACTION_BTN_NEON_GLOW > 0:
            pad = 16
            gx0, gy0 = max(0, px - pad), max(0, py - pad)
            gx1 = min(frame.shape[1], px + panel_w + pad)
            gy1 = min(frame.shape[0], py + panel_h + pad)
            if gx1 > gx0 and gy1 > gy0:
                glow_patch = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
                lx, ly = px - gx0, py - gy0
                cv2.rectangle(glow_patch, (lx, ly), (lx + panel_w, ly + panel_h), panel_border, 8, cv2.LINE_AA)
                glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 6.0)
                feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, ACTION_BTN_NEON_GLOW, feather_px=14)
        cv2.rectangle(frame, (px, py), (px + panel_w - 1, py + panel_h - 1), panel_border, max(1, ACTION_BTN_BORDER_THICKNESS), cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    field_key = button_state.gallery_tag_active_field
    query = button_state.gallery_tag_keyboard_query
    bar_text_color = CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else (240, 240, 240)
    field_label = next((lbl for k, lbl, _ in _TAG_FIELDS if k == field_key), "")
    bar_text = f"{field_label}: {query[:30]}" if field_label else query[:35]
    cv2.putText(frame, bar_text, (px + 8, py + _TK_BAR_H - 10),
                font, 0.48, bar_text_color, 1, cv2.LINE_AA)

    # Keys always use dark text on light gray gradient (never classic white — keys are not classic buttons)
    key_border = _TK_KEY_BORDER
    key_text = _TK_KEY_TEXT
    key_fill = None
    key_y = py + _TK_BAR_H + key_gap
    key_font = 0.66 if key_h > 32 else 0.56
    key_font_special = 0.62 if key_h > 32 else 0.52
    is_symbol = getattr(button_state, "gallery_keyboard_mode", "alpha") == "symbol"
    shift_highlight = getattr(button_state, "gallery_keyboard_shift_next", False)
    if is_symbol:
        draw_standard_symbol_keyboard(
            frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
            "tag_key_", font, key_font, key_font_special,
            key_border_bgr=key_border, key_text_bgr=key_text, key_fill_bgr=key_fill,
        )
    else:
        draw_standard_alpha_keyboard(
            frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
            "tag_key_", font, key_font, key_font_special, shift_highlight,
            key_border_bgr=key_border, key_text_bgr=key_text,
            shift_highlight_color=MENU_ACTIVE_BLUE, key_fill_bgr=key_fill,
        )

    if "tag_keyboard_panel" not in menu_buttons:
        menu_buttons["tag_keyboard_panel"] = Button(px, py, panel_w, panel_h, "")
    else:
        b = menu_buttons["tag_keyboard_panel"]
        b.x, b.y, b.w, b.h = px, py, panel_w, panel_h


def draw_tag_modal(frame: np.ndarray, output_dir: Optional[Path], header_h: int) -> None:
    """Edit Tags form: 3 input fields + keyboard. Extends from left edge to dock."""
    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    dock_x = fw - GRID_SIDE_DOCK_WIDTH
    form_x = MODAL_EDGE_MARGIN
    form_w = dock_x - form_x  # extend to kiss dock (no gap west of storage)
    # Tighter form so keyboard + preview bar move up and get more vertical space
    form_h = 208  # title + 3 fields (reduced from 230 to expand keyboard upward)
    tags_row_top = header_h + 3
    form_y = min(tags_row_top, fh - form_h - 4)
    # No form background: dock draws strip + panel with same growth animation as other modals
    progress = getattr(button_state, "gallery_modal_anim_progress", 0.0)
    t = max(0.0, min(1.0, progress))
    progress_eased = t * t * (3.0 - 2.0 * t)
    content_visible = progress_eased >= 0.999

    if not content_visible:
        if "tag_modal_panel" not in menu_buttons:
            menu_buttons["tag_modal_panel"] = Button(form_x, form_y, form_w, form_h, "")
        else:
            b = menu_buttons["tag_modal_panel"]
            b.x, b.y, b.w, b.h = form_x, form_y, form_w, form_h
        return

    title = "Edit Tags"
    title_scale = 0.60  # ~15% larger than 0.52
    title_color = CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else SEARCH_BAR_TEXT_COLOR
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, 1)
    cv2.putText(frame, title, (form_x + (form_w - tw) // 2, form_y + 34),
                font, title_scale, title_color, 1, cv2.LINE_AA)

    # Determine which file is being edited (first selected in grid)
    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_paths = [r['filepath'] for r in rects if r['idx'] in button_state.gallery_selected_items]
    first_path: Optional[Path] = sel_paths[0] if sel_paths else None

    field_vals = getattr(button_state, 'gallery_tag_field_values', {})
    lbl_w = 108
    input_x = form_x + lbl_w + 20
    input_w = form_w - lbl_w - 36
    input_h = 40
    row_start_y = form_y + 52
    row_gap = 50  # tighter so preview bar + keyboard sit higher

    for fi, (fkey, flabel, placeholder) in enumerate(_TAG_FIELDS):
        ry = row_start_y + fi * row_gap

        # Row label
        form_text_color = CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else SEARCH_BAR_TEXT_COLOR
        (lw, lh), _ = cv2.getTextSize(flabel, font, 0.48, 1)
        cv2.putText(frame, flabel, (form_x + 14, ry + input_h // 2 + lh // 2),
                    font, 0.48, form_text_color, 1, cv2.LINE_AA)

        is_active = (button_state.gallery_tag_active_field == fkey)
        # Selected field: white border (active). Other two: no visible border (match input bg).
        input_bg = (28, 28, 34)
        border = (255, 255, 255) if is_active else input_bg
        border_thick = 2 if is_active else 1
        cv2.rectangle(frame, (input_x, ry), (input_x + input_w, ry + input_h), input_bg, -1)
        cv2.rectangle(frame, (input_x, ry), (input_x + input_w, ry + input_h),
                      border, border_thick, cv2.LINE_AA)

        if is_active:
            display = button_state.gallery_tag_keyboard_query
        else:
            if fkey == "asset_name":
                display = field_vals.get("asset_name", first_path.stem if first_path else "")
            else:
                display = field_vals.get(fkey, "")

        if display:
            cv2.putText(frame, display[:40], (input_x + 9, ry + input_h // 2 + 7),
                        font, 0.48, form_text_color, 1, cv2.LINE_AA)
        elif placeholder:
            placeholder_color = (200, 200, 200) if GALLERY_ACTION_STYLE == "classic" else (120, 120, 128)
            cv2.putText(frame, placeholder, (input_x + 9, ry + input_h // 2 + 7),
                        font, 0.45, placeholder_color, 1, cv2.LINE_AA)

        # Blinking cursor in active field (same as email form)
        if is_active:
            cursor_index = max(0, min(getattr(button_state, "gallery_tag_cursor_index", 0), len(display)))
            prefix = display[:cursor_index]
            (cw, _), _ = cv2.getTextSize(prefix, font, 0.48, 1)
            cursor_x = input_x + 9 + cw
            if int(time.time() * 2) % 2 == 0:
                cy1, cy2 = ry + 6, ry + input_h - 6
                cv2.line(frame, (cursor_x, cy1), (cursor_x, cy2), (255, 255, 255), 2, cv2.LINE_AA)

        bkey = f"tag_field_{fkey}"
        if bkey not in menu_buttons:
            menu_buttons[bkey] = Button(input_x, ry, input_w, input_h, flabel)
        else:
            b = menu_buttons[bkey]
            b.x, b.y, b.w, b.h = input_x, ry, input_w, input_h

    # ── Keyboard fused with modal; maximize space (preview bar + keys upward, spread width) ──
    available_below = fh - (form_y + form_h) - 4
    key_gap = 5  # slightly tighter for larger key surfaces
    n_rows = 5
    keyboard_region_h = max(120, available_below - _TK_BAR_H - 2)
    dims = compute_standard_keyboard_dimensions(form_w, keyboard_region_h, key_gap=key_gap, n_rows=n_rows)
    key_h = dims["key_h"]
    keyboard_h = n_rows * (key_h + key_gap) + key_gap
    keyboard_panel_h = _TK_BAR_H + keyboard_h + 4
    _draw_tag_keyboard(frame, form_y + form_h, form_x, form_w, available_below, fused=True)

    # Single panel covering form + keyboard so they are one fused Edit Tags modal
    if "tag_modal_panel" not in menu_buttons:
        menu_buttons["tag_modal_panel"] = Button(form_x, form_y, form_w, form_h + keyboard_panel_h, "")
    else:
        b = menu_buttons["tag_modal_panel"]
        b.x, b.y, b.w, b.h = form_x, form_y, form_w, form_h + keyboard_panel_h


def _viewer_rubber_band_offset(offset: float, idx: int, n: int) -> float:
    """Apply rubber-band damping when dragging past first or last item."""
    if n <= 0:
        return 0.0
    if idx <= 0 and offset < 0:
        return offset * VIEWER_RUBBER_BAND_FACTOR
    if idx >= n - 1 and offset > 0:
        return offset * VIEWER_RUBBER_BAND_FACTOR
    return offset


def _viewer_at_rest() -> bool:
    """True when not swiping: no drag, no inertia, offset near zero. Show only current card with black borders."""
    off = getattr(button_state, "gallery_viewer_swipe_offset", 0.0)
    dragging = getattr(button_state, "gallery_viewer_swipe_dragging", False)
    inertia = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
    # Snap to full-width sooner: treat as at rest when offset has decayed to ~15px even if inertia still ticking
    if inertia and abs(off) < 15.0:
        return True
    return not dragging and not inertia and abs(off) < VIEWER_AT_REST_EPSILON


def _carousel_indices(idx: int, n: int) -> List[int]:
    """Indices to draw for carousel: current plus prev/next only when in range. No wrap."""
    out = [idx]
    if idx > 0:
        out.insert(0, idx - 1)
    if idx < n - 1:
        out.append(idx + 1)
    return out


def _blit_card_into_view(
    frame: np.ndarray,
    card_buf: np.ndarray,
    card_left: int,
    card_w: int,
    frame_w: int,
    available_h: int,
) -> None:
    """Blit the visible portion of a carousel card (card_w x available_h) into the viewport."""
    dst_x0 = max(0, card_left)
    dst_x1 = min(frame_w, card_left + card_w)
    if dst_x0 >= dst_x1:
        return
    src_x0 = dst_x0 - card_left
    src_x1 = dst_x1 - card_left
    view = frame[0:available_h, dst_x0:dst_x1]
    view[:] = card_buf[0:available_h, src_x0:src_x1]


def _make_image_card(img: np.ndarray, card_w: int, card_h: int) -> np.ndarray:
    """Render image scaled and centered into a (card_w x card_h) card buffer."""
    card = np.zeros((card_h, card_w, 3), dtype=np.uint8)
    card[:] = (20, 20, 20)
    h, w = img.shape[:2]
    scale = min(card_w / w, card_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x0 = (card_w - new_w) // 2
    y0 = (card_h - new_h) // 2
    card[y0 : y0 + new_h, x0 : x0 + new_w] = img_resized
    return card


# Avoid repeatedly opening known-bad/incomplete video files (reduces log spam from OpenCV/GStreamer)
_video_fail_cache: Dict[Path, float] = {}
_VIDEO_FAIL_CACHE_TTL_S = 60.0


def _video_read_frame_at(filepath: Path, frame_idx: int = 0):
    """
    Read one frame at index from video file. Uses CAP_FFMPEG only (no GStreamer fallback)
    to avoid GStreamer dispose/moov-atom errors on incomplete or in-progress MP4s.
    Skips opening for a while if this path recently failed (reduces log spam).
    Returns (frame, total_frames, fps) or (None, 0, 0.0). Caller does not release; we always release.
    """
    now = time.time()
    if filepath in _video_fail_cache and (now - _video_fail_cache[filepath]) < _VIDEO_FAIL_CACHE_TTL_S:
        return None, 0, 0.0
    cap = None
    try:
        cap = cv2.VideoCapture(str(filepath), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            _video_fail_cache[filepath] = now
            return None, 0, 0.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
        ret, frame = cap.read()
        if filepath in _video_fail_cache:
            del _video_fail_cache[filepath]
        return (frame if (ret and frame is not None) else None), total, fps
    except Exception:
        _video_fail_cache[filepath] = now
        return None, 0, 0.0
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _get_video_first_frame(filepath: Path) -> Optional[np.ndarray]:
    """Return first frame of video, cached. Used for carousel peek. Uses safe open/read/release."""
    if filepath in _video_first_frame_cache:
        return _video_first_frame_cache[filepath]
    frame, _, _ = _video_read_frame_at(filepath, 0)
    if frame is not None:
        _video_first_frame_cache[filepath] = frame
        return frame
    return None


def get_gallery_items(output_dir: Path) -> List[Tuple[Path, str, datetime]]:
    """
    Get all screenshots and videos from the output directory.
    Returns list of tuples: (filepath, type, modification_time)
    Sorted by modification time (newest first). Does not apply filter/sort/search.
    """
    items = []

    if not output_dir.exists():
        return items

    for img_file in output_dir.glob("*.png"):
        mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
        items.append((img_file, "image", mtime))

    for vid_file in output_dir.glob("*.mp4"):
        mtime = datetime.fromtimestamp(vid_file.stat().st_mtime)
        items.append((vid_file, "video", mtime))

    items.sort(key=lambda x: x[2], reverse=True)

    return items


def apply_gallery_filter_sort_search(
    items: List[Tuple[Path, str, datetime]],
) -> List[Tuple[Path, str, datetime]]:
    """
    Apply current filter (type), sort (date/name/size), and search (filename) from button_state.
    """
    out = list(items)
    # Filter by type
    ft = getattr(button_state, "gallery_filter_type", "all")
    if ft == "image":
        out = [x for x in out if x[1] == "image"]
    elif ft == "video":
        out = [x for x in out if x[1] == "video"]
    # Search by filename (case-insensitive substring)
    q = (getattr(button_state, "gallery_search_query", "") or "").strip().lower()
    if q:
        out = [x for x in out if q in x[0].name.lower()]
    # Sort
    sort_by = getattr(button_state, "gallery_sort_by", "date")
    if sort_by == "date":
        out.sort(key=lambda x: x[2], reverse=True)
    elif sort_by == "name":
        out.sort(key=lambda x: x[0].name.lower())
    elif sort_by == "size":
        def _size_key(t: Tuple[Path, str, datetime]) -> float:
            try:
                return -t[0].stat().st_size
            except OSError:
                return 0.0
        out.sort(key=_size_key)
    elif sort_by == "priority":
        _priority_order = {"high": 0, "medium": 1, "low": 2, "": 3}
        _prios = getattr(button_state, "gallery_file_priorities", {})
        out.sort(key=lambda x: _priority_order.get(_prios.get(x[0].name, ""), 3))
    return out


def get_displayed_gallery_items(output_dir: Optional[Path]) -> List[Tuple[Path, str, datetime]]:
    """Return gallery items after filter, sort, and search (what the user sees).
    Main gallery excludes files that are in any archive folder (move semantics)."""
    if not output_dir:
        return []
    all_items = get_gallery_items(output_dir)
    folder_filenames = set()
    for f in getattr(button_state, "gallery_archive_folders", []):
        folder_filenames.update(f.get("files", []))
    gallery_items = [(p, t, m) for p, t, m in all_items if p.name not in folder_filenames]
    return apply_gallery_filter_sort_search(gallery_items)


def get_folder_displayed_items(
    output_dir: Optional[Path],
    folder_id: Optional[str],
) -> List[Tuple[Path, str, datetime]]:
    """Return items in an archive folder, filtered/sorted like main gallery."""
    if not output_dir or not folder_id:
        return []
    folders = getattr(button_state, "gallery_archive_folders", [])
    folder = next((f for f in folders if f.get("id") == folder_id), None)
    if not folder:
        return []
    filenames = set(folder.get("files", []))
    all_items = get_gallery_items(output_dir)
    folder_items = [(p, t, m) for p, t, m in all_items if p.name in filenames]
    return apply_gallery_filter_sort_search(folder_items)


def _update_viewer_swipe_inertia(frame_w: int) -> None:
    """Update viewer swipe offset from inertia; may advance to prev/next when crossing threshold."""
    if not button_state.gallery_viewer_swipe_inertia_active:
        return
    now = time.perf_counter()
    dt = max(1e-6, now - button_state.gallery_viewer_swipe_last_inertia_t)
    button_state.gallery_viewer_swipe_last_inertia_t = now
    vel = button_state.gallery_viewer_swipe_velocity
    new_offset = button_state.gallery_viewer_swipe_offset + vel * dt
    decay = max(0.0, 1.0 - ui_cache.VIEWER_SWIPE_FRICTION * dt)
    button_state.gallery_viewer_swipe_velocity = vel * decay
    if abs(button_state.gallery_viewer_swipe_velocity) < ui_cache.VIEWER_SWIPE_STOP_VELOCITY:
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
    # Rubber band at ends: damp offset so it doesn't fly off
    # Positive offset = swipe right = toward prev (newer). Negative = swipe left = toward next (older).
    n_items = getattr(button_state, "_gallery_items_len", 1)
    idx = button_state.gallery_selected_item or 0
    if idx <= 0 and new_offset > 0:
        new_offset *= VIEWER_RUBBER_BAND_FACTOR
    if idx >= n_items - 1 and new_offset < 0:
        new_offset *= VIEWER_RUBBER_BAND_FACTOR

    # Snap or advance: swipe right (positive) -> prev (newer); swipe left (negative) -> next (older)
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    threshold = max(float(ui_cache.VIEWER_SWIPE_THRESHOLD_PX), slot_w * VIEWER_SWIPE_SNAP_RATIO)
    if new_offset >= threshold:
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
        button_state.gallery_selected_item = max((button_state.gallery_selected_item or 0) - 1, 0)
        if button_state.gallery_viewer_mode == "video":
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
        return
    if new_offset <= -threshold:
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_inertia_active = False
        button_state.gallery_selected_item = min(
            (button_state.gallery_selected_item or 0) + 1,
            getattr(button_state, "_gallery_items_len", 1) - 1,
        )
        if button_state.gallery_viewer_mode == "video":
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
        return
    button_state.gallery_viewer_swipe_offset = new_offset


def draw_image_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """Draw image viewer with shared dock and swipe/inertia."""
    menu_buttons.pop("viewer_tag_btn", None)
    frame_h, frame_w = frame.shape[:2]
    frame[:] = _vertical_gradient(frame_h, frame_w, BG_GRADIENT_TOP, BG_GRADIENT_BOT)

    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return

    button_state._gallery_items_len = len(items)
    _update_viewer_swipe_inertia(frame_w)

    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    font = cv2.FONT_HERSHEY_SIMPLEX
    controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=False)

    available_h = controls_y
    dx_float = _viewer_rubber_band_offset(
        button_state.gallery_viewer_swipe_offset,
        button_state.gallery_selected_item or 0,
        len(items),
    )
    idx = button_state.gallery_selected_item or 0
    n_items = len(items)
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    button_state._gallery_swipe_threshold_px = max(50, int(slot_w * VIEWER_SWIPE_SNAP_RATIO))
    current_image_failed = False

    if _viewer_at_rest():
        # Only current card, full width, centered → black side borders
        path, itype, _ = items[idx]
        if itype == "image":
            img = cv2.imread(str(path))
            if img is not None:
                card = _make_image_card(img, frame_w, available_h)
                _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
            else:
                current_image_failed = True
        else:
            card = np.zeros((available_h, frame_w, 3), dtype=np.uint8)
            card[:] = (25, 25, 25)
            cv2.putText(card, "[VID]", (frame_w // 2 - 30, available_h // 2),
                       font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
    else:
        # Carousel: only valid indices; at edge swiping toward void draw only current (no ghost card)
        card_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO)
        gap = int(frame_w * VIEWER_CARD_GAP_RATIO)
        center_offset = (frame_w - card_w) // 2
        draw_indices = _carousel_indices(idx, n_items)
        # At edge: during inertia never show neighbor (no ghost). When dragging, show neighbor only past threshold.
        edge_px = VIEWER_EDGE_PEEK_THRESHOLD_PX
        inertia_active = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
        if idx == 0:
            if inertia_active or dx_float >= -edge_px:
                draw_indices = [0]  # first: only current during inertia or until swipe left past threshold
        elif idx == n_items - 1:
            if inertia_active or dx_float <= edge_px:
                draw_indices = [n_items - 1]  # last: only current during inertia or until swipe right past threshold
        for i in draw_indices:
            if i < 0 or i >= n_items:
                continue
            path, itype, _ = items[i]
            if itype != "image":
                card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                card[:] = (30, 30, 30)
                cv2.putText(card, "[VID]", (card_w // 2 - 30, available_h // 2),
                           font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                img = cv2.imread(str(path))
                if img is None:
                    current_image_failed = current_image_failed or (i == idx)
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (25, 25, 25)
                else:
                    card = _make_image_card(img, card_w, available_h)
            card_left = int(center_offset + (i - idx) * slot_w + dx_float)
            _blit_card_into_view(frame, card, card_left, card_w, frame_w, available_h)
        # Explicit black fill for void at edges so no content bleeds through
        if idx == n_items - 1:
            void_left = min(frame_w, int(center_offset + dx_float + card_w))
            if void_left < frame_w:
                frame[0:available_h, void_left:frame_w] = (0, 0, 0)
        elif idx == 0:
            void_right = max(0, int(center_offset + dx_float))
            if void_right > 0:
                frame[0:available_h, 0:void_right] = (0, 0, 0)

    if current_image_failed:
        msg = "Failed to load image"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.7, 1)
        cv2.putText(frame, msg, ((frame_w - msg_w) // 2, frame_h // 2),
                   font, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    # Prev/next: double-chevron icons with faint neon glow (no square buttons)
    arrow_y = controls_y // 2 - 28
    prev_btn_x, prev_btn_w, prev_btn_h = 20, 50, 56
    next_btn_x, next_btn_w, next_btn_h = frame_w - 70, 50, 56
    prev_cx, prev_cy = prev_btn_x + prev_btn_w // 2, arrow_y + prev_btn_h // 2
    next_cx, next_cy = next_btn_x + next_btn_w // 2, arrow_y + next_btn_h // 2
    if button_state.gallery_selected_item > 0:
        if "gallery_prev" not in menu_buttons:
            menu_buttons["gallery_prev"] = Button(prev_btn_x, arrow_y, prev_btn_w, prev_btn_h, "")
        else:
            menu_buttons["gallery_prev"].x, menu_buttons["gallery_prev"].y = prev_btn_x, arrow_y
            menu_buttons["gallery_prev"].w, menu_buttons["gallery_prev"].h = prev_btn_w, prev_btn_h
        _draw_double_chevron_left(frame, prev_cx, prev_cy)
        draw_viewer_button_feedback(
            frame, "gallery_prev", prev_btn_x, arrow_y, prev_btn_w, prev_btn_h,
            glow_center=(prev_cx - 10, prev_cy),
        )
    else:
        menu_buttons.pop("gallery_prev", None)
    if button_state.gallery_selected_item < len(items) - 1:
        if "gallery_next" not in menu_buttons:
            menu_buttons["gallery_next"] = Button(next_btn_x, arrow_y, next_btn_w, next_btn_h, "")
        else:
            menu_buttons["gallery_next"].x, menu_buttons["gallery_next"].y = next_btn_x, arrow_y
            menu_buttons["gallery_next"].w, menu_buttons["gallery_next"].h = next_btn_w, next_btn_h
        _draw_double_chevron_right(frame, next_cx, next_cy)
        draw_viewer_button_feedback(
            frame, "gallery_next", next_btn_x, arrow_y, next_btn_w, next_btn_h,
            glow_center=(next_cx + 10, next_cy),
        )
    else:
        menu_buttons.pop("gallery_next", None)

    draw_viewer_back_button_on_top(frame)


def draw_video_viewer(frame: np.ndarray, items: List[Tuple[Path, str, datetime]], output_dir: Optional[Path]) -> None:
    """Draw video player with shared dock and swipe/inertia."""
    menu_buttons.pop("viewer_tag_btn", None)
    frame_h, frame_w = frame.shape[:2]
    frame[:] = _vertical_gradient(frame_h, frame_w, BG_GRADIENT_TOP, BG_GRADIENT_BOT)

    if button_state.gallery_selected_item is None or button_state.gallery_selected_item >= len(items):
        return

    button_state._gallery_items_len = len(items)
    _update_viewer_swipe_inertia(frame_w)

    filepath, item_type, mtime = items[button_state.gallery_selected_item]
    button_state.gallery_video_frame_idx = max(0, button_state.gallery_video_frame_idx)
    vid_frame, total_frames, fps = _video_read_frame_at(filepath, button_state.gallery_video_frame_idx)

    if vid_frame is None and total_frames == 0:
        font = cv2.FONT_HERSHEY_SIMPLEX
        msg = "Unable to play (file may be in use or incomplete)"
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, 0.6, 1)
        cv2.putText(frame, msg, ((frame_w - msg_w) // 2, frame_h // 2),
                   font, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
        controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=True,
                                        total_frames=0, fps=0, current_idx=0, play_text="PLAY")
        draw_viewer_back_button_on_top(frame)
        return

    button_state._gallery_video_total_frames = total_frames
    button_state._gallery_video_fps = fps or 30.0
    fps = button_state._gallery_video_fps
    total_frames = max(1, total_frames)

    if button_state.gallery_video_playing:
        button_state.gallery_video_frame_idx += 1
        if button_state.gallery_video_frame_idx >= total_frames:
            button_state.gallery_video_frame_idx = 0

    button_state.gallery_video_frame_idx = max(0, min(button_state.gallery_video_frame_idx, total_frames - 1))
    ret = vid_frame is not None

    play_text = "PAUSE" if button_state.gallery_video_playing else "PLAY"
    controls_y = draw_viewer_chrome(frame, filepath, item_type, mtime, is_video=True,
                                   total_frames=total_frames, fps=fps,
                                   current_idx=button_state.gallery_video_frame_idx, play_text=play_text)

    available_h = controls_y
    dx_float = _viewer_rubber_band_offset(
        button_state.gallery_viewer_swipe_offset,
        button_state.gallery_selected_item or 0,
        len(items),
    )
    idx = button_state.gallery_selected_item or 0
    n_items = len(items)
    slot_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO) + int(frame_w * VIEWER_CARD_GAP_RATIO)
    button_state._gallery_swipe_threshold_px = max(50, int(slot_w * VIEWER_SWIPE_SNAP_RATIO))
    font = cv2.FONT_HERSHEY_SIMPLEX

    if _viewer_at_rest():
        # Only current card, full width, black borders
        if ret and vid_frame is not None:
            card = _make_image_card(vid_frame, frame_w, available_h)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
        else:
            card = np.zeros((available_h, frame_w, 3), dtype=np.uint8)
            card[:] = (25, 25, 25)
            _blit_card_into_view(frame, card, 0, frame_w, frame_w, available_h)
    else:
        card_w = int(frame_w * VIEWER_CARD_WIDTH_RATIO)
        gap = int(frame_w * VIEWER_CARD_GAP_RATIO)
        center_offset = (frame_w - card_w) // 2
        draw_indices = _carousel_indices(idx, n_items)
        edge_px = VIEWER_EDGE_PEEK_THRESHOLD_PX
        inertia_active = getattr(button_state, "gallery_viewer_swipe_inertia_active", False)
        if idx == 0 and (inertia_active or dx_float >= -edge_px):
            draw_indices = [0]
        elif idx == n_items - 1 and (inertia_active or dx_float <= edge_px):
            draw_indices = [n_items - 1]
        for i in draw_indices:
            if i < 0 or i >= n_items:
                continue
            path, itype, _ = items[i]
            if i == idx and ret and vid_frame is not None:
                card = _make_image_card(vid_frame, card_w, available_h)
            elif itype == "video":
                first = _get_video_first_frame(path)
                if first is not None:
                    card = _make_image_card(first, card_w, available_h)
                else:
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (30, 30, 30)
                    cv2.putText(card, "[VID]", (card_w // 2 - 30, available_h // 2),
                               font, 0.8, (120, 120, 120), 1, cv2.LINE_AA)
            else:
                img = cv2.imread(str(path))
                if img is not None:
                    card = _make_image_card(img, card_w, available_h)
                else:
                    card = np.zeros((available_h, card_w, 3), dtype=np.uint8)
                    card[:] = (25, 25, 25)
            card_left = int(center_offset + (i - idx) * slot_w + dx_float)
            _blit_card_into_view(frame, card, card_left, card_w, frame_w, available_h)
        if idx == n_items - 1:
            void_left = min(frame_w, int(center_offset + dx_float + card_w))
            if void_left < frame_w:
                frame[0:available_h, void_left:frame_w] = (0, 0, 0)
        elif idx == 0:
            void_right = max(0, int(center_offset + dx_float))
            if void_right > 0:
                frame[0:available_h, 0:void_right] = (0, 0, 0)

    # Prev/next: double-chevron icons with faint neon glow (no square buttons)
    arrow_y = controls_y // 2 - 28
    prev_btn_x, prev_btn_w, prev_btn_h = 20, 50, 56
    next_btn_x, next_btn_w, next_btn_h = frame_w - 70, 50, 56
    prev_cx, prev_cy = prev_btn_x + prev_btn_w // 2, arrow_y + prev_btn_h // 2
    next_cx, next_cy = next_btn_x + next_btn_w // 2, arrow_y + next_btn_h // 2
    if button_state.gallery_selected_item > 0:
        if "gallery_prev" not in menu_buttons:
            menu_buttons["gallery_prev"] = Button(prev_btn_x, arrow_y, prev_btn_w, prev_btn_h, "")
        else:
            menu_buttons["gallery_prev"].x, menu_buttons["gallery_prev"].y = prev_btn_x, arrow_y
            menu_buttons["gallery_prev"].w, menu_buttons["gallery_prev"].h = prev_btn_w, prev_btn_h
        _draw_double_chevron_left(frame, prev_cx, prev_cy)
        draw_viewer_button_feedback(
            frame, "gallery_prev", prev_btn_x, arrow_y, prev_btn_w, prev_btn_h,
            glow_center=(prev_cx - 10, prev_cy),
        )
    else:
        menu_buttons.pop("gallery_prev", None)
    if button_state.gallery_selected_item < len(items) - 1:
        if "gallery_next" not in menu_buttons:
            menu_buttons["gallery_next"] = Button(next_btn_x, arrow_y, next_btn_w, next_btn_h, "")
        else:
            menu_buttons["gallery_next"].x, menu_buttons["gallery_next"].y = next_btn_x, arrow_y
            menu_buttons["gallery_next"].w, menu_buttons["gallery_next"].h = next_btn_w, next_btn_h
        _draw_double_chevron_right(frame, next_cx, next_cy)
        draw_viewer_button_feedback(
            frame, "gallery_next", next_btn_x, arrow_y, next_btn_w, next_btn_h,
            glow_center=(next_cx + 10, next_cy),
        )
    else:
        menu_buttons.pop("gallery_next", None)

    draw_viewer_back_button_on_top(frame)


def draw_delete_modal(
    frame: np.ndarray,
    title: str = "Delete this item?",
    subtitle: str = "This action cannot be undone.",
    yes_text: str = "YES, DELETE",
    no_text: str = "CANCEL",
) -> None:
    modal_w = 400
    modal_h = 180
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 0.7
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, 2)
    title_x = modal_x + (modal_w - title_w) // 2
    title_y = modal_y + 50
    cv2.putText(frame, title, (title_x, title_y), font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)

    msg_scale = 0.5
    (msg_w, msg_h), _ = cv2.getTextSize(subtitle, font, msg_scale, 1)
    msg_x = modal_x + (modal_w - msg_w) // 2
    msg_y = modal_y + 85
    cv2.putText(frame, subtitle, (msg_x, msg_y), font, msg_scale, (200, 200, 200), 1, cv2.LINE_AA)

    yes_btn_w = 160
    yes_btn_h = 45
    yes_btn_x = modal_x + (modal_w // 2) - yes_btn_w - 15
    yes_btn_y = modal_y + modal_h - yes_btn_h - 25

    if "modal_yes" not in menu_buttons:
        menu_buttons["modal_yes"] = Button(yes_btn_x, yes_btn_y, yes_btn_w, yes_btn_h, yes_text)
    else:
        b = menu_buttons["modal_yes"]
        b.x, b.y, b.w, b.h, b.text = yes_btn_x, yes_btn_y, yes_btn_w, yes_btn_h, yes_text

    menu_buttons["modal_yes"].is_active = True
    menu_buttons["modal_yes"].draw(frame, transparent=True, active_color=(0, 0, 220))

    no_btn_w = 140
    no_btn_h = 45
    no_btn_x = modal_x + (modal_w // 2) + 15
    no_btn_y = modal_y + modal_h - no_btn_h - 25

    if "modal_no" not in menu_buttons:
        menu_buttons["modal_no"] = Button(no_btn_x, no_btn_y, no_btn_w, no_btn_h, no_text)
    else:
        b = menu_buttons["modal_no"]
        b.x, b.y, b.w, b.h, b.text = no_btn_x, no_btn_y, no_btn_w, no_btn_h, no_text

    menu_buttons["modal_no"].draw(frame, transparent=True)


def draw_share_modal(frame: np.ndarray) -> None:
    """Small modal: title, message (multi-line), OK button. Used for share result or warnings. While sending, no OK."""
    title = getattr(button_state, "share_modal_title", "Share")
    message = getattr(button_state, "share_modal_message", "")
    sending = getattr(button_state, "share_modal_sending", False)
    progress = getattr(button_state, "share_modal_progress", 0.0)
    modal_w = 420
    line_h = 24
    lines = message.split("\n") if message else [""]
    msg_h = min(120, len(lines) * line_h + 20)
    # Extra height for progress bar when sending
    progress_bar_h = 28 if sending else 0
    modal_h = 80 + msg_h + 55 + progress_bar_h
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.64
    (tw, title_h), _ = cv2.getTextSize(title, font, title_scale, 2)
    tit_x = modal_x + (modal_w - tw) // 2
    tit_y = modal_y + 38
    cv2.putText(frame, title, (tit_x, tit_y), font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)

    msg_scale = 0.48
    for i, line in enumerate(lines[:5]):
        msg_y = modal_y + 62 + i * line_h
        cv2.putText(frame, line[:60], (modal_x + 20, msg_y), font, msg_scale, (220, 220, 220), 1, cv2.LINE_AA)

    # Progress bar when sending: rectangle with fill increasing left-to-right
    if sending:
        bar_x = modal_x + 24
        bar_y = modal_y + 62 + len(lines[:5]) * line_h + 12
        bar_w = modal_w - 48
        bar_h = 18
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), 1, cv2.LINE_AA)
        fill_w = max(0, int(bar_w * min(1.0, max(0.0, progress))))
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (80, 180, 80), -1, cv2.LINE_AA)

    # OK only when not sending (thread finished or warning)
    if not sending:
        ok_btn_w = 100
        ok_btn_h = 40
        ok_btn_x = modal_x + (modal_w - ok_btn_w) // 2
        ok_btn_y = modal_y + modal_h - ok_btn_h - 18
        if "share_modal_ok" not in menu_buttons:
            menu_buttons["share_modal_ok"] = Button(ok_btn_x, ok_btn_y, ok_btn_w, ok_btn_h, "OK")
        else:
            b = menu_buttons["share_modal_ok"]
            b.x, b.y, b.w, b.h = ok_btn_x, ok_btn_y, ok_btn_w, ok_btn_h
        menu_buttons["share_modal_ok"].is_active = True
        menu_buttons["share_modal_ok"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)
    else:
        if "share_modal_ok" in menu_buttons:
            menu_buttons["share_modal_ok"].w = 0
            menu_buttons["share_modal_ok"].h = 0


def draw_share_confirm_modal(frame: np.ndarray) -> None:
    """Modal before sending: 'Send by email?' with details (To, files, size) and Send / Cancel buttons."""
    to_email = getattr(button_state, "share_confirm_to_email", "") or ""
    n_images = getattr(button_state, "share_confirm_n_images", 0)
    n_videos = getattr(button_state, "share_confirm_n_videos", 0)
    size_str = getattr(button_state, "share_confirm_size_str", "")
    file_count = getattr(button_state, "share_confirm_file_count", 0)
    lines = [
        f"To: {to_email[:50]}" if to_email else "To: (not set)",
        f"Files: {file_count} ({n_images} image(s), {n_videos} video(s))",
        f"Size: {size_str}",
        "",
        "Send these as attachments?",
    ]
    title = "Send by email?"
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 0.72
    msg_scale = 0.52
    line_h = 28
    padding_h = 40
    title_top = 32
    msg_top = 58
    btn_h = 42
    btn_gap = 18
    btn_bottom_margin = 24

    (title_w, title_th), _ = cv2.getTextSize(title, font, title_scale, 2)
    msg_lines = lines[:6]
    content_h = msg_top + len(msg_lines) * line_h + 20 + btn_h + btn_bottom_margin
    modal_w = max(460, title_w + 2 * padding_h)
    modal_h = content_h
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (48, 48, 48), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (120, 120, 120), 2, cv2.LINE_AA)

    # Hit area for modal content (tap outside this closes the modal)
    if "share_confirm_modal_panel" not in menu_buttons:
        menu_buttons["share_confirm_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["share_confirm_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h

    tit_x = modal_x + (modal_w - title_w) // 2
    tit_y = modal_y + title_top
    cv2.putText(frame, title, (tit_x, tit_y), font, title_scale, (255, 255, 255), 2, cv2.LINE_AA)

    for i, line in enumerate(msg_lines):
        text = (line[:62] if line else "").strip()
        if text:
            (lw, lh), _ = cv2.getTextSize(text, font, msg_scale, 1)
            msg_x = modal_x + (modal_w - lw) // 2
            msg_y = modal_y + msg_top + i * line_h + int(line_h * 0.35)
            cv2.putText(frame, text, (msg_x, msg_y), font, msg_scale, (235, 235, 235), 1, cv2.LINE_AA)

    btn_w = 100
    btn_y = modal_y + modal_h - btn_h - btn_bottom_margin
    total_btns_w = btn_w * 2 + btn_gap
    btn_block_x = modal_x + (modal_w - total_btns_w) // 2
    send_btn_x = btn_block_x
    cancel_btn_x = btn_block_x + btn_w + btn_gap
    if "share_confirm_send" not in menu_buttons:
        menu_buttons["share_confirm_send"] = Button(send_btn_x, btn_y, btn_w, btn_h, "Send")
    else:
        b = menu_buttons["share_confirm_send"]
        b.x, b.y, b.w, b.h = send_btn_x, btn_y, btn_w, btn_h
    if "share_confirm_cancel" not in menu_buttons:
        menu_buttons["share_confirm_cancel"] = Button(cancel_btn_x, btn_y, btn_w, btn_h, "Cancel")
    else:
        b = menu_buttons["share_confirm_cancel"]
        b.x, b.y, b.w, b.h = cancel_btn_x, btn_y, btn_w, btn_h
    menu_buttons["share_confirm_send"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)
    menu_buttons["share_confirm_cancel"].draw(frame, transparent=True, active_color=(120, 120, 120))


def draw_move_to_modal(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """Modal to choose a folder when moving selected items. When inside a folder, includes 'Gallery' to remove from folder."""
    folders = getattr(button_state, "gallery_archive_folders", [])
    in_folder_view = bool(getattr(button_state, "gallery_archive_folder_view_id", None))
    if not folders and not in_folder_view:
        button_state.gallery_archive_move_modal_open = False
        return

    # When in folder view, add "Gallery" as first option
    options = []
    if in_folder_view:
        options.append(("__gallery__", "Gallery", 0))
    for folder in folders:
        options.append((folder.get("id", ""), folder.get("name", "Folder") or "Folder", len(folder.get("files", []))))

    modal_w = 320
    row_h = 44
    modal_h = 80 + len(options) * row_h
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Move to folder", (modal_x + 20, modal_y + 35), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (opt_id, name, n_files) in enumerate(options):
        btn_y = modal_y + 55 + i * row_h
        btn_h = row_h - 8
        label = name if opt_id == "__gallery__" else f"{name} ({n_files} files)"
        key = "move_to_gallery" if opt_id == "__gallery__" else "move_to_folder_" + opt_id
        if key not in menu_buttons:
            menu_buttons[key] = Button(modal_x + 15, btn_y, modal_w - 30, btn_h, label)
        else:
            b = menu_buttons[key]
            b.x, b.y, b.w, b.h = modal_x + 15, btn_y, modal_w - 30, btn_h
            b.text = label
        menu_buttons[key].is_active = True
        menu_buttons[key].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=(CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT),
            fill_alpha=ACTION_BTN_FILL_ALPHA,
            neon_border_color=CLASSIC_ACTION_BORDER_BGR,
        )


def _draw_archive_rename_keyboard(frame: np.ndarray, y_top: int, form_x: int, form_w: int) -> None:
    """Unified blue panel for folder rename: title + input at top, standard keyboard below."""
    fh, _ = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    form_h = 100
    form_to_keyboard_gap = 10
    key_gap = 6
    n_rows = 5
    dims = compute_standard_keyboard_dimensions(form_w, 200, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    keyboard_h = n_rows * (key_h + key_gap) + key_gap
    panel_h = form_h + form_to_keyboard_gap + keyboard_h + 6
    panel_w = form_w
    px = form_x
    py = min(y_top, fh - panel_h - 4)

    roi = frame[py: py + panel_h, px: px + panel_w]
    if GALLERY_ACTION_STYLE == "classic":
        fill_top, fill_bot = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
        panel_border = CLASSIC_ACTION_BORDER_BGR
    else:
        fill_top, fill_bot = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
        panel_border = ACTION_BTN_NEON_BORDER_BGR
    grad = _vertical_gradient(panel_h, panel_w, fill_top, fill_bot)
    cv2.addWeighted(grad, ACTION_BTN_FILL_ALPHA, roi, 1.0 - ACTION_BTN_FILL_ALPHA, 0.0, dst=roi)
    if GALLERY_ACTION_STYLE != "classic" and ACTION_BTN_NEON_GLOW > 0:
        pad = 16
        gx0, gy0 = max(0, px - pad), max(0, py - pad)
        gx1 = min(frame.shape[1], px + panel_w + pad)
        gy1 = min(frame.shape[0], py + panel_h + pad)
        if gx1 > gx0 and gy1 > gy0:
            glow_patch = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
            lx, ly = px - gx0, py - gy0
            cv2.rectangle(glow_patch, (lx, ly), (lx + panel_w, ly + panel_h), panel_border, 8, cv2.LINE_AA)
            glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 6.0)
            feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, ACTION_BTN_NEON_GLOW, feather_px=14)
    cv2.rectangle(frame, (px, py), (px + panel_w - 1, py + panel_h - 1), panel_border, max(1, ACTION_BTN_BORDER_THICKNESS), cv2.LINE_AA)

    form_text_color = CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else (240, 240, 240)
    title = "Rename folder"
    (tw, th), _ = cv2.getTextSize(title, font, 0.60, 1)
    cv2.putText(frame, title, (px + (panel_w - tw) // 2, py + 34),
                font, 0.60, form_text_color, 1, cv2.LINE_AA)

    query = getattr(button_state, "gallery_archive_rename_query", "") or ""
    lbl_w = 80
    input_x = px + lbl_w + 20
    input_w = panel_w - lbl_w - 36
    input_h = 40
    ry = py + 50
    border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else DOCK_ROW_WHITE_BORDER
    cv2.putText(frame, "Rename:", (px + 14, ry + input_h // 2 + 7),
                font, 0.48, form_text_color, 1, cv2.LINE_AA)
    input_bg = (28, 28, 34)
    cv2.rectangle(frame, (input_x, ry), (input_x + input_w, ry + input_h), input_bg, -1)
    cv2.rectangle(frame, (input_x, ry), (input_x + input_w, ry + input_h), border, 2, cv2.LINE_AA)
    cv2.putText(frame, query[:40], (input_x + 9, ry + input_h // 2 + 7),
                font, 0.48, form_text_color, 1, cv2.LINE_AA)

    # Keys always dark text on light gray (same as tag keyboard)
    key_border = _TK_KEY_BORDER
    key_text = _TK_KEY_TEXT
    key_y = py + form_h + form_to_keyboard_gap
    key_font = 0.66 if key_h > 32 else 0.56
    key_font_special = 0.62 if key_h > 32 else 0.52
    is_symbol = getattr(button_state, "gallery_keyboard_mode", "alpha") == "symbol"
    shift_highlight = getattr(button_state, "gallery_keyboard_shift_next", False)
    if is_symbol:
        draw_standard_symbol_keyboard(
            frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
            "archive_rename_key_", font, key_font, key_font_special,
            key_border_bgr=key_border, key_text_bgr=key_text,
        )
    else:
        draw_standard_alpha_keyboard(
            frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
            "archive_rename_key_", font, key_font, key_font_special, shift_highlight,
            key_border_bgr=key_border, key_text_bgr=key_text,
            shift_highlight_color=MENU_ACTIVE_BLUE,
        )

    if "archive_rename_keyboard_panel" not in menu_buttons:
        menu_buttons["archive_rename_keyboard_panel"] = Button(px, py, panel_w, panel_h, "")
    else:
        b = menu_buttons["archive_rename_keyboard_panel"]
        b.x, b.y, b.w, b.h = px, py, panel_w, panel_h


def draw_archive_folder_action_modal(frame: np.ndarray, output_dir: Optional[Path], header_h: int = 98) -> None:
    """Modal for Rename/Delete when in select mode and user clicked a folder."""
    folder_id = getattr(button_state, "gallery_archive_folder_action_id", None)
    if not folder_id:
        return

    folders = getattr(button_state, "gallery_archive_folders", [])
    folder = next((f for f in folders if f.get("id") == folder_id), None)
    if not folder:
        button_state.gallery_archive_folder_action_id = None
        return

    in_rename_mode = getattr(button_state, "gallery_archive_rename_folder_id", None) == folder_id
    font = cv2.FONT_HERSHEY_SIMPLEX

    if in_rename_mode:
        # Rename mode: unified blue panel at bottom (title + input + keyboard)
        fh, fw = frame.shape[:2]
        dock_x = fw - GRID_SIDE_DOCK_WIDTH
        form_x = MODAL_EDGE_MARGIN
        form_w = dock_x - form_x

        _draw_archive_rename_keyboard(frame, fh, form_x, form_w)
    else:
        # Action mode: Rename | Delete | Cancel
        folder_name = folder.get("name", "Folder") or "Folder"
        modal_w = 300
        modal_h = 140
        modal_x = (frame.shape[1] - modal_w) // 2
        modal_y = (frame.shape[0] - modal_h) // 2

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
        cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

        cv2.putText(frame, folder_name[:28], (modal_x + 20, modal_y + 35), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Rename or delete folder", (modal_x + 20, modal_y + 58), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        btn_y = modal_y + 75
        btn_h = 36
        _grad = (CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT)
        _border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR

        # Rename button
        if "archive_folder_rename" not in menu_buttons:
            menu_buttons["archive_folder_rename"] = Button(modal_x + 15, btn_y, 85, btn_h, "Rename")
        else:
            menu_buttons["archive_folder_rename"].x, menu_buttons["archive_folder_rename"].y = modal_x + 15, btn_y
            menu_buttons["archive_folder_rename"].w, menu_buttons["archive_folder_rename"].h = 85, btn_h
            menu_buttons["archive_folder_rename"].text = "Rename"
        menu_buttons["archive_folder_rename"].is_active = True
        menu_buttons["archive_folder_rename"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad,
            fill_alpha=ACTION_BTN_FILL_ALPHA,
            neon_border_color=_border,
        )

        # Delete button (same style as gallery; destructive but consistent look)
        if "archive_folder_delete" not in menu_buttons:
            menu_buttons["archive_folder_delete"] = Button(modal_x + 110, btn_y, 85, btn_h, "Delete")
        else:
            menu_buttons["archive_folder_delete"].x, menu_buttons["archive_folder_delete"].y = modal_x + 110, btn_y
            menu_buttons["archive_folder_delete"].w, menu_buttons["archive_folder_delete"].h = 85, btn_h
            menu_buttons["archive_folder_delete"].text = "Delete"
        menu_buttons["archive_folder_delete"].is_active = True
        menu_buttons["archive_folder_delete"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad,
            fill_alpha=ACTION_BTN_FILL_ALPHA,
            neon_border_color=_border,
        )

        # Cancel button
        if "archive_folder_cancel" not in menu_buttons:
            menu_buttons["archive_folder_cancel"] = Button(modal_x + modal_w - 95, btn_y, 80, btn_h, "Cancel")
        else:
            menu_buttons["archive_folder_cancel"].x, menu_buttons["archive_folder_cancel"].y = modal_x + modal_w - 95, btn_y
            menu_buttons["archive_folder_cancel"].w, menu_buttons["archive_folder_cancel"].h = 80, btn_h
            menu_buttons["archive_folder_cancel"].text = "Cancel"
        menu_buttons["archive_folder_cancel"].is_active = True
        menu_buttons["archive_folder_cancel"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad,
            fill_alpha=ACTION_BTN_FILL_ALPHA,
            neon_border_color=_border,
        )

        if "archive_folder_modal_panel" not in menu_buttons:
            menu_buttons["archive_folder_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
        else:
            menu_buttons["archive_folder_modal_panel"].x, menu_buttons["archive_folder_modal_panel"].y = modal_x, modal_y
            menu_buttons["archive_folder_modal_panel"].w, menu_buttons["archive_folder_modal_panel"].h = modal_w, modal_h


def draw_archive_folder_delete_modal(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """Confirmation modal for deleting an archive folder."""
    folder_id = getattr(button_state, "gallery_archive_delete_confirm_folder_id", None)
    if not folder_id:
        return

    folders = getattr(button_state, "gallery_archive_folders", [])
    folder = next((f for f in folders if f.get("id") == folder_id), None)
    if not folder:
        button_state.gallery_archive_delete_confirm_folder_id = None
        return

    folder_name = folder.get("name", "Folder") or "Folder"
    modal_w = 320
    modal_h = 120
    modal_x = (frame.shape[1] - modal_w) // 2
    modal_y = (frame.shape[0] - modal_h) // 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Delete folder \"{folder_name[:20]}\"?", (modal_x + 20, modal_y + 35), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Files stay in gallery.", (modal_x + 20, modal_y + 58), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    btn_y = modal_y + 75
    btn_h = 32
    _grad = (CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT)
    _border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR

    if "archive_delete_yes" not in menu_buttons:
        menu_buttons["archive_delete_yes"] = Button(modal_x + 15, btn_y, 100, btn_h, "Delete")
    else:
        menu_buttons["archive_delete_yes"].x, menu_buttons["archive_delete_yes"].y = modal_x + 15, btn_y
        menu_buttons["archive_delete_yes"].w, menu_buttons["archive_delete_yes"].h = 100, btn_h
        menu_buttons["archive_delete_yes"].text = "Delete"
    menu_buttons["archive_delete_yes"].is_active = True
    menu_buttons["archive_delete_yes"].draw(
        frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
        gradient_colors=_grad,
        fill_alpha=ACTION_BTN_FILL_ALPHA,
        neon_border_color=_border,
    )

    if "archive_delete_no" not in menu_buttons:
        menu_buttons["archive_delete_no"] = Button(modal_x + modal_w - 95, btn_y, 80, btn_h, "Cancel")
    else:
        menu_buttons["archive_delete_no"].x, menu_buttons["archive_delete_no"].y = modal_x + modal_w - 95, btn_y
        menu_buttons["archive_delete_no"].w, menu_buttons["archive_delete_no"].h = 80, btn_h
        menu_buttons["archive_delete_no"].text = "Cancel"
    menu_buttons["archive_delete_no"].is_active = True
    menu_buttons["archive_delete_no"].draw(
        frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
        gradient_colors=_grad,
        fill_alpha=ACTION_BTN_FILL_ALPHA,
        neon_border_color=_border,
    )


def _draw_gallery_header(
    frame: np.ndarray,
    header_h: int,
    items: List[Tuple[Path, str, datetime]],
    folder_title: Optional[str],
) -> None:
    """Draw the gallery header (gradient, title, info, buttons). Call after grid to keep header on top when scrolling."""
    fw = frame.shape[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    dock_row_h = DOCK_ROW_HEIGHT
    action_strip_h = dock_row_h - 2
    back_btn_y = 3

    frame[0:header_h, 0:fw] = _vertical_gradient(header_h, fw, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT)
    cv2.line(frame, (0, header_h), (fw, header_h), (80, 80, 80), 2)

    title = (folder_title or "GALLERY").upper()
    title_scale = 1.2
    title_thick = 2
    (title_w, title_h), _ = cv2.getTextSize(title, font, title_scale, title_thick)
    title_x = (fw - title_w) // 2
    title_y = (action_strip_h + int(title_h)) // 2
    cv2.putText(frame, title, (title_x, title_y), font, title_scale, (255, 255, 255), title_thick, cv2.LINE_AA)

    hint_line_y = action_strip_h + 8
    info_y = hint_line_y + 12

    if button_state.gallery_select_mode and getattr(button_state, "gallery_select_first_hint_until", 0) > 0:
        if time.time() < button_state.gallery_select_first_hint_until:
            hint = "Select one or more items first"
            hint_scale = 0.65
            (hint_w, hint_h), _ = cv2.getTextSize(hint, font, hint_scale, 1)
            hint_x = (fw - hint_w) // 2
            hint_y = hint_line_y + max(0, (26 - hint_h) // 2) - 8
            cv2.putText(frame, hint, (hint_x, hint_y), font, hint_scale, (220, 220, 120), 1, cv2.LINE_AA)
        else:
            button_state.gallery_select_first_hint_until = 0.0

    if button_state.gallery_select_mode and getattr(button_state, "gallery_archive_move_hint_until", 0) > 0:
        if time.time() < button_state.gallery_archive_move_hint_until:
            hint = "Add folders in Archive first"
            hint_scale = 0.65
            (hint_w, hint_h), _ = cv2.getTextSize(hint, font, hint_scale, 1)
            hint_x = (fw - hint_w) // 2
            hint_y = hint_line_y + max(0, (26 - hint_h) // 2) - 8
            cv2.putText(frame, hint, (hint_x, hint_y), font, hint_scale, (220, 220, 120), 1, cv2.LINE_AA)
        else:
            button_state.gallery_archive_move_hint_until = 0.0

    if button_state.gallery_select_mode:
        selected_count = len(button_state.gallery_selected_items)
        if selected_count > 0:
            selected_size = 0
            for idx in button_state.gallery_selected_items:
                if 0 <= idx < len(items):
                    try:
                        selected_size += items[idx][0].stat().st_size
                    except OSError:
                        pass
            size_str = _format_size(selected_size)
            info_text = f"Total: {len(items)} items | Selected: {selected_count} ({size_str}) | Click to select | Swipe to scroll"
        else:
            info_text = f"Total: {len(items)} items | Click to select | Swipe to scroll"
    else:
        info_text = f"Total: {len(items)} items ({sum(1 for _, t, _ in items if t == 'image')} images, {sum(1 for _, t, _ in items if t == 'video')} videos) | Click to view | Swipe to scroll"

    info_scale = 0.45
    (info_w, info_h), _ = cv2.getTextSize(info_text, font, info_scale, 1)
    info_x = (fw - info_w) // 2
    cv2.putText(frame, info_text, (info_x, info_y), font, info_scale, (255, 255, 255), 1, cv2.LINE_AA)

    back_btn_x = 10
    back_btn_w = back_btn_h = BACK_BTN_SIZE
    dock_x = fw - GRID_SIDE_DOCK_WIDTH
    dock_row_w = GRID_SIDE_DOCK_WIDTH - 2 * DOCK_TOP_INSET_X
    action_btn_y = 0
    action_btn_x0 = dock_x + DOCK_TOP_INSET_X

    back_right = back_btn_x + back_btn_w
    delete_btn_x = (back_right + info_x - dock_row_w) // 2
    delete_btn_w = dock_row_w
    delete_btn_h = dock_row_h - 6

    if "gallery_back" not in menu_buttons:
        menu_buttons["gallery_back"] = Button(back_btn_x, back_btn_y, back_btn_w, back_btn_h, "")
    else:
        menu_buttons["gallery_back"].x, menu_buttons["gallery_back"].y = back_btn_x, back_btn_y
        menu_buttons["gallery_back"].w, menu_buttons["gallery_back"].h = back_btn_w, back_btn_h
    menu_buttons["gallery_back"].draw(frame, transparent=True, icon_type="back")

    if button_state.gallery_select_mode and button_state.gallery_selected_items:
        selected_count = len(button_state.gallery_selected_items)
        delete_text = f"DELETE ({selected_count})"
        if "gallery_delete_selected" not in menu_buttons:
            menu_buttons["gallery_delete_selected"] = Button(delete_btn_x, action_btn_y, delete_btn_w, delete_btn_h, delete_text)
        else:
            menu_buttons["gallery_delete_selected"].x, menu_buttons["gallery_delete_selected"].y = delete_btn_x, action_btn_y
            menu_buttons["gallery_delete_selected"].w, menu_buttons["gallery_delete_selected"].h = delete_btn_w, delete_btn_h
            menu_buttons["gallery_delete_selected"].text = delete_text
        menu_buttons["gallery_delete_selected"].is_active = True
        _grad_del = (CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT)
        _border_del = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR
        menu_buttons["gallery_delete_selected"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad_del, fill_alpha=ACTION_BTN_FILL_ALPHA, neon_border_color=_border_del,
        )

    # EMAIL (Share) button: right next to DELETE, only when selection + online + email verified
    email_verified = bool(getattr(app_state, "OUTPUT_DIR", None) and get_email_verified(app_state.OUTPUT_DIR))
    online = bool((HUD.connected_ssid or "").strip())
    show_share_btn = (
        button_state.gallery_select_mode
        and button_state.gallery_selected_items
        and online
        and email_verified
    )
    if show_share_btn:
        email_btn_w = 88  # enough for "EMAIL" label
        email_btn_h = delete_btn_h
        delete_right = delete_btn_x + delete_btn_w
        # Center EMAIL in the gap between DELETE and the "GALLERY" title text
        gap_center = (delete_right + title_x) // 2
        email_btn_x = gap_center - email_btn_w // 2
        email_btn_y = action_btn_y
        if "gallery_share_selected" not in menu_buttons:
            menu_buttons["gallery_share_selected"] = Button(email_btn_x, email_btn_y, email_btn_w, email_btn_h, "EMAIL")
        else:
            menu_buttons["gallery_share_selected"].x, menu_buttons["gallery_share_selected"].y = email_btn_x, email_btn_y
            menu_buttons["gallery_share_selected"].w, menu_buttons["gallery_share_selected"].h = email_btn_w, email_btn_h
        _grad = (CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT)
        _border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR
        menu_buttons["gallery_share_selected"].is_active = True
        menu_buttons["gallery_share_selected"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad, fill_alpha=ACTION_BTN_FILL_ALPHA, neon_border_color=_border,
        )
    else:
        if "gallery_share_selected" in menu_buttons:
            menu_buttons["gallery_share_selected"].w = 0
            menu_buttons["gallery_share_selected"].h = 0

    if items:
        _grad = (CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT)
        _border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR
        select_mode_btn_x, select_mode_btn_y = action_btn_x0, action_btn_y
        if "gallery_select_mode" not in menu_buttons:
            menu_buttons["gallery_select_mode"] = Button(select_mode_btn_x, select_mode_btn_y, dock_row_w, dock_row_h, "SELECT")
        else:
            menu_buttons["gallery_select_mode"].x, menu_buttons["gallery_select_mode"].y = select_mode_btn_x, select_mode_btn_y
            menu_buttons["gallery_select_mode"].w, menu_buttons["gallery_select_mode"].h = dock_row_w, dock_row_h
            menu_buttons["gallery_select_mode"].text = "DONE" if button_state.gallery_select_mode else "SELECT"
        menu_buttons["gallery_select_mode"].is_active = button_state.gallery_select_mode
        menu_buttons["gallery_select_mode"].draw(
            frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
            gradient_colors=_grad, fill_alpha=ACTION_BTN_FILL_ALPHA, neon_border_color=_border,
        )
        if button_state.gallery_select_mode:
            select_all_btn_x = action_btn_x0 - dock_row_w
            move_to_btn_x = select_all_btn_x - dock_row_w
            if "gallery_move_to" not in menu_buttons:
                menu_buttons["gallery_move_to"] = Button(move_to_btn_x, action_btn_y, dock_row_w, dock_row_h, "MOVE TO")
            else:
                menu_buttons["gallery_move_to"].x, menu_buttons["gallery_move_to"].y = move_to_btn_x, action_btn_y
                menu_buttons["gallery_move_to"].w, menu_buttons["gallery_move_to"].h = dock_row_w, dock_row_h
            menu_buttons["gallery_move_to"].is_active = bool(button_state.gallery_selected_items)
            menu_buttons["gallery_move_to"].draw(
                frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
                gradient_colors=_grad, fill_alpha=ACTION_BTN_FILL_ALPHA, neon_border_color=_border,
            )
            if "gallery_select_all" not in menu_buttons:
                menu_buttons["gallery_select_all"] = Button(select_all_btn_x, action_btn_y, dock_row_w, dock_row_h, "SELECT ALL")
            else:
                menu_buttons["gallery_select_all"].x, menu_buttons["gallery_select_all"].y = select_all_btn_x, action_btn_y
                menu_buttons["gallery_select_all"].w, menu_buttons["gallery_select_all"].h = dock_row_w, dock_row_h
                selectable_indices = [i for i in range(len(items)) if items[i][1] in ("image", "video")]
                all_selected = (
                    len(selectable_indices) > 0
                    and len(button_state.gallery_selected_items) == len(selectable_indices)
                    and all(i in button_state.gallery_selected_items for i in selectable_indices)
                )
                menu_buttons["gallery_select_all"].text = "DESELECT ALL" if all_selected else "SELECT ALL"
            menu_buttons["gallery_select_all"].is_active = True
            menu_buttons["gallery_select_all"].draw(
                frame, transparent=True, active_color=MENU_ACTIVE_BLUE,
                gradient_colors=_grad, fill_alpha=ACTION_BTN_FILL_ALPHA, neon_border_color=_border,
            )


def draw_gallery_view(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """
    Draw the gallery view showing all saved screenshots and videos.
    Displays items in a grid layout with thumbnails.
    Supports scrolling and clickable thumbnails.
    When viewing an archive folder, shows folder contents in full-page grid with side dock.
    """
    if output_dir is None:
        return

    in_folder_view = bool(getattr(button_state, "gallery_archive_folder_view_id", None))
    if in_folder_view:
        items = get_folder_displayed_items(output_dir, button_state.gallery_archive_folder_view_id)
        folders = getattr(button_state, "gallery_archive_folders", [])
        view_folder = next((f for f in folders if f.get("id") == button_state.gallery_archive_folder_view_id), None)
        folder_title = (view_folder.get("name", "Folder") or "Folder") if view_folder else "Folder"
    else:
        items = get_displayed_gallery_items(output_dir)
        folder_title = None

    if button_state.gallery_viewer_mode == "image":
        draw_image_viewer(frame, items, output_dir)
        if button_state.gallery_delete_modal_open:
            draw_delete_modal(frame)
        if getattr(button_state, "share_confirm_modal_open", False):
            draw_share_confirm_modal(frame)
        if getattr(button_state, "share_modal_open", False):
            draw_share_modal(frame)
        return
    elif button_state.gallery_viewer_mode == "video":
        draw_video_viewer(frame, items, output_dir)
        if button_state.gallery_delete_modal_open:
            draw_delete_modal(frame)
        if getattr(button_state, "share_confirm_modal_open", False):
            draw_share_confirm_modal(frame)
        if getattr(button_state, "share_modal_open", False):
            draw_share_modal(frame)
        return

    fh, fw = frame.shape[0], frame.shape[1]
    frame[:] = _vertical_gradient(fh, fw, BG_GRADIENT_TOP, BG_GRADIENT_BOT)

    header_h = 98
    font = cv2.FONT_HERSHEY_SIMPLEX
    _draw_gallery_header(frame, header_h, items, folder_title)

    if not items:
        msg = "Folder is empty. Use Move to add items." if in_folder_view else "No captures yet. Use SHOT or REC to create content."
        msg_scale = 0.7
        msg_thick = 1
        (msg_w, msg_h), _ = cv2.getTextSize(msg, font, msg_scale, msg_thick)
        msg_x = (frame.shape[1] - msg_w) // 2
        msg_y = frame.shape[0] // 2
        cv2.putText(frame, msg, (msg_x, msg_y), font, msg_scale, (150, 150, 150), msg_thick, cv2.LINE_AA)
        draw_grid_side_dock(frame, header_h, items, output_dir, 0)
        return

    margin = 20
    grid_start_y = header_h + margin
    grid_start_x = margin
    content_width = frame.shape[1] - GRID_SIDE_DOCK_WIDTH

    thumb_w = 280
    thumb_h = 180
    gap = 15

    cols = (content_width - 2 * margin + gap) // (thumb_w + gap)
    cols = max(2, cols)
    reserve_archive = not in_folder_view
    if reserve_archive:
        cols = max(2, cols)  # need at least 2 so top-right slot exists for archive

    # Row 0: cols-1 media slots when archive reserved (top-right for archive); else full cols
    n = len(items)
    if reserve_archive:
        total_rows = 1 + max(0, (n - (cols - 1) + cols - 1) // cols) if n > cols - 1 else 1
    else:
        total_rows = max(1, (n + cols - 1) // cols)
    item_height = thumb_h + gap + 50
    total_content_height = grid_start_y + (total_rows * item_height)
    footer_space = 40

    visible_height = frame.shape[0] - header_h - footer_space
    max_scroll = max(0, total_content_height - frame.shape[0] + footer_space)

    if button_state.gallery_inertia_active and not button_state.gallery_dragging:
        now = time.perf_counter()
        dt = max(1e-6, now - getattr(button_state, "gallery_last_inertia_t", now))
        button_state.gallery_last_inertia_t = now

        new_scroll = float(button_state.gallery_scroll_offset) + button_state.gallery_scroll_velocity * dt

        FRICTION = 1.2
        decay = max(0.0, 1.0 - FRICTION * dt)
        button_state.gallery_scroll_velocity *= decay

        max_scroll_f = float(getattr(button_state, "gallery_max_scroll", 0))
        if new_scroll < 0.0:
            new_scroll = 0.0
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False
        elif new_scroll > max_scroll_f:
            new_scroll = max_scroll_f
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False

        if abs(button_state.gallery_scroll_velocity) < 15.0:
            button_state.gallery_scroll_velocity = 0.0
            button_state.gallery_inertia_active = False

        button_state.gallery_scroll_offset = int(new_scroll)

    button_state.gallery_max_scroll = int(max_scroll)
    button_state.gallery_scroll_offset = max(0, min(button_state.gallery_scroll_offset, max_scroll))

    clip_y_start = header_h

    if not hasattr(button_state, 'gallery_thumbnail_rects'):
        button_state.gallery_thumbnail_rects = []
    button_state.gallery_thumbnail_rects.clear()

    scroll_offset = button_state.gallery_scroll_offset

    # Archive panel at top-right (0, cols-1) - only in main gallery
    if reserve_archive:
        arch_row, arch_col = archive_panel_grid_pos(cols)
        arch_x = grid_start_x + arch_col * (thumb_w + gap)
        arch_y = grid_start_y + arch_row * (thumb_h + gap + 50) - scroll_offset
        if arch_y + thumb_h + 50 >= header_h and arch_y <= frame.shape[0]:
            _draw_archive_panel(frame, arch_x, arch_y, thumb_w, thumb_h, output_dir, header_h)

    for idx, (filepath, item_type, mtime) in enumerate(items):
        if reserve_archive:
            row, col = item_idx_to_grid_pos(idx, cols)
        else:
            row, col = idx // cols, idx % cols

        x = grid_start_x + col * (thumb_w + gap)
        y = grid_start_y + row * (thumb_h + gap + 50) - scroll_offset

        if y + thumb_h + 50 < header_h or y > frame.shape[0]:
            continue

        button_state.gallery_thumbnail_rects.append({
            'idx': idx,
            'x': x,
            'y': y,
            'w': thumb_w,
            'h': thumb_h,
            'filepath': filepath,
            'type': item_type
        })

        vis_top = header_h
        vis_bot = frame.shape[0]

        x0, y0 = x, y
        x1, y1 = x + thumb_w, y + thumb_h

        cx0 = max(x0, 0)
        cx1 = min(x1, frame.shape[1])
        cy0 = max(y0, vis_top)
        cy1 = min(y1, vis_bot)

        if cx1 <= cx0 or cy1 <= cy0:
            continue

        is_selected = button_state.gallery_select_mode and idx in button_state.gallery_selected_items

        cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), (40, 40, 40), -1)

        # Always draw border so selection highlight stays visible when card is partially on-screen
        border_color = (80, 255, 100) if is_selected else (100, 100, 100)
        border_thickness = 4 if is_selected else 2
        cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), border_color, border_thickness, cv2.LINE_AA)

        if is_selected:
            check_size = 25
            check_x = x + thumb_w - check_size - 10
            check_y = y + 10
            # Only draw checkmark when the whole icon is in the content area (no bleed into header = no ghost flicker)
            if check_y >= vis_top and check_y + check_size <= vis_bot:
                cv2.circle(frame, (check_x + check_size//2, check_y + check_size//2), check_size//2, (80, 255, 100), -1, cv2.LINE_AA)

                check_pts = np.array([
                    [check_x + 6, check_y + check_size//2],
                    [check_x + check_size//2 - 2, check_y + check_size - 8],
                    [check_x + check_size - 6, check_y + 5]
                ], np.int32)
                cv2.polylines(frame, [check_pts], False, (255, 255, 255), 3, cv2.LINE_AA)

        try:
            if item_type == "image":
                mtime_s = filepath.stat().st_mtime
                cached = ui_cache._THUMB_CACHE.get(filepath)
                cached_m = ui_cache._THUMB_CACHE_MTIME.get(filepath)

                if cached is None or cached_m != mtime_s:
                    img = cv2.imread(str(filepath))
                    if img is not None:
                        cached = cv2.resize(img, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                        ui_cache._THUMB_CACHE[filepath] = cached
                        ui_cache._THUMB_CACHE_MTIME[filepath] = mtime_s

                if cached is not None:
                    dst_x0 = x + 2
                    dst_y0 = y + 2
                    dst_x1 = x + thumb_w - 2
                    dst_y1 = y + thumb_h - 2

                    cdx0 = max(dst_x0, 0)
                    cdx1 = min(dst_x1, frame.shape[1])
                    cdy0 = max(dst_y0, vis_top)
                    cdy1 = min(dst_y1, vis_bot)

                    if cdx1 > cdx0 and cdy1 > cdy0:
                        sx0 = cdx0 - dst_x0
                        sy0 = cdy0 - dst_y0
                        sx1 = sx0 + (cdx1 - cdx0)
                        sy1 = sy0 + (cdy1 - cdy0)

                        frame[cdy0:cdy1, cdx0:cdx1] = cached[sy0:sy1, sx0:sx1]
            elif item_type == "video":
                mtime_s = filepath.stat().st_mtime
                cached = ui_cache._THUMB_CACHE.get(filepath)
                cached_m = ui_cache._THUMB_CACHE_MTIME.get(filepath)

                if cached is None or cached_m != mtime_s:
                    vid_frame, _, _ = _video_read_frame_at(filepath, 0)
                    if vid_frame is not None:
                        cached = cv2.resize(vid_frame, (thumb_w - 4, thumb_h - 4), interpolation=cv2.INTER_AREA)
                        ui_cache._THUMB_CACHE[filepath] = cached
                        ui_cache._THUMB_CACHE_MTIME[filepath] = mtime_s

                if cached is not None:
                    dst_x0 = x + 2
                    dst_y0 = y + 2
                    dst_x1 = x + thumb_w - 2
                    dst_y1 = y + thumb_h - 2

                    cdx0 = max(dst_x0, 0)
                    cdx1 = min(dst_x1, frame.shape[1])
                    cdy0 = max(dst_y0, vis_top)
                    cdy1 = min(dst_y1, vis_bot)

                    if cdx1 > cdx0 and cdy1 > cdy0:
                        sx0 = cdx0 - dst_x0
                        sy0 = cdy0 - dst_y0
                        sx1 = sx0 + (cdx1 - cdx0)
                        sy1 = sy0 + (cdy1 - cdy0)

                        frame[cdy0:cdy1, cdx0:cdx1] = cached[sy0:sy1, sx0:sx1]

                    center_x = x + thumb_w // 2
                    center_y = y + thumb_h // 2
                    play_size = 30

                    ox0 = center_x - play_size
                    oy0 = center_y - play_size
                    ox1 = center_x + play_size
                    oy1 = center_y + play_size

                    cox0 = max(ox0, 0)
                    cox1 = min(ox1, frame.shape[1])
                    coy0 = max(oy0, vis_top)
                    coy1 = min(oy1, vis_bot)

                    if cox1 > cox0 and coy1 > coy0:
                        roi = frame[coy0:coy1, cox0:cox1]
                        overlay_roi = roi.copy()

                        ccx = center_x - cox0
                        ccy = center_y - coy0

                        cv2.circle(overlay_roi, (ccx, ccy), play_size, (0, 0, 0), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay_roi, 0.6, roi, 0.4, 0.0, dst=roi)

                        pts = np.array([
                            [center_x - 10, center_y - 15],
                            [center_x - 10, center_y + 15],
                            [center_x + 15, center_y]
                        ], np.int32)

                        if (pts[:, 0].min() >= 0 and pts[:, 0].max() < frame.shape[1] and
                            pts[:, 1].min() >= vis_top and pts[:, 1].max() < vis_bot):
                            cv2.fillPoly(frame, [pts], (255, 255, 255), cv2.LINE_AA)
        except Exception:
            cv2.putText(frame, "Error", (x + 10, y + thumb_h // 2),
                       font, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        label_y = y + thumb_h + 20
        if label_y >= header_h and label_y <= frame.shape[0]:
            type_icon = "[IMG]" if item_type == "image" else "[VID]"
            type_color = (100, 200, 100) if item_type == "image" else (100, 150, 255)
            badge_size = _TYPE_BADGE_SIZE_CACHE.get(type_icon)
            if badge_size is None:
                (badge_w, text_h), _ = cv2.getTextSize(type_icon, font, 0.45, 1)
                _TYPE_BADGE_SIZE_CACHE[type_icon] = (badge_w, text_h)
            else:
                badge_w, text_h = badge_size

            # Line 1: [IMG]/[VID] then filename; priority circle right-aligned under thumbnail
            cv2.putText(frame, type_icon, (x, label_y), font, 0.45, type_color, 1, cv2.LINE_AA)
            filename_x = x + badge_w + 6
            max_fname_chars = 28
            display_name = filepath.stem if filepath.stem else filepath.name
            if len(display_name) > max_fname_chars:
                display_name = display_name[: max_fname_chars - 3] + "..."
            fname_scale = 0.45  # 5% larger than 0.43
            (fname_w, _), _ = cv2.getTextSize(display_name, font, fname_scale, 1)
            cv2.putText(frame, display_name, (filename_x, label_y), font, fname_scale, (255, 255, 255), 1, cv2.LINE_AA)

            priority_radius = 7  # 15% larger than 5
            dot_cy = label_y - text_h // 2  # align circle with type/filename baseline
            gap_px = 4  # min gap between end of filename and priority circle
            dot_cx = max(
                filename_x + fname_w + gap_px + priority_radius,
                x + thumb_w - priority_radius - 4,
            )  # at least gap_px after filename; else right-aligned under thumbnail
            file_priorities = getattr(button_state, 'gallery_file_priorities', {})
            priority = file_priorities.get(filepath.name, "")
            if priority and priority in PRIORITY_COLORS:
                draw_priority_circle_neon(frame, dot_cx, dot_cy, priority_radius, PRIORITY_COLORS[priority])
            else:
                draw_priority_circle_neon(frame, dot_cx, dot_cy, priority_radius, (120, 120, 120), neon=False)

            # Line 2: tag icon + tag text; only show if non-null and has real content (strict)
            def _valid_tag(s) -> bool:
                if s is None:
                    return False
                t = str(s).strip()
                if not t or t.lower() == "null":
                    return False
                # reject placeholders: ?, ??, ???, or only punctuation/whitespace/replacement
                if t in ("?", "??", "???"):
                    return False
                # strict: only show if there is at least one letter or digit (real content)
                if not any(c.isalnum() for c in t):
                    return False
                # reject if only question marks and spaces (any Unicode)
                if not t.replace("?", "").replace(" ", "").replace("\u00a0", "").replace("\u200b", "").replace("\ufffd", "").strip():
                    return False
                return True

            line2_y = label_y + 18
            tag_data_map = getattr(button_state, 'gallery_tag_data', {})
            file_tags_map = getattr(button_state, 'gallery_file_tags', {})
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
            tag_text = " | ".join(tag_parts) if tag_parts else ""  # ASCII separator (· can render as ?? in OpenCV font)

            if tag_text:
                _draw_tag_icon_grid(frame, x + 10, line2_y, (160, 200, 160))
                tag_icon_w = 22  # icon + gap
                max_tag_chars = 32
                if len(tag_text) > max_tag_chars:
                    tag_text = tag_text[: max_tag_chars - 3] + "..."
                tag_scale = 0.51  # 10% larger than 0.46
                cv2.putText(frame, tag_text, (x + tag_icon_w, line2_y + 4),
                            font, tag_scale, (180, 200, 180), 1, cv2.LINE_AA)

            # Date on line 2, right-justified under priority circle (cached)
            date_key = (mtime.year, mtime.month, mtime.day)
            date_str = _DATE_STR_CACHE.get(date_key)
            if date_str is None:
                date_str = mtime.strftime("%b %d, %Y")
                _DATE_STR_CACHE[date_key] = date_str
            date_scale = 0.42
            date_size = ui_cache._DATE_TEXT_SIZE_CACHE.get(date_str)
            if date_size is None:
                (date_w, date_h), _ = cv2.getTextSize(date_str, font, date_scale, 1)
                ui_cache._DATE_TEXT_SIZE_CACHE[date_str] = (date_w, date_h)
            else:
                date_w = date_size[0]
            date_x = x + thumb_w - date_w - 4
            cv2.putText(frame, date_str, (date_x, line2_y + 4),
                        font, date_scale, (150, 150, 160), 1, cv2.LINE_AA)

    # Redraw header on top so scrolled content (archive panel, thumbnails) goes under it
    _draw_gallery_header(frame, header_h, items, folder_title)

    # Archive modals first (lower z-order) so Tags/Priority/Rename/Search/Filter/Sort modals draw on top
    if button_state.gallery_archive_move_modal_open:
        draw_move_to_modal(frame, output_dir)
    if getattr(button_state, "gallery_archive_folder_action_id", None):
        draw_archive_folder_action_modal(frame, output_dir, header_h)
    if getattr(button_state, "gallery_archive_delete_confirm_folder_id", None):
        draw_archive_folder_delete_modal(frame, output_dir)

    # Side dock and action modals (Tags, Priority, Rename, Filter, Sort, Search) on top of archive modals
    draw_grid_side_dock(frame, header_h, items, output_dir, button_state.gallery_scroll_offset)
    if button_state.gallery_tag_modal_open:
        draw_tag_modal(frame, output_dir, header_h)

    if button_state.gallery_delete_modal_open:
        if button_state.gallery_delete_modal_kind == "batch":
            n = len(button_state.gallery_selected_items)
            draw_delete_modal(
                frame,
                title=f"Delete {n} item(s)?",
                subtitle="This action cannot be undone.",
                yes_text="YES, DELETE",
                no_text="CANCEL",
            )
        else:
            draw_delete_modal(frame)

    if getattr(button_state, "share_confirm_modal_open", False):
        draw_share_confirm_modal(frame)
    if getattr(button_state, "share_modal_open", False):
        draw_share_modal(frame)

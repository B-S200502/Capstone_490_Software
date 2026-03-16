"""
Grid side dock: vertical strip on the right in gallery grid view.

Dock is delimited from the grid; the storage bar floats with the viewport
(stays visible) as you scroll up/down.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from ..config import (
    MENU_ACTIVE_BLUE,
    MENU_ACTIVE_BLUE_LIGHT,
    MODAL_ACTIVE_GOLD,
    GALLERY_ACTION_STYLE,
    CLASSIC_ACTION_FILL_TOP,
    CLASSIC_ACTION_FILL_BOT,
    CLASSIC_ACTION_FILL_DARK_TOP,
    CLASSIC_ACTION_FILL_DARK_BOT,
    CLASSIC_ACTION_TEXT_BGR,
    CLASSIC_ACTION_BORDER_BGR,
    CLASSIC_ACTION_GLOW,
    DOCK_GRADIENT_TOP,
    DOCK_GRADIENT_BOT,
    ACTION_BTN_GLOW,
    ACTION_BTN_BORDER_THICKNESS,
    ACTION_BTN_FILL_DARK_TOP,
    ACTION_BTN_FILL_DARK_BOT,
    ACTION_BTN_NEON_BORDER_BGR,
    ACTION_BTN_NEON_GLOW,
    ACTION_BTN_FILL_ALPHA,
    ACTION_BTN_MODAL_FILL_ALPHA,
    ACTION_BTN_SHINE_ALPHA,
)
from ..state import button_state
from .button import Button, menu_buttons, _add_glassy_shine
from .keyboard import (
    ROWS_QWERTY as KEYBOARD_ROWS_QWERTY,
    FULL_KEY_SCALE as KEY_SCALE,
    draw_key_bg_clipped,
    dimensions_for_scale,
    KEY_BORDER_BGR as KEYBOARD_KEY_BORDER,
    KEY_TEXT_BGR as KEYBOARD_KEY_TEXT,
)
from .standard_keyboard import (
    draw_standard_alpha_keyboard,
    draw_standard_symbol_keyboard,
    compute_standard_keyboard_dimensions,
)
from .priority_circle import draw_priority_circle_neon
from .storage_bar import draw_storage_bar, feathered_composite

GRID_SIDE_DOCK_WIDTH = 113

DOCK_BG = (22, 22, 26)
DOCK_EDGE = (55, 55, 60)
DOCK_EDGE_ACCENT = (75, 75, 82)
DOCK_INNER_PAD = 12

# Top dock rows: Search, Filter, and third — flush with inset, dividers between
DOCK_TOP_INSET_X = 2
DOCK_TOP_INSET_Y = 0
DOCK_ROW_HEIGHT = 75  # 50% larger than 50
DOCK_DIVIDER_COLOR = (95, 95, 105)
DOCK_DIVIDER_THICKNESS = 2
# Row labels (third is generic; change as needed)
SEARCH_BAR_PLACEHOLDER = "Search"
DOCK_ROW_FILTER_LABEL = "Filter"
DOCK_ROW_THIRD_LABEL = "Sort"
SEARCH_BAR_TEXT_COLOR = (255, 255, 255)

# Priority colours (BGR)
PRIORITY_COLORS = {
    "high":   (30,  30, 220),   # red
    "medium": (0,  140, 255),   # orange
    "low":    (50, 185,  50),   # green
}
PRIORITY_OPTIONS = [("High", "high"), ("Medium", "medium"), ("Low", "low")]

# Preset tags shown in the tags panel
PRESET_TAGS = ["Important", "Reviewed", "Processed", "Follow-up", "Archive"]


def _vertical_gradient(h: int, w: int, top_bgr: Tuple[int, int, int], bot_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top_bgr[c], bot_bgr[c], h, dtype=np.uint8).reshape(-1, 1)
    return out


def _draw_icon_search(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw magnifying glass icon centered at (cx, cy) — 50% larger."""
    icon_r = 9
    cv2.circle(frame, (cx, cy), icon_r, color, 1, cv2.LINE_AA)
    cv2.line(
        frame,
        (cx + 6, cy + 3),
        (cx + 16, cy + 13),
        color, 1, cv2.LINE_AA
    )


def _draw_icon_filter(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw funnel/filter icon (trapezoid) centered at (cx, cy) — 50% larger."""
    pts = np.array([
        [cx - 12, cy - 7],
        [cx + 12, cy - 7],
        [cx + 6, cy + 7],
        [cx - 6, cy + 7],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)


def _draw_icon_sort(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw stacked lines (ascending) icon for sort — 50% larger."""
    cv2.line(frame, (cx - 7, cy - 7), (cx + 5, cy - 7), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 7, cy), (cx + 7, cy), color, 1, cv2.LINE_AA)
    cv2.line(frame, (cx - 7, cy + 7), (cx + 12, cy + 7), color, 1, cv2.LINE_AA)


def _draw_icon_tag(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a price-tag / label icon centered at (cx, cy)."""
    pts = np.array([
        [cx - 11, cy - 7],
        [cx + 3,  cy - 7],
        [cx + 11, cy],
        [cx + 3,  cy + 7],
        [cx - 11, cy + 7],
    ], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx - 6, cy), 2, color, -1, cv2.LINE_AA)


def _draw_icon_priority(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a flag icon for priority centered at (cx, cy)."""
    cv2.line(frame, (cx - 6, cy - 10), (cx - 6, cy + 10), color, 2, cv2.LINE_AA)
    flag_pts = np.array([
        [cx - 6, cy - 10],
        [cx + 9, cy - 4],
        [cx - 6, cy + 2],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [flag_pts], color)


def _draw_icon_pen(frame: np.ndarray, cx: int, cy: int, color: Tuple[int, int, int]) -> None:
    """Draw a simple pen icon: vertical body + pointed nib at bottom."""
    # Body (vertical rectangle, cap at top)
    body_top = cy - 10
    body_bot = cy + 5
    body_left = cx - 3
    body_right = cx + 3
    cv2.rectangle(frame, (body_left, body_top), (body_right, body_bot), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (body_left, body_top), (body_right, body_bot), color, 1, cv2.LINE_AA)
    # Nib (small triangle pointing down at writing end)
    tip_pts = np.array([
        [body_left, body_bot],
        [body_right, body_bot],
        [cx, cy + 10],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [tip_pts], color)
    cv2.polylines(frame, [tip_pts], True, color, 1, cv2.LINE_AA)


# Subtle border for dock buttons so they read as controls, not flat panels
DOCK_ROW_BORDER = (100, 100, 110)
DOCK_ROW_TOP_HIGHLIGHT = (140, 120, 90)
# White outline for button+modal when modal is open (continuous extension)
DOCK_ROW_WHITE_BORDER = (255, 255, 255)
# Gallery Action surfaces: key/option button border (classic = blue, neon = white)
ACTION_KEY_BORDER = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else DOCK_ROW_WHITE_BORDER
# Modal titles/labels: classic = dark text on light, neon = white with glow
MODAL_TEXT_COLOR = CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else None
MODAL_ANIM_DURATION_S = 0.28


def _put_text_white_glow(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    scale: float,
    pad: int = 10,
    blend: Optional[float] = None,
    color: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw text with optional glow. If color is set (e.g. classic dark text), no glow."""
    text_color = color if color is not None else SEARCH_BAR_TEXT_COLOR
    if color is not None:
        cv2.putText(frame, text, (x, y), font, scale, text_color, 1, cv2.LINE_AA)
        return
    if blend is None:
        blend = ACTION_BTN_GLOW
    fh, fw = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    x0 = max(0, x - pad)
    y0 = max(0, y - th - pad)
    x1 = min(fw, x + tw + pad)
    y1 = min(fh, y + pad)
    if x1 <= x0 or y1 <= y0:
        cv2.putText(frame, text, (x, y), font, scale, text_color, 1, cv2.LINE_AA)
        return
    patch = frame[y0:y1, x0:x1].copy()
    lx, ly = x - x0, y - y0
    cv2.putText(patch, text, (lx, ly), font, scale, text_color, 2, cv2.LINE_AA)
    cv2.putText(patch, text, (lx, ly), font, scale, text_color, 1, cv2.LINE_AA)
    patch = cv2.GaussianBlur(patch, (0, 0), 2.5)
    feathered_composite(frame, y0, y1, x0, x1, patch, blend, feather_px=12)
    cv2.putText(frame, text, (x, y), font, scale, text_color, 1, cv2.LINE_AA)


def _draw_dock_row(
    frame: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    label: str,
    icon_right: Optional[str] = None,
    skip_bg: bool = False,
) -> None:
    """Draw one top-dock row: gradient fill (unless skip_bg), subtle border, icon above text, centered. Neonized."""
    fh, fw = frame.shape[:2]
    if w < 4 or h < 8:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.71
    (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
    gap = 6
    icon_half_h = 10
    block_h = 2 * icon_half_h + gap + th
    block_top = y0 + (h - block_h) // 2
    block_cx = x0 + w // 2
    icon_cx = block_cx
    icon_cy = block_top + icon_half_h
    text_x = block_cx - tw // 2
    text_y = block_top + 2 * icon_half_h + gap + th

    # 1) Fill + border (neon or classic Gallery Action style)
    is_classic = GALLERY_ACTION_STYLE == "classic"
    if not skip_bg:
        roi = frame[y0:y0 + h, x0:x0 + w]
        if is_classic:
            # Closed dock rows: darker blue; expanded strip/panel use CLASSIC_ACTION_FILL_TOP/BOT elsewhere
            fill_top, fill_bot = CLASSIC_ACTION_FILL_DARK_TOP, CLASSIC_ACTION_FILL_DARK_BOT
            row_border = CLASSIC_ACTION_BORDER_BGR
        else:
            fill_top, fill_bot = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
            row_border = ACTION_BTN_NEON_BORDER_BGR
        gradient = _vertical_gradient(h, w, fill_top, fill_bot)
        alpha = ACTION_BTN_FILL_ALPHA
        cv2.addWeighted(gradient, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
        if not is_classic:
            _add_glassy_shine(roi, alpha=ACTION_BTN_SHINE_ALPHA)
        if not is_classic and ACTION_BTN_NEON_GLOW > 0:
            pad_bg = 16
            gx0 = max(0, x0 - pad_bg)
            gy0 = max(0, y0 - pad_bg)
            gx1 = min(fw, x0 + w + pad_bg)
            gy1 = min(fh, y0 + h + pad_bg)
            if gx1 > gx0 and gy1 > gy0:
                glow_patch = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
                lx, ly = x0 - gx0, y0 - gy0
                cv2.rectangle(glow_patch, (lx, ly), (lx + w, ly + h), row_border, 8, cv2.LINE_AA)
                glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 6.0)
                feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, ACTION_BTN_NEON_GLOW, feather_px=14)
        cv2.rectangle(frame, (x0, y0), (x0 + w - 1, y0 + h - 1), row_border, max(1, ACTION_BTN_BORDER_THICKNESS), cv2.LINE_AA)

    # 2) Inner glow: icon + label on patch, blur, feathered composite (neon = stronger, classic = minor)
    row_text_color = CLASSIC_ACTION_TEXT_BGR if is_classic else SEARCH_BAR_TEXT_COLOR
    pad_inner = 18
    ix0 = max(0, block_cx - w // 2 - pad_inner)
    iy0 = max(0, block_top - pad_inner)
    ix1 = min(fw, block_cx + w // 2 + pad_inner)
    iy1 = min(fh, block_top + block_h + pad_inner)
    inner_glow_blend = CLASSIC_ACTION_GLOW if is_classic else ACTION_BTN_GLOW
    if ix1 > ix0 and iy1 > iy0 and inner_glow_blend > 0:
        inner_patch = frame[iy0:iy1, ix0:ix1].copy()
        lcx, lcy = icon_cx - ix0, icon_cy - iy0
        ltx, lty = text_x - ix0, text_y - iy0
        white = row_text_color
        if icon_right == "search":
            _draw_icon_search(inner_patch, lcx, lcy, white)
        elif icon_right == "filter":
            _draw_icon_filter(inner_patch, lcx, lcy, white)
        elif icon_right == "sort":
            _draw_icon_sort(inner_patch, lcx, lcy, white)
        elif icon_right == "tag":
            _draw_icon_tag(inner_patch, lcx, lcy, white)
        elif icon_right == "priority":
            _draw_icon_priority(inner_patch, lcx, lcy, white)
        elif icon_right == "pen":
            _draw_icon_pen(inner_patch, lcx, lcy, white)
        cv2.putText(inner_patch, label, (ltx, lty), font, scale, white, 2, cv2.LINE_AA)
        cv2.putText(inner_patch, label, (ltx, lty), font, scale, white, 1, cv2.LINE_AA)
        inner_patch = cv2.GaussianBlur(inner_patch, (0, 0), 3.0)
        feathered_composite(frame, iy0, iy1, ix0, ix1, inner_patch, inner_glow_blend, feather_px=14)

    # 3) Sharp icon and text on top
    if icon_right == "search":
        _draw_icon_search(frame, icon_cx, icon_cy, row_text_color)
    elif icon_right == "filter":
        _draw_icon_filter(frame, icon_cx, icon_cy, row_text_color)
    elif icon_right == "sort":
        _draw_icon_sort(frame, icon_cx, icon_cy, row_text_color)
    elif icon_right == "tag":
        _draw_icon_tag(frame, icon_cx, icon_cy, row_text_color)
    elif icon_right == "priority":
        _draw_icon_priority(frame, icon_cx, icon_cy, row_text_color)
    elif icon_right == "pen":
        _draw_icon_pen(frame, icon_cx, icon_cy, row_text_color)
    cv2.putText(
        frame, label, (text_x, text_y),
        font, scale, row_text_color, 1, cv2.LINE_AA
    )


def _draw_dock_top_rows(
    frame: np.ndarray,
    dock_x: int,
    dock_y: int,
    dock_w: int,
    active_modal_key: Optional[str] = None,
    progress: float = 0.0,
) -> None:
    """Draw top dock rows.

    Normal mode  → Search / Filter / Sort
    Select mode  → Tags / Priority / Rename
    When active_modal_key is set and progress > 0, that row skips its own bg (combined strip drawn separately).
    """
    if button_state.gallery_select_mode:
        _draw_select_mode_rows(frame, dock_x, dock_y, dock_w, active_modal_key, progress)
        return

    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y
    skip = active_modal_key == "search" and progress > 0
    # Row 1: Search
    _draw_dock_row(frame, x0, y, w, row_h, SEARCH_BAR_PLACEHOLDER, icon_right="search", skip_bg=skip)
    if "gallery_dock_search" not in menu_buttons:
        menu_buttons["gallery_dock_search"] = Button(x0, y, w, row_h, "Search")
    else:
        b = menu_buttons["gallery_dock_search"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    skip = active_modal_key == "filter" and progress > 0
    # Row 2: Filter
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_FILTER_LABEL, icon_right="filter", skip_bg=skip)
    if "gallery_dock_filter" not in menu_buttons:
        menu_buttons["gallery_dock_filter"] = Button(x0, y, w, row_h, "Filter")
    else:
        b = menu_buttons["gallery_dock_filter"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1
    skip = active_modal_key == "sort" and progress > 0
    # Row 3: Sort
    _draw_dock_row(frame, x0, y, w, row_h, DOCK_ROW_THIRD_LABEL, icon_right="sort", skip_bg=skip)
    if "gallery_dock_sort" not in menu_buttons:
        menu_buttons["gallery_dock_sort"] = Button(x0, y, w, row_h, "Sort")
    else:
        b = menu_buttons["gallery_dock_sort"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h


def _draw_select_mode_rows(
    frame: np.ndarray,
    dock_x: int,
    dock_y: int,
    dock_w: int,
    active_modal_key: Optional[str] = None,
    progress: float = 0.0,
) -> None:
    """Draw Tags / Priority / Rename rows shown when gallery_select_mode is active."""
    inset_x = DOCK_TOP_INSET_X
    inset_y = DOCK_TOP_INSET_Y
    row_h = DOCK_ROW_HEIGHT
    x0 = dock_x + inset_x
    w = dock_w - 2 * inset_x
    if w < 10:
        return
    y = dock_y + inset_y

    skip = active_modal_key == "tags" and progress > 0
    # Row 1: Tags
    _draw_dock_row(frame, x0, y, w, row_h, "Tags", icon_right="tag", skip_bg=skip)
    if "gallery_dock_tags" not in menu_buttons:
        menu_buttons["gallery_dock_tags"] = Button(x0, y, w, row_h, "Tags")
    else:
        b = menu_buttons["gallery_dock_tags"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    skip = active_modal_key == "priority" and progress > 0
    # Row 2: Priority
    _draw_dock_row(frame, x0, y, w, row_h, "Priority", icon_right="priority", skip_bg=skip)
    if "gallery_dock_priority" not in menu_buttons:
        menu_buttons["gallery_dock_priority"] = Button(x0, y, w, row_h, "Priority")
    else:
        b = menu_buttons["gallery_dock_priority"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h
    y += row_h
    for _ in range(DOCK_DIVIDER_THICKNESS):
        cv2.line(frame, (dock_x, y), (dock_x + dock_w, y), DOCK_DIVIDER_COLOR, 1, cv2.LINE_AA)
        y += 1

    skip = active_modal_key == "rename" and progress > 0
    # Row 3: Rename
    _draw_dock_row(frame, x0, y, w, row_h, "Rename", icon_right="pen", skip_bg=skip)
    if "gallery_dock_rename" not in menu_buttons:
        menu_buttons["gallery_dock_rename"] = Button(x0, y, w, row_h, "Rename")
    else:
        b = menu_buttons["gallery_dock_rename"]
        b.x, b.y, b.w, b.h = x0, y, w, row_h


# Filter modal: option labels and state values
FILTER_OPTIONS = [("All", "all"), ("Images", "image"), ("Videos", "video")]
# Sort modal: option labels and state values
SORT_OPTIONS = [
    ("Date (newest)", "date"),
    ("Name (A to Z)", "name"),
    ("Size (largest)", "size"),
    ("Priority (Highest)", "priority"),
]

MODAL_PANEL_W = 320
MODAL_PANEL_H = 200  # used for filter modal (3 options)
# Edit Tags form in gallery.py uses this width
TAGS_FORM_W = 590
MODAL_OPTION_H = 44
MODAL_TITLE_H = 50

# Search & Rename keyboards: dimensions from keyboard module (identical size, gray keys)
_KEYBOARD_DIMS = dimensions_for_scale(KEY_SCALE)
KEY_W = _KEYBOARD_DIMS["key_w"]
KEY_H = _KEYBOARD_DIMS["key_h"]
KEY_GAP = _KEYBOARD_DIMS["key_gap"]
KEYBOARD_BAR_H = _KEYBOARD_DIMS["bar_h"]
KEYBOARD_FOOTER_GAP = _KEYBOARD_DIMS["footer_gap"]
KEYBOARD_FONT_BAR = _KEYBOARD_DIMS["font_bar"]
KEYBOARD_FONT_KEY = _KEYBOARD_DIMS["font_key"]
KEYBOARD_FONT_SPECIAL = _KEYBOARD_DIMS["font_special"]
KEYBOARD_MARGIN_BOTTOM = 24


# Modal as extension of dock: gap west of dock, no overlay.
# Connector width so modal blue meets the row (continuous from button to modal).
MODAL_GAP_WEST = 8
MODAL_EDGE_MARGIN = 6  # left edge of screen; modals extend from here to dock
MODAL_CONNECTOR_WIDTH = MODAL_GAP_WEST + DOCK_TOP_INSET_X  # 10px: modal extends to row start


def _get_active_modal_info(
    frame: np.ndarray, dock_x: int, dock_y: int, dock_w: int
) -> Optional[Tuple[str, int, int, int, int, int]]:
    """Return (active_key, row_y, row_h, modal_px, modal_total_w, modal_h) for the open modal, or None."""
    fh = frame.shape[0]
    row_h = DOCK_ROW_HEIGHT
    inset_y = DOCK_TOP_INSET_Y
    div = DOCK_DIVIDER_THICKNESS

    if button_state.gallery_select_mode:
        if button_state.gallery_tag_modal_open:
            row_y = dock_y + inset_y
            content_w = dock_x - MODAL_GAP_WEST - MODAL_EDGE_MARGIN
            px = MODAL_EDGE_MARGIN
            total_w = content_w + MODAL_CONNECTOR_WIDTH
            # Full height so Edit Tags + keyboard are one fused modal
            modal_h = max(280, fh - row_y - 8)
            return ("tags", row_y, row_h, px, total_w, modal_h)
        if button_state.gallery_priority_modal_open:
            panel_h = MODAL_TITLE_H + len(PRIORITY_OPTIONS) * MODAL_OPTION_H + 10
            row_y = dock_y + inset_y + row_h + div
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("priority", row_y, row_h, px, total_w, panel_h)
        if button_state.gallery_rename_modal_open:
            row_y = dock_y + inset_y + 2 * (row_h + div)
            # Use full height below rename row so keys expand to fill the modal
            panel_h = max(
                KEYBOARD_BAR_H + 120,
                fh - row_y - 12,
            )
            content_w = dock_x - MODAL_GAP_WEST - MODAL_EDGE_MARGIN
            px = MODAL_EDGE_MARGIN
            total_w = content_w + MODAL_CONNECTOR_WIDTH
            return ("rename", row_y, row_h, px, total_w, panel_h)
    else:
        if button_state.gallery_search_keyboard_open:
            row_y = dock_y + inset_y
            # Use full height below search row so keys can expand to fill the modal
            panel_h = max(
                KEYBOARD_BAR_H + 120,
                fh - row_y - 12,
            )
            content_w = dock_x - MODAL_GAP_WEST - MODAL_EDGE_MARGIN
            px = MODAL_EDGE_MARGIN
            total_w = content_w + MODAL_CONNECTOR_WIDTH
            return ("search", row_y, row_h, px, total_w, panel_h)
        if button_state.gallery_filter_modal_open:
            row_y = dock_y + inset_y + row_h + div
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("filter", row_y, row_h, px, total_w, MODAL_PANEL_H)
        if button_state.gallery_sort_modal_open:
            row_y = dock_y + inset_y + 2 * (row_h + div)
            panel_h = MODAL_TITLE_H + len(SORT_OPTIONS) * MODAL_OPTION_H + 10
            px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
            total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
            return ("sort", row_y, row_h, px, total_w, panel_h)
    return None


def _draw_combined_modal_strip(
    frame: np.ndarray,
    left: int,
    top: int,
    width: int,
    height: int,
    white_border: bool = False,
    bottom_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw one continuous gradient over button+modal strip (matches Gallery Action style).
    Border drawn separately (P-shape). If bottom_bgr is set, gradient runs top -> bottom_bgr."""
    if width < 1 or height < 1:
        return
    roi = frame[top : top + height, left : left + width]
    if GALLERY_ACTION_STYLE == "classic":
        start_bgr, default_end = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
    else:
        start_bgr, default_end = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
    end_bgr = bottom_bgr if bottom_bgr is not None else default_end
    grad = _vertical_gradient(height, width, start_bgr, end_bgr)
    cv2.addWeighted(grad, ACTION_BTN_MODAL_FILL_ALPHA, roi, 1.0 - ACTION_BTN_MODAL_FILL_ALPHA, 0.0, dst=roi)
    if GALLERY_ACTION_STYLE != "classic":
        _add_glassy_shine(roi, alpha=ACTION_BTN_SHINE_ALPHA)


def _draw_modal_panel_grow_slice(
    frame: np.ndarray,
    px: int,
    py: int,
    content_w: int,
    h: int,
    vis_left: int,
    vis_width: int,
    top_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw only the right vis_width pixels of the modal panel (grow-from-button).
    Fill matches Gallery Action style; if top_bgr is set, gradient runs top_bgr -> bottom (seamless with strip)."""
    if vis_width <= 0 or h <= 0:
        return
    fh, fw = frame.shape[:2]
    total_w = content_w + MODAL_CONNECTOR_WIDTH
    x0 = max(0, vis_left)
    x1 = min(fw, vis_left + vis_width)
    y0 = max(0, py)
    y1 = min(fh, py + h)
    w_actual = x1 - x0
    h_actual = y1 - y0
    if w_actual <= 0 or h_actual <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    if GALLERY_ACTION_STYLE == "classic":
        default_start, default_end = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
    else:
        default_start, default_end = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
    start_bgr = top_bgr if top_bgr is not None else default_start
    grad_full = _vertical_gradient(h, total_w, start_bgr, default_end)
    gy0 = y0 - py
    gy1 = gy0 + h_actual
    gx0 = total_w - w_actual
    gx1 = total_w
    grad_slice = grad_full[gy0:gy1, gx0:gx1]
    if grad_slice.shape[0] != roi.shape[0] or grad_slice.shape[1] != roi.shape[1]:
        return
    cv2.addWeighted(grad_slice, ACTION_BTN_MODAL_FILL_ALPHA, roi, 1.0 - ACTION_BTN_MODAL_FILL_ALPHA, 0.0, dst=roi)
    if GALLERY_ACTION_STYLE != "classic":
        _add_glassy_shine(roi, alpha=ACTION_BTN_SHINE_ALPHA)


def _draw_modal_p_shape_border(
    frame: np.ndarray,
    vis_left: int,
    row_y: int,
    row_h: int,
    modal_right: int,
    modal_h: int,
    dock_x: int,
    dock_w: int,
) -> None:
    """Draw one continuous neon border around the P-shape (modal + button row); optional glow."""
    button_right = dock_x + dock_w - DOCK_TOP_INSET_X
    pts = [
        (vis_left, row_y),
        (button_right, row_y),
        (button_right, row_y + row_h),
        (modal_right, row_y + row_h),
        (modal_right, row_y + modal_h),
        (vis_left, row_y + modal_h),
        (vis_left, row_y),
    ]
    pts_np = np.array(pts, dtype=np.int32)
    pshape_border = CLASSIC_ACTION_BORDER_BGR if GALLERY_ACTION_STYLE == "classic" else ACTION_BTN_NEON_BORDER_BGR
    if GALLERY_ACTION_STYLE != "classic" and ACTION_BTN_NEON_GLOW > 0:
        pad = 12
        bx0 = max(0, min(pts, key=lambda p: p[0])[0] - pad)
        by0 = max(0, min(pts, key=lambda p: p[1])[1] - pad)
        bx1 = min(frame.shape[1], max(pts, key=lambda p: p[0])[0] + pad)
        by1 = min(frame.shape[0], max(pts, key=lambda p: p[1])[1] + pad)
        if bx1 > bx0 and by1 > by0:
            glow_patch = np.zeros((by1 - by0, bx1 - bx0, 3), dtype=np.uint8)
            pts_local = np.array([(x - bx0, y - by0) for x, y in pts], dtype=np.int32)
            cv2.polylines(glow_patch, [pts_local], isClosed=True, color=pshape_border, thickness=8, lineType=cv2.LINE_AA)
            glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 5.0)
            feathered_composite(frame, by0, by1, bx0, bx1, glow_patch, ACTION_BTN_NEON_GLOW, feather_px=12)
    cv2.polylines(frame, [pts_np], isClosed=True, color=pshape_border, thickness=max(1, ACTION_BTN_BORDER_THICKNESS), lineType=cv2.LINE_AA)


def _modal_seam_color(row_h: int, panel_h: int) -> Tuple[int, int, int]:
    """Color at the seam between strip (row) and panel so gradient is continuous (neon or classic)."""
    total_h = row_h + panel_h
    t = row_h / total_h if total_h > 0 else 1.0
    if GALLERY_ACTION_STYLE == "classic":
        top_bgr, bot_bgr = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
    else:
        top_bgr, bot_bgr = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
    return (
        int(top_bgr[0] * (1 - t) + bot_bgr[0] * t),
        int(top_bgr[1] * (1 - t) + bot_bgr[1] * t),
        int(top_bgr[2] * (1 - t) + bot_bgr[2] * t),
    )


def _draw_modal_panel_connected(
    frame: np.ndarray, px: int, py: int, content_w: int, h: int,
    top_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw modal panel with gradient (neon or classic); borders match Gallery Action style. Seamless with strip above if top_bgr set."""
    fh, fw = frame.shape[:2]
    total_w = content_w + MODAL_CONNECTOR_WIDTH
    x0 = max(0, px)
    x1 = min(fw, px + total_w)
    y0 = max(0, py)
    y1 = min(fh, py + h)
    vis_w = x1 - x0
    vis_h = y1 - y0
    if vis_w <= 0 or vis_h <= 0:
        return
    roi = frame[y0:y1, x0:x1]
    if GALLERY_ACTION_STYLE == "classic":
        default_start, default_end = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
        panel_border = CLASSIC_ACTION_BORDER_BGR
    else:
        default_start, default_end = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
        panel_border = ACTION_BTN_NEON_BORDER_BGR
    start_bgr = top_bgr if top_bgr is not None else default_start
    grad_full = _vertical_gradient(h, total_w, start_bgr, default_end)
    gx0 = x0 - px
    gx1 = gx0 + vis_w
    gy0 = y0 - py
    gy1 = gy0 + vis_h
    grad_slice = grad_full[gy0:gy1, gx0:gx1]
    cv2.addWeighted(grad_slice, ACTION_BTN_MODAL_FILL_ALPHA, roi, 1.0 - ACTION_BTN_MODAL_FILL_ALPHA, 0.0, dst=roi)
    if GALLERY_ACTION_STYLE != "classic":
        _add_glassy_shine(roi, alpha=ACTION_BTN_SHINE_ALPHA)
    thick = max(1, ACTION_BTN_BORDER_THICKNESS)
    if x0 < x1:
        cv2.line(frame, (x0, y0), (x0, y1 - 1), panel_border, thick, cv2.LINE_AA)
    if y0 < y1:
        cv2.line(frame, (x0, y0), (x1 - 1, y0), panel_border, thick, cv2.LINE_AA)
        cv2.line(frame, (x0, y1 - 1), (x1 - 1, y1 - 1), panel_border, thick, cv2.LINE_AA)


def _draw_filter_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Draw filter-by-type panel aligned with top of Filter button; no screen dim."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    filter_row_top_y = dock_y + DOCK_TOP_INSET_Y + row_h + DOCK_DIVIDER_THICKNESS
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = filter_row_top_y
    py = max(10, min(py, fh - MODAL_PANEL_H - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, MODAL_PANEL_H,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, MODAL_PANEL_H),
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        _put_text_white_glow(frame, "Filter by type", px + (MODAL_PANEL_W - 180) // 2, py + 32, font, 0.7, color=MODAL_TEXT_COLOR)
    for i, (label, value) in enumerate(FILTER_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass  # skip draw and register when clipped
        else:
            if content_visible:
                active = button_state.gallery_filter_type == value
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), ACTION_KEY_BORDER, 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
                if active:
                    cv2.putText(frame, label, (ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6),
                                font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    _put_text_white_glow(frame, label, ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6, font, 0.55, color=MODAL_TEXT_COLOR)
        key = f"gallery_filter_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h
    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_filter_modal_panel" not in menu_buttons:
        menu_buttons["gallery_filter_modal_panel"] = Button(px, py, total_w, MODAL_PANEL_H, "")
    else:
        menu_buttons["gallery_filter_modal_panel"].x, menu_buttons["gallery_filter_modal_panel"].y = px, py
        menu_buttons["gallery_filter_modal_panel"].w, menu_buttons["gallery_filter_modal_panel"].h = total_w, MODAL_PANEL_H


def _draw_sort_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Draw sort panel aligned with top of Sort button; height scales with number of options."""
    fh, fw = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    sort_row_top_y = dock_y + DOCK_TOP_INSET_Y + 2 * (row_h + DOCK_DIVIDER_THICKNESS)
    panel_h = MODAL_TITLE_H + len(SORT_OPTIONS) * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = sort_row_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(row_h, panel_h),
        )
    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        _put_text_white_glow(frame, "Sort by", px + (MODAL_PANEL_W - 80) // 2, py + 32, font, 0.7, color=MODAL_TEXT_COLOR)
    for i, (label, value) in enumerate(SORT_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                active = button_state.gallery_sort_by == value
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), ACTION_KEY_BORDER, 1, cv2.LINE_AA)
                (tw, _), _ = cv2.getTextSize(label, font, 0.5, 1)
                if active:
                    cv2.putText(frame, label, (ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6),
                                font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    _put_text_white_glow(frame, label, ox + (btn_w - tw) // 2, oy + btn_h // 2 + 6, font, 0.5, color=MODAL_TEXT_COLOR)
        key = f"gallery_sort_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h
    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_sort_modal_panel" not in menu_buttons:
        menu_buttons["gallery_sort_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_sort_modal_panel"].x, menu_buttons["gallery_sort_modal_panel"].y = px, py
        menu_buttons["gallery_sort_modal_panel"].w, menu_buttons["gallery_sort_modal_panel"].h = total_w, panel_h


def _draw_priority_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Priority picker aligned with Row 2 (Priority button) in select mode."""
    fh, _ = frame.shape[:2]
    row_h = DOCK_ROW_HEIGHT
    # Row 2 top = after Row 1 + divider
    row2_top_y = dock_y + DOCK_TOP_INSET_Y + row_h + DOCK_DIVIDER_THICKNESS
    n_opts = len(PRIORITY_OPTIONS)
    panel_h = MODAL_TITLE_H + n_opts * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = row2_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(row_h, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        _put_text_white_glow(frame, "Set Priority", px + (MODAL_PANEL_W - 130) // 2, py + 32, font, 0.7, color=MODAL_TEXT_COLOR)

    # Determine which priority value is held by ALL currently selected items.
    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_names = [r['filepath'].name for r in rects if r['idx'] in button_state.gallery_selected_items]
    priorities_map = getattr(button_state, 'gallery_file_priorities', {})
    selected_priorities: set = set()
    for fname in sel_names:
        pval = priorities_map.get(fname, "")
        if pval:
            selected_priorities.add(pval)

    for i, (label, value) in enumerate(PRIORITY_OPTIONS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                active = value in selected_priorities
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), ACTION_KEY_BORDER, 1, cv2.LINE_AA)
                dot_color = PRIORITY_COLORS[value]
                dot_cx = ox + 18
                dot_cy = oy + btn_h // 2
                draw_priority_circle_neon(frame, dot_cx, dot_cy, 7, dot_color)
                (tw, _), _ = cv2.getTextSize(label, font, 0.55, 1)
                if active:
                    cv2.putText(frame, label, (ox + 34, oy + btn_h // 2 + 6),
                                font, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    _put_text_white_glow(frame, label, ox + 34, oy + btn_h // 2 + 6, font, 0.55, color=MODAL_TEXT_COLOR)

        key = f"gallery_priority_opt_{value}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, label)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h

    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_priority_modal_panel" not in menu_buttons:
        menu_buttons["gallery_priority_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_priority_modal_panel"].x, menu_buttons["gallery_priority_modal_panel"].y = px, py
        menu_buttons["gallery_priority_modal_panel"].w, menu_buttons["gallery_priority_modal_panel"].h = total_w, panel_h


def _draw_tags_modal(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Tags panel aligned with Row 1 (Tags button) in select mode."""
    fh, _ = frame.shape[:2]
    row1_top_y = dock_y + DOCK_TOP_INSET_Y
    n_tags = len(PRESET_TAGS)
    panel_h = MODAL_TITLE_H + n_tags * MODAL_OPTION_H + 10
    px = dock_x - MODAL_GAP_WEST - MODAL_PANEL_W
    py = row1_top_y
    py = max(10, min(py, fh - panel_h - 10))
    if draw_panel and content_visible and vis_left < 0:
        _draw_modal_panel_connected(
            frame, px, py, MODAL_PANEL_W, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible and (vis_left < 0 or px + MODAL_PANEL_W >= vis_left):
        _put_text_white_glow(frame, "Tags", px + (MODAL_PANEL_W - 45) // 2, py + 32, font, 0.7, color=MODAL_TEXT_COLOR)

    for i, tag in enumerate(PRESET_TAGS):
        oy = py + MODAL_TITLE_H + i * MODAL_OPTION_H
        btn_w, btn_h = MODAL_PANEL_W - 24, MODAL_OPTION_H - 8
        ox = px + (MODAL_PANEL_W - btn_w) // 2
        if vis_left >= 0 and ox + btn_w < vis_left:
            pass
        else:
            if content_visible:
                file_tags = getattr(button_state, 'gallery_file_tags', {})
                active = False
                if button_state.gallery_selected_items:
                    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
                    sel_names = [r['filepath'].name for r in rects if r['idx'] in button_state.gallery_selected_items]
                    active = bool(sel_names) and all(tag in file_tags.get(n, []) for n in sel_names)
                if active:
                    cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), MODAL_ACTIVE_GOLD, -1)
                cv2.rectangle(frame, (ox, oy), (ox + btn_w, oy + btn_h), ACTION_KEY_BORDER, 1, cv2.LINE_AA)
                icon_color = (0, 0, 0) if active else (CLASSIC_ACTION_TEXT_BGR if GALLERY_ACTION_STYLE == "classic" else SEARCH_BAR_TEXT_COLOR)
                _draw_icon_tag(frame, ox + 18, oy + btn_h // 2, icon_color)
                if active:
                    cv2.putText(frame, tag, (ox + 34, oy + btn_h // 2 + 6),
                                font, 0.52, (0, 0, 0), 1, cv2.LINE_AA)
                else:
                    _put_text_white_glow(frame, tag, ox + 34, oy + btn_h // 2 + 6, font, 0.52, color=MODAL_TEXT_COLOR)

        key = f"gallery_tag_opt_{tag.lower().replace('-', '_').replace(' ', '_')}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(ox, oy, btn_w, btn_h, tag)
        else:
            menu_buttons[key].x, menu_buttons[key].y = ox, oy
            menu_buttons[key].w, menu_buttons[key].h = btn_w, btn_h

    total_w = MODAL_PANEL_W + MODAL_CONNECTOR_WIDTH
    if "gallery_tags_modal_panel" not in menu_buttons:
        menu_buttons["gallery_tags_modal_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["gallery_tags_modal_panel"].x, menu_buttons["gallery_tags_modal_panel"].y = px, py
        menu_buttons["gallery_tags_modal_panel"].w, menu_buttons["gallery_tags_modal_panel"].h = total_w, panel_h


def _draw_rename_keyboard(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Standard keyboard (alpha/symbol, ?123, Shift, shortcut row); expands to fill modal height."""
    fh, fw = frame.shape[:2]
    panel_w = dock_x - MODAL_GAP_WEST - MODAL_EDGE_MARGIN
    row3_top_y = dock_y + DOCK_TOP_INSET_Y + 2 * (DOCK_ROW_HEIGHT + DOCK_DIVIDER_THICKNESS)
    py = row3_top_y
    # Use full height below rename row so keys expand to fill available space
    panel_h = max(KEYBOARD_BAR_H + 120, fh - py - 12)
    px = MODAL_EDGE_MARGIN
    if draw_panel and content_visible:
        _draw_modal_panel_connected(
            frame, px, py, panel_w, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible:
        display = button_state.gallery_rename_query if button_state.gallery_rename_query else "New name..."
        _put_text_white_glow(frame, display[:40], px + 10, py + KEYBOARD_BAR_H - 10, font, KEYBOARD_FONT_BAR, color=MODAL_TEXT_COLOR)

    keyboard_region_h = panel_h - KEYBOARD_BAR_H - KEYBOARD_FOOTER_GAP
    key_gap = 6
    n_rows = 5
    dims = compute_standard_keyboard_dimensions(panel_w, keyboard_region_h, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    key_y = py + KEYBOARD_BAR_H + key_gap
    key_font = 0.66 if key_h > 32 else 0.56
    key_font_special = 0.62 if key_h > 32 else 0.52
    is_symbol = getattr(button_state, "gallery_keyboard_mode", "alpha") == "symbol"
    shift_highlight = getattr(button_state, "gallery_keyboard_shift_next", False)
    if content_visible:
        if is_symbol:
            draw_standard_symbol_keyboard(
                frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
                "rename_key_", font, key_font, key_font_special,
                key_border_bgr=ACTION_KEY_BORDER, key_text_bgr=KEYBOARD_KEY_TEXT,
            )
        else:
            draw_standard_alpha_keyboard(
                frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
                "rename_key_", font, key_font, key_font_special, shift_highlight,
                key_border_bgr=ACTION_KEY_BORDER, key_text_bgr=KEYBOARD_KEY_TEXT,
                shift_highlight_color=MENU_ACTIVE_BLUE,
            )

    total_w = panel_w + MODAL_CONNECTOR_WIDTH
    if "rename_keyboard_panel" not in menu_buttons:
        menu_buttons["rename_keyboard_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["rename_keyboard_panel"].x, menu_buttons["rename_keyboard_panel"].y = px, py
        menu_buttons["rename_keyboard_panel"].w, menu_buttons["rename_keyboard_panel"].h = total_w, panel_h


def _draw_search_keyboard(
    frame: np.ndarray, dock_x: int, dock_y: int, vis_left: int = -1, draw_panel: bool = True,
    content_visible: bool = True,
) -> None:
    """Standard keyboard (alpha/symbol, ?123, Shift, shortcut row); expands to fill modal height."""
    fh, fw = frame.shape[:2]
    panel_w = dock_x - MODAL_GAP_WEST - MODAL_EDGE_MARGIN
    search_row_top = dock_y + DOCK_TOP_INSET_Y
    py = search_row_top
    # Use full height below search row so keys expand to fill available space
    panel_h = max(KEYBOARD_BAR_H + 120, fh - py - 12)
    px = MODAL_EDGE_MARGIN
    if draw_panel and content_visible:
        _draw_modal_panel_connected(
            frame, px, py, panel_w, panel_h,
            top_bgr=_modal_seam_color(DOCK_ROW_HEIGHT, panel_h),
        )

    font = cv2.FONT_HERSHEY_SIMPLEX
    if content_visible:
        display = button_state.gallery_search_query if button_state.gallery_search_query else "Search..."
        _put_text_white_glow(frame, display[:40], px + 10, py + KEYBOARD_BAR_H - 10, font, KEYBOARD_FONT_BAR, color=MODAL_TEXT_COLOR)

    keyboard_region_h = panel_h - KEYBOARD_BAR_H - KEYBOARD_FOOTER_GAP
    key_gap = 6
    n_rows = 5
    dims = compute_standard_keyboard_dimensions(panel_w, keyboard_region_h, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    key_y = py + KEYBOARD_BAR_H + key_gap
    key_font = 0.66 if key_h > 32 else 0.56
    key_font_special = 0.62 if key_h > 32 else 0.52
    is_symbol = getattr(button_state, "gallery_keyboard_mode", "alpha") == "symbol"
    shift_highlight = getattr(button_state, "gallery_keyboard_shift_next", False)
    if content_visible:
        if is_symbol:
            draw_standard_symbol_keyboard(
                frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
                "search_key_", font, key_font, key_font_special,
                key_border_bgr=ACTION_KEY_BORDER, key_text_bgr=KEYBOARD_KEY_TEXT,
            )
        else:
            draw_standard_alpha_keyboard(
                frame, px, key_y, panel_w, key_w, key_h, sp_w, key_gap,
                "search_key_", font, key_font, key_font_special, shift_highlight,
                key_border_bgr=ACTION_KEY_BORDER, key_text_bgr=KEYBOARD_KEY_TEXT,
                shift_highlight_color=MENU_ACTIVE_BLUE,
            )

    total_w = panel_w + MODAL_CONNECTOR_WIDTH
    if "search_keyboard_panel" not in menu_buttons:
        menu_buttons["search_keyboard_panel"] = Button(px, py, total_w, panel_h, "")
    else:
        menu_buttons["search_keyboard_panel"].x, menu_buttons["search_keyboard_panel"].y = px, py
        menu_buttons["search_keyboard_panel"].w, menu_buttons["search_keyboard_panel"].h = total_w, panel_h


def draw_grid_side_dock(
    frame: np.ndarray,
    header_h: int,
    items: List[Tuple[Path, str, datetime]],
    output_dir: Optional[Path],
    scroll_offset: int = 0,
) -> None:
    """
    Draw the full-height side dock (delimited strip on the right) and the
    storage bar. The bar is viewport-fixed so it floats with you as you
    scroll the grid.
    """
    frame_h, frame_w = frame.shape[:2]
    dock_x = frame_w - GRID_SIDE_DOCK_WIDTH
    dock_y = header_h + 3
    dock_w = GRID_SIDE_DOCK_WIDTH
    dock_h = frame_h - dock_y

    # Dock background: grey-to-black gradient (same as viewer dock / top bar)
    dock_roi = frame[dock_y:dock_y + dock_h, dock_x:dock_x + dock_w]
    dock_roi[:] = _vertical_gradient(dock_h, dock_w, DOCK_GRADIENT_TOP, DOCK_GRADIENT_BOT)
    # Left edge: clear delimiter so it looks like a dock
    cv2.line(
        frame, (dock_x, dock_y), (dock_x, dock_y + dock_h),
        DOCK_EDGE_ACCENT, 2, cv2.LINE_AA
    )
    cv2.line(
        frame, (dock_x + 1, dock_y), (dock_x + 1, dock_y + dock_h),
        DOCK_EDGE, 1, cv2.LINE_AA
    )
    # Bottom edge: delimiter for bottom of side dock
    bottom_y = frame_h - 1
    cv2.line(
        frame, (dock_x, bottom_y), (dock_x + dock_w, bottom_y),
        DOCK_EDGE_ACCENT, 2, cv2.LINE_AA
    )
    cv2.line(
        frame, (dock_x, bottom_y - 1), (dock_x + dock_w, bottom_y - 1),
        DOCK_EDGE, 1, cv2.LINE_AA
    )

    # Modal slide animation: always run when opening/closing or when switching which modal is open
    info = _get_active_modal_info(frame, dock_x, dock_y, dock_w)
    any_open = info is not None
    active_key = info[0] if info else None
    target = 1.0 if any_open else 0.0
    if active_key != getattr(button_state, "gallery_modal_anim_active_key", ""):
        button_state.gallery_modal_anim_active_key = active_key or ""
        button_state.gallery_modal_anim_target = target
        button_state.gallery_modal_anim_start_t = time.time()
        button_state.gallery_modal_anim_progress_at_flip = (
            0.0 if any_open else button_state.gallery_modal_anim_progress
        )
        if any_open:
            button_state.gallery_modal_anim_progress = 0.0
    elif target != button_state.gallery_modal_anim_target:
        button_state.gallery_modal_anim_target = target
        button_state.gallery_modal_anim_start_t = time.time()
        button_state.gallery_modal_anim_progress_at_flip = button_state.gallery_modal_anim_progress
    now = time.time()
    elapsed = now - button_state.gallery_modal_anim_start_t
    t = min(1.0, elapsed / MODAL_ANIM_DURATION_S)
    if button_state.gallery_modal_anim_target >= 0.5:
        button_state.gallery_modal_anim_progress = button_state.gallery_modal_anim_progress_at_flip + (1.0 - button_state.gallery_modal_anim_progress_at_flip) * t
    else:
        button_state.gallery_modal_anim_progress = button_state.gallery_modal_anim_progress_at_flip * (1.0 - t)
    progress = button_state.gallery_modal_anim_progress
    # Ease-in-out (smoothstep) to smooth the grow animation and reduce flicker/secondary movement
    t = progress
    progress_eased = t * t * (3.0 - 2.0 * t) if 0 <= t <= 1 else t
    modal_px = modal_total_w = modal_h = row_y = row_h = 0
    if info is not None:
        _, row_y, row_h, modal_px, modal_total_w, modal_h = info

    # Grow-from-button: visible region is the right (progress * total_w) part of the modal
    vis_left = -1
    if info is not None and progress_eased > 0.001:
        vis_width = max(1, int(progress_eased * modal_total_w))
        vis_left = modal_px + modal_total_w - vis_width
        strip_right = dock_x + dock_w
        strip_width = max(1, strip_right - vis_left)
        # One continuous gradient: strip top->mid, panel mid->bottom so no seam at row_h (classic = lighter/top-dock colors)
        t_mid = row_h / modal_h if modal_h > 0 else 1.0
        if GALLERY_ACTION_STYLE == "classic":
            top_b, bot_b = CLASSIC_ACTION_FILL_TOP, CLASSIC_ACTION_FILL_BOT
        else:
            top_b, bot_b = ACTION_BTN_FILL_DARK_TOP, ACTION_BTN_FILL_DARK_BOT
        mid_bgr: Tuple[int, int, int] = (
            int(top_b[0] * (1 - t_mid) + bot_b[0] * t_mid),
            int(top_b[1] * (1 - t_mid) + bot_b[1] * t_mid),
            int(top_b[2] * (1 - t_mid) + bot_b[2] * t_mid),
        )
        _draw_combined_modal_strip(
            frame, vis_left, row_y, strip_width, row_h,
            white_border=False, bottom_bgr=mid_bgr,
        )
        if modal_h > row_h:
            content_w = modal_total_w - MODAL_CONNECTOR_WIDTH
            _draw_modal_panel_grow_slice(
                frame, modal_px, row_y + row_h, content_w, modal_h - row_h,
                vis_left, vis_width, top_bgr=mid_bgr,
            )

    # Draw modal content only after growth finishes to avoid content appearing before the panel
    content_visible = (info is not None and progress_eased >= 0.999)
    draw_panel = True
    if button_state.gallery_select_mode:
        if button_state.gallery_priority_modal_open:
            _draw_priority_modal(frame, dock_x, dock_y, vis_left if active_key == "priority" else -1, draw_panel, content_visible)
        if button_state.gallery_rename_modal_open:
            _draw_rename_keyboard(frame, dock_x, dock_y, vis_left if active_key == "rename" else -1, draw_panel, content_visible)
    else:
        if button_state.gallery_filter_modal_open:
            _draw_filter_modal(frame, dock_x, dock_y, vis_left if active_key == "filter" else -1, draw_panel, content_visible)
        if button_state.gallery_sort_modal_open:
            _draw_sort_modal(frame, dock_x, dock_y, vis_left if active_key == "sort" else -1, draw_panel, content_visible)
        if button_state.gallery_search_keyboard_open:
            _draw_search_keyboard(frame, dock_x, dock_y, vis_left if active_key == "search" else -1, draw_panel, content_visible)

    _draw_dock_top_rows(frame, dock_x, dock_y, dock_w, active_key, progress_eased)

    # P-shaped continuous outer border (button + modal); grows with the modal
    if info is not None and progress_eased > 0.001 and vis_left >= 0:
        modal_right = modal_px + modal_total_w
        _draw_modal_p_shape_border(frame, vis_left, row_y, row_h, modal_right, modal_h, dock_x, dock_w)

    draw_storage_bar(
        frame, dock_x, dock_w, header_h, items, output_dir
    )

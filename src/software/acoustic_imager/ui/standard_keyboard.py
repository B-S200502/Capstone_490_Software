"""
Standard on-screen keyboard: alpha (QWERTY + shortcut row + ?123 + Shift) and symbol layout.
Used by Email Settings, Tags, Rename, Search, and Archive folder rename.
"""

from __future__ import annotations

from typing import Tuple, List, Optional

import cv2
import numpy as np

from .button import menu_buttons, Button
from .keyboard import (
    ROWS_QWERTY,
    dimensions_for_scale,
    draw_key_bg_clipped,
    draw_key_bg_solid,
    COMPACT_KEY_SCALE,
    SPECIAL_KEYS_COMPACT,
)

# Symbol keyboard rows (same as email modal)
STANDARD_SYMBOL_ROW1 = "1234567890"
STANDARD_SYMBOL_ROW2 = "@._-!#$%&*"
STANDARD_SYMBOL_ROW3 = "()+=[{]};:'\""
STANDARD_SYMBOL_ROW4 = "\"(</>?\\,"  # quote, open paren, < / > ? \ ,

# Alpha shortcut row: (label, value_suffix, key_width_mult)
STANDARD_ALPHA_SHORTCUT_ROW: List[Tuple[str, str, float]] = [
    (".com", "snippet_com", 1.5),
    (".", "dot", 1.0),
    ("@", "at", 1.0),
    ("-", "dash", 1.0),
    ("_", "underscore", 1.0),
    ("space", "space", 2.0),
]

# Default dimensions when not provided (e.g. for computing from panel)
_DEFAULT_DIMS = dimensions_for_scale(COMPACT_KEY_SCALE, width_mult=1.5)
DEFAULT_KEY_TEXT_BGR: Tuple[int, int, int] = (28, 28, 28)
DEFAULT_KEY_BORDER_BGR: Tuple[int, int, int] = (90, 90, 90)


def _symbol_key_suffix(c: str) -> str:
    """Return menu_buttons key suffix for symbol character (escape backslash for key_id)."""
    if c == "\\":
        return "\\"  # one backslash so key_id is e.g. "email_key_\\"
    return c


def _draw_one_key(
    frame: "np.ndarray",
    kx: int, ky: int, kw: int, kh: int,
    label: str,
    font,
    font_scale: float,
    key_id: str,
    key_border_bgr: Tuple[int, int, int],
    key_text_bgr: Tuple[int, int, int],
    key_fill_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw one key and register it in menu_buttons. If key_fill_bgr is set, use solid fill (e.g. white)."""
    if key_fill_bgr is not None:
        draw_key_bg_solid(frame, kx, ky, kw, kh, key_fill_bgr)
    else:
        draw_key_bg_clipped(frame, kx, ky, kw, kh)
    cv2.rectangle(frame, (kx, ky), (kx + kw, ky + kh), key_border_bgr, 1, cv2.LINE_AA)
    fs = font_scale
    if len(label) > 2:
        fs = min(font_scale, 0.52)
    (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
    tx = kx + (kw - tw) // 2
    ty = ky + (kh + th) // 2
    cv2.putText(frame, label, (tx, ty), font, fs, key_text_bgr, 1, cv2.LINE_AA)
    if key_id not in menu_buttons:
        menu_buttons[key_id] = Button(kx, ky, kw, kh, label)
    else:
        menu_buttons[key_id].x, menu_buttons[key_id].y = kx, ky
        menu_buttons[key_id].w, menu_buttons[key_id].h = kw, kh


def draw_standard_alpha_keyboard(
    frame: np.ndarray,
    base_x: int, start_y: int, content_w: int,
    key_w: int, key_h: int, sp_w: int, key_gap: int,
    key_prefix: str,
    font,
    key_font: float, key_font_special: float,
    shift_highlight: bool,
    key_border_bgr: Tuple[int, int, int] = DEFAULT_KEY_BORDER_BGR,
    key_text_bgr: Tuple[int, int, int] = DEFAULT_KEY_TEXT_BGR,
    shift_highlight_color: Tuple[int, int, int] = (80, 120, 255),  # BGR blue
    key_fill_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw standard alpha keyboard: QWERTY + shortcut row + ?123 + Shift, then Back/Clear/Done."""
    key_y = start_y
    rows = list(ROWS_QWERTY)
    for i, row_chars in enumerate(rows):
        if i == 2:  # zxcvbnm row: ?123 (wide), letters, Shift
            q123_w = 2 * key_w + key_gap
            row_w = q123_w + 8 * (key_w + key_gap) - key_gap
            key_x = base_x + (content_w - row_w) // 2
            _draw_one_key(
                frame, key_x, key_y, q123_w, key_h, "?123", font, key_font_special,
                key_prefix + "switch_sym", key_border_bgr, key_text_bgr, key_fill_bgr,
            )
            key_x += q123_w + key_gap
            for c in row_chars:
                kid = key_prefix + c
                _draw_one_key(frame, key_x, key_y, key_w, key_h, c.upper(), font, key_font, kid, key_border_bgr, key_text_bgr, key_fill_bgr)
                key_x += key_w + key_gap
            _draw_one_key(
                frame, key_x, key_y, key_w, key_h, "Shift", font, key_font_special,
                key_prefix + "shift", key_border_bgr, key_text_bgr, key_fill_bgr,
            )
            if shift_highlight:
                cv2.rectangle(frame, (key_x, key_y), (key_x + key_w, key_y + key_h), shift_highlight_color, 2, cv2.LINE_AA)
            key_y += key_h + key_gap
            continue
        row_w = len(row_chars) * (key_w + key_gap) - key_gap
        key_x = base_x + (content_w - row_w) // 2
        for c in row_chars:
            kid = key_prefix + c
            _draw_one_key(frame, key_x, key_y, key_w, key_h, c.upper(), font, key_font, kid, key_border_bgr, key_text_bgr, key_fill_bgr)
            key_x += key_w + key_gap
        key_y += key_h + key_gap
    # Shortcut row
    shortcut_total_w = sum(int(m * key_w) for _, _, m in STANDARD_ALPHA_SHORTCUT_ROW) + (len(STANDARD_ALPHA_SHORTCUT_ROW) - 1) * key_gap
    key_x = base_x + (content_w - shortcut_total_w) // 2
    for label, val, mult in STANDARD_ALPHA_SHORTCUT_ROW:
        kw = int(mult * key_w)
        _draw_one_key(frame, key_x, key_y, kw, key_h, label, font, key_font_special, key_prefix + val, key_border_bgr, key_text_bgr, key_fill_bgr)
        key_x += kw + key_gap
    key_y += key_h + key_gap
    # Back, Clear, Done
    sp_key_x = base_x
    for label, val in SPECIAL_KEYS_COMPACT:
        _draw_one_key(frame, sp_key_x, key_y, sp_w, key_h, label, font, key_font_special, key_prefix + val, key_border_bgr, key_text_bgr, key_fill_bgr)
        sp_key_x += sp_w + key_gap


def draw_standard_symbol_keyboard(
    frame: np.ndarray,
    base_x: int, start_y: int, content_w: int,
    key_w: int, key_h: int, sp_w: int, key_gap: int,
    key_prefix: str,
    font,
    key_font: float, key_font_special: float,
    key_border_bgr: Tuple[int, int, int] = DEFAULT_KEY_BORDER_BGR,
    key_text_bgr: Tuple[int, int, int] = DEFAULT_KEY_TEXT_BGR,
    key_fill_bgr: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Draw standard symbol keyboard: 4 symbol rows + ABC, then Back/Clear/Done."""
    key_y = start_y
    symbol_rows = (STANDARD_SYMBOL_ROW1, STANDARD_SYMBOL_ROW2, STANDARD_SYMBOL_ROW3, STANDARD_SYMBOL_ROW4)
    for row_idx, row_chars in enumerate(symbol_rows):
        if row_idx == 3:
            abc_w = 2 * key_w + key_gap
            row_w = len(row_chars) * (key_w + key_gap) - key_gap + (abc_w + key_gap)
            key_x = base_x + (content_w - row_w) // 2
            for c in row_chars:
                label = c
                kid = key_prefix + _symbol_key_suffix(c)
                _draw_one_key(frame, key_x, key_y, key_w, key_h, label, font, key_font, kid, key_border_bgr, key_text_bgr, key_fill_bgr)
                key_x += key_w + key_gap
            _draw_one_key(
                frame, key_x, key_y, abc_w, key_h, "ABC", font, key_font_special,
                key_prefix + "switch_alpha", key_border_bgr, key_text_bgr, key_fill_bgr,
            )
        else:
            row_w = len(row_chars) * (key_w + key_gap) - key_gap
            key_x = base_x + (content_w - row_w) // 2
            for c in row_chars:
                label = c
                kid = key_prefix + _symbol_key_suffix(c)
                _draw_one_key(frame, key_x, key_y, key_w, key_h, label, font, key_font, kid, key_border_bgr, key_text_bgr, key_fill_bgr)
                key_x += key_w + key_gap
        key_y += key_h + key_gap
    sp_key_x = base_x
    for label, val in SPECIAL_KEYS_COMPACT:
        _draw_one_key(frame, sp_key_x, key_y, sp_w, key_h, label, font, key_font_special, key_prefix + val, key_border_bgr, key_text_bgr, key_fill_bgr)
        sp_key_x += sp_w + key_gap


def compute_standard_keyboard_dimensions(
    content_w: int, keyboard_h: int, key_gap: int = 6, n_rows: int = 5,
) -> dict:
    """Compute key_w, key_h, sp_w to fit standard keyboard in content_w x keyboard_h. Keys expand to fill modal."""
    key_h = (keyboard_h - key_gap * (n_rows + 1)) // n_rows
    key_h = max(20, key_h)
    # Widest alpha row: 11 keys (2+8+1) + 8 gaps
    key_w_alpha = (content_w - 8 * key_gap) // 11 if content_w > 0 else 28
    # Symbol row 1: 10 keys + 9 gaps
    key_w_sym = (content_w - 9 * key_gap) // 10 if content_w > 0 else 28
    # Use full width (cap at 120 so keys don't get absurdly large on very wide screens)
    key_w = min(key_w_alpha, key_w_sym, 120)
    key_w = max(18, key_w)
    sp_w = (content_w - 2 * key_gap) // 3
    sp_w = max(key_w, sp_w)
    return {"key_w": key_w, "key_h": key_h, "sp_w": sp_w, "key_gap": key_gap}

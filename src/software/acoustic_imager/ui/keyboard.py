"""
On-screen keyboard: shared layout, dimensions, and key styling.

Keys use a standard legible gray gradient with good accessibility
(dark text on light gray, WCAG-friendly contrast).
"""

from typing import Tuple, List, Optional

import numpy as np


# ─── Layout (QWERTY + number row + special row) ─────────────────────────────
ROWS_QWERTY: List[str] = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
ROW_NUMBERS: str = "1234567890"

# (display_label, value_key) for special row; label can vary by context
SPECIAL_KEYS_FULL: List[Tuple[str, str]] = [
    ("Back", "backspace"),
    ("Clear", "clear"),
    ("Done", "done"),
]
SPECIAL_KEYS_COMPACT: List[Tuple[str, str]] = [
    ("Back", "backspace"),
    ("Clear", "clear"),
    ("Done", "done"),
]


# ─── Scale presets ─────────────────────────────────────────────────────────
FULL_KEY_SCALE = 1.875   # Search & Rename keyboards (dock)
COMPACT_KEY_SCALE = 1.3  # Edit Tags modal keyboard
# Widen keys (touch targets) without changing height
KEY_WIDTH_MULT = 1.4
KEY_WIDTH_MULT_COMPACT = 1.82  # Edit Tags keyboard: wider touch targets


# ─── Accessible key styling: legible gray gradient ─────────────────────────
# Light gray (top) to mid gray (bottom) for a raised, readable key.
# Text is dark for strong contrast (WCAG AA+).
KEY_GRADIENT_TOP_BGR: Tuple[int, int, int] = (225, 225, 225)  # light gray
KEY_GRADIENT_BOT_BGR: Tuple[int, int, int] = (145, 145, 145)  # darker gray
KEY_BORDER_BGR: Tuple[int, int, int] = (90, 90, 90)
KEY_TEXT_BGR: Tuple[int, int, int] = (28, 28, 28)  # near black


def _vertical_gradient(
    h: int, w: int,
    top_bgr: Tuple[int, int, int], bot_bgr: Tuple[int, int, int]
) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(
            top_bgr[c], bot_bgr[c], h, dtype=np.uint8
        ).reshape(-1, 1)
    return out


def draw_key_bg_clipped(
    frame: np.ndarray, x: int, y: int, w: int, h: int
) -> None:
    """
    Draw one key background with accessible gray gradient.
    Clips to frame bounds so ROI and gradient shapes match (safe during animations).
    """
    fh, fw = frame.shape[:2]
    x0, x1 = max(0, x), min(fw, x + w)
    y0, y1 = max(0, y), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    roi[:] = _vertical_gradient(
        roi.shape[0], roi.shape[1],
        KEY_GRADIENT_TOP_BGR, KEY_GRADIENT_BOT_BGR
    )


def draw_key_bg_solid(
    frame: np.ndarray, x: int, y: int, w: int, h: int,
    bgr: Tuple[int, int, int],
) -> None:
    """Draw one key background with a solid color (e.g. white for black-on-white keys). Clips to frame."""
    fh, fw = frame.shape[:2]
    x0, x1 = max(0, x), min(fw, x + w)
    y0, y1 = max(0, y), min(fh, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = bgr


def dimensions_for_scale(scale: float, width_mult: Optional[float] = None) -> dict:
    """
    Return key dimensions and font scales for a given scale factor.
    Keys: key_w, key_h, key_gap, bar_h, footer_gap, font_bar, font_key, font_special,
    special_key_w_mult (e.g. 2 for double-width special keys).
    Key width is scaled by width_mult (default KEY_WIDTH_MULT); height unchanged.
    """
    w_mult = width_mult if width_mult is not None else KEY_WIDTH_MULT
    key_w = int(28 * scale * w_mult)
    key_h = int(28 * scale)
    key_gap = int(4 * scale)
    bar_h = int(22 * scale) if scale >= 1.5 else int(36 * scale)
    footer_gap = 0 if scale >= 1.5 else 6
    return {
        "key_w": key_w,
        "key_h": key_h,
        "key_gap": key_gap,
        "bar_h": bar_h,
        "footer_gap": footer_gap,
        "font_bar": 0.6 if scale >= 1.5 else 0.48,
        "font_key": 0.82 if scale >= 1.5 else 0.75,
        "font_special": 0.76 if scale >= 1.5 else 0.69,
        "special_key_w_mult": 2,
    }

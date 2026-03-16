# dsp/bars.py
from __future__ import annotations

import numpy as np
import cv2

from dataclasses import dataclass
import time

from ..config import FREQ_BAR_BLUE

# Colormap mapping
COLORMAP_DICT = {
    "MAGMA": cv2.COLORMAP_MAGMA,
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
}

@dataclass
class DbSliderState:
    enabled: bool = False       # double-click toggles this
    dragging: bool = False
    db_max: float = 12.0        # top label / scaling
    span_db: float = 60.0       # db_min = db_max - span_db
    _last_click_t: float = 0.0  # for double click detect
    _last_click_y: int = 0

_DB_COLORBAR_CACHE: dict[tuple[int, int, str], np.ndarray] = {}  # (h, w, colormap) -> colormap image

def db_to_y(db: float, h: int, db_min: float, db_max: float) -> int:
    if h <= 1:
        return 0
    db = float(np.clip(db, db_min, db_max))
    frac = (db - db_min) / max(1e-9, (db_max - db_min))
    # db_max at top (y=0)
    return int((1.0 - frac) * (h - 1))

def y_to_db(y: int, h: int, db_min: float, db_max: float) -> float:
    y = int(np.clip(y, 0, h - 1))
    frac = 1.0 - (y / max(1, (h - 1)))
    return float(db_min + frac * (db_max - db_min))

_PANEL_CACHE: dict[tuple[int, int], np.ndarray] = {}

def _get_panel_bg(h: int, w: int) -> np.ndarray:
    key = (h, w)
    bg = _PANEL_CACHE.get(key)
    if bg is None:
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        _draw_panel_bg(bg)
        _PANEL_CACHE[key] = bg
    return bg

def _draw_vignette(img: np.ndarray, strength: float = 0.35) -> None:
    h, w = img.shape[:2]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    r2 = x * x + y * y
    base = 1.0 - np.clip(r2, 0.0, 1.0)          # center=1, edges=0
    base = base ** 0.9                           # curve
    mask = (1.0 - strength) + strength * base    # edges=(1-strength), center=1
    img[:] = (img.astype(np.float32) * mask[..., None]).astype(np.uint8)


def _draw_panel_bg(bar: np.ndarray) -> None:
    """
    Fancy dark UI panel background:
      - vertical gradient
      - subtle grid
      - glass highlight + edge lines
      - slight vignette
    """
    h, w = bar.shape[:2]

    # ---- base vertical gradient (dark top -> slightly brighter bottom)
    top = np.array([18, 18, 22], dtype=np.float32)[None, None, :]   # (1,1,3)
    bot = np.array([34, 34, 42], dtype=np.float32)[None, None, :]   # (1,1,3)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]   # (h,1,1)
    grad = (top * (1.0 - t) + bot * t).astype(np.uint8)             # (h,1,3)
    bar[:] = np.repeat(grad, w, axis=1)                              # (h,w,3)

    # ---- subtle grid (thin lines)
    grid = bar.copy()
    major = 40   # px
    minor = 20

    # vertical lines
    for x in range(0, w, minor):
        cv2.line(grid, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
    # horizontal lines
    for y in range(0, h, minor):
        cv2.line(grid, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)

    # make every other "major" line a bit stronger
    for x in range(0, w, major):
        cv2.line(grid, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, major):
        cv2.line(grid, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)

    # blend grid in very lightly
    cv2.addWeighted(grid, 0.12, bar, 0.88, 0.0, dst=bar)

    major_layer = np.zeros_like(bar)
    for x in range(0, w, major):
        cv2.line(major_layer, (x, 0), (x, h - 1), (255, 255, 255), 1, cv2.LINE_AA)
    for y in range(0, h, major):
        cv2.line(major_layer, (0, y), (w - 1, y), (255, 255, 255), 1, cv2.LINE_AA)

    cv2.addWeighted(major_layer, 0.10, bar, 0.90, 0.0, dst=bar)

    # ---- "glass" highlight band near the top
    highlight = np.zeros_like(bar, dtype=np.uint8)
    y0 = int(h * 0.001)
    y1 = int(h * 1)
    if y1 > y0:
        for y in range(y0, y1):
            a = 1.0 - (y - y0) / max(1, (y1 - y0))
            cv2.line(highlight, (0, y), (w - 1, y), (255, 255, 255), 1)
            # fade by scaling row
            highlight[y] = (highlight[y].astype(np.float32) * (0.25 * a)).astype(np.uint8)
        cv2.add(bar, highlight, dst=bar)


    # y2 = int(h * 0.75)
    # y3 = int(h * 1)
    # if y2 < y3:
    #     for y in range(y2, y3):
    #         a = 1.0 - (y - y2) / max(1, (y3 - y2))
    #         cv2.line(highlight, (0, y), (w - 1, y), (255, 255, 255), 1)
    #         # fade by scaling row
    #         highlight[y] = (highlight[y].astype(np.float32) * (0.25 * a)).astype(np.uint8)
    #     cv2.add(bar, highlight, dst=bar)

    # ---- panel border + inner edge line
    cv2.rectangle(bar, (0, 0), (w - 1, h - 1), (10, 10, 12), 1, cv2.LINE_AA)
    cv2.rectangle(bar, (1, 1), (w - 2, h - 2), (60, 60, 70), 1, cv2.LINE_AA)

    # ---- vignette for depth
    _draw_vignette(bar, strength=0.15)


def freq_to_y(freq_hz: float, h: int, f_display_max: float) -> int:
    """Map frequency (Hz) -> y pixel (0..h-1)."""
    h = int(h)
    if h <= 1:
        return 0
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    freq_hz = float(np.clip(freq_hz, 0.0, f_display_max))
    return int(h - 1 - (freq_hz / f_display_max) * (h - 1))


def y_to_freq(y: int, h: int, f_display_max: float) -> float:
    """Map y pixel (0..h-1) -> frequency (Hz)."""
    h = int(h)
    if h <= 1:
        return 0.0
    f_display_max = float(f_display_max) if float(f_display_max) > 0 else 1.0
    y = int(np.clip(y, 0, h - 1))
    frac = 1.0 - (y / (h - 1))
    return float(np.clip(frac * f_display_max, 0.0, f_display_max))


def draw_db_colorbar(
    frame: np.ndarray,
    db_min: float,
    db_max: float,
    width: int,
    state: Optional[DbSliderState] = None,
    colormap: str = "MAGMA",
) -> tuple[float, float]:
    """
    Draws the dB colorbar on the LEFT side of the frame,
    with an optional draggable slider + ruler.

    Returns (db_min_out, db_max_out) so caller can use updated scaling.
    """
    h = int(frame.shape[0])
    width = int(max(1, width))

    # ---- cached colormap gradient ----
    key = (h, width, colormap)
    bar_color = _DB_COLORBAR_CACHE.get(key)
    if bar_color is None:
        grad = np.linspace(0, 255, h, dtype=np.uint8)[::-1]   # top bright
        bar = np.repeat(grad[:, None], width, axis=1)
        colormap_cv = COLORMAP_DICT.get(colormap, cv2.COLORMAP_MAGMA)
        bar_color = cv2.applyColorMap(bar, colormap_cv)
        _DB_COLORBAR_CACHE[key] = bar_color

    frame[:, :width] = bar_color

    # ---- if no slider state, show top/bottom labels centered horizontally ----
    if state is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1

        # Top label (db_max)
        top_text = f"{float(db_max):.0f} dB"
        (top_w, top_h), _ = cv2.getTextSize(top_text, font, font_scale, font_thickness)
        top_x = (width - top_w) // 2  # Center horizontally
        top_y = 20
        #cv2.putText(frame, top_text, (top_x, top_y), font, font_scale, (255, 255, 255), font_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, top_text, (top_x, top_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        # Bottom label (db_min)
        bottom_text = f"{float(db_min):.0f} dB"
        (bottom_w, bottom_h), _ = cv2.getTextSize(bottom_text, font, font_scale, font_thickness)
        bottom_x = (width - bottom_w) // 2  # Center horizontally
        bottom_y = h - 10
        # cv2.putText(frame, bottom_text, (bottom_x, bottom_y), font, font_scale, (0, 0, 0), font_thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, bottom_text, (bottom_x, bottom_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

        return float(db_min), float(db_max)

    # ---- slider-controlled scaling ----
    # If disabled, keep inputs; if enabled, use state.db_max/span_db
    if state.enabled:
        db_max = float(state.db_max)
        db_min = float(db_max - state.span_db)

    # ---- ruler ticks flush with left edge ----
    # Major ticks every 10 dB, minor every 5 dB
    tick_col = (220, 220, 220) if state.enabled else (120, 120, 120)
    minor_col = (160, 160, 160) if state.enabled else (90, 90, 90)

    # Choose tick range around current min/max
    d0 = int(np.floor(db_min / 5.0) * 5)
    d1 = int(np.ceil(db_max / 5.0) * 5)
    for d in range(d0, d1 + 1, 5):
        y = db_to_y(d, h, db_min, db_max)
        if d % 10 == 0:
            x0, x1 = 0, min(width - 1, 12)
            cv2.line(frame, (x0, y), (x1, y), tick_col, 1, cv2.LINE_8)
            # small label inside bar
            cv2.putText(frame, f"{d:>3}", (min(width - 24, 2), max(10, y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, tick_col, 1, cv2.LINE_8)
        else:
            x0, x1 = 0, min(width - 1, 7)
            cv2.line(frame, (x0, y), (x1, y), minor_col, 1, cv2.LINE_8)

    # ---- knob ----
    knob_y = db_to_y(db_max, h, db_min, db_max)
    knob_h = 26
    knob_w = min(width - 2, 30)
    knob_x0 = (width - knob_w) // 2
    knob_y0 = int(np.clip(knob_y - knob_h // 2, 0, h - knob_h))

    # Greyed vs solid look
    if state.enabled:
        knob_fill = (235, 235, 235)
        knob_edge = (30, 30, 30)
    else:
        knob_fill = (140, 140, 140)
        knob_edge = (60, 60, 60)

    cv2.rectangle(frame, (knob_x0, knob_y0), (knob_x0 + knob_w, knob_y0 + knob_h),
                  knob_fill, -1, cv2.LINE_8)
    cv2.rectangle(frame, (knob_x0, knob_y0), (knob_x0 + knob_w, knob_y0 + knob_h),
                  knob_edge, 1, cv2.LINE_8)

    # little center groove
    gx = knob_x0 + knob_w // 2
    cv2.line(frame, (gx, knob_y0 + 5), (gx, knob_y0 + knob_h - 5),
             (90, 90, 90), 2, cv2.LINE_8)

    # ---- top/bottom labels ----
    cv2.putText(frame, f"{float(db_max):.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_8)
    cv2.putText(frame, f"{float(db_min):.0f} dB", (0, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_8)

    return float(db_min), float(db_max)


def handle_db_bar_mouse(
    event: int,
    x: int,
    y: int,
    state: DbSliderState,
    bar_width: int,
    frame_h: int,
) -> None:
    # only react if inside left db bar area
    if x < 0 or x >= bar_width or y < 0 or y >= frame_h:
        if event == cv2.EVENT_LBUTTONUP:
            state.dragging = False
        return

    now = time.time()

    if event == cv2.EVENT_LBUTTONDOWN:
        # double click detection (time + proximity)
        dt = now - state._last_click_t
        if dt < 0.35 and abs(y - state._last_click_y) < 20:
            state.enabled = not state.enabled
            state.dragging = False
        else:
            # single click enables ("solidifies") and starts drag
            state.enabled = True
            state.dragging = True

        state._last_click_t = now
        state._last_click_y = y

        # update db_max immediately on click
        db_max = y_to_db(y, frame_h, state.db_max - state.span_db, state.db_max)
        state.db_max = float(db_max)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.enabled and state.dragging:
            db_max = y_to_db(y, frame_h, state.db_max - state.span_db, state.db_max)
            state.db_max = float(db_max)

    elif event == cv2.EVENT_LBUTTONUP:
        state.dragging = False








"""
Shared UI caches (gradients, recording HUD, thumbnails, modal overlays) and config fallbacks.
"""

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

DRAG_PX = 4  # tweak (6-12 feels good)

# Gallery viewer: dock height is in viewer_dock.VIEWER_DOCK_HEIGHT (105)
VIEWER_SWIPE_THRESHOLD_PX = 60   # min drag to consider swipe
VIEWER_SWIPE_VELOCITY_THRESHOLD = 200.0  # px/s for fling
VIEWER_SWIPE_FLING_GAIN = 1.5
VIEWER_SWIPE_FRICTION = 3.5   # higher = quicker return to rest (less step-back delay)
VIEWER_SWIPE_STOP_VELOCITY = 25.0

# Config fallbacks
try:
    from ..config import (
        WIDTH,
        HEIGHT,
        DB_BAR_WIDTH,
        BUTTON_HITPAD_PX,
        USE_CAMERA,
    )
except Exception:
    WIDTH = 1024
    HEIGHT = 600
    DB_BAR_WIDTH = 50
    BUTTON_HITPAD_PX = 14
    USE_CAMERA = True

_GRAD_CACHE: Dict[Tuple[int, int, int, int, int], np.ndarray] = {}
_REC_HUD_CACHE: Dict[Tuple[int, int, bool], np.ndarray] = {}
_REC_TEXT_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}
_DATE_TEXT_SIZE_CACHE: Dict[str, Tuple[int, int]] = {}  # date_str -> (w, h) for gallery labels
_THUMB_CACHE: Dict[Path, np.ndarray] = {}
_THUMB_CACHE_MTIME: Dict[Path, float] = {}

# Modal dim overlay: black buffer per (fw, fh) to avoid frame.copy() every frame
_MODAL_OVERLAY_BLACK: Dict[Tuple[int, int], np.ndarray] = {}


def get_modal_overlay_black(fw: int, fh: int) -> np.ndarray:
    """Return cached fullscreen black buffer (BGR) for modal dim. Avoids copying frame."""
    key = (fw, fh)
    buf = _MODAL_OVERLAY_BLACK.get(key)
    if buf is not None:
        return buf
    buf = np.zeros((fh, fw, 3), dtype=np.uint8)
    buf[:] = (0, 0, 0)
    _MODAL_OVERLAY_BLACK[key] = buf
    return buf


def apply_modal_dim(frame: np.ndarray, alpha: float = 0.5) -> None:
    """Dim frame in-place using cached black overlay (no frame copy). alpha=0.5 => half brightness."""
    fh, fw = frame.shape[:2]
    black = get_modal_overlay_black(fw, fh)
    cv2.addWeighted(black, alpha, frame, 1.0 - alpha, 0, frame)


def get_grad(w: int, h: int, color: Tuple[int, int, int]) -> np.ndarray:
    """Return cached or computed vertical gradient (BGR)."""
    key = (w, h, int(color[0]), int(color[1]), int(color[2]))
    g = _GRAD_CACHE.get(key)
    if g is not None:
        return g

    top = np.clip(np.array(color, np.float32) * 1.15, 0, 255)
    bot = np.clip(np.array(color, np.float32) * 0.85, 0, 255)

    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    grad = (top[None, None, :] * (1.0 - t) + bot[None, None, :] * t).astype(np.uint8)
    grad = np.repeat(grad, w, axis=1)

    _GRAD_CACHE[key] = grad
    return grad

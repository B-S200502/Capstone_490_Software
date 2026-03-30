"""
Battery (charge) icon displayed in the UI.

Reads battery voltage from firmware header (battery_mv); converts to percent for display.
Position varies by view:
- Main heatmap/camera: top-left
- Gallery grid: under STORAGE section in side dock
- Single media viewer (image/video): top-right
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .. import config

# Voltage → percent: 5.8V = 0%, 8.4V = 100%; cap at 100% if >= 8.3V
BATTERY_V_MV_MIN = 5800   # 5.8V = 0%
BATTERY_V_MV_MAX = 8400   # 8.4V = 100%
BATTERY_V_MV_CAP = 8300   # >= 8.3V show 100%

# Session state: when ~5V USB is present the ADC does not see pack voltage
_last_good_soc: Optional[int] = None
_last_trusted_mv: Optional[int] = None
_usb_infer_start_s: Optional[float] = None
_usb_infer_anchor_soc: int = 0
# From battery_snapshot.json: wall time at last shutdown; applied as infer start on first USB-band read
_persisted_infer_start_s: Optional[float] = None
_last_display_soc: Optional[int] = None
_last_battery_snapshot_write_s: float = 0.0
_last_snapshot_reported_mv: Optional[int] = None

_SNAPSHOT_VERSION = 2


def _cfg(name: str, default: float | int) -> float | int:
    return getattr(config, name, default)


def reset_battery_charge_inference_state() -> None:
    """Clear USB charge inference (e.g. after tests)."""
    global _last_good_soc, _last_trusted_mv, _usb_infer_start_s, _usb_infer_anchor_soc
    global _persisted_infer_start_s, _last_display_soc
    global _last_battery_snapshot_write_s, _last_snapshot_reported_mv
    _last_good_soc = None
    _last_trusted_mv = None
    _usb_infer_start_s = None
    _usb_infer_anchor_soc = 0
    _persisted_infer_start_s = None
    _last_display_soc = None
    _last_battery_snapshot_write_s = 0.0
    _last_snapshot_reported_mv = None


def _display_floor_pct() -> int:
    d = int(_cfg("BATTERY_DISPLAY_MIN_WHEN_UNCERTAIN_PCT", 25))
    a = int(_cfg("BATTERY_CHARGE_INFER_ANCHOR_DEFAULT", 25))
    return max(0, min(100, d if d > 0 else a))


def load_persisted_battery_state(path: Optional[Path]) -> None:
    """Restore anchor SOC, mV fields, infer start time, and last display from JSON."""
    global _last_good_soc, _last_trusted_mv, _persisted_infer_start_s, _last_display_soc
    global _last_snapshot_reported_mv
    if path is None:
        return
    try:
        p = Path(path)
        if not p.is_file():
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        ver = int(data.get("version", 0))
        if ver not in (1, _SNAPSHOT_VERSION):
            return
        anchor = data.get("anchor_soc")
        if anchor is None:
            return
        lt = int(data.get("last_trusted_mv", 0) or 0)
        ag = max(0, min(100, int(anchor)))
        if ag == 0 and lt == 0:
            ag = _display_floor_pct()
        _last_good_soc = ag
        _last_trusted_mv = lt if lt > 0 else None
        sa = data.get("saved_at")
        if sa is not None:
            _persisted_infer_start_s = float(sa)
        ld = data.get("last_display_soc")
        pm = int(_cfg("BATTERY_PACK_READ_MIN_MV", BATTERY_V_MV_MIN))
        lrmv_file = int(data.get("last_reported_mv", 0) or 0)
        if ld is not None:
            ldi = max(0, min(100, int(ld)))
            if ldi == 0 and not (lrmv_file >= pm):
                ldi = _display_floor_pct()
            _last_display_soc = ldi
        elif _last_display_soc is None:
            _last_display_soc = _last_good_soc
        if lrmv_file > 0:
            _last_snapshot_reported_mv = lrmv_file
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass


def _write_battery_snapshot_file(path: Optional[Path]) -> None:
    """Write battery_snapshot.json with unix saved_at (always set on each write)."""
    if path is None:
        return
    floor = _display_floor_pct()
    anchor = _last_good_soc if _last_good_soc is not None else _last_display_soc
    if anchor is None:
        anchor = floor
    anchor = max(0, min(100, int(anchor)))
    disp = _last_display_soc if _last_display_soc is not None else anchor
    disp = max(0, min(100, int(disp)))
    mv_trust = int(_last_trusted_mv) if _last_trusted_mv is not None else 0
    mv_rep = int(_last_snapshot_reported_mv) if _last_snapshot_reported_mv is not None else 0
    payload = {
        "version": _SNAPSHOT_VERSION,
        "saved_at": time.time(),
        "anchor_soc": anchor,
        "last_trusted_mv": mv_trust,
        "last_reported_mv": mv_rep,
        "last_display_soc": disp,
    }
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.write("\n")
        tmp.replace(p)
    except OSError:
        pass


def maybe_persist_battery_snapshot(path: Optional[Path], mv: Optional[int]) -> None:
    """Throttled disk write while the app runs (first call writes immediately)."""
    global _last_battery_snapshot_write_s, _last_snapshot_reported_mv
    if path is None:
        return
    if mv is not None and mv > 0:
        _last_snapshot_reported_mv = mv
    interval = float(_cfg("BATTERY_SNAPSHOT_INTERVAL_S", 45.0))
    t = time.time()
    if _last_battery_snapshot_write_s > 0.0 and (t - _last_battery_snapshot_write_s) < interval:
        return
    _last_battery_snapshot_write_s = t
    _write_battery_snapshot_file(path)


def save_persisted_battery_state(path: Optional[Path]) -> None:
    """Write snapshot immediately (shutdown). Always updates saved_at."""
    global _last_battery_snapshot_write_s
    _last_battery_snapshot_write_s = time.time()
    _write_battery_snapshot_file(path)


def _coerce_uncertain_zero(out: Optional[int], mv: Optional[int]) -> Optional[int]:
    """Keep real 0% only for trustworthy pack voltage at empty; else use display floor."""
    if out is None:
        return None
    if out != 0:
        return max(0, min(100, out))
    if mv is not None and mv > 0 and _pack_voltage_trustworthy(mv):
        return 0
    return _display_floor_pct()


def _finalize_display(soc: Optional[int], mv: Optional[int]) -> Optional[int]:
    global _last_display_soc, _last_snapshot_reported_mv
    c = _coerce_uncertain_zero(soc, mv)
    if c is not None:
        _last_display_soc = c
    if mv is not None and mv > 0:
        _last_snapshot_reported_mv = mv
    snap_path = getattr(config, "BATTERY_SNAPSHOT_PATH", None)
    if snap_path is not None:
        maybe_persist_battery_snapshot(snap_path, mv)
    return c


def _in_usb_rail_band(mv: int) -> bool:
    lo = int(_cfg("BATTERY_USB_MV_LOW", 4600))
    hi = int(_cfg("BATTERY_USB_MV_HIGH", 5320))
    return lo <= mv <= hi


def _hold_uncertain_display(anchor_default: int) -> int:
    """Prefer last UI % (e.g. from JSON), then anchor SOC, then default — avoid bogus 0."""
    if _last_display_soc is not None:
        return max(0, min(100, int(_last_display_soc)))
    if _last_good_soc is not None:
        return max(0, min(100, int(_last_good_soc)))
    return max(0, min(100, int(anchor_default)))


def _pack_voltage_trustworthy(mv: int) -> bool:
    return mv >= int(_cfg("BATTERY_PACK_READ_MIN_MV", BATTERY_V_MV_MIN))


def _charge_soc_delta(elapsed_s: float) -> float:
    """Estimated % points added over elapsed_s at constant charge current (crude CC; taper above 85%)."""
    cap_ah = float(_cfg("BATTERY_PACK_AH", 7.2))
    i_a = float(_cfg("BATTERY_CHARGE_CURRENT_A", 1.8))
    if cap_ah <= 0 or elapsed_s <= 0:
        return 0.0
    dt_h = elapsed_s / 3600.0
    linear = (i_a / cap_ah) * dt_h * 100.0
    return linear


def resolve_battery_display_percent(mv: Optional[int], now_s: Optional[float] = None) -> Optional[int]:
    """
    Map firmware battery_mv to a display percent.

    When the pack is on USB 5V, the reported voltage often sits in a ~5V band and the
    linear 5.8–8.4V curve would show 0%. In that band we infer SOC from last known good
    % plus charge time at BATTERY_CHARGE_CURRENT_A into BATTERY_PACK_AH.

    When voltage is at or above BATTERY_PACK_READ_MIN_MV, we use the normal voltage curve
    and refresh last-known good SOC.

    Between BATTERY_USB_MV_HIGH and BATTERY_PACK_READ_MIN_MV we hold last known % (or
    default anchor) instead of mapping to 0%.

    Below BATTERY_USB_MV_LOW (but mv > 0) we also hold — same as stale JSON case where
    the ADC reports e.g. ~4.5 V and would otherwise hit 0% then the display floor.
    """
    global _last_good_soc, _last_trusted_mv, _usb_infer_start_s, _usb_infer_anchor_soc
    global _persisted_infer_start_s

    if now_s is None:
        now_s = time.time()

    if mv is None:
        return _finalize_display(_last_good_soc, mv)

    if mv <= 0:
        return _finalize_display(_last_good_soc, mv)

    if _pack_voltage_trustworthy(mv):
        pct = battery_mv_to_percent(mv)
        if pct is not None:
            _last_good_soc = pct
        _last_trusted_mv = mv
        _usb_infer_start_s = None
        _persisted_infer_start_s = None
        return _finalize_display(pct, mv)

    infer_cap = int(_cfg("BATTERY_CHARGE_INFER_CAP_PCT", 95))
    anchor_default = int(_cfg("BATTERY_CHARGE_INFER_ANCHOR_DEFAULT", 25))

    if _in_usb_rail_band(mv):
        if _usb_infer_start_s is None:
            if _persisted_infer_start_s is not None:
                _usb_infer_start_s = float(_persisted_infer_start_s)
                _persisted_infer_start_s = None
            else:
                _usb_infer_start_s = float(now_s)
            _usb_infer_anchor_soc = (
                _last_good_soc if _last_good_soc is not None else anchor_default
            )
            _usb_infer_anchor_soc = max(0, min(100, _usb_infer_anchor_soc))
        elapsed = max(0.0, float(now_s) - float(_usb_infer_start_s))
        raw_add = _charge_soc_delta(elapsed)
        # Taper growth above 85% (CV phase unknown)
        anchor = _usb_infer_anchor_soc
        projected = anchor + raw_add
        if projected > 85.0:
            over = projected - 85.0
            projected = 85.0 + over * 0.28
        out = int(round(projected))
        out = max(anchor, out)
        out = min(min(infer_cap, 100), out)
        return _finalize_display(out, mv)

    # Between USB band and trustworthy pack: hysteresis — keep inferring if session active
    if _usb_infer_start_s is not None:
        elapsed = max(0.0, float(now_s) - float(_usb_infer_start_s))
        raw_add = _charge_soc_delta(elapsed)
        anchor = _usb_infer_anchor_soc
        projected = anchor + raw_add
        if projected > 85.0:
            over = projected - 85.0
            projected = 85.0 + over * 0.28
        out = int(round(projected))
        out = max(anchor, out)
        out = min(min(infer_cap, 100), out)
        return _finalize_display(out, mv)

    # Between USB rail ceiling and trustworthy pack: not on the 5.8–8.4V curve, but
    # battery_mv_to_percent() treats mv <= 5.8V as 0% — bogus "empty" for this gap.
    pack_min = int(_cfg("BATTERY_PACK_READ_MIN_MV", BATTERY_V_MV_MIN))
    usb_hi = int(_cfg("BATTERY_USB_MV_HIGH", 5320))
    usb_lo = int(_cfg("BATTERY_USB_MV_LOW", 4600))
    if usb_hi < mv < pack_min:
        hold = _hold_uncertain_display(anchor_default)
        return _finalize_display(hold, mv)

    # Below USB low threshold: offset/noisy rail (e.g. ~4.5 V); do not use linear curve → 0%.
    if 0 < mv < usb_lo:
        _usb_infer_start_s = None
        hold = _hold_uncertain_display(anchor_default)
        return _finalize_display(hold, mv)

    _usb_infer_start_s = None
    pct = battery_mv_to_percent(mv)
    if pct is not None and mv >= BATTERY_V_MV_MIN:
        _last_good_soc = pct
    return _finalize_display(pct, mv)


def battery_mv_to_percent(mv: Optional[int]) -> Optional[int]:
    """Convert firmware battery voltage (mV) to 0–100%. Returns None if mv is None."""
    if mv is None:
        return None
    if mv >= BATTERY_V_MV_CAP:
        return 100
    if mv <= BATTERY_V_MV_MIN:
        return 0
    return int(round((mv - BATTERY_V_MV_MIN) / (BATTERY_V_MV_MAX - BATTERY_V_MV_MIN) * 100))



# Icon dimensions (body + tip)
BATTERY_BODY_W = 28
BATTERY_BODY_H = 14
BATTERY_TIP_W = 4
BATTERY_TIP_H = 6
BATTERY_PAD = 2  # inner padding for fill
BORDER_COLOR = (255, 255, 255)
BORDER_THICKNESS = 1

# Fill colors (BGR) by charge level
FILL_HIGH = (0, 200, 80)   # green
FILL_MED = (0, 200, 255)   # yellow
FILL_LOW = (0, 80, 255)    # red


def _fill_color(percent: int) -> tuple:
    """Return BGR color for given charge percentage."""
    if percent > 50:
        return FILL_HIGH
    if percent > 20:
        return FILL_MED
    return FILL_LOW


def _battery_position_for_view(frame: np.ndarray) -> Tuple[int, int]:
    """Return (x, y) top-left for battery based on current view."""
    from ..state import button_state

    h, w = frame.shape[:2]
    bw = BATTERY_BODY_W + BATTERY_TIP_W
    bh = BATTERY_BODY_H
    pad = 12

    if not button_state.gallery_open:
        # Main heatmap: top-left of camera feed segment (right of dB bar)
        from ..config import DB_BAR_WIDTH
        return (DB_BAR_WIDTH + pad, pad)

    if button_state.gallery_viewer_mode in ("image", "video"):
        return (w - pad - bw, pad)  # Single media viewer: top-right

    # Gallery grid: under STORAGE in side dock
    GRID_SIDE_DOCK_WIDTH = 113
    dock_x = w - GRID_SIDE_DOCK_WIDTH
    dock_w = GRID_SIDE_DOCK_WIDTH
    # Center horizontally in dock; place at bottom below storage circle + Free/Used text
    bx = dock_x + (dock_w - bw) // 2
    by = h - 10 - bh  # Just above bottom edge, below storage section
    return (bx, by)


def draw_battery_icon(
    frame: np.ndarray,
    x: int = 12,
    y: int = 12,
    percent: Optional[int] = None,
) -> None:
    """
    Draw a battery/charge icon at the given position.

    Args:
        frame: BGR image to draw on (modified in place).
        x, y: Top-left position of the icon.
        percent: Charge 0-100. None = placeholder (shows 100% for now).
    """
    if percent is None:
        percent = 100  # placeholder until live data is connected

    percent = max(0, min(100, percent))
    h, w = frame.shape[:2]
    if x < 0 or y < 0 or x + BATTERY_BODY_W + BATTERY_TIP_W > w or y + BATTERY_BODY_H > h:
        return

    # Battery body outline (rectangle)
    body_x1, body_y1 = x, y
    body_x2 = x + BATTERY_BODY_W
    body_y2 = y + BATTERY_BODY_H
    cv2.rectangle(
        frame, (body_x1, body_y1), (body_x2, body_y2),
        BORDER_COLOR, BORDER_THICKNESS, cv2.LINE_AA
    )

    # Battery tip (positive terminal) on the right
    tip_x1 = body_x2
    tip_y1 = y + (BATTERY_BODY_H - BATTERY_TIP_H) // 2
    tip_x2 = tip_x1 + BATTERY_TIP_W
    tip_y2 = tip_y1 + BATTERY_TIP_H
    cv2.rectangle(
        frame, (tip_x1, tip_y1), (tip_x2, tip_y2),
        BORDER_COLOR, BORDER_THICKNESS, cv2.LINE_AA
    )

    # Fill level (inner rectangle)
    fill_w = max(0, int((BATTERY_BODY_W - 2 * BATTERY_PAD) * percent / 100))
    if fill_w > 0:
        fill_x1 = body_x1 + BATTERY_PAD
        fill_y1 = body_y1 + BATTERY_PAD
        fill_x2 = fill_x1 + fill_w
        fill_y2 = body_y2 - BATTERY_PAD
        cv2.rectangle(
            frame, (fill_x1, fill_y1), (fill_x2, fill_y2),
            _fill_color(percent), -1, cv2.LINE_AA
        )


def draw_battery_icon_for_view(
    frame: np.ndarray,
    percent: Optional[int] = None,
) -> None:
    """
    Draw battery icon at the appropriate position for the current view:
    - Main heatmap: top-left
    - Gallery grid: under STORAGE in side dock
    - Single media viewer: top-right
    """
    x, y = _battery_position_for_view(frame)
    draw_battery_icon(frame, x=x, y=y, percent=percent)

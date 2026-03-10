"""
Circular GPS map + acoustic radar widget.

- North-up map tiles (OSM) clipped to a circular bezel.
- Center arrow rotates with device heading.
- Acoustic detections are rendered as fading dots, color-coded by dB intensity.
"""

from __future__ import annotations

import math
import queue
import threading
import time
import urllib.request
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np

from .. import config
from ..dsp.bars import COLORMAP_DICT
from ..state import HUD, RadarDetection

_TILE_MEM_CACHE: Dict[Tuple[str, int, int, int], np.ndarray] = {}
_TILE_LAST_ATTEMPT: Dict[Tuple[str, int, int, int], float] = {}
_TILE_INFLIGHT: set[Tuple[str, int, int, int]] = set()
_TILE_LOCK = threading.Lock()
_TILE_FETCH_QUEUE: "queue.Queue[Tuple[str, int, int, int]]" = queue.Queue(maxsize=256)
_TILE_WORKERS_STARTED = False
_COLORMAP_LUT_CACHE: Dict[str, np.ndarray] = {}
_MAP_PATCH_CACHE: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
_CIRCLE_MASK_CACHE: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
_WIDGET_CACHE: Dict[str, Any] = {"img": None, "updated_s": 0.0, "d": 0, "colormap": "", "tile_style": "dark"}
_LAST_HISTORY_PRUNE_S = 0.0
_MAP_LAST_STATUS = "INIT"
_RADAR_DEBUG_LAST_LOG_S = 0.0

# region agent log
_AGENT_DEBUG_LOG_PATH = "/home/acousticlord/Capstone_490_Software/.cursor/debug-a9e491.log"
_AGENT_DEBUG_SESSION = "a9e491"


def _agent_debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    try:
        payload = {
            "sessionId": _AGENT_DEBUG_SESSION,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with open(_AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(payload, separators=(",", ":")) + "\n")
    except Exception:
        pass
# endregion


def _latlon_to_world_px(lat: float, lon: float, zoom: int, tile_size: int) -> Tuple[float, float]:
    lat = max(-85.05112878, min(85.05112878, float(lat)))
    lon = float(lon)
    scale = (2 ** int(zoom)) * int(tile_size)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) * 0.5 * scale
    return x, y


def _tile_cache_path(style: str, z: int, x: int, y: int) -> Path:
    cache_dir = Path(getattr(config, "RADAR_MAP_CACHE_DIR", "data/map_tiles"))
    return cache_dir / style / str(z) / str(x) / f"{y}.png"


def _tile_worker_loop() -> None:
    while True:
        key = _TILE_FETCH_QUEUE.get()
        try:
            style, z, x, y = key
            _fetch_tile_worker(style, z, x, y)
        finally:
            _TILE_FETCH_QUEUE.task_done()


def _ensure_tile_workers() -> None:
    global _TILE_WORKERS_STARTED
    if _TILE_WORKERS_STARTED:
        return
    with _TILE_LOCK:
        if _TILE_WORKERS_STARTED:
            return
        n_workers = max(1, int(getattr(config, "RADAR_MAP_FETCH_WORKERS", 2)))
        for _ in range(n_workers):
            th = threading.Thread(target=_tile_worker_loop, daemon=True)
            th.start()
        _TILE_WORKERS_STARTED = True


def _fetch_tile_worker(style: str, z: int, x: int, y: int) -> None:
    key = (style, z, x, y)
    try:
        if style == "light":
            tile_url = getattr(config, "RADAR_MAP_TILE_URL_LIGHT", "https://tile.openstreetmap.org/{z}/{x}/{y}.png")
        else:
            tile_url = getattr(config, "RADAR_MAP_TILE_URL_DARK", "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png")
        url = tile_url.format(
            z=z, x=x, y=y
        )
        req = urllib.request.Request(
            url, headers={"User-Agent": getattr(config, "RADAR_MAP_USER_AGENT", "acoustic-imager-radar/1.0")}
        )
        timeout = float(getattr(config, "RADAR_MAP_FETCH_TIMEOUT_SEC", 1.5))
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return
        p = _tile_cache_path(style, z, x, y)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), img)
        with _TILE_LOCK:
            _TILE_MEM_CACHE[key] = img
    except Exception:
        pass
    finally:
        with _TILE_LOCK:
            _TILE_INFLIGHT.discard(key)
            _TILE_LAST_ATTEMPT[key] = time.time()


def _get_tile(style: str, z: int, x: int, y: int) -> Optional[np.ndarray]:
    _ensure_tile_workers()
    key = (style, z, x, y)
    with _TILE_LOCK:
        img = _TILE_MEM_CACHE.get(key)
        if img is not None:
            return img
    p = _tile_cache_path(style, z, x, y)
    if p.exists():
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            with _TILE_LOCK:
                _TILE_MEM_CACHE[key] = img
            return img

    retry_s = float(getattr(config, "RADAR_MAP_RETRY_SEC", 3.0))
    now = time.time()
    with _TILE_LOCK:
        last = _TILE_LAST_ATTEMPT.get(key, 0.0)
        if key not in _TILE_INFLIGHT and (now - last) >= retry_s:
            _TILE_INFLIGHT.add(key)
            try:
                _TILE_FETCH_QUEUE.put_nowait(key)
            except queue.Full:
                _TILE_INFLIGHT.discard(key)
                _TILE_LAST_ATTEMPT[key] = now
    return None


def _draw_fallback_map(canvas: np.ndarray) -> None:
    h, w = canvas.shape[:2]
    canvas[:] = (42, 48, 52)
    for y in range(0, h, 18):
        cv2.line(canvas, (0, y), (w - 1, y), (60, 68, 72), 1, cv2.LINE_AA)
    for x in range(0, w, 18):
        cv2.line(canvas, (x, 0), (x, h - 1), (60, 68, 72), 1, cv2.LINE_AA)


def _apply_tile_brightness(img: np.ndarray, gain: float) -> np.ndarray:
    g = float(gain)
    if abs(g - 1.0) < 1e-3:
        return img
    out = np.clip(img.astype(np.float32) * g, 0.0, 255.0)
    return out.astype(np.uint8)


def _render_map_patch(diameter_px: int, tile_style: str) -> np.ndarray:
    global _MAP_LAST_STATUS
    d = int(max(32, diameter_px))
    bin_px = max(1, int(getattr(config, "RADAR_MAP_POS_BIN_PX", 3)))
    fallback_brightness = float(getattr(config, "RADAR_MAP_FALLBACK_BRIGHTNESS", 0.65))
    fallback_brightness_q = int(round(fallback_brightness * 100.0))
    style = str(tile_style).lower()
    if style not in ("dark", "light"):
        style = "dark"
    if style == "light":
        brightness = float(getattr(config, "RADAR_MAP_BRIGHTNESS_LIGHT", 1.0))
    else:
        brightness = float(getattr(config, "RADAR_MAP_BRIGHTNESS_DARK", 1.0))
    brightness_q = int(round(brightness * 100.0))
    pos_lat = HUD.position.lat
    pos_lon = HUD.position.lon
    if pos_lat is not None and pos_lon is not None and HUD.position.source != "none":
        tile_size = int(getattr(config, "RADAR_MAP_TILE_SIZE", 256))
        zoom = int(getattr(config, "RADAR_MAP_ZOOM", 17))
        world_x, world_y = _latlon_to_world_px(pos_lat, pos_lon, zoom, tile_size)
        cache_key: Tuple[Any, ...] = (
            "map",
            d,
            zoom,
            tile_size,
            style,
            brightness_q,
            int(world_x // bin_px),
            int(world_y // bin_px),
        )
    else:
        cache_key = ("fallback", d, style, fallback_brightness_q)
    cached = _MAP_PATCH_CACHE.get(cache_key)
    if cached is not None:
        _MAP_PATCH_CACHE.move_to_end(cache_key)
        return cached.copy()

    out = np.zeros((d, d, 3), dtype=np.uint8)
    if pos_lat is None or pos_lon is None or HUD.position.source == "none":
        _draw_fallback_map(out)
        out = _apply_tile_brightness(out, fallback_brightness)
        _MAP_LAST_STATUS = "NO_POS"
        _MAP_PATCH_CACHE[cache_key] = out.copy()
        max_n = max(1, int(getattr(config, "RADAR_MAP_PATCH_CACHE_SIZE", 12)))
        while len(_MAP_PATCH_CACHE) > max_n:
            _MAP_PATCH_CACHE.popitem(last=False)
        return out

    tile_size = int(getattr(config, "RADAR_MAP_TILE_SIZE", 256))
    zoom = int(getattr(config, "RADAR_MAP_ZOOM", 17))
    world_x, world_y = _latlon_to_world_px(pos_lat, pos_lon, zoom, tile_size)
    cx_tile = int(world_x // tile_size)
    cy_tile = int(world_y // tile_size)
    ox = int(world_x - cx_tile * tile_size)
    oy = int(world_y - cy_tile * tile_size)

    mosaic = np.zeros((tile_size * 3, tile_size * 3, 3), dtype=np.uint8)
    got_any = False
    n_tiles = 2 ** zoom
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            tx = (cx_tile + dx) % n_tiles
            ty = cy_tile + dy
            if ty < 0 or ty >= n_tiles:
                continue
            tile = _get_tile(style, zoom, tx, ty)
            if tile is None:
                continue
            got_any = True
            y0 = (dy + 1) * tile_size
            x0 = (dx + 1) * tile_size
            mosaic[y0:y0 + tile_size, x0:x0 + tile_size] = tile

    if not got_any:
        _draw_fallback_map(out)
        out = _apply_tile_brightness(out, fallback_brightness)
        _MAP_LAST_STATUS = "NO_TILES"
        _MAP_PATCH_CACHE[cache_key] = out.copy()
        max_n = max(1, int(getattr(config, "RADAR_MAP_PATCH_CACHE_SIZE", 12)))
        while len(_MAP_PATCH_CACHE) > max_n:
            _MAP_PATCH_CACHE.popitem(last=False)
        return out

    center_x = tile_size + ox
    center_y = tile_size + oy
    half = d // 2
    x0 = center_x - half
    y0 = center_y - half
    x1 = x0 + d
    y1 = y0 + d
    x0c = max(0, x0)
    y0c = max(0, y0)
    x1c = min(mosaic.shape[1], x1)
    y1c = min(mosaic.shape[0], y1)
    if x1c <= x0c or y1c <= y0c:
        _draw_fallback_map(out)
        out = _apply_tile_brightness(out, fallback_brightness)
        _MAP_LAST_STATUS = "CLIP_FALLBACK"
        _MAP_PATCH_CACHE[cache_key] = out.copy()
        max_n = max(1, int(getattr(config, "RADAR_MAP_PATCH_CACHE_SIZE", 12)))
        while len(_MAP_PATCH_CACHE) > max_n:
            _MAP_PATCH_CACHE.popitem(last=False)
        return out
    crop = np.zeros_like(out)
    dx0 = x0c - x0
    dy0 = y0c - y0
    crop[dy0:dy0 + (y1c - y0c), dx0:dx0 + (x1c - x0c)] = mosaic[y0c:y1c, x0c:x1c]
    crop = _apply_tile_brightness(crop, brightness)
    _MAP_LAST_STATUS = "TILES_OK"
    _MAP_PATCH_CACHE[cache_key] = crop.copy()
    max_n = max(1, int(getattr(config, "RADAR_MAP_PATCH_CACHE_SIZE", 12)))
    while len(_MAP_PATCH_CACHE) > max_n:
        _MAP_PATCH_CACHE.popitem(last=False)
    return crop


def _colormap_lut(colormap_name: str) -> np.ndarray:
    lut = _COLORMAP_LUT_CACHE.get(colormap_name)
    if lut is not None:
        return lut
    cm = COLORMAP_DICT.get(colormap_name, cv2.COLORMAP_MAGMA)
    grad = np.arange(256, dtype=np.uint8).reshape(256, 1)
    lut_img = cv2.applyColorMap(grad, cm)[:, 0, :]
    _COLORMAP_LUT_CACHE[colormap_name] = lut_img
    return lut_img


def update_detection_history(rel_angle_deg: float, db_value: float, heading_deg: float, now_s: Optional[float] = None) -> None:
    global _LAST_HISTORY_PRUNE_S
    if now_s is None:
        now_s = time.time()
    history_sec = float(getattr(config, "RADAR_HISTORY_SEC", 6.0))
    max_points = int(getattr(config, "RADAR_MAX_POINTS", 180))
    world_bearing = (float(heading_deg) + float(rel_angle_deg)) % 360.0
    HUD.radar_detections.append(
        RadarDetection(
            t=float(now_s),
            rel_angle_deg=float(rel_angle_deg),
            world_bearing_deg=float(world_bearing),
            db_value=float(db_value),
        )
    )
    need_prune = (
        (float(now_s) - _LAST_HISTORY_PRUNE_S) >= 0.30
        or len(HUD.radar_detections) > int(max_points * 1.20)
    )
    if need_prune:
        cutoff = float(now_s) - history_sec
        HUD.radar_detections = [d for d in HUD.radar_detections if d.t >= cutoff]
        if len(HUD.radar_detections) > max_points:
            HUD.radar_detections = HUD.radar_detections[-max_points:]
        _LAST_HISTORY_PRUNE_S = float(now_s)


def _prune_history(now_s: float) -> None:
    global _LAST_HISTORY_PRUNE_S
    history_sec = float(getattr(config, "RADAR_HISTORY_SEC", 6.0))
    max_points = int(getattr(config, "RADAR_MAX_POINTS", 180))
    if (float(now_s) - _LAST_HISTORY_PRUNE_S) < 0.30:
        return
    cutoff = float(now_s) - history_sec
    HUD.radar_detections = [d for d in HUD.radar_detections if d.t >= cutoff]
    if len(HUD.radar_detections) > max_points:
        HUD.radar_detections = HUD.radar_detections[-max_points:]
    _LAST_HISTORY_PRUNE_S = float(now_s)


def _get_circle_masks(d: int) -> Tuple[np.ndarray, np.ndarray]:
    cached = _CIRCLE_MASK_CACHE.get(d)
    if cached is not None:
        return cached
    r = d // 2
    mask = np.zeros((d, d), dtype=np.uint8)
    cv2.circle(mask, (r, r), r - 1, 255, -1, cv2.LINE_AA)
    inv = cv2.bitwise_not(mask)
    _CIRCLE_MASK_CACHE[d] = (mask, inv)
    return mask, inv


def _render_widget_image(d: int, heading_deg: float, colormap_mode: str, now_s: float, tile_style: str) -> np.ndarray:
    r = d // 2
    widget = _render_map_patch(d, tile_style)

    # Radar/grid accents.
    cv2.circle(widget, (r, r), int(r * 0.66), (120, 160, 120), 1, cv2.LINE_AA)
    cv2.line(widget, (r, 2), (r, 2 * r - 2), (110, 140, 110), 1, cv2.LINE_AA)
    cv2.line(widget, (2, r), (2 * r - 2, r), (110, 140, 110), 1, cv2.LINE_AA)

    # Detections (north-up: world_bearing is used directly).
    history_sec = float(getattr(config, "RADAR_HISTORY_SEC", 6.0))
    # Match full dB scaler span exactly (same as colorbar normalization).
    db_min = float(getattr(config, "REL_DB_MIN", -60.0))
    db_max = float(getattr(config, "REL_DB_MAX", 0.0))
    lut = _colormap_lut(colormap_mode)
    inner_r = r - 8
    for det in HUD.radar_detections:
        age = float(now_s) - det.t
        if age < 0 or age > history_sec:
            continue
        age_alpha = max(0.0, 1.0 - (age / max(1e-6, history_sec)))
        db_norm = (det.db_value - db_min) / max(1e-6, (db_max - db_min))
        db_norm = float(np.clip(db_norm, 0.0, 1.0))
        rr = int(inner_r * (1.0 - 0.65 * db_norm))
        ang = math.radians(det.world_bearing_deg)
        px = int(r + math.sin(ang) * rr)
        py = int(r - math.cos(ang) * rr)
        if px < 0 or py < 0 or px >= 2 * r or py >= 2 * r:
            continue
        color = tuple(int(c * (0.25 + 0.75 * age_alpha)) for c in lut[int(db_norm * 255.0)])
        dot_r = 1 if age_alpha < 0.35 else 2
        cv2.circle(widget, (px, py), dot_r, color, -1, cv2.LINE_AA)

    # Position accuracy circle (meters -> pixels at current zoom and latitude).
    if HUD.position.accuracy_m is not None and HUD.position.lat is not None and HUD.position.source != "none":
        zoom = int(getattr(config, "RADAR_MAP_ZOOM", 17))
        mpp = 156543.03392 * math.cos(math.radians(HUD.position.lat)) / (2 ** zoom)
        if mpp > 1e-6:
            acc_px = int(max(2, min(inner_r, HUD.position.accuracy_m / mpp)))
            cv2.circle(widget, (r, r), acc_px, (180, 220, 240), 1, cv2.LINE_AA)

    # Heading arrow (device orientation over north-up map).
    ang_h = math.radians(float(heading_deg))
    tip = (int(r + math.sin(ang_h) * (inner_r - 2)), int(r - math.cos(ang_h) * (inner_r - 2)))
    left = (int(r + math.sin(ang_h + 2.55) * 8), int(r - math.cos(ang_h + 2.55) * 8))
    right = (int(r + math.sin(ang_h - 2.55) * 8), int(r - math.cos(ang_h - 2.55) * 8))
    cv2.fillConvexPoly(widget, np.array([tip, left, right], dtype=np.int32), (245, 245, 245), lineType=cv2.LINE_AA)
    cv2.circle(widget, (r, r), 2, (25, 25, 25), -1, cv2.LINE_AA)

    # Bezel + cardinals.
    cv2.circle(widget, (r, r), r - 1, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.circle(widget, (r, r), r - 3, (170, 220, 170), 1, cv2.LINE_AA)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, ax, ay in (("N", r - 4, 11), ("E", 2 * r - 12, r + 4), ("S", r - 4, 2 * r - 6), ("W", 4, r + 4)):
        cv2.putText(widget, label, (ax, ay), font, 0.35, (225, 235, 225), 1, cv2.LINE_AA)
    return widget


def draw_radar_map_widget(
    frame: np.ndarray,
    anchor_rect: Tuple[int, int, int, int],
    heading_deg: float,
    colormap_mode: str,
    tile_style: str = "dark",
    show_debug: bool = False,
    hud_offset_y: float = 0.0,
    now_s: Optional[float] = None,
) -> None:
    global _RADAR_DEBUG_LAST_LOG_S
    if not bool(getattr(config, "RADAR_MAP_ENABLED", True)):
        return
    if now_s is None:
        now_s = time.time()
    d = int(getattr(config, "RADAR_MAP_DIAMETER_PX", 100))
    update_hz = max(1.0, float(getattr(config, "RADAR_MAP_UPDATE_HZ", 15.0)))
    refresh_dt = 1.0 / update_hz
    gap = int(getattr(config, "RADAR_MAP_GAP_PX", 8))
    x, y, w, h = anchor_rect
    cx = x + w // 2
    cy = y + h + gap + d // 2
    r = d // 2
    if r < 12:
        return
    fh, fw = frame.shape[:2]
    if cx - r < 0 or cy - r < 0 or cx + r >= fw or cy + r >= fh:
        return

    # Prune stale detections periodically to keep draw path compact.
    _prune_history(float(now_s))

    # Re-render cached widget only at target update rate.
    needs_refresh = (
        _WIDGET_CACHE.get("img") is None
        or int(_WIDGET_CACHE.get("d", 0)) != d
        or str(_WIDGET_CACHE.get("colormap", "")) != str(colormap_mode)
        or str(_WIDGET_CACHE.get("tile_style", "dark")) != str(tile_style)
        or (float(now_s) - float(_WIDGET_CACHE.get("updated_s", 0.0))) >= refresh_dt
    )
    if needs_refresh:
        _WIDGET_CACHE["img"] = _render_widget_image(d, heading_deg, colormap_mode, float(now_s), str(tile_style))
        _WIDGET_CACHE["updated_s"] = float(now_s)
        _WIDGET_CACHE["d"] = d
        _WIDGET_CACHE["colormap"] = str(colormap_mode)
        _WIDGET_CACHE["tile_style"] = str(tile_style)

    widget_img = _WIDGET_CACHE["img"]
    if widget_img is None:
        return
    if widget_img.shape[:2] != (d, d):
        widget_img = cv2.resize(widget_img, (d, d), interpolation=cv2.INTER_LINEAR)
    mask, inv = _get_circle_masks(d)
    roi = frame[cy - r:cy + r, cx - r:cx + r]
    bg = cv2.bitwise_and(roi, roi, mask=inv)
    fg = cv2.bitwise_and(widget_img, widget_img, mask=mask)
    roi[:] = cv2.add(bg, fg)

    if show_debug:
        line1 = (
            f"X={int(getattr(HUD, 'mag_x_raw', 0))}  "
            f"Y={int(getattr(HUD, 'mag_y_raw', 0))}  "
            f"Z={int(getattr(HUD, 'mag_z_raw', 0))}  "
            f"Heading={int(round(float(getattr(HUD, 'mag_heading_dbg', heading_deg))))}deg"
        )
        x_min = getattr(HUD, "mag_x_min", None)
        x_max = getattr(HUD, "mag_x_max", None)
        y_min = getattr(HUD, "mag_y_min", None)
        y_max = getattr(HUD, "mag_y_max", None)
        if x_min is None or x_max is None or y_min is None or y_max is None:
            line2 = "Range X=[--,--] Y=[--,--]"
            line3 = "Ctr=[--,--]  Hcal=--deg  Pair=XY  Cal=OFF"
            span_txt = "Span=(0,0,0)"
        else:
            x_ctr = int(round((x_min + x_max) * 0.5))
            y_ctr = int(round((y_min + y_max) * 0.5))
            hcal = int(round(float(getattr(HUD, "mag_heading_cal_dbg", heading_deg))))
            cal_on = "ON" if bool(getattr(HUD, "mag_cal_active", False)) else "OFF"
            pair = str(getattr(HUD, "mag_pair_dbg", "XY"))
            sx = int(getattr(HUD, "mag_span_x", 0))
            sy = int(getattr(HUD, "mag_span_y", 0))
            sz = int(getattr(HUD, "mag_span_z", 0))
            line2 = f"Range X=[{int(x_min)},{int(x_max)}] Y=[{int(y_min)},{int(y_max)}]"
            line3 = f"Ctr=[{x_ctr},{y_ctr}]  Hcal={hcal}deg  Pair={pair}  Cal={cal_on}"
            span_txt = f"Span=({sx},{sy},{sz})"
        gps_status = "FIX" if bool(getattr(HUD, "gps_fix_valid", False)) else "NOFIX"
        sats = int(getattr(HUD, "gps_sat_count", 0))
        pos_source = str(getattr(HUD.position, "source", "none")).upper()
        map_status = _MAP_LAST_STATUS
        line4 = f"{span_txt}  SRC={pos_source} GPS={gps_status} SAT={sats} MAP={map_status} CM={colormap_mode}"
        debug_lines = [line1, line2, line3, line4]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        pad = 12
        line_h = 18
        max_w = int(getattr(config, "RADAR_DEBUG_BOX_MAX_WIDTH", 360))

        def _fit_line(text: str, width_px: int) -> str:
            s = text
            while s:
                (tw, _), _ = cv2.getTextSize(s, font, font_scale, thickness)
                if tw <= width_px:
                    return s
                if len(s) <= 4:
                    break
                s = s[:-2]
            return "..."

        max_text_px = max(40, max_w - 2 * pad)
        debug_lines = [_fit_line(line, max_text_px) for line in debug_lines]
        text_w = 0
        for line in debug_lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            text_w = max(text_w, tw)
        box_w = min(text_w + 2 * pad, max_w)
        box_h = line_h * len(debug_lines) + 2 * pad

        # Place box to the right of radar with ~3px gap, fallback left if needed.
        box_x = cx + r + 3
        box_y = cy - r
        if box_x + box_w > fw - 2:
            box_x = max(2, cx - r - 3 - box_w)
        box_y = max(2, min(box_y, fh - box_h - 2))

        # region agent log
        if (float(now_s) - _RADAR_DEBUG_LAST_LOG_S) >= 0.40:
            _RADAR_DEBUG_LAST_LOG_S = float(now_s)
            _agent_debug_log(
                "run2",
                "H4",
                "acoustic_radar_map.py:draw_radar_map_widget",
                "radar_debug_box_position_sample",
                {
                    "show_debug": bool(show_debug),
                    "hud_offset_y": float(hud_offset_y),
                    "anchor_rect": [int(x), int(y), int(w), int(h)],
                    "radar_center": [int(cx), int(cy)],
                    "radar_r": int(r),
                    "box_xywh": [int(box_x), int(box_y), int(box_w), int(box_h)],
                    "screen_wh": [int(fw), int(fh)],
                },
            )
        # endregion

        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (100, 100, 100), 2, cv2.LINE_AA)
        ty = box_y + pad + 13
        for line in debug_lines:
            cv2.putText(frame, line, (box_x + pad, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            ty += line_h


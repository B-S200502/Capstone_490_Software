# dsp/heatmap.py
from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

# Precomputed 1D Gaussian half-kernels for blob drawing (sigma -> g_1d of length r_max+1)
_BLOB_R_MAX = 200
_BLOB_KERNEL_CACHE: Optional[list[tuple[float, np.ndarray]]] = None


def _get_blob_kernel_cache() -> list[tuple[float, np.ndarray]]:
    global _BLOB_KERNEL_CACHE
    if _BLOB_KERNEL_CACHE is not None:
        return _BLOB_KERNEL_CACHE
    sigmas = np.linspace(22.0, 35.0, 8)
    cache = []
    for sig in sigmas:
        sigma2 = float(2.0 * sig * sig + 1e-12)
        idx = np.arange(_BLOB_R_MAX + 1, dtype=np.float32)
        g_1d = np.exp(-(idx * idx) / sigma2).astype(np.float32)
        cache.append((float(sig), g_1d))
    _BLOB_KERNEL_CACHE = cache
    return _BLOB_KERNEL_CACHE


def _nearest_sigma_idx(sigma: float, cache: list[tuple[float, np.ndarray]]) -> int:
    best = 0
    best_diff = abs(cache[0][0] - sigma)
    for k in range(1, len(cache)):
        d = abs(cache[k][0] - sigma)
        if d < best_diff:
            best_diff = d
            best = k
    return best


def percentile_uint8_fast(arr: np.ndarray, pct: float) -> float:
    """
    Approximate percentile for uint8 array (0-255) using 256-bin histogram.
    Single pass, O(256) after; faster than np.percentile on large arrays.
    Returns value in [0, 255] (interpolated between bins).
    """
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.size == 0:
        return 0.0
    hist = np.bincount(arr.ravel(), minlength=256)
    total = float(hist.sum())
    if total <= 0:
        return 0.0
    target = (pct / 100.0) * total
    cum = np.cumsum(hist)
    bin_idx = int(np.searchsorted(cum, target, side="left"))
    bin_idx = min(bin_idx, 255)
    if bin_idx <= 0:
        return 0.0
    lo = cum[bin_idx - 1]
    hi = cum[bin_idx]
    denom = hi - lo
    if denom <= 0:
        return float(bin_idx)
    frac = (target - lo) / denom
    frac = np.clip(frac, 0.0, 1.0)
    return float(bin_idx) + frac


# Colormap mapping
COLORMAP_DICT = {
    "MAGMA": cv2.COLORMAP_MAGMA,
    "JET": cv2.COLORMAP_JET,
    "TURBO": cv2.COLORMAP_TURBO,
    "INFERNO": cv2.COLORMAP_INFERNO,
}


def spectra_to_heatmap_absolute(
    spec_matrix: np.ndarray,
    power_rel: np.ndarray,
    out_width: int,
    out_height: int,
    db_min: float,
    db_max: float,
    x_offset_px: float = 0.0,
    angle_min_deg: float = -90.0,
    angle_max_deg: float = 90.0,
    band_freqs_hz: Optional[np.ndarray] = None,
    f_min_hz: Optional[float] = None,
    f_max_hz: Optional[float] = None,
    projection_mode: str = "linear",
    circle_radius_px: float = 0.0,
    assumed_distance_m: float = 1.0,
    camera_hfov_deg: float = 53.0,
    camera_vfov_deg: float = 0.0,
    angle_x_deg: Optional[np.ndarray] = None,
    angle_y_deg: Optional[np.ndarray] = None,
    heat_out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    SAME signature as your monolith.

    Faster heatmap builder:
      - Finds peak angle per source
      - Creates Gaussian blobs only in a small ROI (not full-frame meshgrid)
      - Accumulates in float32 heatmap, returns uint8 0..255
    """
    Nsrc, Nang = spec_matrix.shape
    if Nsrc == 0 or Nang == 0:
        return np.zeros((out_height, out_width), dtype=np.uint8)

    power_rel = np.maximum(power_rel.astype(np.float32), 1e-12)
    power_db = 10.0 * np.log10(power_rel)  # <= 0
    power_norm = (power_db - float(db_min)) / (float(db_max) - float(db_min) + 1e-12)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    peak_idx = np.argmax(spec_matrix, axis=1).astype(np.int32)

    # Parabolic interpolation for sub-bin angle (fractional peak index)
    peak_idx_frac = np.empty(Nsrc, dtype=np.float64)
    for i in range(Nsrc):
        idx = int(peak_idx[i])
        y0 = float(spec_matrix[i, idx - 1]) if idx > 0 else float(spec_matrix[i, idx])
        y1 = float(spec_matrix[i, idx])
        y2 = float(spec_matrix[i, idx + 1]) if idx < (Nang - 1) else float(spec_matrix[i, idx])
        if idx > 0 and idx < (Nang - 1):
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) >= 1e-12:
                d = 0.5 * (y0 - y2) / denom
                d = np.clip(d, -0.5, 0.5)
            else:
                d = 0.0
            peak_idx_frac[i] = idx + d
        else:
            peak_idx_frac[i] = float(idx)
    peak_idx_frac = np.clip(peak_idx_frac, 0.0, float(Nang - 1))

    sharp = np.empty(Nsrc, dtype=np.float32)
    for i in range(Nsrc):
        idx = int(peak_idx[i])
        p = float(spec_matrix[i, idx])
        left = float(spec_matrix[i, idx - 1]) if idx > 0 else p
        right = float(spec_matrix[i, idx + 1]) if idx < (Nang - 1) else p
        sharp[i] = max(p - 0.5 * (left + right), 1e-12)

    sharp /= (sharp.max() + 1e-12)

    h, w = int(out_height), int(out_width)
    if heat_out is not None and heat_out.shape == (h, w) and heat_out.dtype == np.float32:
        heat_out.fill(0)
        heat = heat_out
    else:
        heat = np.zeros((h, w), dtype=np.float32)

    # Map peak angle to (x, y) via projection mode
    angle_deg = -90.0 + 180.0 * peak_idx_frac.astype(np.float32) / max(1, (Nang - 1))
    angle_rad = np.deg2rad(angle_deg)

    use_dual_angle = (
        projection_mode == "dual_angle"
        and angle_x_deg is not None
        and angle_y_deg is not None
        and getattr(angle_x_deg, "size", len(angle_x_deg)) == Nsrc
        and getattr(angle_y_deg, "size", len(angle_y_deg)) == Nsrc
    )
    if use_dual_angle:
        # Two arrays in software: θ_x → x, θ_y → y (same angle range for both axes)
        angle_span = max(1e-6, float(angle_max_deg) - float(angle_min_deg))
        ax = np.asarray(angle_x_deg, dtype=np.float64).reshape(Nsrc)
        ay = np.asarray(angle_y_deg, dtype=np.float64).reshape(Nsrc)
        t_x = np.clip((ax - float(angle_min_deg)) / angle_span, 0.0, 1.0)
        t_y = np.clip((ay - float(angle_min_deg)) / angle_span, 0.0, 1.0)
        cx_all = (t_x * (w - 1) + float(x_offset_px)).astype(np.float32)
        cy_all = (t_y * (h - 1)).astype(np.float32)  # angle_min -> top, angle_max -> bottom
        cx_all = np.clip(cx_all, 0.0, float(w - 1))
        cy_all = np.clip(cy_all, 0.0, float(h - 1))
    elif projection_mode == "camera_plane":
        cx = (w - 1) / 2.0
        hfov_rad = np.deg2rad(max(1e-6, float(camera_hfov_deg)))
        fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
        # x from DOA angle (0° = center): u = cx + fx*sin(θ)
        cx_all = (cx + fx * np.sin(angle_rad)).astype(np.float32)
        cx_all = np.clip(cx_all, 0.0, float(w - 1))
        # y from frequency so full screen is usable: high freq = top, low freq = bottom (match freq bar)
        if (
            band_freqs_hz is not None
            and band_freqs_hz.size == Nsrc
            and f_min_hz is not None
            and f_max_hz is not None
            and float(f_max_hz) > float(f_min_hz)
        ):
            f_span = float(f_max_hz) - float(f_min_hz)
            t_y = np.clip(
                (band_freqs_hz.astype(np.float64) - float(f_min_hz)) / f_span, 0.0, 1.0
            )
            cy_all = ((1.0 - t_y) * (h - 1)).astype(np.float32)
        else:
            cy = (h - 1) / 2.0
            if camera_vfov_deg > 0:
                vfov_rad = np.deg2rad(camera_vfov_deg)
                fy = (h / 2.0) / np.tan(vfov_rad / 2.0)
            else:
                fy = fx
            cy_all = (cy - fy * np.cos(angle_rad)).astype(np.float32)
        cy_all = np.clip(cy_all, 0.0, float(h - 1))
    elif projection_mode == "camera_circle":
        center_x = (w - 1) / 2.0
        center_y = (h - 1) / 2.0
        r = circle_radius_px if circle_radius_px > 0 else min(w, h) * 0.45
        # Rotate so DOA 0° (forward) is at top of circle instead of right
        angle_rad_rot = angle_rad + np.pi / 2.0
        cx_all = (center_x + r * np.cos(angle_rad_rot)).astype(np.float32)
        cy_all = (center_y - r * np.sin(angle_rad_rot)).astype(np.float32)
        cx_all = np.clip(cx_all, 0.0, float(w - 1))
        cy_all = np.clip(cy_all, 0.0, float(h - 1))
    else:
        # linear: x from effective angle range, y from sin(θ)
        angle_span = max(1e-6, float(angle_max_deg) - float(angle_min_deg))
        t = (angle_deg - float(angle_min_deg)) / angle_span
        t = np.clip(t, 0.0, 1.0)
        cx_all = (t * (w - 1) + float(x_offset_px)).astype(np.float32)
        cx_all = np.clip(cx_all, 0.0, float(w - 1))
        sin_theta = np.sin(angle_rad)
        cy_all = ((1.0 - sin_theta) / 2.0 * (h - 1)).astype(np.float32)
        cy_all = np.clip(cy_all, 0.0, float(h - 1))

    base_radius = 60.0
    blob_cache = _get_blob_kernel_cache()

    for i in range(Nsrc):
        amp = float(power_norm[i])
        if amp <= 0.0:
            continue

        cx = float(cx_all[i])
        cy = float(cy_all[i])

        blob_radius = base_radius * (0.7 + 0.3 * float(sharp[i]))
        sigma = blob_radius / 1.8
        r = int(max(6, min(_BLOB_R_MAX, round(7.0 * sigma))))
        cx_int = int(cx)
        cy_int = int(cy)
        x0 = max(0, cx_int - r)
        x1 = min(w, cx_int + r + 1)
        y0 = max(0, cy_int - r)
        y1 = min(h, cy_int + r + 1)

        kidx = _nearest_sigma_idx(sigma, blob_cache)
        g_1d = blob_cache[kidx][1]
        full_1d = np.concatenate([g_1d[r::-1], g_1d[1 : r + 1]]).astype(np.float32)
        kernel_2d = (full_1d[:, None] * full_1d[None, :]).astype(np.float32)
        k_h, k_w = kernel_2d.shape
        ky0 = r + (y0 - cy_int)
        kx0 = r + (x0 - cx_int)
        roi_h, roi_w = y1 - y0, x1 - x0
        kernel_slice = kernel_2d[ky0 : ky0 + roi_h, kx0 : kx0 + roi_w]
        heat[y0:y1, x0:x1] += amp * kernel_slice

    heat = np.clip(heat, 0.0, 1.0)
    return (heat * 255.0).astype(np.uint8)


def build_w_lut_u8(alpha: float, blend_gamma: float) -> np.ndarray:
    """
    Rebuilds your W_LUT_U8 exactly (so main can call this instead of global init).
    """
    x = np.arange(256, dtype=np.float32) / 255.0
    x = np.power(x, float(blend_gamma)) * float(alpha)
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


# Reusable buffers for blend_heatmap_left (keyed by (content_width, content_height))
_blend_buffers: Optional[dict[tuple[int, int], tuple[np.ndarray, ...]]] = None
# Half-res blend: extra buffers for downscaled heatmap and background
_blend_half_buffers: Optional[dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]] = None
# Fused colormap+weight LUT cache: (colormap_name, tuple(w_lut_u8)) -> (256, 3) uint8
_fused_lut_cache: Optional[dict[tuple[str, tuple], np.ndarray]] = None


def _get_fused_lut(colormap: str, w_lut_u8: np.ndarray) -> np.ndarray:
    """Precompute (colormap(v) * w_lut(v)) // 255 for v in 0..255; returns (256, 3) uint8. Cached by (colormap, w_lut)."""
    global _fused_lut_cache
    if _fused_lut_cache is None:
        _fused_lut_cache = {}
    key = (colormap, tuple(w_lut_u8))
    if key not in _fused_lut_cache:
        colormap_cv = COLORMAP_DICT.get(colormap, cv2.COLORMAP_MAGMA)
        one_col = np.arange(256, dtype=np.uint8).reshape(256, 1)
        cmap = cv2.applyColorMap(one_col, colormap_cv)[:, 0, :]
        w = w_lut_u8.reshape(256, 1).astype(np.uint32)
        fused = np.clip((cmap.astype(np.uint32) * w) // 255, 0, 255).astype(np.uint8)
        _fused_lut_cache[key] = fused
    return _fused_lut_cache[key]


def _get_blend_buffers(h: int, w: int) -> tuple[np.ndarray, ...]:
    """Get or create (w_buf, w3_buf, inv3_buf, c1_buf, c2_buf, c1_ch0, c1_ch1, c1_ch2) for blend ROI shape (h, w). Contiguous 2D buffers for LUT dst."""
    global _blend_buffers
    if _blend_buffers is None:
        _blend_buffers = {}
    key = (w, h)
    if key not in _blend_buffers:
        _blend_buffers[key] = (
            np.empty((h, w), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w, 3), dtype=np.uint8),
            np.empty((h, w), dtype=np.uint8),
            np.empty((h, w), dtype=np.uint8),
            np.empty((h, w), dtype=np.uint8),
        )
    return _blend_buffers[key]


def _get_blend_half_buffers(h2: int, w2: int) -> tuple[np.ndarray, np.ndarray]:
    """Get or create (heatmap_half_buf, left_bg_half_buf) for half-res blend."""
    global _blend_half_buffers
    if _blend_half_buffers is None:
        _blend_half_buffers = {}
    key = (w2, h2)
    if key not in _blend_half_buffers:
        _blend_half_buffers[key] = (
            np.empty((h2, w2), dtype=np.uint8),
            np.empty((h2, w2, 3), dtype=np.uint8),
        )
    return _blend_half_buffers[key]


def _blend_core(
    heat: np.ndarray,
    left_bg: np.ndarray,
    w_lut_u8: np.ndarray,
    fused_lut: np.ndarray,
    inv_lut: np.ndarray,
    w_buf: np.ndarray,
    inv3_buf: np.ndarray,
    c1_buf: np.ndarray,
    c2_buf: np.ndarray,
    c1_ch0: np.ndarray,
    c1_ch1: np.ndarray,
    c1_ch2: np.ndarray,
) -> None:
    """Run fused LUT blend: colored*w into c1_buf via contiguous ch buffers; c2_buf = left_bg*inv; c1_buf += c2_buf."""
    cv2.LUT(heat, fused_lut[:, 0], c1_ch0)
    cv2.LUT(heat, fused_lut[:, 1], c1_ch1)
    cv2.LUT(heat, fused_lut[:, 2], c1_ch2)
    c1_buf[:, :, 0] = c1_ch0
    c1_buf[:, :, 1] = c1_ch1
    c1_buf[:, :, 2] = c1_ch2
    cv2.LUT(heat, inv_lut, w_buf)
    inv3_buf[:] = w_buf[:, :, np.newaxis]
    cv2.multiply(left_bg, inv3_buf, c2_buf, scale=1 / 255.0)
    cv2.add(c1_buf, c2_buf, c1_buf)


def blend_heatmap_left(
    base_frame: np.ndarray,
    heatmap_left: np.ndarray,
    content_width: int,
    content_height: int,
    content_offset_x: int,
    w_lut_u8: np.ndarray,
    colormap: str = "MAGMA",
) -> np.ndarray:
    """
    Blend heatmap onto the content strip (between DB bar and freq bar).
    Uses fused colormap+weight LUT (no applyColorMap + first multiply). Optional half-res then upsample.
    """
    from .. import config
    x0 = content_offset_x
    x1 = content_offset_x + content_width
    left_bg = base_frame[0:content_height, x0:x1, :]
    fused_lut = _get_fused_lut(colormap, w_lut_u8)
    inv_lut = np.subtract(255, w_lut_u8, dtype=np.uint8)

    half_res = bool(getattr(config, "BLEND_HALF_RES", False))
    if half_res and content_height >= 4 and content_width >= 4:
        h2, w2 = content_height // 2, content_width // 2
        heatmap_half_buf, left_bg_half_buf = _get_blend_half_buffers(h2, w2)
        cv2.resize(heatmap_left, (w2, h2), dst=heatmap_half_buf, interpolation=cv2.INTER_LINEAR)
        cv2.resize(left_bg, (w2, h2), dst=left_bg_half_buf, interpolation=cv2.INTER_LINEAR)
        w_buf, w3_buf, inv3_buf, c1_buf, c2_buf, c1_ch0, c1_ch1, c1_ch2 = _get_blend_buffers(h2, w2)
        _blend_core(
            heatmap_half_buf, left_bg_half_buf, w_lut_u8, fused_lut, inv_lut,
            w_buf, inv3_buf, c1_buf, c2_buf, c1_ch0, c1_ch1, c1_ch2,
        )
        cv2.resize(c1_buf, (content_width, content_height), dst=base_frame[0:content_height, x0:x1, :], interpolation=cv2.INTER_LINEAR)
    else:
        w_buf, w3_buf, inv3_buf, c1_buf, c2_buf, c1_ch0, c1_ch1, c1_ch2 = _get_blend_buffers(content_height, content_width)
        _blend_core(
            heatmap_left, left_bg, w_lut_u8, fused_lut, inv_lut,
            w_buf, inv3_buf, c1_buf, c2_buf, c1_ch0, c1_ch1, c1_ch2,
        )
        base_frame[0:content_height, x0:x1, :] = c1_buf
    return base_frame


# Crosshairs: 5mm cross (two lines 5mm each), ~19 px at 96 DPI
CROSSHAIR_LENGTH_PX = 19
CROSSHAIR_HALF = CROSSHAIR_LENGTH_PX // 2
CROSSHAIR_COLOR = (220, 220, 220)
CROSSHAIR_THICKNESS = 1
CROSSHAIR_TOOLTIP_BG = (40, 40, 40)
CROSSHAIR_TOOLTIP_TEXT = (255, 255, 255)
CROSSHAIR_TRACK_RADIUS = 15
CROSSHAIR_DISMISS_RADIUS_PX = 25

# 180° protractor in tooltip (semicircle, flat base at bottom; light fill, dark outline)
PROTRACTOR_W = 76
PROTRACTOR_H = 60
PROTRACTOR_BUFFER_PX = 3   # gap from max(dB, kHz) text to protractor left edge
PROTRACTOR_FILL_BGR = (220, 200, 160)   # light blue-ish
PROTRACTOR_OUTLINE_BGR = (0, 0, 0)
PROTRACTOR_ARROW_BGR = (0, 0, 255)      # red arrow

# Cache: static protractor (no arrow/text) keyed by (w, h)
_PROTRACTOR_STATIC_CACHE: dict[tuple[int, int], np.ndarray] = {}

# EMA smoothing for tooltip (alpha = new value weight; 0.35 = smooth)
_TOOLTIP_EMA_ALPHA = 0.35
_TOOLTIP_SMOOTH: dict[str, float] = {}  # db, f_peak_khz, angle_deg, trend_db, accel_db, distance, tx, ty

# Precomputed max tooltip box size (set once)
_TOOLTIP_BOX_W: Optional[int] = None
_TOOLTIP_BOX_H: Optional[int] = None
_TOOLTIP_PROTRACTOR_X0: Optional[int] = None
_TOOLTIP_LINE_H: Optional[int] = None


def _ensure_tooltip_box_size() -> tuple[int, int, int, int]:
    """Compute max tooltip box (box_w, box_h, protractor_x0, line_h) once; return cached."""
    global _TOOLTIP_BOX_W, _TOOLTIP_BOX_H, _TOOLTIP_PROTRACTOR_X0, _TOOLTIP_LINE_H
    if _TOOLTIP_BOX_W is not None:
        return (_TOOLTIP_BOX_W, _TOOLTIP_BOX_H, _TOOLTIP_PROTRACTOR_X0, _TOOLTIP_LINE_H)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    pad = 6
    worst = ["-60 dB", "99.9 kHz", "99.99 m", "3s: +99.9 dB", "12s: +0.00/3s"]
    box_w = pad * 2
    (_, th), _ = cv2.getTextSize("0 dB", font, scale, thick)
    _TOOLTIP_LINE_H = th
    for txt in worst:
        (tw, _), _ = cv2.getTextSize(txt, font, scale, thick)
        box_w = max(box_w, tw + 2 * pad)
    box_h = pad * 2 + 5 * (th + 2) + 7
    box_h = max(box_h, PROTRACTOR_H + 4)
    (tw1, _), _ = cv2.getTextSize("-60 dB", font, scale, thick)
    (tw2, _), _ = cv2.getTextSize("99.9 kHz", font, scale, thick)
    _TOOLTIP_PROTRACTOR_X0 = pad + max(tw1, tw2) + PROTRACTOR_BUFFER_PX
    _TOOLTIP_BOX_W = max(box_w, _TOOLTIP_PROTRACTOR_X0 + PROTRACTOR_W + 2)
    _TOOLTIP_BOX_H = box_h
    return (_TOOLTIP_BOX_W, _TOOLTIP_BOX_H, _TOOLTIP_PROTRACTOR_X0, _TOOLTIP_LINE_H)


def _tooltip_ema(key: str, raw: float, alpha: float = _TOOLTIP_EMA_ALPHA) -> float:
    """Return EMA-smoothed value; updates _TOOLTIP_SMOOTH[key]. First call uses raw."""
    if key not in _TOOLTIP_SMOOTH:
        _TOOLTIP_SMOOTH[key] = float(raw)
        return float(raw)
    prev = _TOOLTIP_SMOOTH[key]
    out = alpha * raw + (1.0 - alpha) * prev
    _TOOLTIP_SMOOTH[key] = out
    return out


def _get_static_protractor_base(w: int, h: int) -> np.ndarray:
    """Build or return cached static protractor (semicircle, ticks, center dot) on tooltip BG."""
    key = (w, h)
    if key in _PROTRACTOR_STATIC_CACHE:
        return _PROTRACTOR_STATIC_CACHE[key]
    base = np.empty((h, w, 3), dtype=np.uint8)
    base[:] = CROSSHAIR_TOOLTIP_BG
    cx = w // 2
    r = min((w - 4) // 2, (h - 6) // 2)
    if r < 4:
        _PROTRACTOR_STATIC_CACHE[key] = base
        return base
    cy = r + 2
    cv2.ellipse(base, (cx, cy), (r, r), 0, 180, 360, PROTRACTOR_FILL_BGR, -1, cv2.LINE_AA)
    cv2.ellipse(base, (cx, cy), (r, r), 0, 180, 360, PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    cv2.line(base, (cx - r, cy), (cx + r, cy), PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    for deg in [0, 30, 60, 90, 120, 150, 180]:
        a_rad = np.deg2rad(180 - deg)
        tx = cx + int((r - 2) * np.cos(a_rad))
        ty = cy - int((r - 2) * np.sin(a_rad))
        tx2 = cx + int(r * np.cos(a_rad))
        ty2 = cy - int(r * np.sin(a_rad))
        cv2.line(base, (tx, ty), (tx2, ty2), PROTRACTOR_OUTLINE_BGR, 1, cv2.LINE_AA)
    cv2.circle(base, (cx, cy), 2, PROTRACTOR_OUTLINE_BGR, -1, cv2.LINE_AA)
    _PROTRACTOR_STATIC_CACHE[key] = base
    return base


def find_local_max(heatmap: np.ndarray, cx: int, cy: int, radius: int) -> tuple[int, int]:
    """Return (x, y) of pixel with max value in heatmap within radius of (cx, cy)."""
    h, w = heatmap.shape[:2]
    y0 = max(0, cy - radius)
    y1 = min(h, cy + radius + 1)
    x0 = max(0, cx - radius)
    x1 = min(w, cx + radius + 1)
    roi = heatmap[y0:y1, x0:x1]
    if roi.size == 0:
        return (cx, cy)
    flat_idx = np.argmax(roi)
    iy = flat_idx // roi.shape[1]
    ix = flat_idx % roi.shape[1]
    return (x0 + ix, y0 + iy)


# Trend/acceleration colors (BGR): green = rising, red = falling, white = stable
TREND_COLOR_POS = (0, 200, 100)    # green
TREND_COLOR_NEG = (0, 100, 255)    # red
TREND_COLOR_ZERO = (200, 200, 200)  # light gray
TREND_COLOR_PENDING = (120, 120, 120)  # dim gray for placeholder (waiting for first avg)


def _draw_protractor_180(
    frame: np.ndarray,
    x0: int,
    y0: int,
    w: int,
    h: int,
    angle_deg: Optional[float],
) -> None:
    """Draw a 180° protractor: blit cached static shape, then arrow + degree text only."""
    if w < 10 or h < 10:
        return
    cx = x0 + w // 2
    r = min((w - 4) // 2, (h - 6) // 2)
    if r < 4:
        return
    cy = y0 + r + 2
    # Blit cached static protractor (semicircle, ticks, center dot)
    base = _get_static_protractor_base(w, h)
    h_clip = min(h, base.shape[0])
    w_clip = min(w, base.shape[1])
    frame[y0 : y0 + h_clip, x0 : x0 + w_clip] = base[:h_clip, :w_clip]
    # Dynamic arrow only
    arrow_len = int(r * 0.65)
    if angle_deg is not None:
        rad = np.deg2rad(angle_deg)
        ex = int(cx + arrow_len * np.sin(rad))
        ey = int(cy - arrow_len * np.cos(rad))
        cv2.arrowedLine(frame, (cx, cy), (ex, ey), PROTRACTOR_ARROW_BGR, 2, cv2.LINE_AA, tipLength=0.25)
    # Degree text only
    label = f"{angle_deg:.1f} deg" if angle_deg is not None else "-- deg"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    lx = cx - tw // 2
    ly = cy + 2 + th
    cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.35, CROSSHAIR_TOOLTIP_TEXT, 1, cv2.LINE_AA)


def _trend_color(db: float, threshold: float = 0.3) -> tuple:
    """Return BGR color for trend/accel value."""
    if db is None:
        return CROSSHAIR_TOOLTIP_TEXT
    if db > threshold:
        return TREND_COLOR_POS
    if db < -threshold:
        return TREND_COLOR_NEG
    return TREND_COLOR_ZERO


def draw_crosshairs(
    frame: np.ndarray,
    center_x: int,
    center_y: int,
    content_width: int,
    height: int,
    heatmap_left: np.ndarray,
    rel_db_min: float,
    rel_db_max: float,
    f_peak_hz: float,
    trend_db: Optional[float] = None,
    accel_db: Optional[float] = None,
    distance_to_source_m: Optional[float] = None,
    angle_deg: Optional[float] = None,
    content_offset_x: int = 0,
) -> None:
    """Draw 5mm cross and tooltip (dB, kHz, distance, 3s trend, 12s accel, 180° protractor).
    center_x, center_y are in content (heatmap) coords; content_offset_x is added when drawing on frame."""
    if content_width <= 0 or height <= 0 or heatmap_left.size == 0:
        return
    cx = max(0, min(content_width - 1, int(center_x)))
    cy = max(0, min(height - 1, int(center_y)))
    fx = cx + content_offset_x  # frame x for drawing
    x0 = max(0, fx - CROSSHAIR_HALF)
    x1 = min(frame.shape[1], fx + CROSSHAIR_HALF + 1)
    y0 = max(0, cy - CROSSHAIR_HALF)
    y1 = min(height, cy + CROSSHAIR_HALF + 1)
    cv2.line(frame, (x0, cy), (x1, cy), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS, cv2.LINE_AA)
    cv2.line(frame, (fx, y0), (fx, y1), CROSSHAIR_COLOR, CROSSHAIR_THICKNESS, cv2.LINE_AA)

    val = int(heatmap_left[cy, cx])
    db_raw = rel_db_min + (val / 255.0) * (rel_db_max - rel_db_min)
    f_peak_khz_raw = f_peak_hz / 1000.0
    angle_raw = float(angle_deg) if angle_deg is not None else None
    # EMA smoothing to reduce wobble
    db = _tooltip_ema("db", db_raw)
    f_peak_khz = _tooltip_ema("f_peak_khz", f_peak_khz_raw)
    if angle_raw is not None:
        angle_deg_smooth = _tooltip_ema("angle_deg", angle_raw)
        angle_deg_smooth = round(angle_deg_smooth * 2.0) / 2.0  # quantize to 0.5°
    else:
        if "angle_deg" in _TOOLTIP_SMOOTH:
            angle_deg_smooth = _TOOLTIP_SMOOTH["angle_deg"]
        else:
            angle_deg_smooth = None
    if trend_db is not None:
        trend_smooth = _tooltip_ema("trend_db", trend_db)
    else:
        trend_smooth = _TOOLTIP_SMOOTH.get("trend_db")
    if accel_db is not None:
        accel_smooth = _tooltip_ema("accel_db", accel_db)
    else:
        accel_smooth = _TOOLTIP_SMOOTH.get("accel_db")
    dist_smooth = None
    if distance_to_source_m is not None:
        dist_smooth = _tooltip_ema("distance", distance_to_source_m)
    else:
        dist_smooth = _TOOLTIP_SMOOTH.get("distance")

    line1 = f"{db:+.0f} dB"
    line2 = f"{f_peak_khz:.1f} kHz"
    line3 = f"{dist_smooth:.2f} m" if dist_smooth is not None else "-- m"
    lines = [(line1, CROSSHAIR_TOOLTIP_TEXT), (line2, CROSSHAIR_TOOLTIP_TEXT), (line3, CROSSHAIR_TOOLTIP_TEXT)]
    if trend_smooth is not None:
        lines.append((f"3s: {trend_smooth:+.1f} dB", _trend_color(trend_smooth)))
    else:
        lines.append(("3s: -- dB", TREND_COLOR_PENDING))
    if accel_smooth is not None:
        lines.append((f"12s: {accel_smooth:+.2f}/3s", _trend_color(accel_smooth, 0.05)))
    else:
        lines.append(("12s: --/3s", TREND_COLOR_PENDING))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1
    pad = 6
    box_w, box_h, protractor_x0, line_h = _ensure_tooltip_box_size()
    tx_raw = float(cx + 14)
    ty_raw = float(cy - 4)
    if tx_raw + box_w > content_width:
        tx_raw = float(cx - box_w - 14)
    if ty_raw < 0:
        ty_raw = float(cy + 20)
    if ty_raw + box_h > height:
        ty_raw = float(cy - box_h - 8)
    tx_raw = float(max(2, min(content_width - box_w - 2, int(tx_raw))))
    ty_raw = float(max(2, min(height - box_h - 2, int(ty_raw))))
    tx = int(_tooltip_ema("tx", tx_raw))
    ty = int(_tooltip_ema("ty", ty_raw))
    tx = max(2, min(content_width - box_w - 2, tx))
    ty = max(2, min(height - box_h - 2, ty))
    tx_frame = tx + content_offset_x

    cv2.rectangle(frame, (tx_frame, ty), (tx_frame + box_w, ty + box_h), CROSSHAIR_TOOLTIP_BG, -1)
    cv2.rectangle(frame, (tx_frame, ty), (tx_frame + box_w, ty + box_h), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
    _draw_protractor_180(
        frame,
        tx_frame + protractor_x0,
        ty + 2,
        PROTRACTOR_W,
        min(PROTRACTOR_H, box_h - 4),
        angle_deg_smooth,
    )
    y_off = ty + pad
    for i, (txt, color) in enumerate(lines):
        cv2.putText(frame, txt, (tx_frame + pad, y_off + line_h), font, scale, color, thick, cv2.LINE_AA)
        y_off += line_h + 2
        if i == 2:
            bar_y = int(y_off + 2)
            bar_left = tx_frame + pad
            bar_right = tx_frame + box_w - pad
            cv2.line(frame, (bar_left, bar_y), (bar_right, bar_y), CROSSHAIR_COLOR, 1, cv2.LINE_AA)
            y_off = bar_y + 4

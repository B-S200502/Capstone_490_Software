#!/usr/bin/env python3
"""
Acoustic Imager - Fermat Spiral Heatmap (Interactive Bandpass Demo)

Differences vs heatmap_fermat_multisrc.py:
  1) Infinite loop (no FFmpeg recording, live demo only)
  2) Interactive band-pass filter:
       - Two OpenCV sliders: f_min_kHz and f_max_kHz (0–45 kHz)
       - Only frequencies within [f_min, f_max] contribute to the heatmap
       - Right-side frequency bar shows spectrum and highlights the selected band
  3) NEW: Interactive aperture radius slider ("aperture_mm")
       - Controls Fermat spiral radius in millimeters (safe range: 10–35 mm)
       - Mic positions are recomputed and mic signals re-simulated each frame
       - Blob width scales with aperture radius (smaller radius → fatter blobs)
"""

import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import cv2
from numpy.linalg import eigh, pinv, eig

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # type: ignore

from heatmap_pipeline_test import (  # type: ignore
    apply_heatmap_overlay,
    create_background_frame,
)

# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 150000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005
WINDOW_NAME = "Fermat Heatmap + Bandpass (MUSIC + Aperture Slider)"

# Display config
WIDTH = 1024
HEIGHT = 600
FPS = 30
ALPHA = 0.7

# Right-side frequency bar width (pixels)
FREQ_BAR_WIDTH = 200

# Beamforming / scanning grid
ANGLES = np.linspace(-90, 90, 181)  # 1° resolution

# Multiple deterministic sources
SOURCE_FREQS = [9000, 11000, 30000]  # Hz
SOURCE_ANGLES = [-35.0, 0.0, 40.0]   # degrees (initial)
SOURCE_AMPLS = [0.6, 1.0, 2.0]       # relative amplitudes
N_SOURCES = len(SOURCE_ANGLES)

# Frequency axis for FFT
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)

# Max frequency shown / controlled (Hz)
F_DISPLAY_MAX = 45000.0
ABSOLUTE_MAX_POWER = 1e-12

# ===============================================================
# 2. Geometry setup (Fermat spiral, dynamic aperture)
# ===============================================================
golden_angle = np.deg2rad(137.5)

# SAFE RANGE: radius 10 mm → 35 mm
APERTURE_MIN_MM = 10   # 1.0 cm radius
APERTURE_MAX_MM = 100   # 3.5 cm radius

# Default radius: 25 mm (5 cm diameter)
aperture_radius = 0.025  # meters

# Geometry globals (will be filled by update_geometry)
x_coords = np.zeros(N_MICS, dtype=np.float64)
y_coords = np.zeros(N_MICS, dtype=np.float64)
pitch = 0.0
c_geom = 0.0


def update_geometry(radius_m: float) -> None:
    """
    Recompute Fermat spiral microphone coordinates and pitch
    for a given aperture radius (in meters).
    """
    global aperture_radius, x_coords, y_coords, pitch, c_geom

    aperture_radius = radius_m
    c_geom = aperture_radius / np.sqrt(N_MICS - 1)

    xs, ys = [], []
    for n in range(N_MICS):
        r = c_geom * np.sqrt(n)
        theta = n * golden_angle
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))

    x_coords = np.array(xs, dtype=np.float64)
    y_coords = np.array(ys, dtype=np.float64)

    # Approximate average radial spacing, used by ESPRIT
    radii = np.sqrt(x_coords**2 + y_coords**2)
    radii_sorted = np.sort(np.unique(radii))
    if len(radii_sorted) > 1:
        pitch_vals = np.diff(radii_sorted)
        pitch_mean = np.mean(pitch_vals)
    else:
        pitch_mean = 0.0
    pitch = pitch_mean


# Initialize geometry once
update_geometry(aperture_radius)


# ===============================================================
# 3. Multi-source STM32 Frame Generator (deterministic)
# ===============================================================
def generate_fft_frame_from_dataframe(angle_degs: List[float]) -> FFTFrame:
    """
    Simulate one STM32 FFT frame with multiple plane-wave sources,
    using the CURRENT mic geometry (x_coords, y_coords).
    """
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    # Combine all sources with appropriate delays per mic
    for src_idx, angle_deg in enumerate(angle_degs):
        angle_rad = np.deg2rad(angle_deg)
        f = SOURCE_FREQS[src_idx]
        amp = SOURCE_AMPLS[src_idx]

        for i in range(N_MICS):
            delay = -(x_coords[i] * np.cos(angle_rad) +
                      y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
            delayed_t = t - delay
            mic_signals[i, :] += amp * np.sin(2 * np.pi * f * delayed_t)

    # Add noise once
    mic_signals += np.random.normal(0, np.sqrt(NOISE_POWER), mic_signals.shape)

    # Convert to FFT domain (per-mic)
    fft_data = np.fft.rfft(mic_signals, axis=1)
    frame.fft_data = fft_data.astype(np.complex64)
    return frame


# ===============================================================
# 4. Beamforming (MUSIC / ESPRIT)
# ===============================================================
def music_spectrum(R: np.ndarray,
                   angles: np.ndarray,
                   f_signal: float,
                   n_sources: int) -> np.ndarray:
    """
    MUSIC spectrum vs. angle for a given covariance matrix.

    NOTE: We DO NOT normalize the spectrum here (no / max),
    so array geometry changes (aperture_radius) are not hidden.
    """
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]  # noise subspace

    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(
            -1j * 2 * np.pi * f_signal / SPEED_SOUND *
            -(x_coords * np.cos(theta) + y_coords * np.sin(theta))
        )
        a = a[:, np.newaxis]
        P = 1.0 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

    # NO NORMALIZATION HERE
    return np.array(spectrum, dtype=np.float64)


def esprit_estimate(R: np.ndarray,
                    f_signal: float,
                    n_sources: int) -> np.ndarray:
    """ESPIRIT DOA estimate (not used for heatmap, but handy to keep)."""
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch if pitch != 0 else 1.0)
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)


# ===============================================================
# 5. Heatmap mapping (absolute dB, aperture-sensitive blobs)
# ===============================================================
def spectra_to_heatmap_absolute(spec_matrix: np.ndarray,
                                power_per_source: np.ndarray,
                                out_width: int,
                                out_height: int,
                                db_min: float = -40.0,
                                db_max: float = 0.0) -> np.ndarray:
    """
    Convert MUSIC + absolute power into an absolute dB-scaled heatmap.

    - No per-frame normalization
    - Amplitude comes ONLY from absolute |Xf|^2 converted to dB
    - MUSIC determines ANGLE and sharpness
    - Blob width also depends on aperture_radius (smaller radius → wider blobs)
    """

    Nsrc, Nang = spec_matrix.shape

    # 1) Absolute power → dB → [0,1]
    power_abs = np.maximum(power_per_source, 1e-12)
    power_db = 10 * np.log10(power_abs)
    power_norm = (power_db - db_min) / (db_max - db_min + 1e-12)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    # 2) Peak locations and local sharpness
    peak_indices = []
    sharpness = []

    for i in range(Nsrc):
        row = spec_matrix[i]

        idx = int(np.argmax(row))
        peak_indices.append(idx)

        p = row[idx]
        left = row[idx - 1] if idx > 0 else p
        right = row[idx + 1] if idx < Nang - 1 else p
        sh = max(p - 0.5 * (left + right), 1e-12)
        sharpness.append(sh)

    sharpness = np.array(sharpness, dtype=np.float64)
    # Normalize SHAPE only across sources, not across frames
    sharpness /= (sharpness.max() + 1e-12)

    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # 3) Aperture-dependent base radius
    # Use nominal radius = 0.025 m → base ~ 60 px
    nominal_R = 0.025
    base_radius = int(60.0 * (nominal_R / max(aperture_radius, 1e-6)))
    base_radius = int(np.clip(base_radius, 10, 140))

    def ang_to_px(idx: int) -> int:
        return int(idx / max(Nang - 1, 1) * (w - 1))

    # 4) Draw Gaussian blobs
    for i in range(Nsrc):
        cx = ang_to_px(peak_indices[i])
        cy = h // 2

        # Blob width: narrower for large apertures, wider for small apertures
        blob_radius = base_radius * (0.7 + 0.3 * sharpness[i])
        sigma = blob_radius / 1.8

        amp = power_norm[i]  # 0..1 from dB

        blob = amp * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)
        )

        heatmap += blob

    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    return heatmap_u8


def spectra_to_heatmap(spec_matrix: np.ndarray,
                       power_per_source: np.ndarray,
                       out_width: int,
                       out_height: int) -> np.ndarray:
    """
    Legacy heatmap (kept for reference, not used in this script).
    """
    Nsrc, Nang = spec_matrix.shape

    vmin = spec_matrix.min()
    vmax = spec_matrix.max()
    norm = (spec_matrix - vmin) / (vmax - vmin + 1e-12)

    gamma = 1.8
    norm = norm ** gamma

    music_strengths = []
    peak_indices = []

    for i in range(Nsrc):
        row = norm[i]
        peak = np.max(row)
        idx = np.argmax(row)

        peak_indices.append(idx)

        left = row[idx - 1] if idx > 0 else peak
        right = row[idx + 1] if idx < Nang - 1 else peak
        sharpness = max(peak - (left + right) / 2.0, 1e-12)

        music_strengths.append(peak * sharpness)

    music_strengths = np.array(music_strengths, dtype=np.float64)
    music_strengths /= music_strengths.max() + 1e-12

    power = power_per_source.astype(np.float64)
    power /= power.max() + 1e-12

    strengths = music_strengths * power
    strengths /= strengths.max() + 1e-12

    floor = 0.22
    strengths = floor + (1.0 - floor) * strengths

    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def ang_to_px(idx: int) -> int:
        return int(idx / max(Nang - 1, 1) * (w - 1))

    base_radius = 60

    for i in range(Nsrc):
        strength = strengths[i]
        peak_idx = peak_indices[i]

        cx = ang_to_px(peak_idx)
        cy = h // 2

        blob_radius = base_radius * (0.7 + 0.3 * strength)
        sigma = blob_radius / 2.0

        blob = strength * np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)
        )

        heatmap += blob

    heatmap -= heatmap.min()
    heatmap /= heatmap.max() + 1e-12
    heatmap = (heatmap * 255).astype(np.uint8)
    return heatmap


# ===============================================================
# 6. Frequency bar drawing (right side)
# ===============================================================
def draw_frequency_bar(frame: np.ndarray,
                       fft_data: np.ndarray,
                       f_axis: np.ndarray,
                       f_min: float,
                       f_max: float) -> None:
    """
    Draw a vertical frequency bar on the right side:
      - magnitude spectrum (avg across mics)
      - band [f_min, f_max] highlighted
    """
    h, w, _ = frame.shape
    bar_w = FREQ_BAR_WIDTH
    left = w - bar_w
    right = w

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    valid = f_axis <= F_DISPLAY_MAX
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    if mag_valid.size == 0:
        frame[:, left:right, :] = 0
        return

    mag_norm = mag_valid / (mag_valid.max() + 1e-12)
    mag_norm = mag_norm ** 0.4

    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    for f, m in zip(f_valid, mag_norm):
        y = int(h - 1 - (f / F_DISPLAY_MAX) * (h - 1))
        y = np.clip(y, 0, h - 1)

        length = int(m * (bar_w - 20))
        x0 = bar_w - 5 - length
        x1 = bar_w - 5

        if f_min <= f <= f_max:
            color = (0, 255, 255)
        else:
            color = (120, 120, 255)

        if length > 0:
            cv2.line(bar, (x0, y), (x1, y), color, 1)

    def freq_to_y(freq_hz: float) -> int:
        return int(h - 1 - (freq_hz / F_DISPLAY_MAX) * (h - 1))

    y_min = np.clip(freq_to_y(f_min), 0, h - 1)
    y_max = np.clip(freq_to_y(f_max), 0, h - 1)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), (0, 255, 0), 1)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), (0, 255, 0), 1)

    cv2.putText(bar, "Freq", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bar, "45 kHz", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bar, "0", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    frame[:, left:right, :] = bar


def draw_db_colorbar(frame: np.ndarray,
                     db_min: float,
                     db_max: float,
                     width: int = 50):
    """
    Draw a vertical colorbar using the same colormap as the heatmap.
    """
    h = frame.shape[0]
    bar = np.zeros((h, width), dtype=np.uint8)

    for y in range(h):
        val = y / (h - 1)
        bar[h - 1 - y, :] = int(val * 255)

    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_JET)

    frame[:, :width] = bar_color

    cv2.putText(frame, f"{db_max:.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{db_min:.0f} dB", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ===============================================================
# 7. Main loop (infinite, no FFmpeg)
# ===============================================================
def nothing(x):
    pass


CURSOR_POS = (0, 0)


def mouse_move(event, x, y, flags, param):
    global CURSOR_POS
    if event == cv2.EVENT_MOUSEMOVE:
        CURSOR_POS = (x, y)


def main():
    print("Acoustic Imager - Fermat Spiral Bandpass Demo (with Aperture Slider)")
    print("=" * 70)

    GLOBAL_DB_MIN = -60.0
    GLOBAL_DB_MAX = 0.0

    background_full = create_background_frame(WIDTH, HEIGHT)
    left_width = WIDTH - FREQ_BAR_WIDTH

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_move)

    # Band-pass sliders (kHz)
    cv2.createTrackbar("f_min_kHz", WINDOW_NAME, 0, 45, nothing)
    cv2.createTrackbar("f_max_kHz", WINDOW_NAME, 45, 45, nothing)

    # NEW: Aperture radius slider (mm)
    # Trackbar is 0..APERTURE_MAX_MM, but we clamp to [APERTURE_MIN_MM, APERTURE_MAX_MM]
    default_radius_mm = int(aperture_radius * 1000)
    default_radius_mm = int(np.clip(default_radius_mm, APERTURE_MIN_MM, APERTURE_MAX_MM))
    cv2.createTrackbar("aperture_mm", WINDOW_NAME, default_radius_mm, APERTURE_MAX_MM, nothing)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Read band-pass sliders
            f_min_khz = cv2.getTrackbarPos("f_min_kHz", WINDOW_NAME)
            f_max_khz = cv2.getTrackbarPos("f_max_kHz", WINDOW_NAME)
            if f_max_khz < f_min_khz:
                f_max_khz = f_min_khz

            f_min = max(0.0, min(F_DISPLAY_MAX, f_min_khz * 1000.0))
            f_max = max(0.0, min(F_DISPLAY_MAX, f_max_khz * 1000.0))

            # Read aperture slider and update geometry if needed
            slider_val = cv2.getTrackbarPos("aperture_mm", WINDOW_NAME)
            slider_val = int(np.clip(slider_val, APERTURE_MIN_MM, APERTURE_MAX_MM))
            radius_m = slider_val / 1000.0

            if abs(radius_m - aperture_radius) > 1e-6:
                update_geometry(radius_m)

            # Animate sources across FoV
            for k in range(N_SOURCES):
                SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
                if SOURCE_ANGLES[k] > 90.0:
                    SOURCE_ANGLES[k] = -90.0

            # Generate frame with current geometry
            frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

            # Build spec_matrix only for freqs within band
            selected_indices = [
                i for i, f in enumerate(SOURCE_FREQS)
                if f_min <= f <= f_max
            ]

            if not selected_indices:
                heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                LOCAL_DB_MIN = -60.0
                LOCAL_DB_MAX = 0.0
            else:
                n_sel = len(selected_indices)
                spec_matrix = np.zeros((n_sel, len(ANGLES)), dtype=np.float32)
                power_per_source = np.zeros(n_sel, dtype=np.float32)

                for row_idx, src_idx in enumerate(selected_indices):
                    f_sig = SOURCE_FREQS[src_idx]
                    f_idx = int(np.argmin(np.abs(f_axis - f_sig)))
                    Xf = frame.fft_data[:, f_idx][:, np.newaxis]
                    R = Xf @ Xf.conj().T

                    spec = music_spectrum(R, ANGLES, f_sig, n_sources=n_sel)
                    spec_matrix[row_idx, :] = spec

                    power = np.sum(np.abs(Xf) ** 2).real
                    power_per_source[row_idx] = power

                    global ABSOLUTE_MAX_POWER
                    ABSOLUTE_MAX_POWER = max(ABSOLUTE_MAX_POWER, power)

                    # Update global auto-range
                    p_db = 10 * np.log10(power / (ABSOLUTE_MAX_POWER + 1e-12))
                    GLOBAL_DB_MIN = min(GLOBAL_DB_MIN, p_db)
                    GLOBAL_DB_MAX = max(GLOBAL_DB_MAX, p_db)

                heatmap_left = spectra_to_heatmap_absolute(
                    spec_matrix,
                    power_per_source / (ABSOLUTE_MAX_POWER + 1e-12),
                    left_width,
                    HEIGHT,
                    db_min=GLOBAL_DB_MIN,
                    db_max=GLOBAL_DB_MAX
                )

                # Local dB range for colorbar
                p_abs = np.maximum(power_per_source, 1e-12)
                p_db_local = 10 * np.log10(p_abs)
                LOCAL_DB_MIN = float(np.min(p_db_local))
                LOCAL_DB_MAX = float(np.max(p_db_local))

            # Compose background and overlay heatmap
            background = background_full.copy()
            left_bg = background[:, :left_width, :]
            left_out = apply_heatmap_overlay(heatmap_left, left_bg, ALPHA)
            background[:, :left_width, :] = left_out
            output_frame = background

            # Frequency bar on the right
            draw_frequency_bar(output_frame, frame.fft_data, f_axis, f_min, f_max)

            # dB colorbar on the left
            draw_db_colorbar(output_frame,
                             db_min=LOCAL_DB_MIN,
                             db_max=LOCAL_DB_MAX)

            # Overlays (text)
            elapsed = time.time() - start_time
            cv2.putText(output_frame, f"Frame: {frame_count}",
                        (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(output_frame, f"t = {elapsed:.2f}s",
                        (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            fs_text = f"Fs: {SAMPLE_RATE_HZ} Hz"
            (tw, _), _ = cv2.getTextSize(fs_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(output_frame, fs_text,
                        (left_width - tw - 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Aperture text overlay (radius & diameter)
            diam_cm = 2 * aperture_radius * 100.0
            ap_text = f"R = {aperture_radius*1000:.1f} mm  (D = {diam_cm:.1f} cm)"
            cv2.putText(output_frame, ap_text,
                        (100, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 0), 2)

            angle_str = " | ".join(
                f"{f/1000:.1f} kHz: {ang:.1f} deg"
                for f, ang in zip(SOURCE_FREQS, SOURCE_ANGLES)
            )
            cv2.putText(output_frame, angle_str,
                        (100, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

            # Crosshair + tooltip
            cx, cy = CURSOR_POS
            cv2.drawMarker(output_frame, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=1)

            tooltip = ""
            if cx < left_width:
                ang = (cx / left_width) * 180.0 - 90.0

                px_db = LOCAL_DB_MIN + (heatmap_left[cy, cx] / 255.0) * (LOCAL_DB_MAX - LOCAL_DB_MIN)

                tooltip = f"| {ang:.1f} deg | {px_db:.1f} dB "

                if selected_indices:
                    source_angles = []
                    for row in range(spec_matrix.shape[0]):
                        idx = int(np.argmax(spec_matrix[row]))
                        source_angles.append(ANGLES[idx])
                    nearest_idx = int(np.argmin(np.abs(np.array(source_angles) - ang)))
                    tooltip += f"| {SOURCE_FREQS[selected_indices[nearest_idx]]/1000:.1f} kHz |"

            cv2.putText(output_frame, tooltip,
                        (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            # Show
            cv2.imshow(WINDOW_NAME, output_frame)

            key = cv2.waitKey(int(1000 // FPS)) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break
            if key == ord("q") or key == 27:
                print("Quit requested by user.")
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    finally:
        print("Cleaning up...")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

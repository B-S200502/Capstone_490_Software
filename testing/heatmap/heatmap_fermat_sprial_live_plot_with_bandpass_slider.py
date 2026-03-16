#!/usr/bin/env python3
"""
Acoustic Imager - Fermat Spiral Heatmap (Interactive Bandpass Demo)

Differences vs heatmap_fermat_multisrc.py:
  1) Infinite loop (no FFmpeg recording, live demo only)
  2) Interactive band-pass filter:
       - Two OpenCV sliders: f_min_kHz and f_max_kHz (0–45 kHz)
       - Only frequencies within [f_min, f_max] contribute to the heatmap
       - Right-side frequency bar shows spectrum and highlights the selected band
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
WINDOW_NAME = "Fermat Heatmap + Bandpass (MUSIC)"

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

# ===============================================================
# 2. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.025  # 5 cm radius (~10 cm diameter)
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

pitch = np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2)))))


# ===============================================================
# 3. Multi-source STM32 Frame Generator (deterministic)
# ===============================================================
def generate_fft_frame_from_dataframe(angle_degs: List[float]) -> FFTFrame:
    """
    Simulate one STM32 FFT frame with multiple plane-wave sources.
    """
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    # Combine all sources
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
    """MUSIC spectrum vs. angle for a given covariance matrix."""
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

    spec = np.array(spectrum)
    spec /= np.max(spec) + 1e-12
    return spec


def esprit_estimate(R: np.ndarray,
                    f_signal: float,
                    n_sources: int) -> np.ndarray:
    """ESPIRIT DOA estimate (not used for heatmap, but handy to keep)."""
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch)
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)


# ===============================================================
# 5. Heatmap mapping (same logic as your tuned version)
# ===============================================================
def spectra_to_heatmap(spec_matrix: np.ndarray,
                       power_per_source: np.ndarray,
                       out_width: int,
                       out_height: int) -> np.ndarray:
    """
    Convert MUSIC [Nsrc x Nangles] into circular blob-like heatmap.

    Blob intensity and size reflect a combination of:
      - MUSIC peak height / sharpness
      - actual signal power |Xf|^2 per source (across mics)
    """
    Nsrc, Nang = spec_matrix.shape

    vmin = spec_matrix.min()
    vmax = spec_matrix.max()
    norm = (spec_matrix - vmin) / (vmax - vmin + 1e-12)

    # gamma for visual contrast
    gamma = 1.8
    norm = norm ** gamma

    # MUSIC strength per source: peak * sharpness
    music_strengths = []
    peak_indices = []

    for i in range(Nsrc):
        row = norm[i]
        peak = np.max(row)
        idx = np.argmax(row)

        peak_indices.append(idx)

        left = row[idx-1] if idx > 0 else peak
        right = row[idx+1] if idx < Nang-1 else peak
        sharpness = max(peak - (left + right) / 2.0, 1e-12)

        music_strengths.append(peak * sharpness)

    music_strengths = np.array(music_strengths, dtype=np.float64)
    music_strengths /= music_strengths.max() + 1e-12

    # Fold in actual signal power
    power = power_per_source.astype(np.float64)
    power /= power.max() + 1e-12

    #TODO: logarithmic scaling for power?

    strengths = music_strengths * power
    strengths /= strengths.max() + 1e-12

    # Minimum visibility floor so weak sources don't vanish
    floor = 0.22
    strengths = floor + (1.0 - floor) * strengths

    # Build heatmap with Gaussian blobs
    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def ang_to_px(idx: int) -> int:
        return int(idx / Nang * w)

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

    # Compute average power across mics for each bin
    # fft_data shape: (N_MICS, N_bins)
    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    # Focus on 0..F_DISPLAY_MAX
    valid = f_axis <= F_DISPLAY_MAX
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    if mag_valid.size == 0:
        frame[:, left:right, :] = 0
        return

    # Normalize magnitudes (log-ish scaling)
    mag_norm = mag_valid / (mag_valid.max() + 1e-12)
    mag_norm = mag_norm ** 0.4  # flatten dynamic range a bit

    # Prepare bar region
    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)

    # For each freq bin, draw a horizontal line
    for f, m in zip(f_valid, mag_norm):
        # map frequency to vertical coordinate (0 = bottom, F_DISPLAY_MAX = top)
        y = int(h - 1 - (f / F_DISPLAY_MAX) * (h - 1))
        y = np.clip(y, 0, h - 1)

        # line length based on magnitude
        length = int(m * (bar_w - 20))
        x0 = bar_w - 5 - length
        x1 = bar_w - 5

        # color: highlight if inside band
        if f_min <= f <= f_max:
            color = (0, 255, 255)  # yellowish for in-band
        else:
            color = (120, 120, 255)  # bluish background spectrum

        if length > 0:
            cv2.line(bar, (x0, y), (x1, y), color, 1)

    # Draw bandpass lines
    def freq_to_y(freq_hz: float) -> int:
        return int(h - 1 - (freq_hz / F_DISPLAY_MAX) * (h - 1))

    y_min = np.clip(freq_to_y(f_min), 0, h - 1)
    y_max = np.clip(freq_to_y(f_max), 0, h - 1)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), (0, 255, 0), 1)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), (0, 255, 0), 1)

    # Some labels
    cv2.putText(bar, "Freq", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bar, "45 kHz", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bar, "0", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # Insert into right side of frame
    frame[:, left:right, :] = bar


# ===============================================================
# 7. Main loop (infinite, no FFmpeg)
# ===============================================================
def nothing(x):
    pass


def main():
    print("Acoustic Imager - Fermat Spiral Bandpass Demo")
    print("=" * 70)

    background_full = create_background_frame(WIDTH, HEIGHT)
    left_width = WIDTH - FREQ_BAR_WIDTH

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    # Interactive bandpass sliders (kHz)
    cv2.createTrackbar("f_min_kHz", WINDOW_NAME, 0, 45, nothing)
    cv2.createTrackbar("f_max_kHz", WINDOW_NAME, 45, 45, nothing)

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Read slider positions
            f_min_khz = cv2.getTrackbarPos("f_min_kHz", WINDOW_NAME)
            f_max_khz = cv2.getTrackbarPos("f_max_kHz", WINDOW_NAME)
            if f_max_khz < f_min_khz:
                f_max_khz = f_min_khz

            f_min = f_min_khz * 1000.0
            f_max = f_max_khz * 1000.0
            f_min = max(0.0, min(F_DISPLAY_MAX, f_min))
            f_max = max(0.0, min(F_DISPLAY_MAX, f_max))

            # Animate sources (slow sweep across FoV)
            for k in range(N_SOURCES):
                SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
                if SOURCE_ANGLES[k] > 90.0:
                    SOURCE_ANGLES[k] = -90.0

            frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

            # Build spec_matrix only for freqs within band
            selected_indices = [
                i for i, f in enumerate(SOURCE_FREQS)
                if f_min <= f <= f_max
            ]

            # Default: no sources in band => blank heatmap
            if not selected_indices:
                heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
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

                heatmap_full = spectra_to_heatmap(
                    spec_matrix, power_per_source, left_width, HEIGHT
                )
                heatmap_left = heatmap_full  # already correct size

            # Compose background and overlay heatmap on left region
            background = background_full.copy()
            left_bg = background[:, :left_width, :]
            left_out = apply_heatmap_overlay(heatmap_left, left_bg, ALPHA)
            background[:, :left_width, :] = left_out
            output_frame = background

            # Draw frequency bar on the right based on full FFT data
            draw_frequency_bar(output_frame, frame.fft_data, f_axis, f_min, f_max)

            # Overlays (text)
            elapsed = time.time() - start_time
            cv2.putText(
                output_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                output_frame,
                f"t = {elapsed:.2f}s",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            fs_text = f"Fs: {SAMPLE_RATE_HZ} Hz"
            (tw, _), _ = cv2.getTextSize(fs_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(
                output_frame,
                fs_text,
                (left_width - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            angle_str = " | ".join(
                f"{f/1000:.1f} kHz: {ang:.1f} deg"
                for f, ang in zip(SOURCE_FREQS, SOURCE_ANGLES)
            )
            cv2.putText(
                output_frame,
                angle_str,
                (10, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            # Show
            cv2.imshow(WINDOW_NAME, output_frame)

            # Handle key / window close
            key = cv2.waitKey(int(1000 // FPS)) & 0xFF

            # Try to detect X button (still keep 'q' as reliable exit)
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

    #TODO: add db scaler visible, separate from freq slider

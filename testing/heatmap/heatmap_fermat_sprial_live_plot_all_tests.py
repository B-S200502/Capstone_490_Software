#!/usr/bin/env python3
"""
Acoustic Imager - Fermat Spiral Heatmap (Interactive Bandpass Demo)
with Aperture Slider + Realistic Imperfections + Audio Test WAVs (Option B)

New vs previous version:
  - 7 audio test WAVs (from data/signals/) can be toggled ON/OFF individually.
  - Each active WAV is treated as an extra plane-wave source:
        * signal injected at its own DOA angle
        * per frame we find its dominant frequency
        * MUSIC runs at that bin, so it appears as an extra blob
  - Original 3 deterministic tones stay present at all times.
"""

import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
from numpy.linalg import eigh, pinv, eig
from dataclasses import dataclass

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame  # type: ignore

from heatmap_pipeline_test import (  # type: ignore
    apply_heatmap_overlay,
    create_background_frame,
)

# ---------------------------------------------------------------------
# Audio loading (requires `soundfile`: pip install soundfile)
# ---------------------------------------------------------------------
import soundfile as sf  # type: ignore


# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 150000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005
WINDOW_NAME = "Fermat Heatmap + Bandpass (MUSIC + Realistic Aperture + WAVs)"

# Display config
WIDTH = 1024
HEIGHT = 600
FPS = 30
ALPHA = 0.7

# Right-side frequency bar width (pixels)
FREQ_BAR_WIDTH = 200

# Beamforming / scanning grid
ANGLES = np.linspace(-90, 90, 181)  # 1° resolution

# Multiple deterministic sources (your "true" sources)
SOURCE_FREQS = [9000, 11000, 30000]  # Hz
SOURCE_ANGLES = [-35.0, 0.0, 40.0]   # degrees (initial)
SOURCE_AMPLS = [0.6, 1.0, 2.0]       # relative amplitudes
N_SOURCES = len(SOURCE_ANGLES)

# Extra weak clutter sources (random each frame)
N_CLUTTER = 5
CLUTTER_LEVEL = 0.15
CLUTTER_FMIN = 5000.0
CLUTTER_FMAX = 40000.0

# Frequency axis for FFT (for MUSIC bins)
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)

# Max frequency shown / controlled (Hz)
F_DISPLAY_MAX = 45000.0
ABSOLUTE_MAX_POWER = 1e-12

# Audio injection level relative to deterministic tones
AUDIO_LEVEL = 0.8

# Root for WAV files (…/data/signals)
AUDIO_ROOT = Path(__file__).resolve().parents[2] / "data" / "signals"

BUTTON_Y = HEIGHT - 50   # bottom bar

BUTTONS = [
    {"name": "SweepLin",      "x": 60,   "y": BUTTON_Y, "w": 110, "h": 30, "state": False},
    {"name": "SweepLog",      "x": 180,  "y": BUTTON_Y, "w": 110, "h": 30, "state": False},
    {"name": "Aliasing44100", "x": 300,  "y": BUTTON_Y, "w": 140, "h": 30, "state": False},
    {"name": "Aliasing48000", "x": 450,  "y": BUTTON_Y, "w": 140, "h": 30, "state": False},
    {"name": "White",         "x": 600,  "y": BUTTON_Y, "w": 90,  "h": 30, "state": False},
    {"name": "Impulse",       "x": 700,  "y": BUTTON_Y, "w": 110, "h": 30, "state": False},
]



BUTTON_TEST_MAP = {
    "SweepLin": "sweep_lin",
    "SweepLog": "sweep_log",
    "Aliasing44100": "Aliasing44100",
    "Aliasing48000": "Aliasing48000",
    "White": "white",
    "Impulse": "impulse",
}


def update_test_states():
    # Disable all by default
    for test in AUDIO_TESTS:
        test.enabled = False

    for btn in BUTTONS:
        if not btn["state"]:
            continue

        target = BUTTON_TEST_MAP.get(btn["name"])
        if not target:
            continue

        for test in AUDIO_TESTS:
            if test.name == target:
                if test.data is not None:
                    test.enabled = True


# ===============================================================
# 1.1 Audio test description
# ===============================================================
@dataclass
class AudioTest:
    name: str
    filename: str
    angle_deg: float
    enabled: bool = False
    data: Optional[np.ndarray] = None
    samplerate: int = 0
    pos: int = 0              # current read index (in samples)
    last_peak_freq: Optional[float] = None  # Hz in this frame


AUDIO_TESTS: List[AudioTest] = [
    AudioTest("Aliasing44100",  "audiocheck.net_aliasingcheck_44100.wav",      -60.0),
    AudioTest("Aliasing48000",   "audiocheck.net_aliasingcheck_48000.wav",      -30.0),
    AudioTest("sweep_lin",   "audiocheck.net_sweep20-20klin.wav",             0.0),
    AudioTest("sweep_log",   "audiocheck.net_sweep20-20klog.wav",            20.0),
    AudioTest("impulse",     "audiocheck.net_impulse.wav",                   40.0),
    AudioTest("white",       "audiocheck.net_whitenoise.wav",                60.0),
    AudioTest("white_gauss", "audiocheck.net_whitenoisegaussian.wav",        80.0),
]


def load_and_resample_wav(path: Path, target_sr: int) -> Optional[np.ndarray]:
    """
    Load a WAV file and resample to target_sr with simple linear interpolation.
    Returns float32 mono signal in [-1, 1], or None on failure.
    """
    try:
        data, sr = sf.read(str(path), always_2d=False)
    except Exception as e:
        print(f"WARNING: failed to load {path}: {e}")
        return None

    # Ensure mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)
    peak = np.max(np.abs(data)) or 1.0
    data /= peak  # normalize to [-1,1]

    if sr == target_sr:
        return data

    # Resample to target_sr
    duration = len(data) / sr
    t_orig = np.linspace(0.0, duration, num=len(data), endpoint=False)
    t_new = np.linspace(0.0, duration, num=int(duration * target_sr), endpoint=False)
    data_rs = np.interp(t_new, t_orig, data).astype(np.float32)
    return data_rs


def init_audio_tests() -> None:
    """
    Load all WAVs into AUDIO_TESTS (if present).
    """
    print("Loading audio test WAVs...")
    for test in AUDIO_TESTS:
        path = AUDIO_ROOT / test.filename
        if not path.exists():
            print(f"  [MISSING] {path}")
            continue
        sig = load_and_resample_wav(path, SAMPLE_RATE_HZ)
        if sig is None:
            continue
        test.data = sig
        test.samplerate = SAMPLE_RATE_HZ
        test.pos = 0
        print(f"  [OK] {test.name}: {path} ({len(sig)} samples @ {SAMPLE_RATE_HZ} Hz)")


# ===============================================================
# 2. Geometry setup (Fermat spiral, dynamic aperture)
# ===============================================================
golden_angle = np.deg2rad(137.5)

# Slider range: radius 10 mm → 100 mm = 1–10 cm (2–20 cm diameter)
APERTURE_MIN_MM = 10    # 1.0 cm radius
APERTURE_MAX_MM = 100   # 10.0 cm radius

# Default radius: 25 mm (5 cm diameter)
aperture_radius = 0.025  # meters

# Geometry globals (model vs true)
x_model = np.zeros(N_MICS, dtype=np.float64)   # ideal design used by MUSIC
y_model = np.zeros(N_MICS, dtype=np.float64)
x_true = np.zeros(N_MICS, dtype=np.float64)    # “real” hardware positions
y_true = np.zeros(N_MICS, dtype=np.float64)
pitch = 0.0
c_geom = 0.0

# --- Fixed hardware imperfections (do NOT change with slider) ---

# Position jitter (meters) ~ PCB / solder tolerance
POS_JITTER_STD_M = 0.0002  # 0.2 mm
dx_err = np.random.normal(0.0, POS_JITTER_STD_M, N_MICS)
dy_err = np.random.normal(0.0, POS_JITTER_STD_M, N_MICS)

# Per-mic gain mismatch
MIC_GAIN_STD = 0.15  # ≈ ±15%
mic_gain = 1.0 + np.random.normal(0.0, MIC_GAIN_STD, N_MICS)
mic_gain = np.clip(mic_gain, 0.5, 1.5)

# Per-mic additional timing jitter (seconds)
PHASE_JITTER_US = 4.0
mic_delay_jitter = np.random.normal(0.0, PHASE_JITTER_US * 1e-6, N_MICS)


def update_geometry(radius_m: float) -> None:
    """
    Recompute Fermat spiral microphone coordinates and pitch
    for a given *design* aperture radius (in meters).

    x_model, y_model: ideal design used in MUSIC steering vector.
    x_true, y_true:  jittered physical positions used to generate signals.
    """
    global aperture_radius, x_model, y_model, x_true, y_true, pitch, c_geom

    aperture_radius = radius_m
    c_geom = aperture_radius / np.sqrt(N_MICS - 1)

    xs, ys = [], []
    for n in range(N_MICS):
        r = c_geom * np.sqrt(n)
        theta = n * golden_angle
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))

    x_model = np.array(xs, dtype=np.float64)
    y_model = np.array(ys, dtype=np.float64)

    # "True" positions include fixed jitter
    x_true = x_model + dx_err
    y_true = y_model + dy_err

    # Approximate average radial spacing, used by ESPRIT
    radii = np.sqrt(x_model**2 + y_model**2)
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
# 3. Multi-source STM32 Frame Generator (tones + clutter + WAVs)
# ===============================================================
def _advance_audio_segment(test: AudioTest, frame_len: int) -> np.ndarray:
    """
    Return next `frame_len` samples for this test, wrapping around if needed.
    Assumes test.data is not None and already at SAMPLE_RATE_HZ.
    """
    assert test.data is not None
    data = test.data
    N = len(data)

    if N == 0:
        return np.zeros(frame_len, dtype=np.float32)

    start = test.pos
    end = start + frame_len
    if end <= N:
        seg = data[start:end]
        test.pos = end % N
        return seg.astype(np.float32)

    # wrap
    part1 = data[start:]
    remaining = frame_len - len(part1)
    part2 = data[:remaining]
    seg = np.concatenate([part1, part2]).astype(np.float32)
    test.pos = remaining % N
    return seg


def generate_fft_frame_from_dataframe(angle_degs: List[float]) -> FFTFrame:
    """
    Simulate one STM32 FFT frame with:
      - deterministic tones (SOURCE_FREQS at SOURCE_ANGLES)
      - weak clutter
      - active audio-test WAVs (as plane waves),
        including per-frame dominant frequency tracking.
    """
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    # ------------------------------------------------------------------
    # Main deterministic tones
    # ------------------------------------------------------------------
    for src_idx, angle_deg in enumerate(angle_degs):
        angle_rad = np.deg2rad(angle_deg)
        f = SOURCE_FREQS[src_idx]
        base_amp = SOURCE_AMPLS[src_idx]

        for i in range(N_MICS):
            geom_delay = -(x_true[i] * np.cos(angle_rad) +
                           y_true[i] * np.sin(angle_rad)) / SPEED_SOUND
            total_delay = geom_delay + mic_delay_jitter[i]
            mic_signals[i, :] += (
                mic_gain[i] * base_amp *
                np.sin(2 * np.pi * f * (t - total_delay))
            )

    # ------------------------------------------------------------------
    # Audio tests (each as a separate plane-wave source)
    # ------------------------------------------------------------------
    for test in AUDIO_TESTS:
        if not test.enabled or test.data is None:
            test.last_peak_freq = None
            continue

        seg = _advance_audio_segment(test, SAMPLES_PER_CHANNEL)  # mono, float32

        # Estimate dominant frequency in this frame
        audio_fft = np.fft.rfft(seg)
        mag = np.abs(audio_fft)
        valid = f_axis <= F_DISPLAY_MAX
        mag_valid = mag[valid]
        if mag_valid.size == 0 or np.all(mag_valid == 0):
            test.last_peak_freq = None
        else:
            idx_peak = int(np.argmax(mag_valid))
            test.last_peak_freq = float(f_axis[valid][idx_peak])

        # Inject as plane wave at test.angle_deg
        angle_rad = np.deg2rad(test.angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for i in range(N_MICS):
            geom_delay = -(x_true[i] * cos_a +
                           y_true[i] * sin_a) / SPEED_SOUND
            total_delay = geom_delay + mic_delay_jitter[i]

            # Fractional delay via interpolation
            shifted = np.interp(t - total_delay, t, seg, left=0.0, right=0.0)
            mic_signals[i, :] += mic_gain[i] * AUDIO_LEVEL * shifted

    # ------------------------------------------------------------------
    # Weak random clutter plane waves (new each frame)
    # ------------------------------------------------------------------
    for _ in range(N_CLUTTER):
        angle_rad = np.deg2rad(np.random.uniform(-90.0, 90.0))
        f_c = np.random.uniform(CLUTTER_FMIN, CLUTTER_FMAX)
        amp_c = CLUTTER_LEVEL

        cos_c = np.cos(angle_rad)
        sin_c = np.sin(angle_rad)

        for i in range(N_MICS):
            geom_delay = -(x_true[i] * cos_c +
                           y_true[i] * sin_c) / SPEED_SOUND
            total_delay = geom_delay + mic_delay_jitter[i]
            mic_signals[i, :] += (
                mic_gain[i] * amp_c *
                np.sin(2 * np.pi * f_c * (t - total_delay))
            )

    # Add white noise
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
    so geometry changes (aperture_radius) and imperfections are visible.
    Steering uses the *ideal* design coordinates x_model, y_model.
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
            -(x_model * np.cos(theta) + y_model * np.sin(theta))
        )
        a = a[:, np.newaxis]
        P = 1.0 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

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
    denom = 2 * np.pi * f_signal * (pitch if pitch != 0 else 1.0)
    val = -(psi * SPEED_SOUND) / denom
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
    sharpness /= (sharpness.max() + 1e-12)

    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    # 3) Aperture-dependent base radius: nominal 0.025 m → ~60 px
    nominal_R = 0.025
    base_radius = int(60.0 * (nominal_R / max(aperture_radius, 1e-6)))
    base_radius = int(np.clip(base_radius, 10, 140))

    def ang_to_px(idx: int) -> int:
        return int(idx / max(Nang - 1, 1) * (w - 1))

    # 4) Draw Gaussian blobs
    for i in range(Nsrc):
        cx = ang_to_px(peak_indices[i])
        cy = h // 2

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

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    valid = f_axis <= F_DISPLAY_MAX
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    if mag_valid.size == 0:
        frame[:, left:, :] = 0
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

    frame[:, left:, :] = bar


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
    CURSOR_POS = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        for btn in BUTTONS:
            bx, by, bw, bh = btn["x"], btn["y"], btn["w"], btn["h"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                btn["state"] = not btn["state"]
                print(f"{btn['name']} toggled -> {btn['state']}")



def draw_buttons(frame):
    """Draw clickable buttons."""
    for btn in BUTTONS:
        x, y, w, h = btn["x"], btn["y"], btn["w"], btn["h"]

        # Color: green = ON, red = OFF
        color = (0, 200, 0) if btn["state"] else (0, 0, 200)

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Text label centered
        text = f"[ {btn['name']} ]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2
        cv2.putText(frame, text, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def main():
    print("Acoustic Imager - Fermat Spiral Bandpass Demo (Realistic + WAVs)")
    print("=" * 70)

    init_audio_tests()

    GLOBAL_DB_MIN = -60.0
    GLOBAL_DB_MAX = 0.0

    background_full = create_background_frame(WIDTH, HEIGHT)
    left_width = WIDTH - FREQ_BAR_WIDTH

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_move)

    # Band-pass sliders (kHz)
    cv2.createTrackbar("f_min_kHz", WINDOW_NAME, 0, 45, nothing)
    cv2.createTrackbar("f_max_kHz", WINDOW_NAME, 45, 45, nothing)

    # Aperture radius slider (mm) – clamp to [APERTURE_MIN_MM, APERTURE_MAX_MM]
    default_radius_mm = int(aperture_radius * 1000)
    default_radius_mm = int(np.clip(default_radius_mm,
                                    APERTURE_MIN_MM, APERTURE_MAX_MM))
    cv2.createTrackbar("aperture_mm", WINDOW_NAME,
                       default_radius_mm, APERTURE_MAX_MM, nothing)

    # Seven 0/1 buttons (trackbars) for audio tests


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
            slider_val = int(np.clip(slider_val,
                                     APERTURE_MIN_MM, APERTURE_MAX_MM))
            radius_m = slider_val / 1000.0
            if abs(radius_m - aperture_radius) > 1e-6:
                update_geometry(radius_m)



            # Animate deterministic sources across FoV
            for k in range(N_SOURCES):
                SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
                if SOURCE_ANGLES[k] > 90.0:
                    SOURCE_ANGLES[k] = -90.0

            update_test_states()

            frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

            # ----------------------------------------------------------
            # Build MUSIC analysis list: tones + active audio tests
            # Each entry: (kind, idx, f_sig)
            # ----------------------------------------------------------
            analysis_items = []
            analysis_freqs = []

            # Deterministic tones
            for i, f in enumerate(SOURCE_FREQS):
                if f_min <= f <= f_max:
                    analysis_items.append(("tone", i, float(f)))
                    analysis_freqs.append(float(f))

            # Audio tests
            for i, test in enumerate(AUDIO_TESTS):
                f_peak = test.last_peak_freq
                if (test.enabled and f_peak is not None and
                        f_min <= f_peak <= f_max):
                    analysis_items.append(("audio", i, float(f_peak)))
                    analysis_freqs.append(float(f_peak))

            if not analysis_items:
                heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                LOCAL_DB_MIN = -60.0
                LOCAL_DB_MAX = 0.0
            else:
                n_sel = len(analysis_items)
                spec_matrix = np.zeros((n_sel, len(ANGLES)), dtype=np.float32)
                power_per_source = np.zeros(n_sel, dtype=np.float32)

                for row_idx, (_, _, f_sig) in enumerate(analysis_items):
                    f_sig = float(f_sig)
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

            # Crosshair + tooltip
            cx, cy = CURSOR_POS
            cv2.drawMarker(output_frame, (cx, cy), (255, 255, 255),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=12, thickness=1)

            tooltip = ""
            if cx < left_width and analysis_items:
                ang = (cx / left_width) * 180.0 - 90.0
                px_db = LOCAL_DB_MIN + (heatmap_left[cy, cx] / 255.0) * (LOCAL_DB_MAX - LOCAL_DB_MIN)
                tooltip = f"| {ang:.1f} deg | {px_db:.1f} dB "

                # pick nearest MUSIC peak in angle to cursor
                source_angles = []
                for row in range(spec_matrix.shape[0]):
                    idx = int(np.argmax(spec_matrix[row]))
                    source_angles.append(ANGLES[idx])
                nearest_idx = int(np.argmin(np.abs(np.array(source_angles) - ang)))
                tooltip += f"| {analysis_freqs[nearest_idx]/1000:.1f} kHz |"

            cv2.putText(output_frame, tooltip,
                        (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)


            draw_buttons(output_frame)
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

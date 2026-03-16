#!/usr/bin/env python3
"""
Acoustic Imager - Fermat Spiral Heatmap (SIM + SPI selectable) + Interactive Bandpass

LOOPBACK VERSION (no STM32 required)

SPI mode behavior:
  - We generate a full framed packet locally (header + payload + crc + end magic)
  - We transmit it on MOSI and read it back on MISO in the same transfer
  - This works as a loopback test ONLY if MOSI is physically connected to MISO.

Payload format (unchanged):
  - float32 MAG + float32 PHASE (polar) per mic/bin
  - 8 bytes per bin per mic
  - We convert to complex64 internally.

THIS REV:
  1) SPI “virtual sources” are FEWER, STABLE, and have DIFFERENT magnitudes,
     plus 2 extra detections in upper frequencies.
  2) SPI stats remain STACKED vertically in first column (+ FPS + Throughput).
  3) Top-right MENU dropdown:
       - Segmented FPS selector: 30 / 60 / MAX
       - Gain Mode toggle (LOW / HIGH) placeholder
  4) SPI sources made more "sound-like" by spreading energy across neighboring bins.
  5) SPI heatmap made brighter (per-frame normalization + mild compression).

FIXES APPLIED IN THIS PRINT:
  #1) Camera capture moved to a "latest frame" worker thread (main loop never blocks on camera)
  #2) Background resize uses preallocated dst buffers (no cv2.resize allocation each frame)
"""

import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import struct
import zlib
import subprocess
import threading  # <-- FIX #1

import numpy as np
import cv2
from numpy.linalg import eigh, pinv, eig

# SPI optional
try:
    import spidev  # type: ignore
    SPIDEV_AVAILABLE = True
except ImportError:
    spidev = None
    SPIDEV_AVAILABLE = False

# Try picamera2
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    Picamera2 = None

# ---------------------------------------------------------------------
# Local imports
# ---------------------------------------------------------------------
try:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
    from fftframe import FFTFrame  # type: ignore

    from heatmap_pipeline_test import (  # type: ignore
        create_background_frame,
    )
    sys.path.append(str(Path(__file__).resolve().parents[2] / "utilities"))
    from stage_profiler import StageProfiler
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print(f"CWD: {Path.cwd()}")
    print(f"Script: {Path(__file__).resolve()}")
    sys.exit(1)

# ===============================================================
# 1. Configuration
# ===============================================================
WINDOW_NAME = "Fermat Heatmap (SIM + SPI LOOPBACK) + Bandpass (MUSIC)"
N_MICS = 16
SAMPLES_PER_CHANNEL = 512

SAMPLE_RATE_HZ = 100000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005

WIDTH = 1024
HEIGHT = 600
ALPHA = 0.7

FREQ_BAR_WIDTH = 200
DB_BAR_WIDTH = 50  # left colorbar

# FPS defaults (MENU controls 30/60/MAX at runtime)
FPS_TARGET = 60

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = WIDTH
CAMERA_HEIGHT = HEIGHT
USE_CAMERA = True
USE_LIBCAMERA = True

# Bandpass
F_DISPLAY_MAX = 45000.0
F_MIN_HZ = 0.0
F_MAX_HZ = F_DISPLAY_MAX

DRAG_ACTIVE = False
DRAG_TARGET = None  # "min" or "max"
DRAG_MARGIN_PX = 18

ANGLES = np.linspace(-90, 90, 181)

# SIM sources (unchanged)
SIM_SOURCE_FREQS = [9000, 11000, 30000]
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]
SIM_N_SOURCES = len(SIM_SOURCE_ANGLES)

f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 257

# dB mapping for heatmap intensity
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0

# ===============================================================
# 2. SPI frame format (float32 MAG + float32 PHASE)
# ===============================================================
MAGIC_START = 0x46524654  # 'TFRF' (example)
MAGIC_END = 0x454E4421    # 'END!' (example)
VERSION = 1

HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

TRAILER_FMT = "<II"  # crc32(u32), end_magic(u32)
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

PAYLOAD_LEN = N_MICS * N_BINS * 2 * 4   # 8 bytes per bin per mic
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN

# SPI config
SPI_BUS = 0
SPI_DEV = 0
SPI_MODE = 0
SPI_BITS = 8

SPI_MAX_SPEED_HZ = 80_000_000
SPI_XFER_CHUNK = 8192

# For debugging if FPS is too slow, we can pre-generate a static frame and reuse it.
STATIC_TX_FRAME = None

# --- LOOPBACK SPI "virtual sources" ---
SPI_SIM_BINS   = [35, 80, 160, 220]
SPI_SIM_AMPLS  = [6.0, 3.0, 5.0, 4.0]          # boosted
SPI_SIM_ANGLES = [-25.0, 35.0, -5.0, 60.0]
SPI_SIM_DRIFT_DEG_PER_SEC = [1.2, -0.6, 0.4, -0.3]

CRC_EVERY_N = 30

# ===============================================================
# 3. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.010
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

pitch = np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2))))) if N_MICS > 2 else 0.0

# ===============================================================
# 4. Beamforming (MUSIC / ESPRIT)
# ===============================================================
def music_spectrum(R: np.ndarray,
                   angles: np.ndarray,
                   f_signal: float,
                   n_sources: int) -> np.ndarray:
    """
    Vectorized MUSIC spectrum over all angles (no Python loop).
    Same output semantics: float32, normalized to max=1.
    """
    # Eigendecomposition
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]

    # Noise subspace
    En = eigvecs[:, n_sources:]              # (M, M-nsrc)
    Pn = En @ En.conj().T                    # (M, M) noise projector

    # Precompute geometry projection for all angles:
    # phase = j*k*(x cos + y sin) with k = 2πf/c
    theta = np.deg2rad(angles).astype(np.float32)          # (A,)
    cth = np.cos(theta).astype(np.float32)                 # (A,)
    sth = np.sin(theta).astype(np.float32)                 # (A,)

    proj = (x_coords[:, None] * cth[None, :] +
            y_coords[:, None] * sth[None, :]).astype(np.float32)  # (M, A)

    k = (2.0 * np.pi * float(f_signal) / SPEED_SOUND)
    A = np.exp(1j * k * proj).astype(np.complex64)         # (M, A)

    # MUSIC pseudo-spectrum:
    # P(θ) = 1 / Re(a^H Pn a)
    # Do it in a vectorized way for all angles:
    PA = Pn @ A                                            # (M, A)
    denom = np.einsum("ma,ma->a", A.conj(), PA).real        # (A,)
    denom = np.maximum(denom, 1e-12)

    spec = (1.0 / denom).astype(np.float32)

    # Normalize
    m = float(spec.max()) if spec.size else 1.0
    spec /= (m + 1e-12)
    return spec


def esprit_estimate(R: np.ndarray, f_signal: float, n_sources: int) -> np.ndarray:
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    if pitch == 0:
        return np.zeros(n_sources)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch)
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)

# ===============================================================
# 5. Heatmap mapping
# ===============================================================
def spectra_to_heatmap_absolute(spec_matrix: np.ndarray,
                                power_rel: np.ndarray,
                                out_width: int,
                                out_height: int,
                                db_min: float = REL_DB_MIN,
                                db_max: float = REL_DB_MAX) -> np.ndarray:
    """
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
    power_norm = (power_db - db_min) / (db_max - db_min + 1e-12)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    # Peak index + a simple "sharpness" estimate (still cheap)
    peak_idx = np.argmax(spec_matrix, axis=1).astype(np.int32)  # (Nsrc,)

    sharp = np.empty(Nsrc, dtype=np.float32)
    for i in range(Nsrc):
        idx = int(peak_idx[i])
        p = float(spec_matrix[i, idx])
        left = float(spec_matrix[i, idx - 1]) if idx > 0 else p
        right = float(spec_matrix[i, idx + 1]) if idx < (Nang - 1) else p
        sharp[i] = max(p - 0.5 * (left + right), 1e-12)

    sharp /= (sharp.max() + 1e-12)

    h, w = int(out_height), int(out_width)
    heat = np.zeros((h, w), dtype=np.float32)

    # map peak angle index -> x pixel (precompute)
    # note: clamp to [0, w-1]
    cx_all = (peak_idx.astype(np.float32) * (w - 1) / max(1, (Nang - 1))).astype(np.int32)
    cx_all = np.clip(cx_all, 0, w - 1)
    cy = h // 2

    base_radius = 60.0

    for i in range(Nsrc):
        amp = float(power_norm[i])
        if amp <= 0.0:
            continue

        cx = int(cx_all[i])

        blob_radius = base_radius * (0.7 + 0.3 * float(sharp[i]))
        sigma = blob_radius / 1.8
        sigma2 = float(2.0 * sigma * sigma + 1e-12)

        # ROI bounds (3*sigma is plenty)
        r = int(max(6, min(200, round(3.0 * sigma))))
        x0 = max(0, cx - r)
        x1 = min(w, cx + r + 1)
        y0 = max(0, cy - r)
        y1 = min(h, cy + r + 1)

        # build small grids
        xs = (np.arange(x0, x1, dtype=np.float32) - cx)
        ys = (np.arange(y0, y1, dtype=np.float32) - cy)

        # separable gaussian: exp(-(x^2+y^2)/2s^2) = exp(-x^2/2s^2)*exp(-y^2/2s^2)
        gx = np.exp(-(xs * xs) / sigma2)  # (Wx,)
        gy = np.exp(-(ys * ys) / sigma2)  # (Hy,)

        blob = amp * (gy[:, None] * gx[None, :]).astype(np.float32)
        heat[y0:y1, x0:x1] += blob

    # Normalize / clamp
    heat = np.clip(heat, 0.0, 1.0)
    return (heat * 255.0).astype(np.uint8)


# ===============================================================
# 6. Frequency bar + dB colorbar
# ===============================================================
def freq_to_y(freq_hz: float, h: int) -> int:
    freq_hz = float(np.clip(freq_hz, 0.0, F_DISPLAY_MAX))
    return int(h - 1 - (freq_hz / F_DISPLAY_MAX) * (h - 1))

def y_to_freq(y: int, h: int) -> float:
    y = int(np.clip(y, 0, h - 1))
    frac = 1.0 - (y / (h - 1))
    return float(np.clip(frac * F_DISPLAY_MAX, 0.0, F_DISPLAY_MAX))

def draw_frequency_bar(frame: np.ndarray,
                       fft_data: np.ndarray,
                       f_axis: np.ndarray,
                       f_min: float,
                       f_max: float) -> None:
    h, w, _ = frame.shape

    area_left = w - FREQ_BAR_WIDTH
    area_right = w

    # FULL WIDTH for freq bar (no reserved menu column)
    bar_w = FREQ_BAR_WIDTH
    bar_left = area_left
    bar_right = area_right

    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real

    valid = f_axis <= F_DISPLAY_MAX
    f_valid = f_axis[valid]
    mag_valid = mag2[valid]

    bar = np.zeros((h, bar_w, 3), dtype=np.uint8)
    if mag_valid.size > 0:
        mag_norm = mag_valid / (mag_valid.max() + 1e-12)
        mag_norm = mag_norm ** 0.4

        for f, m in zip(f_valid, mag_norm):
            y = int(h - 1 - (f / F_DISPLAY_MAX) * (h - 1))
            y = np.clip(y, 0, h - 1)
            length = int(m * (bar_w - 20))
            x0 = bar_w - 5 - length
            x1 = bar_w - 5
            color = (0, 255, 255) if (f_min <= f <= f_max) else (120, 120, 255)
            if length > 0:
                cv2.line(bar, (x0, y), (x1, y), color, 1)

    y_min = np.clip(freq_to_y(f_min, h), 0, h - 1)
    y_max = np.clip(freq_to_y(f_max, h), 0, h - 1)

    # --- label the draggable band edges (kHz) ---
    # Put text a bit to the right of the green line
    label_x = 8

    fmin_khz = f_min / 1000.0
    fmax_khz = f_max / 1000.0

    # keep text on-screen (avoid negative y / overflow)
    y_min_txt = int(np.clip(y_min - 6, 12, h - 6))
    y_max_txt = int(np.clip(y_max - 6, 12, h - 6))

    cv2.putText(bar, f"{fmin_khz:5.1f} kHz", (label_x, y_min_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{fmax_khz:5.1f} kHz", (label_x, y_max_txt),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.line(bar, (0, y_min), (bar_w - 1, y_min), (0, 255, 0), 1)
    cv2.line(bar, (0, y_max), (bar_w - 1, y_max), (0, 255, 0), 1)

    cv2.putText(bar, "Freq", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(bar, f"{int(F_DISPLAY_MAX/1000)} kHz", (5, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
    cv2.putText(bar, "0", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    handle_x = bar_w // 2
    cv2.circle(bar, (handle_x, int(y_min)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_max)), 7, (0, 255, 0), -1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_min)), 7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(bar, (handle_x, int(y_max)), 7, (0, 0, 0), 1, cv2.LINE_AA)

    frame[:, bar_left:bar_right, :] = bar


def draw_db_colorbar(frame: np.ndarray,
                     db_min: float,
                     db_max: float,
                     width: int = DB_BAR_WIDTH) -> None:
    h = frame.shape[0]
    bar = np.zeros((h, width), dtype=np.uint8)
    for y in range(h):
        val = y / (h - 1)
        bar[h - 1 - y, :] = int(val * 255)

    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_MAGMA)
    frame[:, :width] = bar_color

    cv2.putText(frame, f"{db_max:.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"{db_min:.0f} dB", (5, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ===============================================================
# 7. Screenshot + Recording
# ===============================================================
def save_screenshot(frame: np.ndarray, output_dir: Path) -> Optional[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"screenshot_{timestamp}.png"
    if cv2.imwrite(str(filepath), frame):
        print(f"Screenshot saved: {filepath}")
        return str(filepath)
    print(f"Failed to save screenshot: {filepath}")
    return None

class VideoRecorder:
    def __init__(self, output_dir: Path, width: int, height: int, fps: int = 30):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_recording = False
        self.is_paused = False
        self.current_file = None
        self.frame_count = 0

    def start_recording(self) -> bool:
        if self.is_recording:
            return False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.output_dir / f"recording_{timestamp}.mp4"
            command = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-preset", "fast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(self.current_file),
            ]
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8
            )
            self.is_recording = True
            self.is_paused = False
            self.frame_count = 0
            print(f"Recording started: {self.current_file}")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False

    def write_frame(self, frame: np.ndarray) -> bool:
        if not self.is_recording or self.process is None:
            return False
        if self.is_paused:
            return True
        try:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False

    def pause_recording(self) -> bool:
        if not self.is_recording or self.is_paused:
            return False
        self.is_paused = True
        return True

    def resume_recording(self) -> bool:
        if not self.is_recording or not self.is_paused:
            return False
        self.is_paused = False
        return True

    def stop_recording(self) -> Optional[str]:
        if not self.is_recording or self.process is None:
            return None
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
            out = str(self.current_file)
            print(f"Recording stopped: {out} ({self.frame_count} frames)")
            return out
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None
        finally:
            self.is_recording = False
            self.is_paused = False
            self.current_file = None
            self.frame_count = 0
            self.process = None

    def cleanup(self) -> None:
        if self.is_recording:
            self.stop_recording()

# ===============================================================
# 8. UI Buttons + MENU dropdown
# ===============================================================
class ButtonState:
    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.camera_enabled = USE_CAMERA
        self.source_mode = "SIM"  # SIM or SPI

        # MENU states
        self.menu_open = False
        self.fps_mode = "60"   # "30" | "60" | "MAX"
        self.gain_mode = "LOW"  # placeholder toggle
        self.debug_enabled = True

button_state = ButtonState()
video_recorder: Optional[VideoRecorder] = None

FPS_MODE_TO_TARGET = {"30": 30, "60": 60}

def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        color,
        thickness,
        lineType=cv2.LINE_AA
    )


class Button:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.is_hovered = False
        self.is_active = False

    def contains(self, mx, my) -> bool:
        return (self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h)

    def draw(self, frame: np.ndarray) -> None:
        base = (60, 60, 60)
        hover = (85, 85, 85)
        active = (70, 90, 70)
        border = (230, 230, 230)

        color = active if self.is_active else (hover if self.is_hovered else base)

        # Solid filled rounded rectangle (no overlay, no blending)
        _rounded_rect(frame, self.x, self.y, self.w, self.h,
                    r=10, color=color, thickness=-1)

        # Border
        _rounded_rect(frame, self.x, self.y, self.w, self.h,
                    r=10, color=border, thickness=2)

        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2

        cv2.putText(frame, self.text, (tx, ty),
                    font, scale, (255, 255, 255),
                    thick, cv2.LINE_AA)


buttons = {}
menu_buttons = {}

def init_buttons(left_width: int, camera_available: bool) -> None:
    global buttons

    n = 3   # <--- was 2
    margin = 10
    y = HEIGHT - 50 - 10
    h = 50

    left_pad = DB_BAR_WIDTH + 12
    right_pad = 12
    avail = left_width - left_pad - right_pad

    w = (avail - (n - 1) * margin) // n
    w = int(max(160, min(240, w)))

    total = n * w + (n - 1) * margin
    x0 = left_pad + (avail - total) // 2

    buttons = {}

    cam_text = "Camera: N/A" if not camera_available else ("Camera: ON" if button_state.camera_enabled else "Camera: OFF")
    buttons["camera"] = Button(x0 + 0 * (w + margin), y, w, h, cam_text)
    buttons["camera"].is_active = camera_available and button_state.camera_enabled

    buttons["source"] = Button(x0 + 1 * (w + margin), y, w, h, f"Source: {button_state.source_mode}")
    buttons["source"].is_active = True

    dbg_text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
    buttons["debug"] = Button(x0 + 2 * (w + margin), y, w, h, dbg_text)
    buttons["debug"].is_active = button_state.debug_enabled


def init_menu_buttons(left_width: int) -> None:
    global menu_buttons
    menu_buttons = {}

    menu_x = 590
    menu_y = 10
    menu_w = 220
    menu_h = 60

    menu_buttons["menu"] = Button(menu_x, menu_y, menu_w, menu_h, "MENU")

    item_h = 40
    gap = 8
    y0 = menu_y + menu_h + gap

    # Segmented FPS buttons (30 | 60 | MAX)
    seg_gap = 6
    seg_w = (menu_w - 2 * seg_gap) // 3
    menu_buttons["fps30"]  = Button(menu_x + 0 * (seg_w + seg_gap), y0, seg_w, item_h, "30FPS")
    menu_buttons["fps60"]  = Button(menu_x + 1 * (seg_w + seg_gap), y0, seg_w, item_h, "60FPS")
    menu_buttons["fpsmax"] = Button(menu_x + 2 * (seg_w + seg_gap), y0, seg_w, item_h, "MAX")

    # Gain toggle (full width)
    gain_y = y0 + (item_h + gap)
    menu_buttons["gain"] = Button(menu_x, gain_y, menu_w, item_h, f"GAIN: {button_state.gain_mode}")

    # NEW: segmented control row under GAIN (SHOT | REC | PAUSE)
    tools_y = gain_y + (item_h + gap)
    tool_gap = 6
    tool_w = (menu_w - 2 * tool_gap) // 3

    menu_buttons["shot"]  = Button(menu_x + 0 * (tool_w + tool_gap), tools_y, tool_w, item_h, "SHOT")
    menu_buttons["rec"]   = Button(menu_x + 1 * (tool_w + tool_gap), tools_y, tool_w, item_h, "REC")
    menu_buttons["pause"] = Button(menu_x + 2 * (tool_w + tool_gap), tools_y, tool_w, item_h, "PAUSE")


def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

    if "menu" in menu_buttons:
        menu_buttons["menu"].is_hovered = menu_buttons["menu"].contains(mx, my)

    keys = ("fps30", "fps60", "fpsmax", "gain", "shot", "rec", "pause")

    if button_state.menu_open:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)
    else:
        for k in keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = False

def draw_buttons(frame: np.ndarray) -> None:
    for b in buttons.values():
        b.draw(frame)

def draw_menu(frame: np.ndarray) -> None:
    if "menu" not in menu_buttons:
        return

    menu_buttons["menu"].is_active = button_state.menu_open
    menu_buttons["menu"].draw(frame)

    if not button_state.menu_open:
        return

    x = menu_buttons["menu"].x
    y = menu_buttons["menu"].y + menu_buttons["menu"].h + 6
    w = menu_buttons["menu"].w
    h = 3 * 40 + 2 * 8 + 20

    overlay = frame.copy()
    _rounded_rect(overlay, x-2, y-2, w+4, h+4, r=12, color=(20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    menu_buttons["fps30"].is_active  = (button_state.fps_mode == "30")
    menu_buttons["fps60"].is_active  = (button_state.fps_mode == "60")
    menu_buttons["fpsmax"].is_active = (button_state.fps_mode == "MAX")

    menu_buttons["gain"].is_active = (button_state.gain_mode == "HIGH")

    # Tool button actives reflect current state
    menu_buttons["rec"].is_active = button_state.is_recording
    menu_buttons["pause"].is_active = button_state.is_paused

    menu_buttons["fps30"].draw(frame)
    menu_buttons["fps60"].draw(frame)
    menu_buttons["fpsmax"].draw(frame)
    menu_buttons["gain"].draw(frame)

    menu_buttons["shot"].draw(frame)
    menu_buttons["rec"].draw(frame)
    menu_buttons["pause"].draw(frame)


# ===============================================================
# 9. SPI framing helpers (loopback)
# ===============================================================
def crc_validate_frame(buf: bytes) -> bool:
    if len(buf) != FRAME_BYTES:
        return False
    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(HEADER_FMT, buf, 0)
    except struct.error:
        return False

    if magic != MAGIC_START or ver != VERSION or hdr_len != HEADER_LEN:
        return False
    if mic != N_MICS or fft_size != SAMPLES_PER_CHANNEL or fs != SAMPLE_RATE_HZ or bins != N_BINS:
        return False
    if pay_len != PAYLOAD_LEN:
        return False

    crc_rx, magic_end = struct.unpack_from(TRAILER_FMT, buf, HEADER_LEN + PAYLOAD_LEN)
    if magic_end != MAGIC_END:
        return False

    header = buf[:HEADER_LEN]
    payload = buf[HEADER_LEN:HEADER_LEN + PAYLOAD_LEN]
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return crc == crc_rx

def framing_validate_frame(buf: bytes) -> tuple[bool, str]:
    if len(buf) != FRAME_BYTES:
        return False, "len"
    try:
        (magic,) = struct.unpack_from("<I", buf, 0)
    except struct.error:
        return False, "hdr_unpack0"
    if magic != MAGIC_START:
        return False, "magic_start"

    try:
        (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(HEADER_FMT, buf, 0)
    except struct.error:
        return False, "hdr_unpack"

    if ver != VERSION or hdr_len != HEADER_LEN:
        return False, "hdr_fields"
    if mic != N_MICS or fft_size != SAMPLES_PER_CHANNEL or fs != SAMPLE_RATE_HZ or bins != N_BINS:
        return False, "cfg_mismatch"
    if pay_len != PAYLOAD_LEN:
        return False, "pay_len"

    try:
        (_, magic_end) = struct.unpack_from(TRAILER_FMT, buf, HEADER_LEN + PAYLOAD_LEN)
    except struct.error:
        return False, "trl_unpack"
    if magic_end != MAGIC_END:
        return False, "magic_end"

    return True, "ok"

def make_payload_mag_phase(seq: int, t_sec: float) -> bytes:
    """
    Loopback payload with stable virtual sources.
    Now with "sound-like" spread across neighboring bins (thicker spectral bands).
    """
    mp = np.zeros((N_MICS, N_BINS, 2), dtype=np.float32)

    noise_mag = 0.006
    noise_phase = 0.015

    mp[:, :, 0] = noise_mag * (1.0 + 0.25 * np.random.randn(N_MICS, N_BINS)).astype(np.float32)
    mp[:, :, 1] = noise_phase * np.random.randn(N_MICS, N_BINS).astype(np.float32)

    spread_bins = 3      # +/- bins around center
    sigma_bins = 1.2     # gaussian width in bins

    for i, b0 in enumerate(SPI_SIM_BINS):
        if b0 < 0 or b0 >= N_BINS:
            continue
        if i >= len(SPI_SIM_AMPLS) or i >= len(SPI_SIM_ANGLES) or i >= len(SPI_SIM_DRIFT_DEG_PER_SEC):
            continue

        f0 = float(f_axis[b0])
        ang = SPI_SIM_ANGLES[i] + SPI_SIM_DRIFT_DEG_PER_SEC[i] * t_sec
        ang = ((ang + 90.0) % 180.0) - 90.0
        theta = np.deg2rad(ang)

        a0 = np.exp(
            -1j * 2 * np.pi * f0 / SPEED_SOUND *
            -(x_coords * np.cos(theta) + y_coords * np.sin(theta))
        ).astype(np.complex64)

        for dbin in range(-spread_bins, spread_bins + 1):
            b = b0 + dbin
            if b < 0 or b >= N_BINS:
                continue

            w = float(np.exp(-0.5 * (dbin / sigma_bins) ** 2))
            amp = float(SPI_SIM_AMPLS[i]) * w

            ph0 = 0.15 * dbin
            X = amp * a0 * np.exp(1j * ph0)

            mp[:, b, 0] += np.abs(X).astype(np.float32)
            mp[:, b, 1] += np.angle(X).astype(np.float32)

    return mp.tobytes(order="C")

def make_frame(seq: int, t_sec: float) -> bytes:
    payload = make_payload_mag_phase(seq, t_sec)
    header = struct.pack(
        HEADER_FMT,
        MAGIC_START, VERSION, HEADER_LEN, seq,
        N_MICS, SAMPLES_PER_CHANNEL, SAMPLE_RATE_HZ,
        N_BINS, 0, len(payload),
    )
    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    trailer = struct.pack(TRAILER_FMT, crc, MAGIC_END)
    return header + payload + trailer

def spi_xfer_bytes(spi, tx: bytes) -> bytes:
    rx = bytearray(len(tx))
    mv = memoryview(tx)
    offset = 0

    while offset < len(tx):
        end = min(offset + SPI_XFER_CHUNK, len(tx))
        chunk = mv[offset:end]
        r = spi.xfer3(chunk)
        rx[offset:end] = bytes(r)
        offset = end

    return bytes(rx)

# ===============================================================
# 10. Sources: SIM + SPI LOOPBACK
# ===============================================================
class SimFFTSource:
    def __init__(self):
        self.frame_id = 0

    def read_frame(self) -> FFTFrame:
        global SIM_SOURCE_ANGLES

        for k in range(SIM_N_SOURCES):
            SIM_SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
            if SIM_SOURCE_ANGLES[k] > 90.0:
                SIM_SOURCE_ANGLES[k] = -90.0

        self.frame_id += 1
        return self._generate_fft_frame(SIM_SOURCE_ANGLES)

    def _generate_fft_frame(self, angle_degs: List[float]) -> FFTFrame:
        frame = FFTFrame()
        frame.channel_count = N_MICS
        frame.sampling_rate = SAMPLE_RATE_HZ
        frame.fft_size = SAMPLES_PER_CHANNEL
        frame.frame_id = self.frame_id

        t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
        mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

        for src_idx, angle_deg in enumerate(angle_degs):
            angle_rad = np.deg2rad(angle_deg)
            f = SIM_SOURCE_FREQS[src_idx]
            amp = SIM_SOURCE_AMPLS[src_idx]
            for i in range(N_MICS):
                delay = -(x_coords[i] * np.cos(angle_rad) + y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
                mic_signals[i, :] += amp * np.sin(2 * np.pi * f * (t - delay))

        mic_signals += np.random.normal(0, np.sqrt(NOISE_POWER), mic_signals.shape).astype(np.float32)

        fft_data = np.fft.rfft(mic_signals, axis=1).astype(np.complex64)
        frame.fft_data = fft_data
        return frame

class SpiFFTSource:
    def __init__(self, bus=SPI_BUS, dev=SPI_DEV, max_speed_hz=SPI_MAX_SPEED_HZ):
        self.bus = bus
        self.dev = dev
        self.max_speed_hz = int(max_speed_hz)
        self.spi = None

        self.frames_ok = 0
        self.bad_parse = 0
        self.bad_crc = 0
        self.last_err = ""
        self.seq_seen = 0
        self.sclk_hz_rep = 0

    def open(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self.last_err = "spidev not installed"
            return False
        try:
            self.spi = spidev.SpiDev()
            self.spi.open(self.bus, self.dev)
            self.spi.mode = SPI_MODE
            self.spi.bits_per_word = SPI_BITS
            self.spi.max_speed_hz = int(self.max_speed_hz)
            self.sclk_hz_rep = int(self.spi.max_speed_hz)
            self.last_err = ""
            return True
        except Exception as e:
            self.last_err = f"SPI open failed: {e}"
            self.spi = None
            return False

    def close(self) -> None:
        if self.spi is not None:
            try:
                self.spi.close()
            except Exception:
                pass
        self.spi = None

    def read_frame(self) -> Optional[FFTFrame]:
        if self.spi is None:
            if not self.open():
                return None

        try:
            self.seq_seen += 1
            t_sec = time.time()

            tx = make_frame(self.seq_seen, t_sec)
            rx = spi_xfer_bytes(self.spi, tx)

            nz = sum(b != 0 for b in rx[:256])
            self.last_err = f"rx_nonzero256={nz}"

            ok, why = framing_validate_frame(rx)
            if not ok:
                self.bad_parse += 1
                self.last_err = f"parse:{why} nz256={nz}"
                return None

            if CRC_EVERY_N and ((self.frames_ok % CRC_EVERY_N) == 0):
                if not crc_validate_frame(rx):
                    self.bad_crc += 1
                    self.last_err = "crc"
                    return None

            payload = rx[HEADER_LEN:HEADER_LEN + PAYLOAD_LEN]
            mp = np.frombuffer(payload, dtype=np.float32).reshape(N_MICS, N_BINS, 2)
            mag = mp[:, :, 0]
            phase = mp[:, :, 1]

            fft_data = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)

            f = FFTFrame()
            f.channel_count = N_MICS
            f.sampling_rate = SAMPLE_RATE_HZ
            f.fft_size = SAMPLES_PER_CHANNEL
            f.frame_id = self.seq_seen
            f.fft_data = fft_data

            self.frames_ok += 1
            return f

        except Exception as e:
            self.bad_parse += 1
            self.last_err = f"spi_exc:{e}"
            return None

# ===============================================================
# 11. Mouse callback (buttons + bandpass drag + MENU)
# ===============================================================
CURSOR_POS = (0, 0)
CURRENT_FRAME: Optional[np.ndarray] = None
OUTPUT_DIR: Optional[Path] = None
CAMERA_AVAILABLE = False

sim_source = SimFFTSource()
spi_source = SpiFFTSource()

# ===============================================================
# 11.5 Camera "latest frame" thread (FIX #1)
# ===============================================================
class _LatestCamFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ok = False

_latest = _LatestCamFrame()
_stop_cam = False
_cam_thread = None


def _cam_worker_picam2(picam2):
    global _stop_cam
    while not _stop_cam:
        try:
            fr = picam2.capture_array()
            if fr is not None and getattr(fr, "size", 0) > 0:
                with _latest.lock:
                    _latest.frame = fr
                    _latest.ok = True
        except Exception:
            time.sleep(0.005)


def _cam_worker_opencv(cam):
    global _stop_cam
    while not _stop_cam:
        try:
            if cam is None:
                time.sleep(0.01)
                continue

            ok = cam.grab()
            if not ok:
                time.sleep(0.005)
                continue

            ok, fr = cam.retrieve()
            if ok and fr is not None and getattr(fr, "size", 0) > 0:
                with _latest.lock:
                    _latest.frame = fr
                    _latest.ok = True
        except Exception:
            time.sleep(0.005)


def handle_menu_click(x: int, y: int) -> None:
    global video_recorder

    if "menu" not in menu_buttons:
        return

    if menu_buttons["menu"].contains(x, y):
        button_state.menu_open = not button_state.menu_open
        return

    if not button_state.menu_open:
        return

    # FPS segmented
    if "fps30" in menu_buttons and menu_buttons["fps30"].contains(x, y):
        button_state.fps_mode = "30"
        return
    if "fps60" in menu_buttons and menu_buttons["fps60"].contains(x, y):
        button_state.fps_mode = "60"
        return
    if "fpsmax" in menu_buttons and menu_buttons["fpsmax"].contains(x, y):
        button_state.fps_mode = "MAX"
        return

    # Gain toggle
    if "gain" in menu_buttons and menu_buttons["gain"].contains(x, y):
        button_state.gain_mode = "HIGH" if button_state.gain_mode == "LOW" else "LOW"
        menu_buttons["gain"].text = f"GAIN: {button_state.gain_mode}"
        return

    # --- NEW TOOLS ---
    if "shot" in menu_buttons and menu_buttons["shot"].contains(x, y):
        if CURRENT_FRAME is not None and OUTPUT_DIR is not None:
            save_screenshot(CURRENT_FRAME, OUTPUT_DIR)
        return

    if "rec" in menu_buttons and menu_buttons["rec"].contains(x, y):
        # lazy-create recorder only when needed (keeps overhead near zero)
        if video_recorder is None and OUTPUT_DIR is not None:
            video_recorder = VideoRecorder(OUTPUT_DIR, WIDTH, HEIGHT, fps=30)

        if video_recorder is None:
            return

        button_state.is_recording = not button_state.is_recording
        if button_state.is_recording:
            if video_recorder.start_recording():
                button_state.is_paused = False
            else:
                button_state.is_recording = False
        else:
            video_recorder.stop_recording()
            button_state.is_paused = False
        return

    if "pause" in menu_buttons and menu_buttons["pause"].contains(x, y):
        if not button_state.is_recording or video_recorder is None:
            return

        button_state.is_paused = not button_state.is_paused
        if button_state.is_paused:
            video_recorder.pause_recording()
            menu_buttons["pause"].text = "RESUME"
        else:
            video_recorder.resume_recording()
            menu_buttons["pause"].text = "PAUSE"
        return


def handle_button_click(x, y, current_frame=None, output_dir=None, camera_available=False):
    handle_menu_click(x, y)

    if buttons["camera"].contains(x, y):
        if camera_available:
            button_state.camera_enabled = not button_state.camera_enabled
            buttons["camera"].is_active = button_state.camera_enabled
            buttons["camera"].text = "Camera: ON" if button_state.camera_enabled else "Camera: OFF"
        return

    if buttons["source"].contains(x, y):
        button_state.source_mode = "SPI" if button_state.source_mode == "SIM" else "SIM"
        buttons["source"].text = f"Source: {button_state.source_mode}"
        if button_state.source_mode != "SPI":
            spi_source.close()
        return

    if buttons["debug"].contains(x, y):
        button_state.debug_enabled = not button_state.debug_enabled
        buttons["debug"].is_active = button_state.debug_enabled
        buttons["debug"].text = "DEBUG: ON" if button_state.debug_enabled else "DEBUG: OFF"
        return


def mouse_move(event, x, y, flags, param):
    global CURSOR_POS, DRAG_ACTIVE, DRAG_TARGET, F_MIN_HZ, F_MAX_HZ

    CURSOR_POS = (x, y)
    update_button_states(x, y)

    left_width, h = param
    bar_left = left_width
    freq_right_edge = WIDTH  # FULL width now (no reserved menu column)

    if event == cv2.EVENT_LBUTTONDOWN:
        for b in buttons.values():
            if b.contains(x, y):
                handle_button_click(x, y, CURRENT_FRAME, OUTPUT_DIR, CAMERA_AVAILABLE)
                return

        if "menu" in menu_buttons and menu_buttons["menu"].contains(x, y):
            handle_button_click(x, y, CURRENT_FRAME, OUTPUT_DIR, CAMERA_AVAILABLE)
            return
        if button_state.menu_open:
            for k in ("fps30", "fps60", "fpsmax", "gain", "shot", "rec", "pause"):
                if k in menu_buttons and menu_buttons[k].contains(x, y):
                    handle_button_click(x, y, CURRENT_FRAME, OUTPUT_DIR, CAMERA_AVAILABLE)
                    return

        # bandpass drag only on freq bar area
        if x >= bar_left and x < freq_right_edge:
            y_min = freq_to_y(F_MIN_HZ, h)
            y_max = freq_to_y(F_MAX_HZ, h)
            dmin = abs(y - y_min)
            dmax = abs(y - y_max)

            DRAG_TARGET = "min" if dmin <= dmax else "max"
            DRAG_ACTIVE = True

            f = y_to_freq(y, h)
            if DRAG_TARGET == "min":
                F_MIN_HZ = min(f, F_MAX_HZ)
            else:
                F_MAX_HZ = max(f, F_MIN_HZ)

    elif event == cv2.EVENT_MOUSEMOVE:
        if DRAG_ACTIVE and x >= bar_left and x < freq_right_edge:
            f = y_to_freq(y, h)
            if DRAG_TARGET == "min":
                F_MIN_HZ = min(f, F_MAX_HZ)
            elif DRAG_TARGET == "max":
                F_MAX_HZ = max(f, F_MIN_HZ)

    elif event == cv2.EVENT_LBUTTONUP:
        DRAG_ACTIVE = False
        DRAG_TARGET = None

    F_MIN_HZ = float(np.clip(F_MIN_HZ, 0.0, F_DISPLAY_MAX))
    F_MAX_HZ = float(np.clip(F_MAX_HZ, 0.0, F_DISPLAY_MAX))
    if F_MIN_HZ > F_MAX_HZ:
        F_MIN_HZ, F_MAX_HZ = F_MAX_HZ, F_MIN_HZ

# ===============================================================
# 12. Main
# ===============================================================
def main():
    global OUTPUT_DIR, CURRENT_FRAME, video_recorder, CAMERA_AVAILABLE
    global _stop_cam, _cam_thread
        # ---- PROFILER (DEBUG) ----
    prof = StageProfiler(keep=120)
    PRINT_EVERY = 60  # frames
    # --------------------------

    print("Acoustic Imager - SIM + SPI LOOPBACK Bandpass Demo")
    print("=" * 70)
    print(f"FPS default: 60 (segmented: 30/60/MAX)")
    print(f"SPI available: {SPIDEV_AVAILABLE} | Picamera2 available: {PICAMERA2_AVAILABLE}")
    print(f"SPI frame bytes: {FRAME_BYTES} (payload float32 mag+phase {N_MICS}x{N_BINS})")
    print("LOOPBACK REQUIREMENT: MOSI must be connected to MISO for SPI mode to work.")
    print("=" * 70)

    background_full = create_background_frame(WIDTH, HEIGHT)
    left_width = WIDTH - FREQ_BAR_WIDTH

    OUTPUT_DIR = Path(__file__).parent / "heatmap_captures"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    video_recorder = VideoRecorder(OUTPUT_DIR, WIDTH, HEIGHT, fps=30)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, mouse_move, param=(left_width, HEIGHT))

    # --------------------------
    # FIX #2: Preallocated buffers (no per-frame resize allocation)
    # --------------------------
    base_frame = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)  # holds background (camera or static)
    cam_bgr = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)     # used for RGB->BGR conversion (picamera2)

    # Camera init
    cam = None
    picam2 = None
    use_camera = USE_CAMERA
    camera_type = None

    if USE_CAMERA:
        if USE_LIBCAMERA and PICAMERA2_AVAILABLE:
            try:
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                picam2.configure(config)
                try:
                    picam2.set_controls({"AwbEnable": True, "AwbMode": 1})
                except Exception:
                    pass
                picam2.start()
                time.sleep(1.5)
                test_frame = picam2.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    camera_type = "picamera2"
            except Exception:
                if picam2 is not None:
                    try:
                        picam2.stop()
                        picam2.close()
                    except Exception:
                        pass
                picam2 = None

        if camera_type is None and USE_LIBCAMERA:
            try:
                gst = (
                    f"libcamerasrc ! video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate=30/1 ! "
                    f"videoconvert ! appsink"
                )
                cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
                if cam.isOpened():
                    ret, tf = cam.read()
                    if ret and tf is not None and tf.size > 0:
                        camera_type = "opencv"
                    else:
                        cam.release()
                        cam = None
            except Exception:
                if cam is not None:
                    cam.release()
                cam = None

        if camera_type is None:
            try:
                test_cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
                if not test_cam.isOpened():
                    test_cam = cv2.VideoCapture(CAMERA_INDEX)
                if test_cam.isOpened():
                    test_cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                    test_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                    test_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    for _ in range(3):
                        test_cam.read()
                    ret, tf = test_cam.read()
                    if ret and tf is not None and tf.size > 0:
                        cam = test_cam
                        camera_type = "opencv"
                    else:
                        test_cam.release()
            except Exception:
                pass

    if camera_type is None:
        use_camera = False
        CAMERA_AVAILABLE = False
    else:
        CAMERA_AVAILABLE = True

    # --------------------------
    # FIX #1: Start camera worker thread (latest-frame)
    # --------------------------
    _stop_cam = False
    with _latest.lock:
        _latest.frame = None
        _latest.ok = False

    if use_camera and button_state.camera_enabled and camera_type is not None:
        if camera_type == "picamera2" and picam2 is not None:
            _cam_thread = threading.Thread(target=_cam_worker_picam2, args=(picam2,), daemon=True)
            _cam_thread.start()
        elif camera_type == "opencv" and cam is not None:
            _cam_thread = threading.Thread(target=_cam_worker_opencv, args=(cam,), daemon=True)
            _cam_thread.start()

    init_buttons(left_width, CAMERA_AVAILABLE)
    init_menu_buttons(left_width)

    frame_count = 0
    start_time = time.time()

    last_spi_fft_data = None

    # FPS estimator + throttle tick
    fps_ema = 0.0
    last_t = time.perf_counter()
    next_tick = time.perf_counter()

    try:
        while True:
            prof.start_frame()

            # --- FPS estimator (EMA) ---
            now_t = time.perf_counter()
            dt = now_t - last_t
            last_t = now_t
            inst_fps = (1.0 / dt) if dt > 1e-6 else 0.0
            fps_ema = (0.92 * fps_ema + 0.08 * inst_fps) if fps_ema > 0 else inst_fps
            prof.mark("fps_est")

            # --- Throttle based on segmented FPS mode ---
            if button_state.fps_mode in ("30", "60"):
                fps_target = FPS_MODE_TO_TARGET[button_state.fps_mode]
                period = 1.0 / max(1, fps_target)

                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()
            else:
                # MAX = unthrottled
                next_tick = time.perf_counter()
            prof.mark("throttle")

            f_min = F_MIN_HZ
            f_max = F_MAX_HZ

            # --- Read source ---
            if button_state.source_mode == "SIM":
                fft_frame = sim_source.read_frame()
                source_label = "SIM"
                source_err = ""
            else:
                fft_frame = spi_source.read_frame()
                source_label = "SPI"
                source_err = spi_source.last_err
            prof.mark("read_source")

            # --- fft_data extract/fallback ---
            if fft_frame is None or fft_frame.fft_data is None:
                if source_label == "SPI" and last_spi_fft_data is not None:
                    fft_data = last_spi_fft_data
                else:
                    fft_data = np.zeros((N_MICS, N_BINS), dtype=np.complex64)
            else:
                fft_data = fft_frame.fft_data
                if source_label == "SPI":
                    last_spi_fft_data = fft_data
            prof.mark("fft_extract")

            # --- Heatmap build ---
            if source_label == "SIM":
                selected_indices = [i for i, f in enumerate(SIM_SOURCE_FREQS) if f_min <= f <= f_max]
                if not selected_indices:
                    heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                else:
                    n_sel = len(selected_indices)
                    spec_matrix = np.zeros((n_sel, len(ANGLES)), dtype=np.float32)
                    power = np.zeros(n_sel, dtype=np.float32)

                    running_max_power = 1e-12
                    for row_idx, src_idx in enumerate(selected_indices):
                        f_sig = float(SIM_SOURCE_FREQS[src_idx])
                        f_idx = int(np.argmin(np.abs(f_axis - f_sig)))
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T

                        spec_matrix[row_idx, :] = music_spectrum(R, ANGLES, f_sig, n_sources=n_sel)
                        p = float(np.sum(np.abs(Xf) ** 2).real)
                        power[row_idx] = p
                        running_max_power = max(running_max_power, p)

                    power_rel = power / (running_max_power + 1e-12)
                    heatmap_left = spectra_to_heatmap_absolute(spec_matrix, power_rel, left_width, HEIGHT)

            else:
                bins = [b for b in SPI_SIM_BINS
                        if 0 <= b < N_BINS and (f_min <= float(f_axis[b]) <= f_max)]

                if not bins:
                    heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                else:
                    spec_matrix = np.zeros((len(bins), len(ANGLES)), dtype=np.float32)
                    power = np.zeros(len(bins), dtype=np.float32)

                    for i, f_idx in enumerate(bins):
                        f_sig = float(f_axis[f_idx])
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T
                        spec_matrix[i, :] = music_spectrum(R, ANGLES, f_sig, n_sources=len(bins))
                        power[i] = float(np.sum(np.abs(Xf) ** 2).real)

                    # brighter SPI: normalize within selected bins each frame
                    power_rel = power / (power.max() + 1e-12)
                    power_rel = np.power(power_rel, 0.6)
                    heatmap_left = spectra_to_heatmap_absolute(spec_matrix, power_rel, left_width, HEIGHT)
            prof.mark("beamform+heatmap")

            # -------------------------------------------------------
            # Background (FIX #1 + #2)
            # - never block on camera (use latest)
            # - never allocate on resize (use dst buffers)
            # -------------------------------------------------------
            if use_camera and camera_type is not None and button_state.camera_enabled:
                cam_frame = None
                with _latest.lock:
                    if _latest.ok and _latest.frame is not None:
                        cam_frame = _latest.frame

                if cam_frame is not None and getattr(cam_frame, "size", 0) > 0:
                    try:
                        if camera_type == "picamera2":
                            # picamera2 is RGB888 at correct size -> convert into prealloc cam_bgr
                            cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR, dst=cam_bgr)
                            # copy into base_frame (cheap memcopy, but avoids any new allocations)
                            base_frame[:] = cam_bgr
                        else:
                            # opencv path is already BGR; resize into base_frame (dst=)
                            if cam_frame.shape[1] == WIDTH and cam_frame.shape[0] == HEIGHT:
                                base_frame[:] = cam_frame
                            else:
                                cv2.resize(cam_frame, (WIDTH, HEIGHT), dst=base_frame, interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        base_frame[:] = background_full
                else:
                    base_frame[:] = background_full
            else:
                base_frame[:] = background_full
            prof.mark("background")

            # --- Blend heatmap left ---
            left_bg = base_frame[:, :left_width, :]
            colored = cv2.applyColorMap(heatmap_left, cv2.COLORMAP_MAGMA)

            mask = (heatmap_left.astype(np.float32) / 255.0)
            mask = np.power(mask, 0.5)
            mask3 = np.stack([mask] * 3, axis=-1)

            left_out = (colored * mask3 * ALPHA + left_bg * (1 - mask3 * ALPHA)).astype(np.uint8)
            base_frame[:, :left_width, :] = left_out
            output_frame = base_frame
            prof.mark("blend")

            # --- Bars ---
            draw_frequency_bar(output_frame, fft_data, f_axis, f_min, f_max)
            draw_db_colorbar(output_frame, REL_DB_MIN, REL_DB_MAX)
            prof.mark("bars")

            # --- Text/UI ---
            if button_state.debug_enabled:
                elapsed = time.time() - start_time
                text_x = DB_BAR_WIDTH + 12
                cv2.putText(output_frame, f"Frame: {frame_count}", (text_x, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"t = {elapsed:.2f}s", (text_x, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(output_frame, f"Source: {source_label}", (text_x, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if source_label == "SPI":
                    mhz = (spi_source.sclk_hz_rep / 1e6) if spi_source.sclk_hz_rep else (SPI_MAX_SPEED_HZ / 1e6)
                    y0 = 120
                    dy = 22

                    bytes_per_s = FRAME_BYTES * fps_ema
                    mbps_bytes = bytes_per_s / 1e6
                    mbps_bits  = (bytes_per_s * 8) / 1e6

                    cv2.putText(output_frame, f"SPI {mhz:.0f}MHz", (text_x, y0),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"ok: {spi_source.frames_ok}", (text_x, y0 + 1*dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"badParse: {spi_source.bad_parse}   badCRC: {spi_source.bad_crc}",
                                (text_x, y0 + 2*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"FPS: {fps_ema:5.1f}", (text_x, y0 + 3*dy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    cv2.putText(output_frame, f"Throughput: {mbps_bytes:.2f} MB/s  ({mbps_bits:.1f} Mb/s)",
                                (text_x, y0 + 4*dy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    if source_err:
                        cv2.putText(output_frame, f"lastErr: {source_err[:60]}", (text_x, y0 + 5*dy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            draw_buttons(output_frame)
            draw_menu(output_frame)
            prof.mark("ui")

            CURRENT_FRAME = output_frame.copy()
            prof.mark("copy_frame")

            if button_state.is_recording and video_recorder:
                video_recorder.write_frame(output_frame)
            prof.mark("record")

            cv2.imshow(WINDOW_NAME, output_frame)
            prof.mark("imshow")

            key = cv2.waitKey(1) & 0xFF
            prof.mark("waitKey")

            prof.end_frame()

            if (frame_count % PRINT_EVERY) == 0 and frame_count > 0:
                print(
                    "ms avg | "
                    f"read={prof.ms('read_source'):.2f} "
                    f"heat={prof.ms('beamform+heatmap'):.2f} "
                    f"bg={prof.ms('background'):.2f} "
                    f"blend={prof.ms('blend'):.2f} "
                    f"bars={prof.ms('bars'):.2f} "
                    f"ui={prof.ms('ui'):.2f} "
                    f"imshow={prof.ms('imshow'):.2f} "
                    f"waitKey={prof.ms('waitKey'):.2f} "
                    f"total={prof.ms('frame_total'):.2f}"
                )

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("q") or key == 27:
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Cleaning up...")

        # FIX #1: stop cam thread
        _stop_cam = True
        time.sleep(0.02)

        if video_recorder:
            video_recorder.cleanup()
        spi_source.close()
        if cam is not None:
            try:
                cam.release()
            except Exception:
                pass
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


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

Fixes included:
  - Chunked SPI xfer2 so we never exceed 4096 bytes
  - Clean loopback framing (no rolling resync buffer needed)
"""

import sys
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import struct
import zlib
import subprocess

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

FPS_THROTTLE_ENABLED = False
FPS_TARGET = 60

# IMPORTANT: spidev xfer2 list length limit
SPI_XFER_CHUNK = 4096

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

# SIM sources
SIM_SOURCE_FREQS = [9000, 11000, 30000]
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]
SIM_N_SOURCES = len(SIM_SOURCE_ANGLES)

f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 257

# dB mapping for heatmap intensity: relative-to-running-max
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0

# ===============================================================
# 2. SPI frame format (float32 MAG + float32 PHASE)
# ===============================================================
MAGIC_START = 0x46524654  # 'TFRF' (example)
MAGIC_END = 0x454E4421    # 'END!' (example)
VERSION = 1

# magic(u32), ver(u16), hdr_len(u16),
# seq(u32),
# mic_count(u16), fft_size(u16),
# sample_rate(u32),
# bins(u16), reserved(u16),
# payload_len(u32)
HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

TRAILER_FMT = "<II"  # crc32(u32), end_magic(u32)
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

# Payload: (N_MICS, N_BINS, 2) float32 => [mag, phase]
PAYLOAD_LEN = N_MICS * N_BINS * 2 * 4   # 8 bytes per bin per mic
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN

# SPI config
SPI_BUS = 0
SPI_DEV = 0
SPI_MODE = 0
SPI_BITS = 8
SPI_MAX_SPEED_HZ = 60_000_000

# CRC sampling (set to 0 to disable CRC checks)
CRC_EVERY_N = 1

# ===============================================================
# 3. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.010  # if you want 5cm diameter => 0.025
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
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]

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

    spec = np.array(spectrum, dtype=np.float32)
    spec /= (np.max(spec) + 1e-12)
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
    spec_matrix: [Nsrc x Nang], normalized 0..1
    power_rel:   [Nsrc] power relative to running max (0..1]
    Maps power_rel to dB in [-60..0] by default.
    """
    Nsrc, Nang = spec_matrix.shape
    power_rel = np.maximum(power_rel.astype(np.float32), 1e-12)

    power_db = 10.0 * np.log10(power_rel)  # <= 0
    power_norm = (power_db - db_min) / (db_max - db_min + 1e-12)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    peak_indices = []
    sharpness = []

    for i in range(Nsrc):
        row = spec_matrix[i]
        idx = int(np.argmax(row))
        peak_indices.append(idx)

        p = float(row[idx])
        left = float(row[idx - 1]) if idx > 0 else p
        right = float(row[idx + 1]) if idx < Nang - 1 else p
        sh = max(p - 0.5 * (left + right), 1e-12)
        sharpness.append(sh)

    sharpness = np.array(sharpness, dtype=np.float32)
    sharpness /= (sharpness.max() + 1e-12)

    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def ang_to_px(idx: int) -> int:
        return int(idx / Nang * w)

    base_radius = 60

    for i in range(Nsrc):
        cx = ang_to_px(peak_indices[i])
        cy = h // 2

        blob_radius = base_radius * (0.7 + 0.3 * float(sharpness[i]))
        sigma = blob_radius / 1.8

        amp = float(power_norm[i])
        blob = amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
        heatmap += blob

    heatmap = np.clip(heatmap, 0.0, 1.0)
    return (heatmap * 255).astype(np.uint8)

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
    bar_w = FREQ_BAR_WIDTH
    left = w - bar_w
    right = w

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

    frame[:, left:right, :] = bar

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
# 8. UI Buttons (rounded, uniform)
# ===============================================================
class ButtonState:
    def __init__(self):
        self.is_recording = False
        self.is_paused = False
        self.camera_enabled = USE_CAMERA
        self.source_mode = "SIM"  # SIM or SPI

button_state = ButtonState()
video_recorder: Optional[VideoRecorder] = None

def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    r = int(max(0, min(r, min(w, h)//2)))
    if thickness < 0:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1)
        cv2.circle(img, (x+r, y+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+w-r, y+r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+r, y+h-r), r, color, -1, cv2.LINE_AA)
        cv2.circle(img, (x+w-r, y+h-r), r, color, -1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, thickness)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, thickness)
        cv2.circle(img, (x+r, y+r), r, color, thickness, cv2.LINE_AA)
        cv2.circle(img, (x+w-r, y+r), r, color, thickness, cv2.LINE_AA)
        cv2.circle(img, (x+r, y+h-r), r, color, thickness, cv2.LINE_AA)
        cv2.circle(img, (x+w-r, y+h-r), r, color, thickness, cv2.LINE_AA)

class Button:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = x, y, w, h
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

        overlay = frame.copy()
        _rounded_rect(overlay, self.x, self.y, self.w, self.h, r=10, color=color, thickness=-1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        _rounded_rect(frame, self.x, self.y, self.w, self.h, r=10, color=border, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.52
        thick = 1
        tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
        tx = self.x + (self.w - tw) // 2
        ty = self.y + (self.h + th) // 2
        cv2.putText(frame, self.text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

buttons = {}

def init_buttons(left_width: int, camera_available: bool) -> None:
    global buttons

    n = 5
    margin = 10
    y = HEIGHT - 50 - 10
    h = 50

    left_pad = DB_BAR_WIDTH + 12
    right_pad = 12
    avail = left_width - left_pad - right_pad

    w = (avail - (n - 1) * margin) // n
    w = int(max(110, min(160, w)))

    total = n * w + (n - 1) * margin
    x0 = left_pad + (avail - total) // 2

    buttons = {}
    buttons["screenshot"] = Button(x0 + 0 * (w + margin), y, w, h, "Screenshot")
    buttons["record"]     = Button(x0 + 1 * (w + margin), y, w, h, "Record")
    buttons["pause"]      = Button(x0 + 2 * (w + margin), y, w, h, "Pause")

    cam_text = "Camera: N/A" if not camera_available else ("Camera: ON" if button_state.camera_enabled else "Camera: OFF")
    buttons["camera"] = Button(x0 + 3 * (w + margin), y, w, h, cam_text)
    buttons["camera"].is_active = camera_available and button_state.camera_enabled

    buttons["source"] = Button(x0 + 4 * (w + margin), y, w, h, f"Source: {button_state.source_mode}")
    buttons["source"].is_active = True

def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

def draw_buttons(frame: np.ndarray) -> None:
    for b in buttons.values():
        b.draw(frame)

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

def make_payload_mag_phase(seq: int) -> bytes:
    """
    Deterministic synthetic payload for loopback testing.
    Shape: (N_MICS, N_BINS, 2) float32 => [mag, phase]
    """
    base = (seq % 1024) * 0.01
    mp = np.empty((N_MICS, N_BINS, 2), dtype=np.float32)

    bins_vec = (np.arange(N_BINS, dtype=np.float32) * 1e-4)

    for ch in range(N_MICS):
        mag = base + ch + bins_vec
        phase = (0.1 * ch + 0.001 * np.arange(N_BINS, dtype=np.float32))  # radians-ish
        mp[ch, :, 0] = mag
        mp[ch, :, 1] = phase

    return mp.tobytes(order="C")

def make_frame(seq: int) -> bytes:
    payload = make_payload_mag_phase(seq)

    header = struct.pack(
        HEADER_FMT,
        MAGIC_START,
        VERSION,
        HEADER_LEN,
        seq,
        N_MICS,
        SAMPLES_PER_CHANNEL,
        SAMPLE_RATE_HZ,
        N_BINS,
        0,
        len(payload),
    )

    crc = zlib.crc32(header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    trailer = struct.pack(TRAILER_FMT, crc, MAGIC_END)

    return header + payload + trailer

def spi_xfer_bytes(spi, tx: bytes) -> bytes:
    """
    Chunked xfer2 so we never exceed spidev list length limits.
    """
    rx = bytearray(len(tx))
    mv = memoryview(tx)
    offset = 0

    while offset < len(tx):
        end = min(offset + SPI_XFER_CHUNK, len(tx))
        chunk = mv[offset:end]
        r = spi.xfer2(chunk)  # accepts bytes-like
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
    """
    Loopback SPI source (no STM32):
      - We generate frame bytes locally and xfer them.
      - RX must match TX if MOSI is connected to MISO.
    """
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
            self.spi.max_speed_hz = self.max_speed_hz
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
            tx = make_frame(self.seq_seen)
            rx = spi_xfer_bytes(self.spi, tx)

            # quick debug
            nz = sum(b != 0 for b in rx[:256])
            self.last_err = f"rx_nonzero256={nz} first16={rx[:16].hex()}"

            ok, why = framing_validate_frame(rx)
            if not ok:
                self.bad_parse += 1
                self.last_err = f"parse:{why} rx_nonzero256={nz} first16={rx[:16].hex()}"
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
# 11. Mouse callback (buttons + bandpass drag)
# ===============================================================
CURSOR_POS = (0, 0)
CURRENT_FRAME: Optional[np.ndarray] = None
OUTPUT_DIR: Optional[Path] = None
CAMERA_AVAILABLE = False

sim_source = SimFFTSource()
spi_source = SpiFFTSource()

def handle_button_click(x, y, current_frame=None, output_dir=None, camera_available=False):
    global video_recorder

    if buttons["screenshot"].contains(x, y):
        if current_frame is not None and output_dir is not None:
            save_screenshot(current_frame, output_dir)
        return

    if buttons["record"].contains(x, y):
        button_state.is_recording = not button_state.is_recording
        if button_state.is_recording:
            if video_recorder and video_recorder.start_recording():
                buttons["record"].text = "Stop"
                buttons["record"].is_active = True
            else:
                button_state.is_recording = False
        else:
            if video_recorder:
                video_recorder.stop_recording()
            buttons["record"].text = "Start Recording"
            buttons["record"].is_active = False
            button_state.is_paused = False
            buttons["pause"].is_active = False
            buttons["pause"].text = "Pause"
        return

    if buttons["pause"].contains(x, y):
        if button_state.is_recording and video_recorder:
            button_state.is_paused = not button_state.is_paused
            buttons["pause"].is_active = button_state.is_paused
            if button_state.is_paused:
                video_recorder.pause_recording()
                buttons["pause"].text = "Resume"
            else:
                video_recorder.resume_recording()
                buttons["pause"].text = "Pause"
        return

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

def mouse_move(event, x, y, flags, param):
    global CURSOR_POS, DRAG_ACTIVE, DRAG_TARGET, F_MIN_HZ, F_MAX_HZ

    CURSOR_POS = (x, y)
    update_button_states(x, y)

    left_width, h = param
    bar_left = left_width

    if event == cv2.EVENT_LBUTTONDOWN:
        for b in buttons.values():
            if b.contains(x, y):
                handle_button_click(x, y, CURRENT_FRAME, OUTPUT_DIR, CAMERA_AVAILABLE)
                return

        if x >= bar_left:
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
        if DRAG_ACTIVE and x >= bar_left:
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

    print("Acoustic Imager - SIM + SPI LOOPBACK Bandpass Demo")
    print("=" * 70)
    print(f"FPS mode: {'THROTTLED' if FPS_THROTTLE_ENABLED else 'UNTHROTTLED'}")
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

    init_buttons(left_width, CAMERA_AVAILABLE)

    if FPS_THROTTLE_ENABLED:
        next_tick = time.perf_counter()
        period = 1.0 / max(1, FPS_TARGET)
    else:
        next_tick = 0.0
        period = 0.0

    frame_count = 0
    start_time = time.time()

    running_max_power = 1e-12
    last_spi_fft_data = None

    try:
        while True:
            if FPS_THROTTLE_ENABLED:
                next_tick += period
                sleep = next_tick - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.perf_counter()

            f_min = F_MIN_HZ
            f_max = F_MAX_HZ

            # Read source
            if button_state.source_mode == "SIM":
                fft_frame = sim_source.read_frame()
                source_label = "SIM"
                source_err = ""
            else:
                fft_frame = spi_source.read_frame()
                source_label = "SPI"
                source_err = spi_source.last_err

            if fft_frame is None or fft_frame.fft_data is None:
                if source_label == "SPI" and last_spi_fft_data is not None:
                    fft_data = last_spi_fft_data
                else:
                    fft_data = np.zeros((N_MICS, N_BINS), dtype=np.complex64)
            else:
                fft_data = fft_frame.fft_data
                if source_label == "SPI":
                    last_spi_fft_data = fft_data

            # Heatmap build
            if source_label == "SIM":
                selected_indices = [i for i, f in enumerate(SIM_SOURCE_FREQS) if f_min <= f <= f_max]
                if not selected_indices:
                    heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                else:
                    n_sel = len(selected_indices)
                    spec_matrix = np.zeros((n_sel, len(ANGLES)), dtype=np.float32)
                    power = np.zeros(n_sel, dtype=np.float32)

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
                band_mask = (f_axis >= f_min) & (f_axis <= f_max)
                if not np.any(band_mask):
                    heatmap_left = np.zeros((HEIGHT, left_width), dtype=np.uint8)
                else:
                    K = 3
                    band_idx = np.where(band_mask)[0]
                    mag2 = np.sum(np.abs(fft_data) ** 2, axis=0).real
                    mag2_band = mag2[band_idx]

                    topk_local = np.argsort(mag2_band)[-K:][::-1]
                    topk_bins = band_idx[topk_local]

                    spec_matrix = np.zeros((len(topk_bins), len(ANGLES)), dtype=np.float32)
                    power = np.zeros(len(topk_bins), dtype=np.float32)

                    for i, f_idx in enumerate(topk_bins):
                        f_sig = float(f_axis[f_idx])
                        Xf = fft_data[:, f_idx][:, np.newaxis]
                        R = Xf @ Xf.conj().T
                        spec_matrix[i, :] = music_spectrum(R, ANGLES, f_sig, n_sources=len(topk_bins))
                        p = float(np.sum(np.abs(Xf) ** 2).real)
                        power[i] = p
                        running_max_power = max(running_max_power, p)

                    power_rel = power / (running_max_power + 1e-12)
                    heatmap_left = spectra_to_heatmap_absolute(spec_matrix, power_rel, left_width, HEIGHT)

            # Background
            if use_camera and camera_type is not None and button_state.camera_enabled:
                try:
                    if camera_type == "picamera2" and picam2 is not None:
                        cam_frame = picam2.capture_array()
                        cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)
                    elif camera_type == "opencv" and cam is not None:
                        ret, cam_frame = cam.read()
                        if not ret:
                            cam_frame = None
                    else:
                        cam_frame = None

                    if cam_frame is not None and cam_frame.size > 0:
                        background = cv2.resize(cam_frame, (WIDTH, HEIGHT))
                    else:
                        background = background_full.copy()
                except Exception:
                    background = background_full.copy()
            else:
                background = background_full.copy()

            # Blend heatmap left
            left_bg = background[:, :left_width, :]
            colored = cv2.applyColorMap(heatmap_left, cv2.COLORMAP_MAGMA)

            mask = (heatmap_left.astype(np.float32) / 255.0)
            mask = np.power(mask, 0.5)
            mask3 = np.stack([mask] * 3, axis=-1)

            left_out = (colored * mask3 * ALPHA + left_bg * (1 - mask3 * ALPHA)).astype(np.uint8)
            background[:, :left_width, :] = left_out
            output_frame = background

            # Bars
            draw_frequency_bar(output_frame, fft_data, f_axis, f_min, f_max)
            draw_db_colorbar(output_frame, REL_DB_MIN, REL_DB_MAX)

            # Text
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
                msg = f"SPI {mhz:.0f}MHz | ok {spi_source.frames_ok} | badParse {spi_source.bad_parse} | badCRC {spi_source.bad_crc}"
                if source_err:
                    msg += f" | lastErr {source_err}"
                cv2.putText(output_frame, msg, (text_x, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            draw_buttons(output_frame)

            CURRENT_FRAME = output_frame.copy()

            if button_state.is_recording and video_recorder:
                video_recorder.write_frame(output_frame)

            cv2.imshow(WINDOW_NAME, output_frame)

            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("q") or key == 27:
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        print("Cleaning up...")
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

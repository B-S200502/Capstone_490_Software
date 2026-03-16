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
import threading
import subprocess

# Try to import picamera2 (modern Raspberry Pi camera library)
try:
    from picamera2 import Picamera2
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
        apply_heatmap_overlay,
        create_background_frame,
    )
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    sys.exit(1)

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

# >>> CAMERA ADDITION >>>
# ===============================================================
# CAMERA CONFIGURATION
# ===============================================================
CAMERA_INDEX = 0
CAMERA_WIDTH = WIDTH
CAMERA_HEIGHT = HEIGHT
USE_CAMERA = True
# For Raspberry Pi, you may need to use libcamera or picamera2
# Set to True to try libcamera-based capture (Raspberry Pi Camera Module)
USE_LIBCAMERA = True  # Try libcamera first (for Pi Camera Module)
# <<< CAMERA ADDITION <<<

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
# 2. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.010  # 2.5 cm radius → 5 cm diameter (correct)
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
    - MUSIC only determines ANGLE and spatial sharpness
    """

    Nsrc, Nang = spec_matrix.shape

    # ===========================================================
    # 1) Convert absolute power to dB (does NOT depend on other sources)
    # ===========================================================
    power_abs = np.maximum(power_per_source, 1e-12)
    power_db  = 10 * np.log10(power_abs)

    # Normalize into [0..1] based on fixed absolute dB range
    power_norm = (power_db - db_min) / (db_max - db_min)
    power_norm = np.clip(power_norm, 0.0, 1.0)

    # ===========================================================
    # 2) Find MUSIC peaks for the ANGLE ONLY (no amplitude scaling)
    # ===========================================================
    peak_indices = []
    sharpness = []

    for i in range(Nsrc):
        row = spec_matrix[i]

        # get location of peak
        idx = np.argmax(row)
        peak_indices.append(idx)

        # compute sharpness for blob width (NOT for amplitude)
        p  = row[idx]
        left  = row[idx-1] if idx > 0 else p
        right = row[idx+1] if idx < Nang-1 else p
        sh = max(p - 0.5*(left + right), 1e-12)
        sharpness.append(sh)

    sharpness = np.array(sharpness)
    sharpness /= sharpness.max() + 1e-12   # only affects width, not amplitude

    # ===========================================================
    # 3) Create heatmap canvas
    # ===========================================================
    h, w = out_height, out_width
    heatmap = np.zeros((h, w), dtype=np.float32)

    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def ang_to_px(idx: int) -> int:
        return int(idx / Nang * w)

    # ===========================================================
    # 4) Draw Gaussian blobs with absolute dB-scaling
    # ===========================================================
    base_radius = 60

    for i in range(Nsrc):
        cx = ang_to_px(peak_indices[i])
        cy = h // 2

        # blob width from MUSIC sharpness
        blob_radius = base_radius * (0.7 + 0.3 * sharpness[i])
        sigma = blob_radius / 1.8

        # ABSOLUTE amplitude = dB-normalized
        amp = power_norm[i]     # 0..1

        blob = amp * np.exp(
            -((xx - cx)**2 + (yy - cy)**2) / (2*sigma*sigma)
        )

        heatmap += blob

    # clip and convert to uint8
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_u8 = (heatmap * 255).astype(np.uint8)

    return heatmap_u8

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

def draw_db_colorbar(frame: np.ndarray,
                     db_min: float,
                     db_max: float,
                     width: int = 50):
    """
    Draw a vertical colorbar using the same colormap as the heatmap.
    """
    h = frame.shape[0]
    bar = np.zeros((h, width), dtype=np.uint8)

    # gradient bottom→top
    for y in range(h):
        val = y / (h - 1)         # 0 (bottom) → 1 (top)
        bar[h - 1 - y, :] = int(val * 255)

    # apply colormap
    bar_color = cv2.applyColorMap(bar, cv2.COLORMAP_TURBO)

    # overlay
    frame[:, :width] = bar_color

    # labels
    cv2.putText(frame, f"{db_max:.0f} dB", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(frame, f"{db_min:.0f} dB", (5, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

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
    print("Acoustic Imager - Fermat Spiral Bandpass Demo")
    print("=" * 70)

    GLOBAL_DB_MIN = -60.0
    GLOBAL_DB_MAX = 0.0

    background_full = create_background_frame(WIDTH, HEIGHT)
    left_width = WIDTH - FREQ_BAR_WIDTH

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, mouse_move)

    # >>> CAMERA ADDITION >>>
    cam = None
    picam2 = None
    use_camera = USE_CAMERA
    camera_type = None  # Will be 'picamera2', 'opencv', or None
    
    if USE_CAMERA:
        # ============================================================
        # METHOD 1: Try Picamera2 (Raspberry Pi Camera Module)
        # ============================================================
        if USE_LIBCAMERA and PICAMERA2_AVAILABLE:
            try:
                print("Attempting to open Raspberry Pi camera using picamera2...")
                picam2 = Picamera2()
                
                # Configure camera for video capture
                config = picam2.create_preview_configuration(
                    main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
                )
                picam2.configure(config)
                picam2.start()
                
                # Give camera time to warm up
                print("  Warming up camera...")
                time.sleep(2)
                
                # Test capture
                test_frame = picam2.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    print(f"  ✓ Picamera2 initialized successfully! Frame shape: {test_frame.shape}")
                    camera_type = 'picamera2'
                    use_camera = True
                else:
                    print("  Picamera2 opened but cannot capture frames")
                    picam2.stop()
                    picam2.close()
                    picam2 = None
            except Exception as e:
                print(f"  Picamera2 failed: {e}")
                if picam2 is not None:
                    try:
                        picam2.stop()
                        picam2.close()
                    except:
                        pass
                    picam2 = None
        
        # ============================================================
        # METHOD 2: Try libcamera-vid with GStreamer pipeline
        # ============================================================
        if camera_type is None and USE_LIBCAMERA:
            try:
                print("Attempting to open camera using GStreamer with libcamera...")
                # GStreamer pipeline for libcamera
                gst_pipeline = (
                    f"libcamerasrc ! "
                    f"video/x-raw,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={FPS}/1 ! "
                    f"videoconvert ! appsink"
                )
                
                cam = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                
                if cam.isOpened():
                    print("  Testing GStreamer libcamera pipeline...")
                    time.sleep(1)
                    ret, test_frame = cam.read()
                    
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"  ✓ GStreamer libcamera pipeline working! Frame shape: {test_frame.shape}")
                        camera_type = 'opencv'
                        use_camera = True
                    else:
                        print("  GStreamer pipeline opened but cannot read frames")
                        cam.release()
                        cam = None
                else:
                    print("  Could not open GStreamer libcamera pipeline")
                    cam = None
            except Exception as e:
                print(f"  GStreamer libcamera failed: {e}")
                if cam is not None:
                    cam.release()
                    cam = None
        
        # ============================================================
        # METHOD 3: Try standard OpenCV VideoCapture (USB cameras)
        # ============================================================
        if camera_type is None:
            try:
                print("Attempting to open camera using standard OpenCV...")
                camera_index = CAMERA_INDEX
                test_indices = [camera_index, 0, 1, 2] if camera_index != 0 else [0, 1, 2]
                test_indices = list(dict.fromkeys(test_indices))  # Remove duplicates
                
                for test_idx in test_indices:
                    print(f"  Trying camera index {test_idx}...")
                    
                    # Try V4L2 backend first
                    test_cam = cv2.VideoCapture(test_idx, cv2.CAP_V4L2)
                    if not test_cam.isOpened():
                        test_cam = cv2.VideoCapture(test_idx)
                    
                    if test_cam.isOpened():
                        # Configure camera
                        test_cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                        test_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                        test_cam.set(cv2.CAP_PROP_FPS, FPS)
                        test_cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Warm up and test
                        time.sleep(0.5)
                        for _ in range(5):  # Discard first few frames
                            test_cam.read()
                        
                        ret, test_frame = test_cam.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            if test_frame.shape[0] > 0 and test_frame.shape[1] > 0:
                                print(f"  ✓ OpenCV camera at index {test_idx} working! Frame shape: {test_frame.shape}")
                                cam = test_cam
                                camera_type = 'opencv'
                                use_camera = True
                                break
                        
                        test_cam.release()
                
                if camera_type is None:
                    print("  Could not find working camera with OpenCV")
            except Exception as e:
                print(f"  OpenCV camera failed: {e}")
                if cam is not None:
                    cam.release()
                    cam = None
        
        # ============================================================
        # Final status
        # ============================================================
        if camera_type is None:
            print("\n" + "="*70)
            print("ERROR: Could not initialize camera with any method")
            print("="*70)
            print("\nTroubleshooting for Raspberry Pi Camera Module:")
            print("  1. Enable camera: sudo raspi-config → Interface Options → Camera")
            print("  2. Install picamera2: sudo apt install -y python3-picamera2")
            print("  3. Test camera: libcamera-hello")
            print("  4. Check camera detection: libcamera-hello --list-cameras")
            print("\nTroubleshooting for USB cameras:")
            print("  5. List video devices: ls -l /dev/video*")
            print("  6. Check permissions: sudo usermod -a -G video $USER")
            print("  7. Check if in use: sudo lsof /dev/video*")
            print("\nFalling back to static background.")
            print("="*70 + "\n")
            use_camera = False
        else:
            print(f"\n✓ Camera initialized successfully using: {camera_type}")
    # <<< CAMERA ADDITION <<<

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
                    global ABSOLUTE_MAX_POWER
                    ABSOLUTE_MAX_POWER = max(ABSOLUTE_MAX_POWER, power)

                    p_db = 10*np.log10(power/(ABSOLUTE_MAX_POWER+1e-12))
                    GLOBAL_DB_MIN = min(GLOBAL_DB_MIN, p_db)
                    GLOBAL_DB_MAX = max(GLOBAL_DB_MAX, p_db)

                heatmap_full = spectra_to_heatmap_absolute(
                    spec_matrix,
                    power_per_source / (ABSOLUTE_MAX_POWER + 1e-12),
                    left_width,
                    HEIGHT,
                    db_min=GLOBAL_DB_MIN,
                    db_max=GLOBAL_DB_MAX
                )

                heatmap_left = heatmap_full

            # ===================================================
            # CAMERA BACKGROUND SUBSTITUTION (CORRECT LOCATION)
            # ===================================================
            if use_camera and camera_type is not None:
                cam_frame = None
                
                try:
                    if camera_type == 'picamera2' and picam2 is not None:
                        # Capture from picamera2
                        cam_frame = picam2.capture_array()
                        # picamera2 returns RGB, OpenCV uses BGR
                        if cam_frame is not None and cam_frame.size > 0:
                            cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)
                    
                    elif camera_type == 'opencv' and cam is not None:
                        # Capture from OpenCV VideoCapture
                        ret, cam_frame = cam.read()
                        if not ret or cam_frame is None:
                            cam_frame = None
                    
                    # Process captured frame
                    if cam_frame is not None and cam_frame.size > 0:
                        if cam_frame.shape[0] > 0 and cam_frame.shape[1] > 0:
                            background = cv2.resize(cam_frame, (WIDTH, HEIGHT))
                        else:
                            if frame_count % 60 == 0:
                                print(f"WARNING: Camera frame has invalid dimensions: {cam_frame.shape}")
                            background = background_full.copy()
                    else:
                        if frame_count % 60 == 0:
                            print(f"WARNING: Camera frame capture failed")
                        background = background_full.copy()
                
                except Exception as e:
                    if frame_count % 60 == 0:
                        print(f"WARNING: Camera capture exception: {e}")
                    background = background_full.copy()
            else:
                background = background_full.copy()

            # Compose background and overlay heatmap on left region
            left_bg = background[:, :left_width, :]
            
            # Apply heatmap with transparency for zero values
            # Convert heatmap to colored version
            colored_heatmap = cv2.applyColorMap(heatmap_left, cv2.COLORMAP_TURBO)
            
            # Create alpha mask based on heatmap intensity
            # Where heatmap is zero (or very low), make it fully transparent
            heatmap_mask = heatmap_left.astype(np.float32) / 255.0
            heatmap_mask = np.power(heatmap_mask, 0.5)  # Adjust curve for better visibility
            heatmap_mask = np.stack([heatmap_mask] * 3, axis=-1)  # Convert to 3-channel
            
            # Blend: only show heatmap where there's actual signal
            left_out = (colored_heatmap * heatmap_mask * ALPHA + 
                       left_bg * (1 - heatmap_mask * ALPHA)).astype(np.uint8)
            
            background[:, :left_width, :] = left_out
            output_frame = background

            # Draw frequency bar on the right
            draw_frequency_bar(output_frame, frame.fft_data, f_axis, f_min, f_max)

            # AUTO-SCALE dB RANGE BASED ON WHAT IS ACTUALLY VISIBLE
            if selected_indices:
                p_abs = np.maximum(power_per_source, 1e-12)
                p_db  = 10 * np.log10(p_abs)
                LOCAL_DB_MIN = np.min(p_db)
                LOCAL_DB_MAX = np.max(p_db)
            else:
                LOCAL_DB_MIN = -60
                LOCAL_DB_MAX = 0

            draw_db_colorbar(output_frame,
                             db_min=LOCAL_DB_MIN,
                             db_max=LOCAL_DB_MAX)

            elapsed = time.time() - start_time
            cv2.putText(output_frame, f"Frame: {frame_count}",
                        (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(output_frame, f"t = {elapsed:.2f}s",
                        (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, output_frame)

            key = cv2.waitKey(int(1000 // FPS)) & 0xFF
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord("q") or key == 27:
                break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nERROR: Unexpected exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if cam is not None:
            try:
                cam.release()
            except:
                pass
        if picam2 is not None:
            try:
                picam2.stop()
                picam2.close()
            except:
                pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
"""
Array geometry for the Acoustic Imager (measured layout).

The app's source of truth for coordinates/geometry at runtime is config.py. This module
holds the measured geometry (FreeCAD); run update_config_geometry.py to push these
values into config.py. Structure matches config: N_MICS, x_coords, y_coords, pitch.

Source: FreeCAD measurements. CSV X,Y,Z = vector from camera to mic.
"Array Center → Camera" = vector from array center to camera. A 180° rotation
in the XY plane is applied so the view matches physical setup (camera top-left
of array). Run as script for table, plot, CSV, and MUSIC summary.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# 1. Configuration
# ===============================================================
N_MICS = 16
SPEED_SOUND = 343.0
REF_FREQ_HZ = 30000.0  # reference for wavelength (middle of typical bandpass)

OUTPUT_CSV = Path(__file__).resolve().parent / "array_geometry.csv"

# Packed RFFT layout (one mic): matches firmware / unpack_packed_rfft_to_complex in spi_protocol.py
FFT_FRAME_SIZE = 512
FFT_HALF = 256
FFT_N_BINS = FFT_HALF + 1  # 257
# float array layout: fft_data[0]=Re(DC), fft_data[1]=Re(Nyquist); for i=1..255: fft_data[2*i]=Re(bin i), fft_data[2*i+1]=Im(bin i)

# ===============================================================
# 2. Raw measurement data (FreeCAD)
# ===============================================================
# Vector from array center to camera (mm). Source: CENTER_TO_CENTER = camera center → mic array center;
# we store center → camera, so negate: (-43.61, -29.65, -7.39).
ARRAY_CENTER_TO_CAMERA_MM = (-43.61, -29.65, -7.39)
ARRAY_CENTER_TO_CAMERA_DIST_MM = 53.25

# Mics: (label, min_dist_mm, x_mm, y_mm, z_mm); x,y,z = from camera to mic (source of truth for measurements)
MICS_RAW = [
    ("U1", 47.40, 41.71, 21.27, 7.39),
    ("U2", 33.02, 19.81, 5.37, 7.39),
    ("U3", 38.50, 30.31, 22.57, 7.39),
    ("U4", 36.68, 33.91, 11.87, 7.39),
    ("U6", 62.42, 55.21, 28.17, 7.39),
    ("U7", 46.73, 45.81, 5.57, 7.39),
    ("U8", 53.19, 50.41, 15.27, 7.39),
    ("U9", 65.23, 61.91, 19.17, 7.39),
    ("U11", 45.08, 25.31, 36.57, 7.39),
    ("U12", 75.85, 56.41, 50.17, 7.39),
    ("U13", 65.67, 46.61, 45.67, 7.39),
    ("U14", 62.74, 36.11, 50.77, 7.39),
    ("U16", 53.59, 35.71, 39.27, 7.39),
    ("U17", 52.76, 42.51, 30.37, 7.39),   # center most
    ("U18", 71.63, 60.41, 37.77, 7.39),
    ("U19", 59.17, 47.31, 34.77, 7.39),
]

# Payload index -> physical mic (firmware/SPI order; fft_data[row, :] = payload[row])
# Used to build x_coords/y_coords in payload order so MUSIC aligns with SPI.
PAYLOAD_TO_MIC = (
    "U3", "U2", "U1", "U4", "U8", "U6", "U9", "U7",
    "U18", "U16", "U19", "U17", "U13", "U12", "U11", "U14",
)

# Payload index -> (adc_ch, pin, mic_label) for table/reference
PAYLOAD_ADC_PIN_MIC = (
    (0, "ADC1 ch1 PA0", "U3"),
    (1, "ADC1 ch2 PA1", "U2"),
    (2, "ADC1 ch3 PA2", "U1"),
    (3, "ADC1 ch4 PA3", "U4"),
    (4, "ADC2 ch3 PA6", "U8"),
    (5, "ADC2 ch4 PA7", "U6"),
    (6, "ADC2 ch5 PC4", "U9"),
    (7, "ADC2 ch6 PC0", "U7"),
    (8, "ADC3 ch1 PB1", "U18"),
    (9, "ADC3 ch2 PE9", "U16"),
    (10, "ADC3 ch3 PE13", "U19"),
    (11, "ADC3 ch4 PE7", "U17"),
    (12, "ADC4 ch1 PE14", "U13"),
    (13, "ADC4 ch2 PE15", "U12"),
    (14, "ADC4 ch3 PB12", "U11"),
    (15, "ADC4 ch4 PB14", "U14"),
)


def _to_array_center_xy(x_cam_mm: float, y_cam_mm: float) -> tuple[float, float]:
    """Convert camera-frame (from camera to point) to array-center frame XY (mm)."""
    cx, cy = ARRAY_CENTER_TO_CAMERA_MM[0], ARRAY_CENTER_TO_CAMERA_MM[1]
    return (x_cam_mm + cx, y_cam_mm + cy)


def _flip_180(x_mm: float, y_mm: float) -> tuple[float, float]:
    """180° rotation in XY plane so camera appears top-left in 2D view."""
    return (-x_mm, -y_mm)


# ===============================================================
# 3. Geometry setup (measured array, flipped frame, payload order)
# ===============================================================
# Camera in array-center frame then flipped
_cam_center_x = ARRAY_CENTER_TO_CAMERA_MM[0]
_cam_center_y = ARRAY_CENTER_TO_CAMERA_MM[1]
cam_fx, cam_fy = _flip_180(_cam_center_x, _cam_center_y)

# Build per-mic lookup from MICS_RAW (label -> computed row)
_mics_by_label: dict[str, dict] = {}
for label, min_dist_mm, x_cam, y_cam, z_cam in MICS_RAW:
    x_center, y_center = _to_array_center_xy(x_cam, y_cam)
    x_f, y_f = _flip_180(x_center, y_center)
    dist_to_camera_mm = math.hypot(x_cam, y_cam, z_cam)
    dist_to_center_mm = math.hypot(x_center, y_center)
    azimuth_deg = math.degrees(math.atan2(y_f, x_f))
    x_m = x_f / 1000.0
    y_m = y_f / 1000.0
    _mics_by_label[label] = {
        "mic_label": label,
        "x_mm": round(x_f, 2),
        "y_mm": round(y_f, 2),
        "x_mm_raw": x_f,
        "y_mm_raw": y_f,
        "x_m": x_m,
        "y_m": y_m,
        "dist_to_camera_mm": round(dist_to_camera_mm, 2),
        "dist_to_center_mm": round(dist_to_center_mm, 2),
        "azimuth_deg": round(azimuth_deg, 2),
        "min_dist_mm": min_dist_mm,
    }

# Build geometry_table, x_coords, y_coords, x_mm_list, y_mm_list in payload order
# so index i = payload index i (matches fft_data[row, :] from SPI).
geometry_table = []
for payload_idx in range(N_MICS):
    mic = PAYLOAD_TO_MIC[payload_idx]
    row = {k: v for k, v in _mics_by_label[mic].items() if k not in ("x_mm_raw", "y_mm_raw")}
    row["payload_index"] = payload_idx
    geometry_table.append(row)

x_mm_list = [_mics_by_label[PAYLOAD_TO_MIC[i]]["x_mm_raw"] for i in range(N_MICS)]
y_mm_list = [_mics_by_label[PAYLOAD_TO_MIC[i]]["y_mm_raw"] for i in range(N_MICS)]

# Arrays in meters, payload order (same names as config.py for drop-in use)
x_coords = np.array([_mics_by_label[PAYLOAD_TO_MIC[i]]["x_m"] for i in range(N_MICS)], dtype=np.float64)
y_coords = np.array([_mics_by_label[PAYLOAD_TO_MIC[i]]["y_m"] for i in range(N_MICS)], dtype=np.float64)

# Pitch: mean radial spacing (m), used by ESPRIT
radii = sorted(set(round(math.hypot(x, y), 6) for x, y in zip(x_mm_list, y_mm_list)))
if len(radii) > 1:
    pitch = (sum(radii[i + 1] - radii[i] for i in range(len(radii) - 1)) / (len(radii) - 1)) / 1000.0
else:
    pitch = 0.0

# Aperture and wavelength (for MUSIC sanity checks)
aperture_radius = max(math.hypot(x_mm, y_mm) for x_mm, y_mm in zip(x_mm_list, y_mm_list)) / 1000.0
wavelength_ref = SPEED_SOUND / REF_FREQ_HZ
half_wavelength = wavelength_ref / 2.0

pair_dists = []
for i in range(N_MICS):
    for j in range(i + 1, N_MICS):
        pair_dists.append(math.hypot(x_coords[i] - x_coords[j], y_coords[i] - y_coords[j]))
min_pair_m = min(pair_dists) if pair_dists else 0.0
max_pair_m = max(pair_dists) if pair_dists else 0.0
mean_pair_m = sum(pair_dists) / len(pair_dists) if pair_dists else 0.0

# ===============================================================
# 4. FFT packed layout (reference only)
# ===============================================================
# Per-mic payload from firmware: FRAME_SIZE=512 floats, half=256.
# Layout: fft_data[0]=Re(DC bin 0), fft_data[1]=Re(Nyquist bin 256);
# for i=1..255: fft_data[2*i]=Re(bin i), fft_data[2*i+1]=Im(bin i).
# Implemented in acoustic_imager.spi.spi_protocol.unpack_packed_rfft_to_complex.


def run() -> None:
    """Print table, show 2D plot, write CSV, print MUSIC summary."""
    # ---- 2D plot (matplotlib popup) ----
    disp_cam_x, disp_cam_y = 0.0, 0.0
    disp_mic_x = [x_mm - cam_fx for x_mm in x_mm_list]
    disp_mic_y = [y_mm - cam_fy for y_mm in y_mm_list]
    min_dx = min(disp_cam_x, *disp_mic_x)
    min_dy = min(disp_cam_y, *disp_mic_y)
    margin_mm = 5.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_xlim(min_dx - margin_mm, 0 + margin_mm)
    ax.set_ylim(min_dy - margin_mm, 0 + margin_mm)
    ax.invert_xaxis()
    ax.set_xlabel("x (mm), camera at origin")
    ax.set_ylabel("y (mm)")
    ax.set_title("Array geometry — camera top-left, mic array (180° flip applied)")

    ax.plot(disp_cam_x, disp_cam_y, "k^", markersize=12, label="Camera", zorder=5)
    ax.annotate("Camera", (disp_cam_x, disp_cam_y), xytext=(5, -5), textcoords="offset points", fontsize=9)

    for label, (dx, dy) in zip(PAYLOAD_TO_MIC, zip(disp_mic_x, disp_mic_y)):
        ax.plot(dx, dy, "o", color="C0", markersize=6, zorder=3)
        ax.annotate(label, (dx, dy), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # ---- Table to stdout ----
    print("TABLE (flipped frame, array center = origin; row index = payload index)")
    print("-" * 110)
    header = "pay  label   x_mm    y_mm    x_m      y_m      dist_cam_mm  dist_center_mm  azimuth_deg  min_dist_mm"
    print(header)
    print("-" * 110)
    for r in geometry_table:
        print(
            f"{r['payload_index']:2}   {r['mic_label']:<6} "
            f"{r['x_mm']:7.2f} {r['y_mm']:7.2f} "
            f"{r['x_m']:.6f} {r['y_m']:.6f} "
            f"{r['dist_to_camera_mm']:12.2f} {r['dist_to_center_mm']:15.2f} "
            f"{r['azimuth_deg']:11.2f} {r['min_dist_mm']:11.2f}"
        )
    print()

    # ---- Write CSV ----
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "payload_index", "mic_label", "x_mm", "y_mm", "x_m", "y_m",
                "dist_to_camera_mm", "dist_to_center_mm", "azimuth_deg", "min_dist_mm",
            ],
        )
        w.writeheader()
        w.writerows(geometry_table)
    print(f"Wrote table to: {OUTPUT_CSV}")
    print()

    # ---- MUSIC-ready summary ----
    print("MUSIC-ready summary (use in config/beamforming)")
    print("-" * 60)
    print("Payload order (row index = payload index): 0=U3, 1=U2, 2=U1, 3=U4, 4=U8, 5=U6, 6=U9, 7=U7,")
    print("                                           8=U18, 9=U16, 10=U19, 11=U17, 12=U13, 13=U12, 14=U11, 15=U14")
    print()
    print("x_coords (m) = np.array([")
    print("    " + ", ".join(f"{x:.6f}" for x in x_coords))
    print("])")
    print()
    print("y_coords (m) = np.array([")
    print("    " + ", ".join(f"{y:.6f}" for y in y_coords))
    print("])")
    print()
    print(f"pitch (m) = {pitch:.6f}")
    print(f"speed_sound (m/s) = {SPEED_SOUND:.1f}")
    print(f"aperture_radius (m) = {aperture_radius:.6f}")
    print(f"wavelength at {REF_FREQ_HZ/1000:.0f} kHz (m) = {wavelength_ref:.6f}  (λ/2 = {half_wavelength:.6f})")
    print(f"pairwise distances (m): min = {min_pair_m:.6f}, max = {max_pair_m:.6f}, mean = {mean_pair_m:.6f}")
    print("  (keep max_pair < λ/2 to avoid grating lobes at ref frequency)")
    print()


if __name__ == "__main__":
    run()

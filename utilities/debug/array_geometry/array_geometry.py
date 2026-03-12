#!/usr/bin/env python3
"""
Array geometry debug/utility script.

Purpose: Geometry sanity check for calibrating audio signals to the heatmap.
- Prints a 2D representation of the camera and mic array (correct perspective:
  camera top-left of array; measurements were taken in that perspective).
- Outputs a table of metrics (distances, angles, mic labels) to stdout and to
  array_geometry.csv for cross-check and for feeding into the MUSIC pipeline.

Source: FreeCAD measurements. CSV X,Y,Z are vectors from camera to each mic.
"Array Center → Camera" is the vector from array center to camera.
A 180° rotation in the XY plane is applied so the 2D view matches the real
orientation (camera top-left of array). All geometry and MUSIC-ready arrays
use this flipped frame. Channel order is fixed and documented below.

Run: python utilities/array_geometry.py  (no arguments; all inputs hardcoded.)
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Hardcoded data from FreeCAD (camera-centric: X,Y,Z = vector from camera to mic)
# "Array Center → Camera" = vector from array center to camera (58 mm, 47.05, 32.80, 8.64)
# Channel order: 0=U1, 1=U2, 2=U3, 3=U4, 4=U6, 5=U7, 6=U8, 7=U9, 8=U11, 9=U12,
#                10=U13, 11=U14, 12=U16, 13=U17, 14=U18, 15=U19
# -----------------------------------------------------------------------------

# Vector from array center to camera (mm)
ARRAY_CENTER_TO_CAMERA_MM = (47.05, 32.80, 8.64)
ARRAY_CENTER_TO_CAMERA_DIST_MM = 58.00

# Mics: (label, min_dist_mm, x_mm, y_mm, z_mm) in channel order; x,y,z = from camera to mic
MICS_RAW = [
    ("U1", 51.86, 45.00, 24.30, 8.64),
    ("U2", 37.61, 23.10, 28.40, 8.64),
    ("U3", 43.11, 33.60, 25.60, 8.64),
    ("U4", 40.99, 37.20, 14.90, 8.64),
    ("U6", 66.86, 58.50, 31.20, 8.64),
    ("U7", 50.59, 49.10, 8.60, 8.64),
    ("U8", 57.38, 53.70, 18.30, 8.64),
    ("U9", 69.41, 65.20, 22.20, 8.64),
    ("U11", 49.60, 28.60, 39.60, 8.64),
    ("U12", 80.43, 59.70, 53.20, 8.64),
    ("U13", 70.26, 49.90, 48.70, 8.64),
    ("U14", 67.24, 39.40, 53.80, 8.64),
    ("U16", 58.18, 39.00, 42.30, 8.64),
    ("U17", 57.34, 45.80, 33.40, 8.64),   # center most
    ("U18", 76.14, 63.70, 40.80, 8.64),
    ("U19", 63.75, 50.60, 37.80, 8.64),
]

# Output file (next to this script)
OUTPUT_CSV = Path(__file__).resolve().parent / "array_geometry.csv"


def _to_array_center_xy(x_cam_mm: float, y_cam_mm: float) -> tuple[float, float]:
    """Convert camera-frame (from camera to point) to array-center frame XY (mm)."""
    cx, cy = ARRAY_CENTER_TO_CAMERA_MM[0], ARRAY_CENTER_TO_CAMERA_MM[1]
    # In camera frame, array center is at (-cx, -cy). So point in camera frame (x_cam, y_cam)
    # is at (x_cam - (-cx), y_cam - (-cy)) = (x_cam + cx, y_cam + cy) from array center.
    return (x_cam_mm + cx, y_cam_mm + cy)


def _flip_180(x_mm: float, y_mm: float) -> tuple[float, float]:
    """180° rotation in XY plane so camera appears top-left in 2D view."""
    return (-x_mm, -y_mm)


def main() -> None:
    # Camera in array-center frame: "Array Center → Camera" = vector from center to camera
    cam_center_x = ARRAY_CENTER_TO_CAMERA_MM[0]
    cam_center_y = ARRAY_CENTER_TO_CAMERA_MM[1]
    cam_fx, cam_fy = _flip_180(cam_center_x, cam_center_y)

    # Build per-mic data in flipped array-center frame (mm then m)
    rows: list[dict] = []
    x_mm_list: list[float] = []
    y_mm_list: list[float] = []

    for ch, (label, min_dist_mm, x_cam, y_cam, z_cam) in enumerate(MICS_RAW):
        x_center, y_center = _to_array_center_xy(x_cam, y_cam)
        x_f, y_f = _flip_180(x_center, y_center)

        x_mm_list.append(x_f)
        y_mm_list.append(y_f)

        dist_to_camera_mm = math.hypot(x_cam, y_cam, z_cam)
        dist_to_center_mm = math.hypot(x_center, y_center)
        azimuth_deg = math.degrees(math.atan2(y_f, x_f))

        x_m = x_f / 1000.0
        y_m = y_f / 1000.0

        rows.append({
            "channel_index": ch,
            "mic_label": label,
            "x_mm": round(x_f, 2),
            "y_mm": round(y_f, 2),
            "x_m": x_m,
            "y_m": y_m,
            "dist_to_camera_mm": round(dist_to_camera_mm, 2),
            "dist_to_center_mm": round(dist_to_center_mm, 2),
            "azimuth_deg": round(azimuth_deg, 2),
            "min_dist_mm": min_dist_mm,
        })

    # Pitch (mean radial spacing) in meters, from flipped positions
    radii = sorted(set(round(math.hypot(x, y), 6) for x, y in zip(x_mm_list, y_mm_list)))
    if len(radii) > 1:
        pitch_m = (sum(radii[i + 1] - radii[i] for i in range(len(radii) - 1)) / (len(radii) - 1)) / 1000.0
    else:
        pitch_m = 0.0

    # ---- 2D plot (matplotlib popup) ----
    # Display coords: camera at (0,0), mics at (dx, dy) with dx, dy negative (array on other side).
    disp_cam_x, disp_cam_y = 0.0, 0.0
    disp_mic_x = [x_mm - cam_fx for x_mm in x_mm_list]
    disp_mic_y = [y_mm - cam_fy for y_mm in y_mm_list]
    min_dx = min(disp_cam_x, *disp_mic_x)
    min_dy = min(disp_cam_y, *disp_mic_y)
    margin_mm = 5.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    # Camera at (0, 0) top-left: x from 0 to min_dx, y from 0 to min_dy; invert x so 0 is left
    ax.set_xlim(min_dx - margin_mm, 0 + margin_mm)
    ax.set_ylim(min_dy - margin_mm, 0 + margin_mm)
    ax.invert_xaxis()
    ax.set_xlabel("x (mm), camera at origin")
    ax.set_ylabel("y (mm)")
    ax.set_title("Array geometry — camera top-left, mic array (180° flip applied)")

    ax.plot(disp_cam_x, disp_cam_y, "k^", markersize=12, label="Camera", zorder=5)
    ax.annotate("Camera", (disp_cam_x, disp_cam_y), xytext=(5, -5), textcoords="offset points", fontsize=9)

    for (label, _, _, _, _), (dx, dy) in zip(MICS_RAW, zip(disp_mic_x, disp_mic_y)):
        ax.plot(dx, dy, "o", color="C0", markersize=6, zorder=3)
        ax.annotate(label, (dx, dy), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.grid(True, linestyle=":", alpha=0.7)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # ---- Table to stdout ----
    print("TABLE (flipped frame, array center = origin)")
    print("-" * 100)
    header = "ch  label   x_mm    y_mm    x_m      y_m      dist_cam_mm  dist_center_mm  azimuth_deg  min_dist_mm"
    print(header)
    print("-" * 100)
    for r in rows:
        print(
            f"{r['channel_index']:2}   {r['mic_label']:<6} "
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
                "channel_index", "mic_label", "x_mm", "y_mm", "x_m", "y_m",
                "dist_to_camera_mm", "dist_to_center_mm", "azimuth_deg", "min_dist_mm",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote table to: {OUTPUT_CSV}")
    print()

    # ---- MUSIC-ready summary ----
    x_coords_m = [r["x_m"] for r in rows]
    y_coords_m = [r["y_m"] for r in rows]

    print("MUSIC-ready summary (use in config/beamforming)")
    print("-" * 60)
    print("Channel order: 0=U1, 1=U2, 2=U3, 3=U4, 4=U6, 5=U7, 6=U8, 7=U9,")
    print("               8=U11, 9=U12, 10=U13, 11=U14, 12=U16, 13=U17, 14=U18, 15=U19")
    print()
    print("x_coords (m) = np.array([")
    print("    " + ", ".join(f"{x:.6f}" for x in x_coords_m))
    print("])")
    print()
    print("y_coords (m) = np.array([")
    print("    " + ", ".join(f"{y:.6f}" for y in y_coords_m))
    print("])")
    print()
    print(f"pitch (m) = {pitch_m:.6f}")
    print()


if __name__ == "__main__":
    main()

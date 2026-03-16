#!/usr/bin/env python3
"""
Print current metrics useful for debugging: config values, and with --live per-mic levels
and suggested gains. Stop the acoustic imager before using --live.
When gain control (GPIO) is available, --live runs the test at LOW gain then HIGH gain and
prints both tables in the same output.

Averaging: Temporal averaging (e.g. over multiple FFT frames or covariance) can be done
on the Pi for a smoother heatmap or in firmware for lower-rate updates. The app pipeline
is per-frame; firmware may add its own averaging later.

Usage (from repo root):
  python3 utilities/calibration/metrics_debug.py --config
  python3 utilities/calibration/metrics_debug.py --live
  python3 utilities/calibration/metrics_debug.py --live --frames 5 --average
  python3 utilities/calibration/metrics_debug.py --live --frames 5 --average --write-config  # update config.SPI_MIC_GAIN
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC_SOFTWARE = os.path.join(_REPO_ROOT, "src", "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

import numpy as np

try:
    from acoustic_imager import config
except ImportError:
    config = None


def print_config_metrics() -> None:
    """Print current heatmap/SPI-related config for debugging."""
    if config is None:
        print("Config not available (run from repo root).")
        return
    print("--- Config (heatmap / SPI) ---")
    print(f"  N_MICS                    = {getattr(config, 'N_MICS', 'N/A')}")
    print(f"  N_BINS                    = {getattr(config, 'N_BINS', 'N/A')}")
    print(f"  SPI_TOP_K_BINS            = {getattr(config, 'SPI_TOP_K_BINS', 'N/A')}")
    print(f"  SPI_NOISE_FLOOR_DB        = {getattr(config, 'SPI_NOISE_FLOOR_DB', 'N/A')}")
    print(f"  SPI_MUSIC_N_SOURCES       = {getattr(config, 'SPI_MUSIC_N_SOURCES', 'N/A')}")
    print(f"  SPI_DIRECTIVITY_MIN       = {getattr(config, 'SPI_DIRECTIVITY_MIN', 'N/A')}")
    print(f"  SPI_ANGLE_STABILITY_DEG   = {getattr(config, 'SPI_ANGLE_STABILITY_DEG', 'N/A')}")
    print(f"  SPI_PER_MIC_NORMALIZE     = {getattr(config, 'SPI_PER_MIC_NORMALIZE', 'N/A')}")
    print(f"  SPI_ARRAY_GAIN            = {getattr(config, 'SPI_ARRAY_GAIN', 'N/A')}")
    gains = getattr(config, "SPI_MIC_GAIN", None)
    if gains is not None:
        g = list(gains) if hasattr(gains, "__iter__") else [gains]
        print(f"  SPI_MIC_GAIN              = {g}")
    print(f"  HEATMAP_SMOOTH_ALPHA      = {getattr(config, 'HEATMAP_SMOOTH_ALPHA', 'N/A')}")
    print(f"  HEATMAP_LEVEL_FLOOR       = {getattr(config, 'HEATMAP_LEVEL_FLOOR', 'N/A')}")
    print(f"  HEATMAP_LEVEL_REFERENCE    = {getattr(config, 'HEATMAP_LEVEL_REFERENCE', 'N/A')}")
    x_hw = getattr(config, "x_coords_hw", None)
    if x_hw is not None and getattr(config, "y_coords_hw", None) is not None:
        y_hw = config.y_coords_hw
        print(f"  Geometry (HW): x_coords_hw, y_coords_hw  shape {getattr(x_hw, 'shape', (len(x_hw),))}, {getattr(y_hw, 'shape', (len(y_hw),))}")
    else:
        print("  Geometry (HW): not set")


# Max FFT magnitude per bin to avoid overflow in L2 norm (sum of squares).
_FFT_MAG_CAP = 1e10
# Cap suggested gain in table so we don't print 1e13 (show "100+" when higher).
_SUGGESTED_GAIN_CAP = 100.0
# Payload index -> ADC number (1-4). Matches hardware: mics 0-3 ADC1, 4-7 ADC2, 8-11 ADC3, 12-15 ADC4.
PAYLOAD_TO_ADC = (1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)


def _sanitize_fft_for_metrics(fft_data: np.ndarray) -> np.ndarray:
    """Replace non-finite with 0 and cap magnitude so L2 norm won't overflow."""
    out = np.asarray(fft_data, dtype=np.complex64).copy()
    out[~np.isfinite(out)] = 0.0
    mag = np.abs(out)
    scale = np.where(mag > _FFT_MAG_CAP, _FFT_MAG_CAP / (mag + 1e-30), 1.0)
    out = (out * scale).astype(np.complex64)
    return out


def per_mic_levels(fft_data: np.ndarray) -> np.ndarray:
    """L2 norm per mic across bins (length N_MICS). Call after _sanitize_fft_for_metrics to avoid overflow."""
    if fft_data is None or fft_data.size == 0:
        return np.array([])
    return np.sqrt(np.sum(np.abs(fft_data) ** 2, axis=1)).astype(np.float64)


def _write_spi_mic_gain_to_config(config_path: str, gains: list[float]) -> bool:
    """Replace SPI_MIC_GAIN line in config.py with the given tuple of gains. Returns True on success."""
    if not gains:
        return False
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError as e:
        print(f"  Failed to read config: {e}")
        return False
    # Match line like: SPI_MIC_GAIN = (1.0,) * N_MICS   or   SPI_MIC_GAIN = (1.0, 2.0, ...)
    new_line = "SPI_MIC_GAIN = (" + ", ".join(f"{g:.2f}" for g in gains) + ")"
    pattern = re.compile(r"^SPI_MIC_GAIN\s*=\s*\([^)]*\).*$", re.MULTILINE)
    if not pattern.search(content):
        print("  SPI_MIC_GAIN line not found in config.")
        return False
    new_content = pattern.sub(new_line, content, count=1)
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(new_content)
    except OSError as e:
        print(f"  Failed to write config: {e}")
        return False
    return True


def _collect_frames(mgr, num_frames: int, duration_sec: float) -> list:
    """Collect num_frames from SPIManager within duration_sec. Returns list of fft_data."""
    frames_fft = []
    deadline = time.monotonic() + duration_sec
    while len(frames_fft) < num_frames and time.monotonic() < deadline:
        lf = mgr.get_latest()
        if lf.ok and lf.fft_data is not None:
            frames_fft.append(lf.fft_data.copy())
        time.sleep(max(0.05, duration_sec / max(1, num_frames)))
    return frames_fft


def _print_metrics_table(
    frames_fft: list,
    label: str,
    average: bool,
) -> list[float]:
    """
    Compute levels from frames, print full per-mic table and per-ADC summary.
    Returns suggested_gains (length n_mics) for possible --write-config.
    """
    if not frames_fft:
        return []

    if average and len(frames_fft) > 1:
        fft_data = np.mean(frames_fft, axis=0).astype(np.complex64)
        sub = f"averaged over {len(frames_fft)} frames"
    else:
        fft_data = frames_fft[-1].copy()
        sub = f"latest of {len(frames_fft)} frames"
    if label:
        print(f"--- Per-mic levels ({label}, {sub}) ---")
    else:
        print(f"--- Per-mic levels ({sub}) ---")

    if np.any(~np.isfinite(fft_data)):
        print("Warning: frame contains inf/nan; metrics may be invalid.")
        print("  Run on Pi with GPIO for frame-aligned reads, or check firmware for bad FFT output.")
    fft_data = _sanitize_fft_for_metrics(fft_data)

    n_mics = fft_data.shape[0]
    levels = per_mic_levels(fft_data)
    if levels.size == 0:
        print("No data.")
        return []

    levels = np.nan_to_num(levels, nan=0.0, posinf=0.0, neginf=0.0)
    levels = np.clip(levels, 0.0, 1e10)

    total_power = float(np.sum(levels ** 2)) + 1e-12
    max_level = float(np.max(levels)) + 1e-12
    min_level = float(np.min(levels)) + 1e-12
    ratio_max_min = max_level / min_level if min_level > 0 else 0.0
    if not np.isfinite(ratio_max_min) or ratio_max_min > 1e6:
        ratio_max_min_str = ">1e6"
    else:
        ratio_max_min_str = f"{ratio_max_min:.2f}"

    print(f"  Total power (sum of squared L2 per mic): {total_power:.6e}")
    print(f"  Max mic L2: {max_level:.6e}  Min mic L2: {min_level:.6e}")
    print(f"  Ratio max/min: {ratio_max_min_str}")
    print("")
    print("  mic   ADC   L2 norm      dB vs max   ratio    suggested gain (to match max)")
    print("  ---   ---   ---------   ---------   -----    -----------------------------")

    suggested_gains: list[float] = []
    db_vals: list[float] = []
    for m in range(n_mics):
        adc = PAYLOAD_TO_ADC[m] if m < len(PAYLOAD_TO_ADC) else "-"
        L = float(levels[m]) + 1e-12
        if max_level <= 0:
            db_val, ratio_val, suggested_val = 0.0, 0.0, 1.0
        else:
            ratio_val = L / max_level
            if ratio_val > 0 and np.isfinite(ratio_val):
                db_val = 10.0 * np.log10(ratio_val)
            else:
                db_val = 0.0
            if L > 0 and np.isfinite(max_level / L):
                suggested_val = max_level / L
            else:
                suggested_val = 1.0
            if not np.isfinite(db_val):
                db_val = 0.0
            if not np.isfinite(ratio_val):
                ratio_val = 0.0
            if not np.isfinite(suggested_val):
                suggested_val = 1.0
        suggested_gains.append(min(suggested_val, _SUGGESTED_GAIN_CAP))
        db_vals.append(db_val)
        if suggested_val > _SUGGESTED_GAIN_CAP:
            suggested_str = f"{_SUGGESTED_GAIN_CAP:.0f}+"
        else:
            suggested_str = f"{suggested_val:.2f}"
        adc_str = f"ADC{adc}" if adc != "-" else "  -"
        print(f"  {m:3d}   {adc_str:4}   {L:.6e}   {db_val:+.1f} dB    {ratio_val:.3f}     {suggested_str}")

    if len(db_vals) == n_mics and n_mics <= len(PAYLOAD_TO_ADC):
        print("")
        for adc_num in (1, 2, 3, 4):
            indices = [m for m in range(n_mics) if m < len(PAYLOAD_TO_ADC) and PAYLOAD_TO_ADC[m] == adc_num]
            if indices:
                mean_db = float(np.mean([db_vals[m] for m in indices]))
                print(f"  ADC{adc_num} (mics {indices}): mean dB vs max = {mean_db:+.1f} dB")

    weak_threshold_ratio = 0.35
    weak = [m for m in range(n_mics) if (float(levels[m]) + 1e-12) < max_level * weak_threshold_ratio]
    if weak:
        print("")
        print(f"  Mics with level < {weak_threshold_ratio*100:.0f}% of max (candidates for SPI_MIC_GAIN boost): {weak}")
        print("  Example: set those indices to suggested gain in config.SPI_MIC_GAIN.")
    print("")
    return suggested_gains


def run_live_metrics(
    num_frames: int = 5,
    duration_sec: float = 2.0,
    average: bool = False,
    write_config: bool = False,
    write_from: str = "high",
) -> None:
    """Read frames from HW SPI and print per-mic levels. With gain control: run at LOW then HIGH gain, both in same output."""
    if config is None:
        print("Config not available.")
        return
    try:
        from acoustic_imager.io.spi_manager import SPIManager
    except ImportError as e:
        print(f"SPIManager import failed: {e}")
        return

    gain_control = None
    try:
        from acoustic_imager.io.gain_control import GAIN_CONTROL
        if getattr(GAIN_CONTROL, "enabled", False):
            gain_control = GAIN_CONTROL
    except Exception:
        pass

    mgr = SPIManager(use_frame_ready=True)
    mgr.start()
    try:
        if gain_control is not None:
            gain_control.set_mode("LOW")
            time.sleep(0.35)
            frames_low = _collect_frames(mgr, num_frames, duration_sec)
            if not frames_low:
                print("No valid frames at LOW gain. Is the acoustic imager stopped and HW connected?")
                return
            suggested_low = _print_metrics_table(frames_low, "LOW gain", average)

            gain_control.set_mode("HIGH")
            time.sleep(0.35)
            frames_high = _collect_frames(mgr, num_frames, duration_sec)
            if not frames_high:
                print("No valid frames at HIGH gain.")
                return
            suggested_high = _print_metrics_table(frames_high, "HIGH gain", average)
            n_mics = getattr(config, "N_MICS", 16)
            suggested_gains = suggested_high if write_from == "high" else suggested_low
            if write_config and config is not None and len(suggested_gains) == n_mics:
                config_path = getattr(config, "__file__", None)
                if not config_path or not os.path.isfile(config_path):
                    config_path = os.path.join(_REPO_ROOT, "src", "software", "acoustic_imager", "config.py")
                if _write_spi_mic_gain_to_config(config_path, suggested_gains):
                    print(f"  Wrote SPI_MIC_GAIN to {config_path} (from {write_from.upper()} gain run)")
                    print("  Restart the acoustic imager for changes to take effect.")
                else:
                    print("  --write-config: could not update config (see errors above).")
        else:
            frames_fft = _collect_frames(mgr, num_frames, duration_sec)
            if not frames_fft:
                print("No valid frames received. Is the acoustic imager stopped and HW connected?")
                return
            suggested_gains = _print_metrics_table(frames_fft, "", average)
            n_mics = getattr(config, "N_MICS", 16)
            if write_config and config is not None and len(suggested_gains) == n_mics:
                config_path = getattr(config, "__file__", None)
                if not config_path or not os.path.isfile(config_path):
                    config_path = os.path.join(_REPO_ROOT, "src", "software", "acoustic_imager", "config.py")
                if _write_spi_mic_gain_to_config(config_path, suggested_gains):
                    print(f"  Wrote SPI_MIC_GAIN to {config_path}")
                    print("  Restart the acoustic imager for changes to take effect.")
                else:
                    print("  --write-config: could not update config (see errors above).")
    finally:
        mgr.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Print debug metrics: config and/or live per-mic levels")
    ap.add_argument("--config", action="store_true", help="Print heatmap/SPI config values")
    ap.add_argument("--live", action="store_true", help="Read HW SPI and print per-mic levels (stop app first)")
    ap.add_argument("--frames", type=int, default=5, help="Frames to read for --live (default 5)")
    ap.add_argument("--sec", type=float, default=2.0, help="Max seconds to collect --live frames (default 2)")
    ap.add_argument("--average", action="store_true", help="Average FFT over collected frames before computing levels")
    ap.add_argument("--write-config", action="store_true", help="After --live, write suggested gains to config.SPI_MIC_GAIN")
    ap.add_argument("--write-from", choices=("low", "high"), default="high",
                    help="With --write-config and dual gain: which run to use (default: high)")
    args = ap.parse_args()

    if not args.config and not args.live:
        ap.print_help()
        print("\nUse --config and/or --live.")
        return 0

    if args.config:
        print_config_metrics()
        print("")

    if args.live:
        run_live_metrics(
            num_frames=args.frames,
            duration_sec=args.sec,
            average=args.average,
            write_config=args.write_config,
            write_from=args.write_from,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

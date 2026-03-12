#!/usr/bin/env python3
"""
Print current metrics useful for debugging: config values, and with --live per-mic levels
and suggested gains. Stop the acoustic imager before using --live.

Averaging: Temporal averaging (e.g. over multiple FFT frames or covariance) can be done
on the Pi for a smoother heatmap or in firmware for lower-rate updates. The app pipeline
is per-frame; firmware may add its own averaging later.

Usage (from repo root):
  python3 utilities/debug/metrics_debug.py --config
  python3 utilities/debug/metrics_debug.py --live
  python3 utilities/debug/metrics_debug.py --live --frames 5 --average
  python3 utilities/debug/metrics_debug.py --live --frames 5 --average --write-config  # update config.SPI_MIC_GAIN
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


# Max FFT magnitude per bin to avoid overflow in L2 norm (sum of squares).
_FFT_MAG_CAP = 1e10
# Cap suggested gain in table so we don't print 1e13 (show "100+" when higher).
_SUGGESTED_GAIN_CAP = 100.0


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


def run_live_metrics(
    num_frames: int = 5,
    duration_sec: float = 2.0,
    average: bool = False,
    write_config: bool = False,
) -> None:
    """Read frames from HW SPI and print per-mic levels and suggested gains."""
    if config is None:
        print("Config not available.")
        return
    try:
        from acoustic_imager.io.spi_manager import SPIManager
    except ImportError as e:
        print(f"SPIManager import failed: {e}")
        return

    mgr = SPIManager(use_frame_ready=True)
    mgr.start()
    frames_fft = []
    deadline = time.monotonic() + duration_sec
    try:
        while len(frames_fft) < num_frames and time.monotonic() < deadline:
            lf = mgr.get_latest()
            if lf.ok and lf.fft_data is not None:
                frames_fft.append(lf.fft_data.copy())
            time.sleep(max(0.05, duration_sec / max(1, num_frames)))
    finally:
        mgr.stop()

    if not frames_fft:
        print("No valid frames received. Is the acoustic imager stopped and HW connected?")
        return

    if average and len(frames_fft) > 1:
        fft_data = np.mean(frames_fft, axis=0).astype(np.complex64)
        print(f"--- Per-mic levels (averaged over {len(frames_fft)} frames) ---")
    else:
        fft_data = frames_fft[-1].copy()
        print(f"--- Per-mic levels (latest of {len(frames_fft)} frames) ---")

    if np.any(~np.isfinite(fft_data)):
        print("Warning: frame contains inf/nan; metrics may be invalid.")
        print("  Run on Pi with GPIO for frame-aligned reads, or check firmware for bad FFT output.")
    fft_data = _sanitize_fft_for_metrics(fft_data)

    n_mics = fft_data.shape[0]
    levels = per_mic_levels(fft_data)
    if levels.size == 0:
        print("No data.")
        return

    # Sanitize levels so overflow/garbage never produces inf/nan in summary or table
    levels = np.nan_to_num(levels, nan=0.0, posinf=0.0, neginf=0.0)
    levels = np.clip(levels, 0.0, 1e10)

    total_power = float(np.sum(levels ** 2)) + 1e-12
    max_level = float(np.max(levels)) + 1e-12
    min_level = float(np.min(levels)) + 1e-12
    # Keep ratio finite for display; cap huge values for readability
    ratio_max_min = max_level / min_level if min_level > 0 else 0.0
    if not np.isfinite(ratio_max_min) or ratio_max_min > 1e6:
        ratio_max_min_str = ">1e6"
    else:
        ratio_max_min_str = f"{ratio_max_min:.2f}"

    print(f"  Total power (sum of squared L2 per mic): {total_power:.6e}")
    print(f"  Max mic L2: {max_level:.6e}  Min mic L2: {min_level:.6e}")
    print(f"  Ratio max/min: {ratio_max_min_str}")
    print("")
    print("  mic   L2 norm      dB vs max   ratio    suggested gain (to match max)")
    print("  ---   ---------   ---------   -----    -----------------------------")

    suggested_gains: list[float] = []
    for m in range(n_mics):
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
        suggested_gains.append(min(suggested_val, _SUGGESTED_GAIN_CAP))  # cap for writing to config
        # Cap suggested gain for display (e.g. avoid 1e13 for near-silent mics)
        if suggested_val > _SUGGESTED_GAIN_CAP:
            suggested_str = f"{_SUGGESTED_GAIN_CAP:.0f}+"
        else:
            suggested_str = f"{suggested_val:.2f}"
        print(f"  {m:3d}   {L:.6e}   {db_val:+.1f} dB    {ratio_val:.3f}     {suggested_str}")

    weak_threshold_ratio = 0.35
    weak = [m for m in range(n_mics) if (float(levels[m]) + 1e-12) < max_level * weak_threshold_ratio]
    if weak:
        print("")
        print(f"  Mics with level < {weak_threshold_ratio*100:.0f}% of max (candidates for SPI_MIC_GAIN boost): {weak}")
        print("  Example: set those indices to suggested gain in config.SPI_MIC_GAIN.")

    if write_config and config is not None and len(suggested_gains) == n_mics:
        config_path = getattr(config, "__file__", None)
        if not config_path or not os.path.isfile(config_path):
            config_path = os.path.join(_REPO_ROOT, "src", "software", "acoustic_imager", "config.py")
        if _write_spi_mic_gain_to_config(config_path, suggested_gains):
            print("")
            print(f"  Wrote SPI_MIC_GAIN to {config_path}")
            print("  Restart the acoustic imager for changes to take effect.")
        else:
            print("")
            print("  --write-config: could not update config (see errors above).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Print debug metrics: config and/or live per-mic levels")
    ap.add_argument("--config", action="store_true", help="Print heatmap/SPI config values")
    ap.add_argument("--live", action="store_true", help="Read HW SPI and print per-mic levels (stop app first)")
    ap.add_argument("--frames", type=int, default=5, help="Frames to read for --live (default 5)")
    ap.add_argument("--sec", type=float, default=2.0, help="Max seconds to collect --live frames (default 2)")
    ap.add_argument("--average", action="store_true", help="Average FFT over collected frames before computing levels")
    ap.add_argument("--write-config", action="store_true", help="After --live, write suggested gains to config.SPI_MIC_GAIN")
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
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

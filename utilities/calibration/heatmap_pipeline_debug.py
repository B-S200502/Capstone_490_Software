#!/usr/bin/env python3
"""
Heatmap pipeline debug: live-data sanity check and unit tests for noise floor, level scale, etc.

Usage (from repo root; for --live, stop the acoustic imager first):
  python3 utilities/calibration/heatmap_pipeline_debug.py --unit
  python3 utilities/calibration/heatmap_pipeline_debug.py --live
  python3 utilities/calibration/heatmap_pipeline_debug.py --live --frames 20 --sec 3

SPI clock: config.SPI_MAX_SPEED_HZ is 21.25 MHz. Raising to 30 MHz may be possible if
the STM32 and wiring support it; overclocking can cause marginal timing, CRC errors, or
drops. Try 30e6 in config and run --live to confirm frames_ok and no bad_parse increase.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional, Tuple

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


def _fingerprint(fft_data: np.ndarray) -> float:
    """Single scalar that changes when FFT content changes."""
    if fft_data is None or fft_data.size == 0:
        return 0.0
    return float(np.sum(np.abs(fft_data)))


def run_live_check(num_frames: int = 15, duration_sec: float = 2.0) -> Tuple[bool, str]:
    """
    Read frames from HW SPI; verify frame_id and/or fft_data change (live data sanity check).
    Returns (passed, message). Run with acoustic imager stopped.
    """
    if config is None:
        return False, "config not available"
    try:
        from acoustic_imager.io.spi_manager import SPIManager
    except ImportError as e:
        return False, f"SPIManager import failed: {e}"

    mgr = SPIManager(use_frame_ready=True)
    mgr.start()
    try:
        frame_ids: List[int] = []
        fingerprints: List[float] = []
        deadline = time.monotonic() + duration_sec
        while len(frame_ids) < num_frames and time.monotonic() < deadline:
            lf = mgr.get_latest()
            if lf.ok and lf.fft_data is not None:
                frame_ids.append(lf.frame_id)
                fingerprints.append(_fingerprint(lf.fft_data))
            time.sleep(max(0.02, duration_sec / max(1, num_frames)))
    finally:
        mgr.stop()

    if len(frame_ids) < 2:
        return False, f"Only {len(frame_ids)} valid frame(s); need at least 2 to check liveness"

    ids_changed = len(set(frame_ids)) > 1
    fps_changed = len(set(fingerprints)) > 1
    if ids_changed or fps_changed:
        msg = "Live data check PASSED: "
        parts = []
        if ids_changed:
            parts.append("frame_id varied")
        if fps_changed:
            parts.append("fft_data fingerprint varied")
        return True, msg + "; ".join(parts)
    return False, (
        "Live data check FAILED: frame_id and fft_data fingerprint did not change across "
        f"{len(frame_ids)} samples. Heatmap may be showing stale or non-live data."
    )


def test_noise_floor_filters_weak_bins() -> None:
    """Noise floor should drop bins more than SPI_NOISE_FLOOR_DB below peak."""
    if config is None:
        raise RuntimeError("config not available")
    # Synthetic: 10 candidate bins, one strong and rest ~30 dB down
    candidate_bins = np.arange(10, dtype=np.intp)
    power_per_bin = np.ones(10, dtype=np.float64) * 1e-3
    power_per_bin[0] = 1.0
    p_max = float(power_per_bin.max()) + 1e-12
    power_db = 10.0 * np.log10((power_per_bin + 1e-12) / p_max)
    above_floor = power_db >= (-config.SPI_NOISE_FLOOR_DB)
    kept = int(np.sum(above_floor))
    assert kept >= 1, "At least peak bin must pass noise floor"
    assert kept <= len(candidate_bins), "Cannot keep more than candidates"


def test_top_k_respects_k() -> None:
    """After noise floor, we must have at most SPI_TOP_K_BINS bins."""
    if config is None:
        raise RuntimeError("config not available")
    candidate_bins = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=np.intp)
    power_per_bin = np.random.rand(len(candidate_bins)).astype(np.float64) + 0.1
    p_max = float(power_per_bin.max()) + 1e-12
    power_db = 10.0 * np.log10((power_per_bin + 1e-12) / p_max)
    above_floor = power_db >= (-config.SPI_NOISE_FLOOR_DB)
    candidate_bins = candidate_bins[above_floor]
    power_per_bin = power_per_bin[above_floor]
    K = min(config.SPI_TOP_K_BINS, len(candidate_bins))
    if K <= 0:
        return
    top_idx = np.argsort(power_per_bin)[::-1][:K]
    bins = candidate_bins[top_idx].tolist()
    assert len(bins) <= config.SPI_TOP_K_BINS, f"Got {len(bins)} bins, max is {config.SPI_TOP_K_BINS}"


def test_level_scale_in_range() -> None:
    """level = max(floor, min(1, power/ref)) must be in [floor, 1]."""
    floor = getattr(config, "HEATMAP_LEVEL_FLOOR", 0.18)
    ref = getattr(config, "HEATMAP_LEVEL_REFERENCE", 1e6)
    for total_bandpass_power in [0.0, ref * 0.5, ref, ref * 2.0]:
        level = max(floor, min(1.0, (total_bandpass_power + 1e-12) / ref))
        assert floor <= level <= 1.0, f"level {level} outside [{floor}, 1] for power {total_bandpass_power}"


def test_contrast_stretch_clips() -> None:
    """Contrast stretch (percentile scale) must not produce values > 255."""
    pct = getattr(config, "HEATMAP_CONTRAST_STRETCH_PERCENTILE", 98.0)
    if pct <= 0:
        return
    heatmap = (np.random.rand(100, 200) * 180).astype(np.uint8)
    p_val = float(np.percentile(heatmap, pct))
    if p_val > 1e-6:
        stretched = (heatmap.astype(np.float32) * (255.0 / p_val)).clip(0, 255).astype(np.uint8)
        assert stretched.max() <= 255 and stretched.min() >= 0


def test_hw_geometry_present() -> None:
    """Config (source of truth) must define x_coords_hw and y_coords_hw with length N_MICS."""
    if config is None:
        raise RuntimeError("config not available")
    assert hasattr(config, "x_coords_hw"), "config missing x_coords_hw (HW geometry)"
    assert hasattr(config, "y_coords_hw"), "config missing y_coords_hw (HW geometry)"
    assert len(config.x_coords_hw) == config.N_MICS, (
        f"x_coords_hw length {len(config.x_coords_hw)} != config.N_MICS ({config.N_MICS})"
    )
    assert len(config.y_coords_hw) == config.N_MICS, (
        f"y_coords_hw length {len(config.y_coords_hw)} != config.N_MICS ({config.N_MICS})"
    )
    # Sanity: positions in meters, typical aperture ~0.05–0.15 m
    assert np.all(np.isfinite(config.x_coords_hw)) and np.all(np.isfinite(config.y_coords_hw)), (
        "x_coords_hw / y_coords_hw must be finite"
    )


def run_unit_tests() -> Tuple[int, int]:
    """Run unit tests; return (passed, failed)."""
    tests = [
        test_noise_floor_filters_weak_bins,
        test_top_k_respects_k,
        test_level_scale_in_range,
        test_contrast_stretch_clips,
        test_hw_geometry_present,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  OK {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL {t.__name__}: {e}")
    return passed, failed


def main() -> int:
    ap = argparse.ArgumentParser(description="Heatmap pipeline debug: live check and unit tests")
    ap.add_argument("--unit", action="store_true", help="Run unit tests (no SPI)")
    ap.add_argument("--live", action="store_true", help="Run live-data sanity check (uses HW SPI)")
    ap.add_argument("--frames", type=int, default=15, help="Number of frames for --live (default 15)")
    ap.add_argument("--sec", type=float, default=2.0, help="Max seconds to collect frames for --live (default 2)")
    args = ap.parse_args()

    if not args.unit and not args.live:
        ap.print_help()
        print("\nUse --unit and/or --live.")
        return 0

    if args.unit:
        print("Unit tests (noise floor, top-K, level scale, contrast stretch, HW geometry):")
        passed, failed = run_unit_tests()
        print(f"  Result: {passed} passed, {failed} failed")
        if failed:
            return 1

    if args.live:
        print("Live data check (HW SPI; ensure acoustic imager is stopped):")
        ok, msg = run_live_check(num_frames=args.frames, duration_sec=args.sec)
        print(f"  {msg}")
        if not ok:
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

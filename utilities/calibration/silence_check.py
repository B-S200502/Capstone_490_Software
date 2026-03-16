#!/usr/bin/env python3
"""
Pre-test: check that the environment is quiet enough for gain calibration.
Captures 1--2 s of SPI frames, computes total power (L2 norm over mics/bins);
exits 0 if below threshold, 1 otherwise. Run with acoustic imager stopped.

Usage (from repo root):
  python3 utilities/calibration/silence_check.py
  python3 utilities/calibration/silence_check.py --sec 1.5 --threshold 1e9
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_SOFTWARE = str(_REPO_ROOT / "src" / "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

try:
    from acoustic_imager import config
except ImportError:
    config = None


def run(duration_sec: float = 1.5, threshold: float = 1e9) -> tuple[bool, float, str]:
    """
    Capture frames for duration_sec, return (passed, max_total_power, message).
    passed = True if max total power over the run is below threshold.
    """
    if config is None:
        return False, 0.0, "config not available"
    try:
        from acoustic_imager.io.spi_manager import SPIManager
    except ImportError as e:
        return False, 0.0, f"SPIManager import failed: {e}"

    mgr = SPIManager(use_frame_ready=True)
    mgr.start()
    try:
        deadline = time.monotonic() + duration_sec
        max_power = 0.0
        count = 0
        while time.monotonic() < deadline:
            lf = mgr.get_latest()
            if lf.ok and lf.fft_data is not None:
                # Total power: sum of squared magnitudes over all mics and bins
                power = float(np.sum(np.abs(lf.fft_data) ** 2))
                if power > max_power:
                    max_power = power
                count += 1
            time.sleep(0.05)
        if count == 0:
            return False, 0.0, "no frames received (SPI busy or app running?)"
        passed = max_power < threshold
        msg = f"max_total_power={max_power:.3e} threshold={threshold:.3e} frames={count} -> {'PASS' if passed else 'FAIL (env too loud)'}"
        return passed, max_power, msg
    finally:
        mgr.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description="Check environment is quiet enough for calibration")
    ap.add_argument("--sec", type=float, default=1.5, help="Capture duration (default 1.5)")
    ap.add_argument("--threshold", type=float, default=1e9, help="Max allowed total power (default 1e9)")
    args = ap.parse_args()
    passed, power, msg = run(duration_sec=args.sec, threshold=args.threshold)
    print(msg)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

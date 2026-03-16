#!/usr/bin/env python3
"""
Pin monitor for calibration suite: runs N iterations and exits.
Uses SPI0 pinout; all non-GND pins must read 0 or 1 for success.
Run from repo root: python3 utilities/calibration/pin_monitor.py --runs 5
"""
import argparse
import sys
import time
from pathlib import Path

_CAL_DIR = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description="Pin monitor (N runs, exit 0 if all pins 0/1)")
    ap.add_argument("--runs", type=int, default=5, help="Number of read runs")
    args = ap.parse_args()

    try:
        from pin_monitor_spi0 import PINS_USED, read_one, spi_status, SPI0_DEVICE
    except ImportError:
        # Run from repo root: calibration dir not on path
        sys.path.insert(0, str(_CAL_DIR))
        from pin_monitor_spi0 import PINS_USED, read_one, spi_status, SPI0_DEVICE

    width = 58
    header = f"{'Physical':<10} {'BCM':<6} {'Usage':<26} {'Value':<6}"
    print("Pin monitor (SPI0, %d runs)" % args.runs)
    print(header)
    print("-" * width)

    all_ok = True
    for run in range(args.runs):
        for physical, bcm, usage in PINS_USED:
            bcm_str = str(bcm) if bcm is not None else "-"
            if bcm is None:
                val = "-"
            else:
                val = read_one(bcm)
                if val not in ("0", "1"):
                    all_ok = False
            print(f"{physical:<10} {bcm_str:<6} {usage:<26} {val:<6}")
        exists, in_use = spi_status(SPI0_DEVICE)
        print("SPI: %s  exists=%s  in_use=%s" % (SPI0_DEVICE, "yes" if exists else "no", "yes" if in_use else "no"))
        print()
        if run < args.runs - 1:
            time.sleep(0.2)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

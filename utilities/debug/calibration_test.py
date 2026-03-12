#!/usr/bin/env python3
"""
Print and optionally validate the camera–array calibration test setup.
See calibration_test.md for layout (5 in distance, 7 in offset, camera left / board right).

Usage (from repo root):
  python3 utilities/debug/calibration_test.py
  python3 utilities/debug/calibration_test.py --validate
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC_SOFTWARE = os.path.join(_REPO_ROOT, "src", "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

try:
    from acoustic_imager import config
except ImportError:
    config = None

# Canonical values from calibration_test.md (single source of truth for the script)
CALIBRATION_DISTANCE_INCHES = 5.0
CALIBRATION_OFFSET_INCHES = 7.0
CALIBRATION_NOTE = "Camera left, board right, same heading"


def main() -> int:
    ap = argparse.ArgumentParser(description="Print or validate calibration test setup")
    ap.add_argument("--validate", action="store_true", help="Assert config matches calibration_test.md values")
    args = ap.parse_args()

    print("Calibration test setup (see utilities/debug/calibration_test.md):")
    print(f"  distance_inches = {CALIBRATION_DISTANCE_INCHES}")
    print(f"  offset_inches   = {CALIBRATION_OFFSET_INCHES} (+ = array right of camera center)")
    print(f"  note            = {CALIBRATION_NOTE}")

    if config is not None:
        print("\nConfig (config.py):")
        print(f"  CALIBRATION_DISTANCE_INCHES = {getattr(config, 'CALIBRATION_DISTANCE_INCHES', 'N/A')}")
        print(f"  CALIBRATION_OFFSET_INCHES   = {getattr(config, 'CALIBRATION_OFFSET_INCHES', 'N/A')}")
        print(f"  CALIBRATION_NOTE             = {getattr(config, 'CALIBRATION_NOTE', 'N/A')!r}")

        if args.validate:
            if getattr(config, "CALIBRATION_DISTANCE_INCHES", None) != CALIBRATION_DISTANCE_INCHES:
                print(f"\nValidate FAILED: config.CALIBRATION_DISTANCE_INCHES != {CALIBRATION_DISTANCE_INCHES}")
                return 1
            if getattr(config, "CALIBRATION_OFFSET_INCHES", None) != CALIBRATION_OFFSET_INCHES:
                print(f"\nValidate FAILED: config.CALIBRATION_OFFSET_INCHES != {CALIBRATION_OFFSET_INCHES}")
                return 1
            print("\nValidate OK: config matches calibration_test.md")
    else:
        print("\nConfig not available (run from repo root with src/software on path for config check)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

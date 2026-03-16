#!/usr/bin/env python3
"""
Update config.py HW geometry (x_coords_hw, y_coords_hw, pitch_hw) from array_geometry.

config.py is the app's source of truth for coordinates/geometry at runtime. This script
writes the measured geometry from array_geometry into config. Run after changing
MICS_RAW or PAYLOAD_TO_MIC in array_geometry.py. Usage from repo root:

  python3 utilities/calibration/update_config_geometry.py

Optional: --dry-run prints the new block without writing config.py.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_CAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CAL_DIR.parents[1]
_CONFIG_PY = _REPO_ROOT / "src" / "software" / "acoustic_imager" / "config.py"

# Import array_geometry (same path setup as run_suite)
if str(_CAL_DIR) not in sys.path:
    sys.path.insert(0, str(_CAL_DIR))
import array_geometry.array_geometry as ag


def _format_array(values, indent: str = "    ", per_line: int = 6) -> str:
    """Format a 16-element array as in config: 6 + 6 + 4 per line."""
    parts = [indent + ", ".join("%.6f" % v for v in values[i : i + per_line]) for i in range(0, 16, per_line)]
    return ",\n".join(parts)


def build_hw_block() -> str:
    header = """# ===============================================================
# 4. HW geometry (measured, payload order)
# From utilities/calibration/array_geometry; payload 0=U3 .. 15=U14.
# Used only for SRC:HW and LOOP; SIM keeps x_coords/y_coords above.
# ===============================================================
x_coords_hw = np.array([
"""
    x_block = _format_array(ag.x_coords) + ",\n])\ny_coords_hw = np.array([\n"
    y_block = _format_array(ag.y_coords) + ",\n])\npitch_hw = %.6f" % float(ag.pitch)
    return header + x_block + y_block


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Update config.py HW geometry from array_geometry.")
    ap.add_argument("--dry-run", action="store_true", help="Print new block only, do not write config.")
    args = ap.parse_args()

    if not _CONFIG_PY.exists():
        print("ERROR: config.py not found at %s" % _CONFIG_PY, file=sys.stderr)
        return 1

    new_block = build_hw_block()
    if args.dry_run:
        print(new_block)
        return 0

    text = _CONFIG_PY.read_text()
    # Match from "# 4. HW geometry" comment block through "pitch_hw = <number>"
    pattern = re.compile(
        r"(# ===============================================================\n"
        r"# 4\. HW geometry \(measured, payload order\).*?"
        r"pitch_hw = [^\n]+)",
        re.DOTALL,
    )
    if not pattern.search(text):
        print("ERROR: Could not find HW geometry section in config.py", file=sys.stderr)
        return 1
    new_text = pattern.sub(new_block, text, count=1)
    if new_text == text:
        print("ERROR: Replacement did not change file", file=sys.stderr)
        return 1
    _CONFIG_PY.write_text(new_text)
    print("Updated %s with x_coords_hw, y_coords_hw, pitch_hw from array_geometry." % _CONFIG_PY)
    return 0


if __name__ == "__main__":
    sys.exit(main())

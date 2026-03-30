"""
Four-point (N/E/S/W) compass calibration: infer fixed XY/XZ/YZ plane + heading offset.
Persists to COMPASS_CAL_DIR / compass_cal.json (repo utilities/calibration; set in main.py).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

from .. import config

CAL_VERSION = 1
CAL_FILENAME = "compass_cal.json"
DEFAULT_RMS_THRESHOLD_DEG = 35.0


def _default_compass_cal_dir() -> Path:
    """Repo root utilities/calibration when main has not set COMPASS_CAL_DIR yet."""
    return Path(__file__).resolve().parents[4] / "utilities" / "calibration"


def cal_dir() -> Path:
    d = getattr(config, "COMPASS_CAL_DIR", None)
    if d is not None:
        return Path(d)
    return _default_compass_cal_dir()


def cal_file_path() -> Path:
    return cal_dir() / CAL_FILENAME


def angle_in_plane(plane: str, x: float, y: float, z: float) -> float:
    p = plane.strip().upper()
    if p == "XZ":
        a, b = float(x), float(z)
    elif p == "YZ":
        a, b = float(y), float(z)
    else:
        a, b = float(x), float(y)
    h = math.degrees(math.atan2(b, a))
    if h < 0:
        h += 360.0
    return h


def _wrap180(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def score_plane(samples: List[Tuple[int, int, int]], plane: str) -> Tuple[float, List[float]]:
    thetas = [angle_in_plane(plane, float(s[0]), float(s[1]), float(s[2])) for s in samples]
    expected = [0.0, 90.0, 180.0, 270.0]
    t0 = thetas[0]
    errs: List[float] = []
    for i in range(4):
        delta_meas = _wrap180(thetas[i] - t0)
        delta_exp = expected[i]
        errs.append(_wrap180(delta_meas - delta_exp))
    rms = math.sqrt(sum(e * e for e in errs) / 4.0)
    return rms, thetas


def solve_from_samples(
    samples: List[Tuple[int, int, int]],
    rms_threshold_deg: float = DEFAULT_RMS_THRESHOLD_DEG,
) -> Tuple[Optional[dict[str, Any]], Optional[str]]:
    if len(samples) != 4:
        return None, "need 4 samples"
    best_plane = "XY"
    best_rms = float("inf")
    best_thetas: List[float] = []
    for plane in ("XY", "XZ", "YZ"):
        rms, thetas = score_plane(samples, plane)
        if rms < best_rms:
            best_rms = rms
            best_plane = plane
            best_thetas = list(thetas)
    if best_rms > rms_threshold_deg:
        return None, f"fit poor (rms={best_rms:.1f} deg)"
    offset = (-best_thetas[0]) % 360.0
    return (
        {
            "plane": best_plane,
            "offset_deg": float(offset),
            "rms_error": float(best_rms),
            "thetas_deg": best_thetas,
        },
        None,
    )


def save_calibration(
    plane: str,
    offset_deg: float,
    samples: Optional[List[Tuple[int, int, int]]] = None,
) -> str:
    d = cal_dir()
    d.mkdir(parents=True, mode=0o755, exist_ok=True)
    path = cal_file_path()
    data: dict[str, Any] = {
        "version": CAL_VERSION,
        "plane": plane.upper(),
        "offset_deg": float(offset_deg),
        "timestamp": time.time(),
    }
    if samples is not None:
        data["samples"] = [list(s) for s in samples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return str(path)


def load_calibration_file() -> Optional[dict[str, Any]]:
    path = cal_file_path()
    try:
        if not path.is_file():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def clear_user_calibration(cfg: Any) -> None:
    """Remove saved file and clear runtime user-cal flags on cfg."""
    try:
        p = cal_file_path()
        if p.is_file():
            p.unlink()
    except OSError:
        pass
    setattr(cfg, "COMPASS_USER_CAL_VALID", False)
    setattr(cfg, "COMPASS_USER_CAL_PLANE", None)
    setattr(cfg, "COMPASS_USER_CAL_OFFSET_DEG", 0.0)


def apply_loaded_calibration_to_config(cfg: Any, data: Optional[dict[str, Any]] = None) -> bool:
    if data is None:
        data = load_calibration_file()
    if not data or int(data.get("version", 0)) < 1:
        setattr(cfg, "COMPASS_USER_CAL_VALID", False)
        setattr(cfg, "COMPASS_USER_CAL_PLANE", None)
        setattr(cfg, "COMPASS_USER_CAL_OFFSET_DEG", 0.0)
        return False
    plane = str(data.get("plane", "XY")).strip().upper()
    if plane not in ("XY", "XZ", "YZ"):
        plane = "XY"
    try:
        off = float(data.get("offset_deg", 0.0))
    except (TypeError, ValueError):
        off = 0.0
    setattr(cfg, "COMPASS_USER_CAL_VALID", True)
    setattr(cfg, "COMPASS_USER_CAL_PLANE", plane)
    setattr(cfg, "COMPASS_USER_CAL_OFFSET_DEG", off)
    return True

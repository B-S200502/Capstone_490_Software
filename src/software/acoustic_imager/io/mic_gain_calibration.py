"""
Persisted per-mic SPI_MIC_GAIN from the calibration suite (metrics_debug --write-config).

JSON is written beside compass cal under utilities/calibration/mic_gain_calibration.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def apply_mic_gain_calibration_json(cfg: Any, path: Path) -> bool:
    """If JSON exists and is valid, set cfg.SPI_MIC_GAIN. Returns True if applied."""
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if int(data.get("version", 0)) != 1:
        return False
    raw = data.get("spi_mic_gain")
    if not isinstance(raw, list) or not raw:
        return False
    n = int(getattr(cfg, "N_MICS", 16))
    if len(raw) != n:
        return False
    try:
        tup = tuple(float(x) for x in raw)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(g) and g > 0 for g in tup):
        return False
    setattr(cfg, "SPI_MIC_GAIN", tup)
    return True

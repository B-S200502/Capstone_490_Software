"""Persist selected config.py assignments (heading trim, etc.) without rewriting the whole file."""

from __future__ import annotations

import re
from pathlib import Path


def config_py_path() -> Path:
    import acoustic_imager.config as cfg

    return Path(cfg.__file__).resolve()


def save_mag_heading_display_offset_deg(value: float) -> bool:
    """Rewrite the MAG_HEADING_DISPLAY_OFFSET_DEG line in config.py. Returns True on success."""
    path = config_py_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    new_line = f"MAG_HEADING_DISPLAY_OFFSET_DEG = {float(value)!r}"
    pattern = r"^MAG_HEADING_DISPLAY_OFFSET_DEG\s*=\s*[^\n]+"
    if not re.search(pattern, text, flags=re.MULTILINE):
        return False
    new_text = re.sub(pattern, new_line, text, count=1, flags=re.MULTILINE)
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError:
        return False
    return True

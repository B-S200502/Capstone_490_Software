"""
Persist gallery metadata (tags, priority, tag_data) in a single JSON file per output dir.

File: {output_dir}/gallery_metadata.json
Lightweight: one small JSON, no extra deps. Load once at startup (or when gallery opens),
save after any change to priorities, file_tags, or tag_data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

METADATA_FILENAME = "gallery_metadata.json"
VERSION = 1


def _metadata_path(output_dir: Path) -> Path:
    return output_dir / METADATA_FILENAME


def load_metadata(output_dir: Path) -> tuple[dict[str, str], dict[str, list[str]], dict[str, dict[str, str]]]:
    """
    Load priorities, file_tags, and tag_data from gallery_metadata.json.
    Returns (priorities, file_tags, tag_data). Empty dicts if file missing or invalid.
    """
    path = _metadata_path(output_dir)
    if not path.exists():
        return {}, {}, {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, {}, {}

    if not isinstance(data, dict):
        return {}, {}, {}

    priorities: dict[str, str] = {}
    if isinstance(data.get("priorities"), dict):
        for k, v in data["priorities"].items():
            if isinstance(k, str) and isinstance(v, str):
                priorities[k] = v

    file_tags: dict[str, list[str]] = {}
    if isinstance(data.get("file_tags"), dict):
        for k, v in data["file_tags"].items():
            if isinstance(k, str) and isinstance(v, list):
                file_tags[k] = [str(t) for t in v if isinstance(t, str)]

    tag_data: dict[str, dict[str, str]] = {}
    if isinstance(data.get("tag_data"), dict):
        for k, v in data["tag_data"].items():
            if isinstance(k, str) and isinstance(v, dict):
                tag_data[k] = {str(a): str(b) for a, b in v.items() if isinstance(a, str) and isinstance(b, str)}

    return priorities, file_tags, tag_data


def save_metadata(
    output_dir: Path,
    priorities: dict[str, str],
    file_tags: dict[str, list[str]],
    tag_data: dict[str, dict[str, str]],
) -> None:
    """Write priorities, file_tags, and tag_data to gallery_metadata.json."""
    path = _metadata_path(output_dir)
    payload: dict[str, Any] = {
        "version": VERSION,
        "priorities": priorities,
        "file_tags": file_tags,
        "tag_data": tag_data,
    }
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except OSError:
        pass

"""
Archive panel: folders for organizing gallery media.

The top-right grid slot is reserved for the archive panel. Users can create
up to 4 folders, rename them, and move selected media into folders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

ARCHIVE_METADATA_FILENAME = "archive_folders.json"
MAX_FOLDERS = 4


def _archive_path(output_dir: Path) -> Path:
    return output_dir / ARCHIVE_METADATA_FILENAME


def load_archive_folders(output_dir: Optional[Path]) -> List[dict]:
    """
    Load archive folders from archive_folders.json.
    Returns list of {"id": str, "name": str, "files": [str]}.
    """
    if not output_dir or not output_dir.exists():
        return []

    path = _archive_path(output_dir)
    if not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, dict) or "folders" not in data:
        return []

    folders: List[dict] = []
    for item in data.get("folders", []):
        if isinstance(item, dict) and "id" in item and "name" in item:
            fid = str(item["id"])
            name = str(item.get("name", "Folder"))
            files = item.get("files", [])
            if isinstance(files, list):
                files = [str(f) for f in files if isinstance(f, str)]
            else:
                files = []
            folders.append({"id": fid, "name": name, "files": files})
            if len(folders) >= MAX_FOLDERS:
                break

    return folders


def save_archive_folders(output_dir: Path, folders: List[dict]) -> None:
    """Persist archive folders."""
    path = _archive_path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "folders": folders}, f, indent=2)
    except OSError:
        pass


def add_folder(output_dir: Path, folders: List[dict]) -> List[dict]:
    """Add a new folder. Returns updated list."""
    if len(folders) >= MAX_FOLDERS:
        return folders

    used_ids = {f["id"] for f in folders}
    n = 1
    while f"folder_{n}" in used_ids:
        n += 1

    new = {"id": f"folder_{n}", "name": f"Folder {n}", "files": []}
    out = folders + [new]
    save_archive_folders(output_dir, out)
    return out


def rename_folder(output_dir: Path, folders: List[dict], folder_id: str, new_name: str) -> List[dict]:
    """Rename a folder."""
    out = []
    for f in folders:
        if f["id"] == folder_id:
            out.append({**f, "name": new_name.strip() or f["name"]})
        else:
            out.append(dict(f))
    save_archive_folders(output_dir, out)
    return out


def delete_folder(output_dir: Path, folders: List[dict], folder_id: str) -> List[dict]:
    """Remove a folder. Returns updated list."""
    out = [f for f in folders if f["id"] != folder_id]
    save_archive_folders(output_dir, out)
    return out


def move_files_to_folder(
    output_dir: Path, folders: List[dict], folder_id: str, filenames: List[str]
) -> List[dict]:
    """Add filenames to a folder's file list (remove from other folders first)."""
    out = []
    for f in folders:
        d = dict(f)
        files = list(d["files"])
        if f["id"] == folder_id:
            for fn in filenames:
                if fn not in files:
                    files.append(fn)
            d["files"] = files
        else:
            d["files"] = [x for x in files if x not in filenames]
        out.append(d)
    save_archive_folders(output_dir, out)
    return out


def remove_files_from_all_folders(
    output_dir: Path, folders: List[dict], filenames: List[str]
) -> List[dict]:
    """Remove filenames from all folders (move to Gallery)."""
    out = []
    for f in folders:
        d = dict(f)
        d["files"] = [x for x in d.get("files", []) if x not in filenames]
        out.append(d)
    save_archive_folders(output_dir, out)
    return out


def item_idx_to_grid_pos(idx: int, cols: int) -> tuple:
    """
    Map item index to grid (row, col). Slot (0, cols-1) is reserved for archive.
    """
    if idx < cols - 1:
        return (0, idx)
    row = 1 + (idx - (cols - 1)) // cols
    col = (idx - (cols - 1)) % cols
    return (row, col)


def archive_panel_grid_pos(cols: int) -> tuple:
    """Grid position of the archive panel (top-right)."""
    return (0, cols - 1)

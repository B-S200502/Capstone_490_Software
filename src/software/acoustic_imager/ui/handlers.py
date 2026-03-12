"""
Click and mouse handlers for menu, buttons, and gallery.
"""

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import time

from . import ui_cache
from .button import buttons, menu_buttons
from .gallery import get_displayed_gallery_items, get_folder_displayed_items, _viewer_rubber_band_offset, _video_read_frame_at
from .archive_panel import load_archive_folders, add_folder, rename_folder as archive_rename_folder, delete_folder as archive_delete_folder, move_files_to_folder, remove_files_from_all_folders, save_archive_folders
from .viewer_dock import trigger_viewer_button_feedback
from .storage_bar import _format_size
from .menu import get_recording_timestamp_rect
from .screenshot import save_screenshot
from ..config import SOURCE_MODES, SOURCE_DEFAULT
from ..state import button_state, HUD
from ..io.gallery_metadata import save_metadata
from ..io.gain_control import GAIN_CONTROL
from ..io.email_config import (
    load_config,
    load_provider_config,
    save_config,
    save_provider_config,
    send_test_email,
    set_email_verified,
    send_share_email,
    get_email_verified,
    get_share_recipient,
    SHARE_ATTACHMENT_LIMIT_BYTES,
    SMTP_PRESETS,
)
from .video_recorder import VideoRecorder


def _email_cursor_index_from_click(displayed_text: str, click_x_in_field: int, font_scale: float = 0.52) -> int:
    """Return cursor index (0 to len(displayed_text)) from click x relative to field content start."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if not displayed_text:
        return 0
    best = 0
    for i in range(1, len(displayed_text) + 1):
        (w, _), _ = cv2.getTextSize(displayed_text[:i], font, font_scale, 1)
        if w <= click_x_in_field:
            best = i
    return best


def _save_email_config(output_dir: Optional[Path]) -> None:
    """Persist current email form for the selected provider to email_config.json."""
    if not output_dir:
        return
    provider = button_state.email_modal_provider or "gmail"
    form_data = {
        "email": (button_state.email_form_email or "").strip(),
        "password": (button_state.email_form_password or "").strip(),
        "default_to": (button_state.email_form_default_to or "").strip(),
        "smtp_host": (button_state.email_form_smtp_host or "").strip(),
        "smtp_port": (button_state.email_form_smtp_port or "587").strip(),
        "use_tls": button_state.email_form_use_tls,
    }
    save_provider_config(output_dir, provider, form_data)


def handle_email_modal_click(x: int, y: int, output_dir: Optional[Path]) -> bool:
    """Handle clicks when Email Settings modal is open. Returns True if click was consumed."""
    if not button_state.email_settings_modal_open:
        return False

    # Provider screen
    if button_state.email_modal_screen == "provider":
        for _, key in [("Gmail", "gmail"), ("Outlook", "outlook"), ("Yahoo", "yahoo"), ("Other", "other")]:
            k = f"email_provider_{key}"
            if k in menu_buttons and menu_buttons[k].contains(x, y):
                button_state.email_modal_screen = "form"
                button_state.email_modal_provider = key
                if output_dir:
                    cfg = load_provider_config(output_dir, key)
                    button_state.email_form_email = (cfg.get("email") or "").strip()
                    button_state.email_form_password = (cfg.get("password") or "").strip()
                    button_state.email_form_default_to = (cfg.get("default_to") or "").strip()
                    button_state.email_form_smtp_host = (cfg.get("smtp_host") or "").strip()
                    button_state.email_form_smtp_port = str(cfg.get("smtp_port", 587))
                    button_state.email_form_use_tls = bool(cfg.get("use_tls", True))
                button_state.email_focused_field = "email"  # first field active by default
                button_state.email_cursor_index = len(button_state.email_form_email or "")
                button_state.email_shift_next = False
                button_state.email_password_visible = False
                button_state.email_test_status = ""
                button_state.email_test_message = ""
                return True
        if "email_modal_panel" in menu_buttons and menu_buttons["email_modal_panel"].contains(x, y):
            return True
        button_state.email_settings_modal_open = False
        HUD.settings_modal_open = True  # return to System Settings
        return True

    # Form screen: field focus and click-to-position cursor
    for field_key in ("email_field_email", "email_field_password", "email_field_default_to", "email_field_smtp_host", "email_field_smtp_port"):
        if field_key in menu_buttons and menu_buttons[field_key].contains(x, y):
            field_id = field_key.replace("email_field_", "")
            button_state.email_focused_field = field_id
            # Cursor index from click position (content starts at button.x + 8)
            b = menu_buttons[field_key]
            rel_x = max(0, x - (b.x + 8))
            if field_id == "email":
                val = button_state.email_form_email or ""
                scale = 0.52
            elif field_id == "password":
                val = button_state.email_form_password or ""
                displayed = "*" * len(val)
                button_state.email_cursor_index = _email_cursor_index_from_click(displayed, rel_x, 0.52)
                return True
            elif field_id == "default_to":
                val = button_state.email_form_default_to or ""
                scale = 0.52
            elif field_id == "smtp_host":
                val = button_state.email_form_smtp_host or ""
                scale = 0.44
            else:
                val = button_state.email_form_smtp_port or "587"
                scale = 0.44
            button_state.email_cursor_index = _email_cursor_index_from_click(val, rel_x, scale)
            return True

    if "email_password_toggle_visible" in menu_buttons and menu_buttons["email_password_toggle_visible"].contains(x, y):
        button_state.email_password_visible = not getattr(button_state, "email_password_visible", False)
        return True

    # Keyboard keys
    focused = button_state.email_focused_field
    is_alpha = getattr(button_state, "email_keyboard_mode", "alpha") == "alpha"
    if focused:
        # Alpha mode: letters only (numbers are in ?123). Symbol mode: numbers only.
        chars = "qwertyuiopasdfghjklzxcvbnm" if is_alpha else "1234567890"
        for c in chars:
            k = f"email_key_{c}"
            if k in menu_buttons and menu_buttons[k].contains(x, y):
                if c.isalpha():
                    shift_on = getattr(button_state, "email_shift_next", False)
                    char = c.upper() if shift_on else c.lower()
                else:
                    char = c
                cur = max(0, min(button_state.email_cursor_index, len(getattr(button_state, f"email_form_{focused}", "") or "")))
                if focused == "email":
                    v = button_state.email_form_email or ""
                    button_state.email_form_email = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "password":
                    v = button_state.email_form_password or ""
                    button_state.email_form_password = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "default_to":
                    v = button_state.email_form_default_to or ""
                    button_state.email_form_default_to = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "smtp_host":
                    v = button_state.email_form_smtp_host or ""
                    button_state.email_form_smtp_host = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "smtp_port" and char.isdigit():
                    v = button_state.email_form_smtp_port or ""
                    button_state.email_form_smtp_port = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                return True
        if "email_key_shift" in menu_buttons and menu_buttons["email_key_shift"].contains(x, y):
            button_state.email_shift_next = not getattr(button_state, "email_shift_next", False)
            return True
        # Alpha shortcut row: .com . @ - _ space
        for key_id, insert_text in (
            ("email_key_snippet_com", ".com"),
            ("email_key_dot", "."),
            ("email_key_at", "@"),
            ("email_key_dash", "-"),
            ("email_key_underscore", "_"),
            ("email_key_space", " "),
        ):
            if key_id in menu_buttons and menu_buttons[key_id].contains(x, y):
                cur = max(0, min(button_state.email_cursor_index, len(getattr(button_state, f"email_form_{focused}", "") or "")))
                if focused == "email":
                    v = button_state.email_form_email or ""
                    button_state.email_form_email = v[:cur] + insert_text + v[cur:]
                    button_state.email_cursor_index = cur + len(insert_text)
                elif focused == "password":
                    v = button_state.email_form_password or ""
                    button_state.email_form_password = v[:cur] + insert_text + v[cur:]
                    button_state.email_cursor_index = cur + len(insert_text)
                elif focused == "default_to":
                    v = button_state.email_form_default_to or ""
                    button_state.email_form_default_to = v[:cur] + insert_text + v[cur:]
                    button_state.email_cursor_index = cur + len(insert_text)
                elif focused == "smtp_host":
                    v = button_state.email_form_smtp_host or ""
                    button_state.email_form_smtp_host = v[:cur] + insert_text + v[cur:]
                    button_state.email_cursor_index = cur + len(insert_text)
                elif focused == "smtp_port" and insert_text.isdigit():
                    v = button_state.email_form_smtp_port or ""
                    button_state.email_form_smtp_port = v[:cur] + insert_text + v[cur:]
                    button_state.email_cursor_index = cur + len(insert_text)
                return True
        if "email_key_backspace" in menu_buttons and menu_buttons["email_key_backspace"].contains(x, y):
            cur = button_state.email_cursor_index
            if cur <= 0:
                return True
            if focused == "email":
                v = button_state.email_form_email or ""
                button_state.email_form_email = v[: cur - 1] + v[cur:]
                button_state.email_cursor_index = cur - 1
            elif focused == "password":
                v = button_state.email_form_password or ""
                button_state.email_form_password = v[: cur - 1] + v[cur:]
                button_state.email_cursor_index = cur - 1
            elif focused == "default_to":
                v = button_state.email_form_default_to or ""
                button_state.email_form_default_to = v[: cur - 1] + v[cur:]
                button_state.email_cursor_index = cur - 1
            elif focused == "smtp_host":
                v = button_state.email_form_smtp_host or ""
                button_state.email_form_smtp_host = v[: cur - 1] + v[cur:]
                button_state.email_cursor_index = cur - 1
            elif focused == "smtp_port":
                v = button_state.email_form_smtp_port or ""
                button_state.email_form_smtp_port = v[: cur - 1] + v[cur:] or "587"
                button_state.email_cursor_index = cur - 1
            return True
        if "email_key_clear" in menu_buttons and menu_buttons["email_key_clear"].contains(x, y):
            if focused == "email":
                button_state.email_form_email = ""
            elif focused == "password":
                button_state.email_form_password = ""
            elif focused == "default_to":
                button_state.email_form_default_to = ""
            elif focused == "smtp_host":
                button_state.email_form_smtp_host = ""
            elif focused == "smtp_port":
                button_state.email_form_smtp_port = "587"
            button_state.email_cursor_index = 0
            return True
        if "email_key_done" in menu_buttons and menu_buttons["email_key_done"].contains(x, y):
            button_state.email_focused_field = ""
            button_state.email_cursor_index = 0
            return True
        if "email_key_switch_sym" in menu_buttons and menu_buttons["email_key_switch_sym"].contains(x, y):
            button_state.email_keyboard_mode = "symbol"
            return True
        if "email_key_switch_alpha" in menu_buttons and menu_buttons["email_key_switch_alpha"].contains(x, y):
            button_state.email_keyboard_mode = "alpha"
            return True
        # Symbol-mode keys (single chars like @ . _ - etc.)
        for k, char in (
            ("email_key_@", "@"), ("email_key_.", "."), ("email_key__", "_"), ("email_key_-", "-"),
            ("email_key_!", "!"), ("email_key_#", "#"), ("email_key_$", "$"), ("email_key_%", "%"),
            ("email_key_&", "&"), ("email_key_*", "*"), ("email_key_(", "("), ("email_key_)", ")"),
            ("email_key_+", "+"), ("email_key_=", "="), ("email_key_,", ","), ("email_key_<", "<"),
            ("email_key_>", ">"), ("email_key_/", "/"), ("email_key_?", "?"), ("email_key_:", ":"),
            ("email_key_;", ";"), ("email_key_'", "'"), ('email_key_"', '"'), ("email_key_[", "["),
            ("email_key_]", "]"), ("email_key_{", "{"), ("email_key_}", "}"), ("email_key_\\", "\\"),
        ):
            if k in menu_buttons and menu_buttons[k].contains(x, y):
                focused = button_state.email_focused_field
                cur = max(0, min(button_state.email_cursor_index, len(getattr(button_state, f"email_form_{focused}", "") or "")))
                if focused == "email":
                    v = button_state.email_form_email or ""
                    button_state.email_form_email = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "password":
                    v = button_state.email_form_password or ""
                    button_state.email_form_password = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "default_to":
                    v = button_state.email_form_default_to or ""
                    button_state.email_form_default_to = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "smtp_host":
                    v = button_state.email_form_smtp_host or ""
                    button_state.email_form_smtp_host = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                elif focused == "smtp_port" and char.isdigit():
                    v = button_state.email_form_smtp_port or ""
                    button_state.email_form_smtp_port = v[:cur] + char + v[cur:]
                    button_state.email_cursor_index = cur + 1
                return True

    # Send Test: only when all required fields are filled; run in background thread
    if "email_send_test" in menu_buttons and menu_buttons["email_send_test"].contains(x, y):
        provider = button_state.email_modal_provider or "gmail"
        email_ok = (button_state.email_form_email or "").strip()
        password_ok = (button_state.email_form_password or "").strip()
        smtp_ok = provider != "other" or bool((button_state.email_form_smtp_host or "").strip())
        if email_ok and password_ok and smtp_ok:
            form_data = {
                "email": (button_state.email_form_email or "").strip(),
                "password": (button_state.email_form_password or "").strip(),
                "default_to": (button_state.email_form_default_to or "").strip(),
                "smtp_host": (button_state.email_form_smtp_host or "").strip(),
                "smtp_port": (button_state.email_form_smtp_port or "587").strip(),
                "use_tls": button_state.email_form_use_tls,
            }
            button_state.email_test_status = "sending"
            button_state.email_test_message = ""

            def _run_send_test() -> None:
                ok, msg = send_test_email(provider, form_data)
                button_state.email_test_status = "ok" if ok else "error"
                button_state.email_test_message = msg
                if ok and output_dir:
                    set_email_verified(output_dir, True)

            thread = threading.Thread(target=_run_send_test, daemon=True)
            thread.start()
        return True

    if "email_save" in menu_buttons and menu_buttons["email_save"].contains(x, y):
        _save_email_config(output_dir)
        button_state.email_settings_modal_open = False
        HUD.settings_modal_open = True  # return to System Settings
        button_state.email_modal_screen = "provider"
        button_state.email_modal_provider = ""
        button_state.email_focused_field = ""
        button_state.email_cursor_index = 0
        button_state.email_keyboard_mode = "alpha"
        button_state.email_shift_next = False
        button_state.email_password_visible = False
        button_state.email_test_status = ""
        button_state.email_test_message = ""
        return True

    if "email_cancel" in menu_buttons and menu_buttons["email_cancel"].contains(x, y):
        button_state.email_settings_modal_open = False
        HUD.settings_modal_open = True  # return to System Settings
        button_state.email_modal_screen = "provider"
        button_state.email_modal_provider = ""
        button_state.email_focused_field = ""
        button_state.email_cursor_index = 0
        button_state.email_keyboard_mode = "alpha"
        button_state.email_shift_next = False
        button_state.email_password_visible = False
        button_state.email_test_status = ""
        button_state.email_test_message = ""
        return True

    # Consume click if inside form panel or keyboard region (so only negative space left/right of form closes)
    in_form = "email_modal_panel" in menu_buttons and menu_buttons["email_modal_panel"].contains(x, y)
    in_keyboard = "email_modal_keyboard_region" in menu_buttons and menu_buttons["email_modal_keyboard_region"].contains(x, y)
    if in_form or in_keyboard:
        return True
    # Click outside (e.g. negative space left/right of form above keyboard): close without saving (like Cancel)
    button_state.email_settings_modal_open = False
    HUD.settings_modal_open = True  # return to System Settings
    button_state.email_modal_screen = "provider"
    button_state.email_modal_provider = ""
    button_state.email_focused_field = ""
    button_state.email_cursor_index = 0
    button_state.email_keyboard_mode = "alpha"
    button_state.email_shift_next = False
    button_state.email_password_visible = False
    button_state.email_test_status = ""
    button_state.email_test_message = ""
    return True


def _persist_gallery_metadata(output_dir: Optional[Path]) -> None:
    """Write current tags, priority, and tag_data to gallery_metadata.json."""
    if not output_dir:
        return
    save_metadata(
        output_dir,
        getattr(button_state, "gallery_file_priorities", {}),
        getattr(button_state, "gallery_file_tags", {}),
        getattr(button_state, "gallery_tag_data", {}),
    )


def _close_select_mode_modals() -> None:
    """Close all panels that are only visible in select mode."""
    button_state.gallery_priority_modal_open = False
    button_state.gallery_tags_modal_open = False
    button_state.gallery_rename_modal_open = False
    button_state.gallery_tag_modal_open = False
    button_state.gallery_tag_active_field = ""
    button_state.gallery_archive_move_modal_open = False
    button_state.gallery_archive_folder_action_id = None
    button_state.gallery_archive_rename_folder_id = None
    button_state.gallery_archive_rename_query = ""
    button_state.gallery_archive_delete_confirm_folder_id = None


def _get_current_gallery_items(output_dir: Optional[Path]):
    """Return items for current view (main gallery or folder)."""
    if not output_dir:
        return []
    view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
    return get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)


def _apply_rename(output_dir: Optional[Path]) -> None:
    """Rename the selected file(s) using gallery_rename_query as the new stem.
    Single selection: name becomes stem + extension.
    Multiple: names become stem_1, stem_2, ... (same extension). No-op if none selected."""
    if not output_dir:
        return
    new_stem = (button_state.gallery_rename_query or "").strip()
    if not new_stem:
        return
    items = _get_current_gallery_items(output_dir)
    sel = sorted(button_state.gallery_selected_items)
    for i, idx in enumerate(sel):
        if idx >= len(items):
            continue
        old_path = items[idx][0]
        if len(sel) == 1:
            new_name = new_stem + old_path.suffix
        else:
            new_name = f"{new_stem}_{i + 1}{old_path.suffix}"
        new_path = old_path.parent / new_name
        if new_path.exists() and new_path != old_path:
            continue
        try:
            old_path.rename(new_path)
            # Evict old thumbnail from cache
            ui_cache._THUMB_CACHE.pop(old_path, None)
            ui_cache._THUMB_CACHE_MTIME.pop(old_path, None)
            # Transfer priority/tags to new filename
            old_fn = old_path.name
            new_fn = new_path.name
            prios = getattr(button_state, 'gallery_file_priorities', {})
            if old_fn in prios:
                prios[new_fn] = prios.pop(old_fn)
            tags = getattr(button_state, 'gallery_file_tags', {})
            if old_fn in tags:
                tags[new_fn] = tags.pop(old_fn)
        except Exception as e:
            print(f"Rename failed: {e}")
    button_state.gallery_storage_dirty = True
    _persist_gallery_metadata(output_dir)


def _open_tag_modal(output_dir: Optional[Path]) -> None:
    """Pre-fill tag field values from the first selected file and open the tag modal."""
    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_paths = [r['filepath'] for r in rects if r['idx'] in button_state.gallery_selected_items]
    first_path = sel_paths[0] if sel_paths else None

    tag_data = getattr(button_state, 'gallery_tag_data', {})
    existing = tag_data.get(first_path.name, {}) if first_path else {}

    asset_name_val = first_path.stem if first_path else ""
    button_state.gallery_tag_field_values = {
        "asset_name": asset_name_val,
        "asset_type": existing.get("asset_type", ""),
        "leak_type":  existing.get("leak_type",  ""),
    }
    # Auto-select Asset Name so user can edit immediately
    button_state.gallery_tag_active_field = "asset_name"
    button_state.gallery_tag_keyboard_query = asset_name_val
    button_state.gallery_tag_cursor_index = len(asset_name_val)
    button_state.gallery_keyboard_mode = "alpha"
    button_state.gallery_keyboard_shift_next = False
    button_state.gallery_tag_modal_open = True
    button_state.gallery_priority_modal_open = False
    button_state.gallery_rename_modal_open = False


def _commit_active_tag_field() -> None:
    """Commit current keyboard input into field_vals (in-memory only)."""
    if button_state.gallery_tag_active_field:
        button_state.gallery_tag_field_values[button_state.gallery_tag_active_field] = (
            button_state.gallery_tag_keyboard_query
        )


def _apply_tag_save(output_dir: Optional[Path]) -> None:
    """Commit tag modal values: rename first file if Asset Name changed; apply Asset Type and Leak Type to all selected.
    Multi-select: same tags (asset_type, leak_type) apply to every selected item; only the first can be renamed by name."""
    if not output_dir:
        return
    # Commit any active field first
    _commit_active_tag_field()
    button_state.gallery_tag_active_field = ""

    field_vals = getattr(button_state, 'gallery_tag_field_values', {})
    new_asset_name = field_vals.get("asset_name", "").strip()
    new_asset_type = field_vals.get("asset_type", "").strip()
    new_leak_type  = field_vals.get("leak_type",  "").strip()

    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
    sel_rects = [r for r in rects if r['idx'] in button_state.gallery_selected_items]
    if not sel_rects:
        return

    tag_data = getattr(button_state, 'gallery_tag_data', {})
    first_path: Optional[Path] = sel_rects[0]['filepath']
    effective_name = first_path.name

    # Rename first selected file if Asset Name changed
    if new_asset_name and new_asset_name != first_path.stem:
        new_file = first_path.parent / (new_asset_name + first_path.suffix)
        if not (new_file.exists() and new_file != first_path):
            try:
                first_path.rename(new_file)
                # Transfer data to new key
                if first_path.name in tag_data:
                    tag_data[new_file.name] = tag_data.pop(first_path.name)
                prios = getattr(button_state, 'gallery_file_priorities', {})
                if first_path.name in prios:
                    prios[new_file.name] = prios.pop(first_path.name)
                file_tags = getattr(button_state, "gallery_file_tags", {})
                if first_path.name in file_tags:
                    file_tags[new_file.name] = file_tags.pop(first_path.name)
                ui_cache._THUMB_CACHE.pop(first_path, None)
                ui_cache._THUMB_CACHE_MTIME.pop(first_path, None)
                effective_name = new_file.name
            except Exception as e:
                print(f"Tag save rename failed: {e}")

    # Save asset_type + leak_type for ALL selected files
    for rect in sel_rects:
        fp: Path = rect['filepath']
        fname = effective_name if fp == first_path else fp.name
        entry = tag_data.get(fname, {})
        entry["asset_type"] = new_asset_type
        entry["leak_type"]  = new_leak_type
        tag_data[fname] = entry

    button_state.gallery_tag_data = tag_data
    button_state.gallery_storage_dirty = True
    _persist_gallery_metadata(output_dir)


def handle_gallery_click(x: int, y: int, output_dir: Optional[Path]) -> bool:
    """
    Handle clicks in gallery view.
    Returns True if click was handled in gallery, False otherwise.
    """
    if not button_state.gallery_open:
        return False

    if "gallery_back" in menu_buttons and menu_buttons["gallery_back"].contains(x, y):
        trigger_viewer_button_feedback("gallery_back")
        if button_state.gallery_viewer_mode == "grid":
            if getattr(button_state, "gallery_archive_folder_view_id", None):
                # In folder view: go back to main gallery
                button_state.gallery_archive_folder_view_id = None
                button_state.gallery_scroll_offset = 0
                button_state.gallery_selected_item = None
                button_state.gallery_selected_items.clear()
            else:
                # In main gallery: close gallery
                button_state.gallery_open = False
                button_state.gallery_scroll_offset = 0
                button_state.gallery_selected_item = None
                button_state.gallery_archive_folder_action_id = None
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                button_state.gallery_archive_delete_confirm_folder_id = None
                button_state.gallery_select_mode = False
                button_state.gallery_selected_items.clear()
                _close_select_mode_modals()

        else:
            button_state.gallery_viewer_mode = "grid"
            button_state.gallery_video_playing = False
            button_state.gallery_video_frame_idx = 0
            button_state.gallery_tag_info_open = False

        return True

    if button_state.gallery_archive_move_modal_open:
        folders = getattr(button_state, "gallery_archive_folders", [])
        view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
        items = get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)
        filenames = [items[i][0].name for i in button_state.gallery_selected_items if i < len(items)]

        if "move_to_gallery" in menu_buttons and menu_buttons["move_to_gallery"].contains(x, y):
            if output_dir and filenames and view_id:
                button_state.gallery_archive_folders = remove_files_from_all_folders(
                    output_dir, folders, filenames
                )
            button_state.gallery_selected_items.clear()
            button_state.gallery_archive_move_modal_open = False
            return True

        for folder in folders:
            key = "move_to_folder_" + folder.get("id", "")
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                if output_dir and filenames:
                    button_state.gallery_archive_folders = move_files_to_folder(
                        output_dir, folders, folder["id"], filenames
                    )
                button_state.gallery_selected_items.clear()
                button_state.gallery_archive_move_modal_open = False
                return True
        # Click outside folder buttons closes modal
        button_state.gallery_archive_move_modal_open = False
        return True

    if button_state.gallery_delete_modal_open:
        if "modal_yes" in menu_buttons and menu_buttons["modal_yes"].contains(x, y):
            view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
            items = get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)

            try:
                if button_state.gallery_delete_modal_kind == "batch":
                    deleted = 0
                    prios = getattr(button_state, "gallery_file_priorities", {})
                    tags = getattr(button_state, "gallery_file_tags", {})
                    tag_data = getattr(button_state, "gallery_tag_data", {})
                    for idx in sorted(button_state.gallery_selected_items, reverse=True):
                        if 0 <= idx < len(items):
                            path = items[idx][0]
                            try:
                                path.unlink()
                                deleted += 1
                                fn = path.name
                                prios.pop(fn, None)
                                tags.pop(fn, None)
                                tag_data.pop(fn, None)
                            except Exception as e:
                                print(f"Failed to delete {path}: {e}")

                    button_state.gallery_selected_items.clear()
                    button_state.gallery_select_mode = False
                    button_state.gallery_delete_modal_open = False
                    button_state.gallery_delete_modal_kind = "single"
                    if deleted > 0:
                        button_state.gallery_storage_dirty = True
                        _persist_gallery_metadata(output_dir)
                    return True

                if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                    filepath = items[button_state.gallery_selected_item][0]
                    filepath.unlink()
                    fn = filepath.name
                    prios = getattr(button_state, "gallery_file_priorities", {})
                    tags = getattr(button_state, "gallery_file_tags", {})
                    tag_data = getattr(button_state, "gallery_tag_data", {})
                    prios.pop(fn, None)
                    tags.pop(fn, None)
                    tag_data.pop(fn, None)
                    button_state.gallery_storage_dirty = True
                    _persist_gallery_metadata(output_dir)

                    if len(items) <= 1:
                        button_state.gallery_viewer_mode = "grid"
                        button_state.gallery_selected_item = None
                    else:
                        if button_state.gallery_selected_item >= len(items) - 1:
                            button_state.gallery_selected_item = len(items) - 2

                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            except Exception:
                button_state.gallery_delete_modal_open = False
                button_state.gallery_delete_modal_kind = "single"
            return True

        if "modal_no" in menu_buttons and menu_buttons["modal_no"].contains(x, y):
            button_state.gallery_delete_modal_open = False
            button_state.gallery_delete_modal_kind = "single"
            return True

        return True

    # Move to button (select mode only)
    if button_state.gallery_viewer_mode == "grid" and button_state.gallery_select_mode:
        if "gallery_move_to" in menu_buttons and menu_buttons["gallery_move_to"].contains(x, y):
            if button_state.gallery_selected_items:
                folders = getattr(button_state, "gallery_archive_folders", [])
                if folders:
                    button_state.gallery_archive_move_modal_open = True
                else:
                    button_state.gallery_archive_move_hint_until = time.time() + 2.5
            return True

    # Archive folder delete confirmation modal
    if getattr(button_state, "gallery_archive_delete_confirm_folder_id", None):
        folder_id = button_state.gallery_archive_delete_confirm_folder_id
        if "archive_delete_yes" in menu_buttons and menu_buttons["archive_delete_yes"].contains(x, y):
            if output_dir and folder_id:
                folders = getattr(button_state, "gallery_archive_folders", [])
                button_state.gallery_archive_folders = archive_delete_folder(output_dir, folders, folder_id)
            button_state.gallery_archive_delete_confirm_folder_id = None
            button_state.gallery_archive_folder_action_id = None
            if getattr(button_state, "gallery_archive_folder_view_id", None) == folder_id:
                button_state.gallery_archive_folder_view_id = None
            return True
        if "archive_delete_no" in menu_buttons and menu_buttons["archive_delete_no"].contains(x, y):
            button_state.gallery_archive_delete_confirm_folder_id = None
            return True
        return True

    # Archive folder action modal (Rename/Delete when in select mode) and rename sub-modal
    folder_action_id = getattr(button_state, "gallery_archive_folder_action_id", None)
    if folder_action_id:
        in_rename = getattr(button_state, "gallery_archive_rename_folder_id", None) == folder_action_id
        if in_rename:
            # Rename mode: standard keyboard (alpha/symbol, ?123, Shift, shortcut row)
            if "archive_rename_key_switch_sym" in menu_buttons and menu_buttons["archive_rename_key_switch_sym"].contains(x, y):
                button_state.gallery_keyboard_mode = "symbol"
                return True
            if "archive_rename_key_switch_alpha" in menu_buttons and menu_buttons["archive_rename_key_switch_alpha"].contains(x, y):
                button_state.gallery_keyboard_mode = "alpha"
                return True
            if "archive_rename_key_shift" in menu_buttons and menu_buttons["archive_rename_key_shift"].contains(x, y):
                button_state.gallery_keyboard_shift_next = not button_state.gallery_keyboard_shift_next
                return True
            q = getattr(button_state, "gallery_archive_rename_query", "") or ""
            shift = getattr(button_state, "gallery_keyboard_shift_next", False)
            is_alpha = getattr(button_state, "gallery_keyboard_mode", "alpha") == "alpha"
            if is_alpha:
                for c in "qwertyuiopasdfghjklzxcvbnm":
                    key = f"archive_rename_key_{c}"
                    if key in menu_buttons and menu_buttons[key].contains(x, y):
                        button_state.gallery_archive_rename_query = q + (c.upper() if shift else c.lower())
                        if shift:
                            button_state.gallery_keyboard_shift_next = False
                        return True
                for suffix, insert in [("snippet_com", ".com"), ("dot", "."), ("at", "@"), ("dash", "-"), ("underscore", "_"), ("space", " ")]:
                    key = f"archive_rename_key_{suffix}"
                    if key in menu_buttons and menu_buttons[key].contains(x, y):
                        button_state.gallery_archive_rename_query = q + insert
                        return True
            else:
                _sym = "1234567890@._-!#$%&*()+=[{]};:'\"\"(</>?\\,"
                for c in _sym:
                    suf = "\\" if c == "\\" else c
                    key = "archive_rename_key_" + suf
                    if key in menu_buttons and menu_buttons[key].contains(x, y):
                        button_state.gallery_archive_rename_query = q + c
                        return True
            if "archive_rename_key_backspace" in menu_buttons and menu_buttons["archive_rename_key_backspace"].contains(x, y):
                button_state.gallery_archive_rename_query = q[:-1]
                return True
            if "archive_rename_key_clear" in menu_buttons and menu_buttons["archive_rename_key_clear"].contains(x, y):
                button_state.gallery_archive_rename_query = ""
                return True
            if "archive_rename_key_done" in menu_buttons and menu_buttons["archive_rename_key_done"].contains(x, y):
                new_name = (getattr(button_state, "gallery_archive_rename_query", "") or "").strip()
                if new_name and output_dir:
                    folders = getattr(button_state, "gallery_archive_folders", [])
                    button_state.gallery_archive_folders = archive_rename_folder(
                        output_dir, folders, folder_action_id, new_name
                    )
                button_state.gallery_archive_rename_folder_id = None
                button_state.gallery_archive_rename_query = ""
                button_state.gallery_archive_folder_action_id = None
                return True
        else:
            # Action mode: Rename | Delete | Cancel
            if "archive_folder_rename" in menu_buttons and menu_buttons["archive_folder_rename"].contains(x, y):
                folder = next((f for f in getattr(button_state, "gallery_archive_folders", []) if f.get("id") == folder_action_id), None)
                if folder:
                    button_state.gallery_archive_rename_folder_id = folder_action_id
                    button_state.gallery_archive_rename_query = folder.get("name", "Folder") or "Folder"
                    button_state.gallery_keyboard_mode = "alpha"
                    button_state.gallery_keyboard_shift_next = False
                return True
            if "archive_folder_delete" in menu_buttons and menu_buttons["archive_folder_delete"].contains(x, y):
                button_state.gallery_archive_delete_confirm_folder_id = folder_action_id
                return True
            if "archive_folder_cancel" in menu_buttons and menu_buttons["archive_folder_cancel"].contains(x, y):
                button_state.gallery_archive_folder_action_id = None
                return True
        # Absorb clicks on modal/rename panels
        if in_rename:
            if "archive_rename_keyboard_panel" in menu_buttons and menu_buttons["archive_rename_keyboard_panel"].contains(x, y):
                return True
        elif "archive_folder_modal_panel" in menu_buttons and menu_buttons["archive_folder_modal_panel"].contains(x, y):
            return True
        # Click outside: close
        button_state.gallery_archive_folder_action_id = None
        button_state.gallery_archive_rename_folder_id = None
        button_state.gallery_archive_rename_query = ""
        button_state.gallery_archive_delete_confirm_folder_id = None
        return True

    # Archive panel (only visible in main gallery, not in folder view).
    # Stop propagation: when an action modal (Tags, Priority, Rename, Filter, Sort, Search) is open
    # and the click is inside its panel, do not activate the archive panel.
    _on_action_modal_panel = False
    if button_state.gallery_tag_modal_open and (
        ("tag_modal_panel" in menu_buttons and menu_buttons["tag_modal_panel"].contains(x, y))
        or ("gallery_tags_modal_panel" in menu_buttons and menu_buttons["gallery_tags_modal_panel"].contains(x, y))
    ):
        _on_action_modal_panel = True
    elif button_state.gallery_priority_modal_open and "gallery_priority_modal_panel" in menu_buttons and menu_buttons["gallery_priority_modal_panel"].contains(x, y):
        _on_action_modal_panel = True
    elif button_state.gallery_rename_modal_open and "rename_keyboard_panel" in menu_buttons and menu_buttons["rename_keyboard_panel"].contains(x, y):
        _on_action_modal_panel = True
    elif button_state.gallery_filter_modal_open and "gallery_filter_modal_panel" in menu_buttons and menu_buttons["gallery_filter_modal_panel"].contains(x, y):
        _on_action_modal_panel = True
    elif button_state.gallery_sort_modal_open and "gallery_sort_modal_panel" in menu_buttons and menu_buttons["gallery_sort_modal_panel"].contains(x, y):
        _on_action_modal_panel = True
    elif getattr(button_state, "gallery_search_keyboard_open", False) and "search_keyboard_panel" in menu_buttons and menu_buttons["search_keyboard_panel"].contains(x, y):
        _on_action_modal_panel = True

    # Skip archive panel when click is in header area (y < 98) so header buttons (SELECT ALL, etc.)
    # take priority when the archive panel has scrolled up and overlaps the header.
    GALLERY_HEADER_H = 98
    if not getattr(button_state, "gallery_archive_folder_view_id", None) and y >= GALLERY_HEADER_H and "archive_panel" in menu_buttons and menu_buttons["archive_panel"].contains(x, y) and not _on_action_modal_panel:
        if "archive_add_btn" in menu_buttons and menu_buttons["archive_add_btn"].contains(x, y):
            if output_dir:
                folders = getattr(button_state, "gallery_archive_folders", [])
                button_state.gallery_archive_folders = add_folder(output_dir, folders)
        else:
            # Check folder clicks
            for folder in getattr(button_state, "gallery_archive_folders", []):
                key = f"archive_folder_{folder.get('id', '')}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    if button_state.gallery_select_mode:
                        button_state.gallery_archive_folder_action_id = folder["id"]
                    else:
                        button_state.gallery_archive_folder_view_id = folder["id"]
                    return True
        return True

    # Dock row buttons have priority.
    # In select mode: Tags / Priority / Rename  (Search / Filter / Sort hidden)
    # Require at least one selected item to open; otherwise show hint and do not open.
    if button_state.gallery_viewer_mode == "grid" and button_state.gallery_select_mode:
        if "gallery_dock_tags" in menu_buttons and menu_buttons["gallery_dock_tags"].contains(x, y):
            if button_state.gallery_tag_modal_open:
                button_state.gallery_tag_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                _open_tag_modal(output_dir)
            return True
        if "gallery_dock_priority" in menu_buttons and menu_buttons["gallery_dock_priority"].contains(x, y):
            if button_state.gallery_priority_modal_open:
                button_state.gallery_priority_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                button_state.gallery_priority_modal_open = True
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tags_modal_open = False
                button_state.gallery_rename_modal_open = False
            return True
        if "gallery_dock_rename" in menu_buttons and menu_buttons["gallery_dock_rename"].contains(x, y):
            if button_state.gallery_rename_modal_open:
                button_state.gallery_rename_modal_open = False
            elif not button_state.gallery_selected_items:
                button_state.gallery_select_first_hint_until = time.time() + 2.5
            else:
                items = _get_current_gallery_items(output_dir)
                sel = sorted(button_state.gallery_selected_items)
                if sel and sel[0] < len(items):
                    first_path = items[sel[0]][0]
                    button_state.gallery_rename_query = first_path.stem
                else:
                    button_state.gallery_rename_query = ""
                button_state.gallery_rename_modal_open = True
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tags_modal_open = False
                button_state.gallery_priority_modal_open = False
                button_state.gallery_keyboard_mode = "alpha"
                button_state.gallery_keyboard_shift_next = False
            return True

    # Priority modal option clicks
    if button_state.gallery_priority_modal_open:
        for value in ("high", "medium", "low"):
            key = f"gallery_priority_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                items = _get_current_gallery_items(output_dir)
                priorities = getattr(button_state, 'gallery_file_priorities', {})
                for idx in button_state.gallery_selected_items:
                    if idx < len(items):
                        fname = items[idx][0].name
                        # Toggle: clicking the same priority again removes it
                        if priorities.get(fname) == value:
                            priorities[fname] = ""
                        else:
                            priorities[fname] = value
                button_state.gallery_file_priorities = priorities
                _persist_gallery_metadata(output_dir)
                return True
        if "gallery_priority_modal_panel" in menu_buttons and menu_buttons["gallery_priority_modal_panel"].contains(x, y):
            return True
        button_state.gallery_priority_modal_open = False

    # Rename keyboard (standard: alpha/symbol, ?123, Shift, shortcut row)
    if button_state.gallery_rename_modal_open:
        if "rename_key_switch_sym" in menu_buttons and menu_buttons["rename_key_switch_sym"].contains(x, y):
            button_state.gallery_keyboard_mode = "symbol"
            return True
        if "rename_key_switch_alpha" in menu_buttons and menu_buttons["rename_key_switch_alpha"].contains(x, y):
            button_state.gallery_keyboard_mode = "alpha"
            return True
        if "rename_key_shift" in menu_buttons and menu_buttons["rename_key_shift"].contains(x, y):
            button_state.gallery_keyboard_shift_next = not button_state.gallery_keyboard_shift_next
            return True
        q = button_state.gallery_rename_query or ""
        shift = getattr(button_state, "gallery_keyboard_shift_next", False)
        is_alpha = getattr(button_state, "gallery_keyboard_mode", "alpha") == "alpha"
        if is_alpha:
            for c in "qwertyuiopasdfghjklzxcvbnm":
                key = f"rename_key_{c}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_rename_query = q + (c.upper() if shift else c.lower())
                    if shift:
                        button_state.gallery_keyboard_shift_next = False
                    return True
            for suffix, insert in [("snippet_com", ".com"), ("dot", "."), ("at", "@"), ("dash", "-"), ("underscore", "_"), ("space", " ")]:
                key = f"rename_key_{suffix}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_rename_query = q + insert
                    return True
        else:
            _sym = "1234567890@._-!#$%&*()+=[{]};:'\"\"(</>?\\,"
            for c in _sym:
                suf = "\\" if c == "\\" else c
                key = "rename_key_" + suf
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_rename_query = q + c
                    return True
        if "rename_key_done" in menu_buttons and menu_buttons["rename_key_done"].contains(x, y):
            _apply_rename(output_dir)
            button_state.gallery_rename_modal_open = False
            return True
        if "rename_key_clear" in menu_buttons and menu_buttons["rename_key_clear"].contains(x, y):
            button_state.gallery_rename_query = ""
            return True
        if "rename_key_backspace" in menu_buttons and menu_buttons["rename_key_backspace"].contains(x, y):
            button_state.gallery_rename_query = q[:-1]
            return True
        if "rename_keyboard_panel" in menu_buttons and menu_buttons["rename_keyboard_panel"].contains(x, y):
            return True
        button_state.gallery_rename_modal_open = False
        return True

    # ── Tag edit modal (standard keyboard: alpha/symbol, ?123, Shift, shortcut row) ─
    if button_state.gallery_tag_modal_open:
        if button_state.gallery_tag_active_field:
            # Mode / shift
            if "tag_key_switch_sym" in menu_buttons and menu_buttons["tag_key_switch_sym"].contains(x, y):
                button_state.gallery_keyboard_mode = "symbol"
                return True
            if "tag_key_switch_alpha" in menu_buttons and menu_buttons["tag_key_switch_alpha"].contains(x, y):
                button_state.gallery_keyboard_mode = "alpha"
                return True
            if "tag_key_shift" in menu_buttons and menu_buttons["tag_key_shift"].contains(x, y):
                button_state.gallery_keyboard_shift_next = not button_state.gallery_keyboard_shift_next
                return True
            q = button_state.gallery_tag_keyboard_query or ""
            shift = getattr(button_state, "gallery_keyboard_shift_next", False)
            is_alpha = getattr(button_state, "gallery_keyboard_mode", "alpha") == "alpha"
            # Alpha: letters (with shift) + shortcut row
            if is_alpha:
                for c in "qwertyuiopasdfghjklzxcvbnm":
                    bk = f"tag_key_{c}"
                    if bk in menu_buttons and menu_buttons[bk].contains(x, y):
                        button_state.gallery_tag_keyboard_query = q + (c.upper() if shift else c.lower())
                        button_state.gallery_tag_cursor_index = len(button_state.gallery_tag_keyboard_query)
                        if shift:
                            button_state.gallery_keyboard_shift_next = False
                        return True
                for suffix, insert in [("snippet_com", ".com"), ("dot", "."), ("at", "@"), ("dash", "-"), ("underscore", "_"), ("space", " ")]:
                    if f"tag_key_{suffix}" in menu_buttons and menu_buttons[f"tag_key_{suffix}"].contains(x, y):
                        button_state.gallery_tag_keyboard_query = q + insert
                        button_state.gallery_tag_cursor_index = len(button_state.gallery_tag_keyboard_query)
                        return True
            else:
                # Symbol mode: all symbol chars (key_id uses \ for backslash, " for quote)
                _sym = "1234567890@._-!#$%&*()+=[{]};:'\"\"(</>?\\,"
                for c in _sym:
                    suf = "\\" if c == "\\" else c
                    k = "tag_key_" + suf
                    if k in menu_buttons and menu_buttons[k].contains(x, y):
                        button_state.gallery_tag_keyboard_query = q + c
                        button_state.gallery_tag_cursor_index = len(button_state.gallery_tag_keyboard_query)
                        return True
            if "tag_key_backspace" in menu_buttons and menu_buttons["tag_key_backspace"].contains(x, y):
                button_state.gallery_tag_keyboard_query = q[:-1]
                button_state.gallery_tag_cursor_index = max(0, len(button_state.gallery_tag_keyboard_query))
                return True
            if "tag_key_clear" in menu_buttons and menu_buttons["tag_key_clear"].contains(x, y):
                button_state.gallery_tag_keyboard_query = ""
                button_state.gallery_tag_cursor_index = 0
                return True
            if "tag_key_done" in menu_buttons and menu_buttons["tag_key_done"].contains(x, y):
                _apply_tag_save(output_dir)
                button_state.gallery_tag_modal_open = False
                button_state.gallery_tag_active_field = ""
                button_state.gallery_tag_cursor_index = 0
                return True
        if "tag_keyboard_panel" in menu_buttons and menu_buttons["tag_keyboard_panel"].contains(x, y):
            return True

        # Field clicks → switch field (auto-save current first)
        for fkey, flabel, _ in [("asset_name", "Asset Name", ""), ("asset_type", "Asset Type", ""), ("leak_type", "Leak Type", "")]:
            btn_key = f"tag_field_{fkey}"
            if btn_key in menu_buttons and menu_buttons[btn_key].contains(x, y):
                _apply_tag_save(output_dir)  # auto-save before switching
                if fkey == "asset_name":
                    rects = getattr(button_state, 'gallery_thumbnail_rects', [])
                    sel_paths = [r['filepath'] for r in rects if r['idx'] in button_state.gallery_selected_items]
                    default_stem = sel_paths[0].stem if sel_paths else ""
                    prefill = button_state.gallery_tag_field_values.get("asset_name", default_stem)
                else:
                    prefill = button_state.gallery_tag_field_values.get(fkey, "")
                button_state.gallery_tag_active_field = fkey
                button_state.gallery_tag_keyboard_query = prefill
                button_state.gallery_tag_cursor_index = len(prefill)
                return True

        # Absorb clicks on modal panel
        if "tag_modal_panel" in menu_buttons and menu_buttons["tag_modal_panel"].contains(x, y):
            return True
        # Click outside modal: auto-save and close
        _apply_tag_save(output_dir)
        button_state.gallery_tag_modal_open = False
        button_state.gallery_tag_active_field = ""
        button_state.gallery_tag_cursor_index = 0
        return True

    # ── Viewer tag info panel ─────────────────────────────────────────────────
    if button_state.gallery_viewer_mode in ("image", "video"):
        if "tag_info_close" in menu_buttons and menu_buttons["tag_info_close"].contains(x, y):
            button_state.gallery_tag_info_open = False
            return True
        if "tag_info_panel" in menu_buttons and menu_buttons["tag_info_panel"].contains(x, y):
            return True
        if "viewer_tag_btn" in menu_buttons and menu_buttons["viewer_tag_btn"].contains(x, y):
            button_state.gallery_tag_info_open = not button_state.gallery_tag_info_open
            return True

    # Search / Filter / Sort (only in normal grid mode, not select mode)
    if button_state.gallery_viewer_mode == "grid" and not button_state.gallery_select_mode:
        if "gallery_dock_search" in menu_buttons and menu_buttons["gallery_dock_search"].contains(x, y):
            button_state.gallery_search_keyboard_open = not button_state.gallery_search_keyboard_open
            if button_state.gallery_search_keyboard_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_sort_modal_open = False
                button_state.gallery_keyboard_mode = "alpha"
                button_state.gallery_keyboard_shift_next = False
            return True
        if "gallery_dock_filter" in menu_buttons and menu_buttons["gallery_dock_filter"].contains(x, y):
            button_state.gallery_filter_modal_open = not button_state.gallery_filter_modal_open
            if button_state.gallery_filter_modal_open:
                button_state.gallery_sort_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True
        if "gallery_dock_sort" in menu_buttons and menu_buttons["gallery_dock_sort"].contains(x, y):
            button_state.gallery_sort_modal_open = not button_state.gallery_sort_modal_open
            if button_state.gallery_sort_modal_open:
                button_state.gallery_filter_modal_open = False
                button_state.gallery_search_keyboard_open = False
            return True

    # Filter modal: option click only sets filter (modal stays open); click outside closes and fall through
    if button_state.gallery_filter_modal_open:
        for value in ("all", "image", "video"):
            key = f"gallery_filter_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_filter_type = value
                return True
        if "gallery_filter_modal_panel" in menu_buttons and menu_buttons["gallery_filter_modal_panel"].contains(x, y):
            return True
        button_state.gallery_filter_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    # Sort modal: option click only sets sort (modal stays open); click outside closes and fall through
    if button_state.gallery_sort_modal_open:
        for value in ("date", "name", "size", "priority"):
            key = f"gallery_sort_opt_{value}"
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                button_state.gallery_sort_by = value
                return True
        if "gallery_sort_modal_panel" in menu_buttons and menu_buttons["gallery_sort_modal_panel"].contains(x, y):
            return True
        button_state.gallery_sort_modal_open = False
        # fall through so the same click can hit thumbnails or other controls

    if button_state.gallery_search_keyboard_open:
        if "search_key_switch_sym" in menu_buttons and menu_buttons["search_key_switch_sym"].contains(x, y):
            button_state.gallery_keyboard_mode = "symbol"
            return True
        if "search_key_switch_alpha" in menu_buttons and menu_buttons["search_key_switch_alpha"].contains(x, y):
            button_state.gallery_keyboard_mode = "alpha"
            return True
        if "search_key_shift" in menu_buttons and menu_buttons["search_key_shift"].contains(x, y):
            button_state.gallery_keyboard_shift_next = not button_state.gallery_keyboard_shift_next
            return True
        q = button_state.gallery_search_query or ""
        shift = getattr(button_state, "gallery_keyboard_shift_next", False)
        is_alpha = getattr(button_state, "gallery_keyboard_mode", "alpha") == "alpha"
        if is_alpha:
            for c in "qwertyuiopasdfghjklzxcvbnm":
                key = f"search_key_{c}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_search_query = q + (c.upper() if shift else c.lower())
                    if shift:
                        button_state.gallery_keyboard_shift_next = False
                    return True
            for suffix, insert in [("snippet_com", ".com"), ("dot", "."), ("at", "@"), ("dash", "-"), ("underscore", "_"), ("space", " ")]:
                key = f"search_key_{suffix}"
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_search_query = q + insert
                    return True
        else:
            _sym = "1234567890@._-!#$%&*()+=[{]};:'\"\"(</>?\\,"
            for c in _sym:
                suf = "\\" if c == "\\" else c
                key = "search_key_" + suf
                if key in menu_buttons and menu_buttons[key].contains(x, y):
                    button_state.gallery_search_query = q + c
                    return True
        if "search_key_done" in menu_buttons and menu_buttons["search_key_done"].contains(x, y):
            button_state.gallery_search_keyboard_open = False
            return True
        if "search_key_clear" in menu_buttons and menu_buttons["search_key_clear"].contains(x, y):
            button_state.gallery_search_query = ""
            return True
        if "search_key_backspace" in menu_buttons and menu_buttons["search_key_backspace"].contains(x, y):
            button_state.gallery_search_query = q[:-1]
            return True
        if "search_keyboard_panel" in menu_buttons and menu_buttons["search_keyboard_panel"].contains(x, y):
            return True
        button_state.gallery_search_keyboard_open = False
        return True

    if button_state.gallery_viewer_mode in ("image", "video", "grid"):
        if "gallery_delete" in menu_buttons and menu_buttons["gallery_delete"].contains(x, y):
            button_state.gallery_delete_modal_open = True
            return True

    if button_state.gallery_viewer_mode in ("image", "video"):
        if "gallery_prev" in menu_buttons and menu_buttons["gallery_prev"].contains(x, y):
            trigger_viewer_button_feedback("gallery_prev")
            if button_state.gallery_selected_item > 0:
                button_state.gallery_selected_item -= 1
                items = get_displayed_gallery_items(output_dir)

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

        if "gallery_next" in menu_buttons and menu_buttons["gallery_next"].contains(x, y):
            trigger_viewer_button_feedback("gallery_next")
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item < len(items) - 1:
                button_state.gallery_selected_item += 1

                if button_state.gallery_selected_item < len(items):
                    new_item_type = items[button_state.gallery_selected_item][1]
                    button_state.gallery_viewer_mode = new_item_type

                    if new_item_type == "video":
                        button_state.gallery_video_playing = False
                        button_state.gallery_video_frame_idx = 0
                # Instant transition: no swipe animation
                button_state.gallery_viewer_swipe_offset = 0.0
                button_state.gallery_viewer_swipe_velocity = 0.0
                button_state.gallery_viewer_swipe_inertia_active = False
            return True

    if button_state.gallery_viewer_mode == "video":
        if "gallery_play" in menu_buttons and menu_buttons["gallery_play"].contains(x, y):
            trigger_viewer_button_feedback("gallery_play")
            button_state.gallery_video_playing = not button_state.gallery_video_playing
            return True

        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            items = get_displayed_gallery_items(output_dir)
            if button_state.gallery_selected_item is not None and button_state.gallery_selected_item < len(items):
                filepath = items[button_state.gallery_selected_item][0]
                _, total_frames, _ = _video_read_frame_at(filepath, 0)
                if total_frames > 0:
                    progress_btn = menu_buttons["gallery_progress"]
                    click_pos = (x - progress_btn.x) / progress_btn.w
                    click_pos = max(0.0, min(1.0, click_pos))

                    button_state.gallery_video_frame_idx = int(total_frames * click_pos)
                    button_state.gallery_video_playing = False
            return True

    if button_state.gallery_viewer_mode == "grid":

        if "gallery_select_mode" in menu_buttons:
            btn = menu_buttons["gallery_select_mode"]
            if btn.contains(x, y):
                button_state.gallery_select_mode = not button_state.gallery_select_mode
                if not button_state.gallery_select_mode:
                    button_state.gallery_selected_items.clear()
                    _close_select_mode_modals()
                return True

        if button_state.gallery_select_mode and "gallery_select_all" in menu_buttons:
            if menu_buttons["gallery_select_all"].contains(x, y):
                items = _get_current_gallery_items(output_dir)
                # Only select images and videos (never folders)
                selectable_indices = [i for i in range(len(items)) if items[i][1] in ("image", "video")]
                if selectable_indices:
                    all_selected = (
                        len(button_state.gallery_selected_items) == len(selectable_indices)
                        and all(i in button_state.gallery_selected_items for i in selectable_indices)
                    )
                    if all_selected:
                        button_state.gallery_selected_items.clear()
                    else:
                        button_state.gallery_selected_items = set(selectable_indices)

                return True

        if button_state.gallery_select_mode and "gallery_delete_selected" in menu_buttons:
            if menu_buttons["gallery_delete_selected"].contains(x, y):
                if button_state.gallery_selected_items:
                    button_state.gallery_delete_modal_open = True
                    button_state.gallery_delete_modal_kind = "batch"
                return True

        if getattr(button_state, "share_confirm_modal_open", False):
            if "share_confirm_send" in menu_buttons and menu_buttons["share_confirm_send"].contains(x, y):
                # User confirmed: close confirm modal and start send in background
                paths_to_send = [Path(p) for p in button_state.share_confirm_paths]
                button_state.share_confirm_modal_open = False
                button_state.share_confirm_paths = []
                button_state.share_modal_open = True
                button_state.share_modal_sending = True
                button_state.share_modal_title = "Share"
                button_state.share_modal_message = "Sending..."
                button_state.share_modal_progress = 0.0

                def _run_share() -> None:
                    def _on_progress(p: float, _phase: str) -> None:
                        button_state.share_modal_progress = p

                    ok, msg, details = send_share_email(output_dir, paths_to_send, progress_callback=_on_progress)
                    button_state.share_modal_sending = False
                    if ok:
                        button_state.share_modal_title = "Sent"
                        to_email = details.get("to_email", "")
                        ni = details.get("n_images", 0)
                        nv = details.get("n_videos", 0)
                        parts = [f"Sent to {to_email}"]
                        if ni or nv:
                            parts.append(f"{ni} image(s), {nv} video(s)")
                        button_state.share_modal_message = "\n".join(parts)
                    else:
                        button_state.share_modal_title = "Error"
                        button_state.share_modal_message = msg

                thread = threading.Thread(target=_run_share, daemon=True)
                thread.start()
                return True
            if "share_confirm_cancel" in menu_buttons and menu_buttons["share_confirm_cancel"].contains(x, y):
                button_state.share_confirm_modal_open = False
                button_state.share_confirm_paths = []
                return True
            # Tap outside modal (dimmed area): close without sending
            if "share_confirm_modal_panel" in menu_buttons and not menu_buttons["share_confirm_modal_panel"].contains(x, y):
                button_state.share_confirm_modal_open = False
                button_state.share_confirm_paths = []
                return True

        if getattr(button_state, "share_modal_open", False):
            if "share_modal_ok" in menu_buttons and menu_buttons["share_modal_ok"].contains(x, y):
                button_state.share_modal_open = False
                return True

        if button_state.gallery_select_mode and "gallery_share_selected" in menu_buttons and menu_buttons["gallery_share_selected"].w > 0:
            if menu_buttons["gallery_share_selected"].contains(x, y) and output_dir:
                view_id = getattr(button_state, "gallery_archive_folder_view_id", None)
                items = get_folder_displayed_items(output_dir, view_id) if view_id else get_displayed_gallery_items(output_dir)
                paths = [items[i][0] for i in sorted(button_state.gallery_selected_items) if 0 <= i < len(items)]
                if not paths:
                    button_state.share_modal_open = True
                    button_state.share_modal_title = "Share"
                    button_state.share_modal_message = "No files selected."
                    return True
                total_size = 0
                for p in paths:
                    try:
                        total_size += p.stat().st_size
                    except OSError:
                        pass
                if total_size > SHARE_ATTACHMENT_LIMIT_BYTES:
                    button_state.share_modal_open = True
                    button_state.share_modal_title = "Too many files"
                    button_state.share_modal_message = "Attachments too large (max 25 MB)."
                    return True

                # Open confirm modal with request details; send only after user clicks Send
                n_images = sum(1 for p in paths if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
                n_videos = sum(1 for p in paths if p.suffix.lower() in (".mp4", ".avi", ".webm", ".mov"))
                size_str = _format_size(total_size)
                button_state.share_confirm_modal_open = True
                button_state.share_confirm_to_email = get_share_recipient(output_dir)
                button_state.share_confirm_n_images = n_images
                button_state.share_confirm_n_videos = n_videos
                button_state.share_confirm_size_str = size_str
                button_state.share_confirm_file_count = len(paths)
                button_state.share_confirm_paths = [str(p) for p in paths]
                return True

        if hasattr(button_state, 'gallery_thumbnail_rects'):
            for thumb in button_state.gallery_thumbnail_rects:
                if (thumb['x'] <= x <= thumb['x'] + thumb['w'] and
                    thumb['y'] <= y <= thumb['y'] + thumb['h']):
                    idx = thumb['idx']

                    if button_state.gallery_select_mode:
                        if idx in button_state.gallery_selected_items:
                            button_state.gallery_selected_items.remove(idx)
                        else:
                            button_state.gallery_selected_items.add(idx)
                    else:
                        button_state.gallery_selected_item = idx
                        item_type = thumb['type']
                        if item_type == "image":
                            button_state.gallery_viewer_mode = "image"
                        else:
                            button_state.gallery_viewer_mode = "video"
                            button_state.gallery_video_playing = False
                            button_state.gallery_video_frame_idx = 0
                    return True

    return True


def handle_menu_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
) -> Optional[VideoRecorder]:
    """
    Returns (possibly newly created) video_recorder, same logic as original.
    """
    if "menu" not in menu_buttons:
        return video_recorder

    if button_state.is_recording and video_recorder is not None:
        rect = get_recording_timestamp_rect()
        if rect is not None:
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                button_state.is_paused = not button_state.is_paused
                if button_state.is_paused:
                    video_recorder.pause_recording()
                else:
                    video_recorder.resume_recording()
                return video_recorder

    if menu_buttons["menu"].contains(x, y):
        button_state.menu_open = not button_state.menu_open
        return video_recorder

    # Bottom HUD pills (shot, rec, gallery) are independent of menu – handle regardless of menu_open
    if "shot" in menu_buttons and menu_buttons["shot"].w > 0 and menu_buttons["shot"].contains(x, y):
        if current_frame is not None and output_dir is not None:
            save_screenshot(current_frame, output_dir)
            button_state.gallery_storage_dirty = True
        return video_recorder

    if "gallery" in menu_buttons and menu_buttons["gallery"].w > 0 and menu_buttons["gallery"].contains(x, y):
        if button_state.is_recording and video_recorder is not None:
            video_recorder.stop_recording()
            button_state.is_recording = False
            button_state.is_paused = False
            button_state.gallery_storage_dirty = True
        button_state.gallery_open = True
        button_state.menu_open = False
        button_state.gallery_storage_dirty = True
        if output_dir:
            button_state.gallery_archive_folders = load_archive_folders(output_dir)
        return video_recorder

    if button_state.is_recording and button_state.is_paused:
        if "rec_resume" in menu_buttons and menu_buttons["rec_resume"].w > 0 and menu_buttons["rec_resume"].contains(x, y):
            if video_recorder is not None:
                video_recorder.resume_recording()
                button_state.is_paused = False
            return video_recorder
        if "rec_stop" in menu_buttons and menu_buttons["rec_stop"].w > 0 and menu_buttons["rec_stop"].contains(x, y):
            if video_recorder is not None:
                video_recorder.stop_recording()
                button_state.is_recording = False
                button_state.is_paused = False
                button_state.gallery_storage_dirty = True
            return video_recorder

    if "rec" in menu_buttons and menu_buttons["rec"].w > 0 and menu_buttons["rec"].contains(x, y):
        if video_recorder is None and output_dir is not None:
            video_recorder = VideoRecorder(output_dir, width, height, fps=30)
        if video_recorder is None:
            return None
        if not button_state.is_recording:
            if video_recorder.start_recording():
                button_state.is_recording = True
                button_state.is_paused = False
        else:
            button_state.is_paused = True
            video_recorder.pause_recording()
        return video_recorder

    # Dropdown items (fps, gain, etc.) only when menu is open
    if not button_state.menu_open:
        return video_recorder

    if "fps30" in menu_buttons and menu_buttons["fps30"].contains(x, y):
        button_state.fps_mode = "30"
        return video_recorder
    # if "fps60" in menu_buttons and menu_buttons["fps60"].contains(x, y):
    #     button_state.fps_mode = "60"
    #     return video_recorder  # 60FPS button commented out
    if "fpsmax" in menu_buttons and menu_buttons["fpsmax"].contains(x, y):
        button_state.fps_mode = "MAX"
        return video_recorder

    if "wifi" in menu_buttons and menu_buttons["wifi"].contains(x, y):
        HUD.wifi_modal_open = True
        HUD.wifi_modal_screen = "list"
        if not HUD.wifi_networks:
            HUD.wifi_networks = []
        return video_recorder

    if "main_menu_settings" in menu_buttons and menu_buttons["main_menu_settings"].contains(x, y):
        HUD.settings_modal_open = True
        return video_recorder

    if "gain" in menu_buttons and menu_buttons["gain"].contains(x, y):
        button_state.gain_mode = "HIGH" if button_state.gain_mode == "LOW" else "LOW"
        menu_buttons["gain"].text = f"GAIN: {button_state.gain_mode}"
        GAIN_CONTROL.set_mode(button_state.gain_mode)
        return video_recorder

    if "source" in menu_buttons and menu_buttons["source"].contains(x, y):
        modes = list(SOURCE_MODES)
        cur = button_state.source_mode
        try:
            i = modes.index(cur)
        except ValueError:
            i = modes.index(SOURCE_DEFAULT)
        button_state.source_mode = modes[(i + 1) % len(modes)]
        menu_buttons["source"].text = f"SRC: {button_state.source_mode}"
        return video_recorder

    if "spectrum_analyzer" in menu_buttons and menu_buttons["spectrum_analyzer"].contains(x, y):
        # Cycle dB -> NORM -> dBA -> dB
        next_mode = {"dB": "NORM", "NORM": "dBA", "dBA": "dB"}.get(
            button_state.spectrum_analyzer_mode, "dB"
        )
        button_state.spectrum_analyzer_mode = next_mode
        menu_buttons["spectrum_analyzer"].text = f"SPECTRUM: {button_state.spectrum_analyzer_mode}"
        return video_recorder

    return video_recorder


def handle_button_click(
    x: int,
    y: int,
    current_frame: Optional[np.ndarray],
    output_dir: Optional[Path],
    camera_available: bool,
    video_recorder: Optional[VideoRecorder],
    width: int,
    height: int,
) -> Optional[VideoRecorder]:
    """
    Handles button clicks:
      - Calls handle_menu_click first
      - Camera toggle
      - Source toggle (SIM <-> SPI) and calls optional callbacks
      - Debug toggle

    Returns updated video_recorder (may be created by menu click).
    """
    return handle_menu_click(
        x, y,
        current_frame=current_frame,
        output_dir=output_dir,
        video_recorder=video_recorder,
        width=width,
        height=height,
    )


def handle_gallery_viewer_mouse(event, x, y, flags, output_dir) -> bool:
    """Handle horizontal swipe/drag with inertia in image or video viewer."""
    if not button_state.gallery_open or button_state.gallery_viewer_mode not in ("image", "video"):
        return False

    now = time.perf_counter()
    items = get_displayed_gallery_items(output_dir)
    n = len(items)
    idx = button_state.gallery_selected_item
    if idx is None or n == 0:
        return False

    # Progress bar scrub (video only): takes priority over swipe on bar
    if event == cv2.EVENT_LBUTTONDOWN and button_state.gallery_viewer_mode == "video":
        if "gallery_progress" in menu_buttons and menu_buttons["gallery_progress"].contains(x, y):
            button_state.gallery_progress_dragging = True
            total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
            if total_frames > 0:
                prog = menu_buttons["gallery_progress"]
                t = (x - prog.x) / prog.w
                t = max(0.0, min(1.0, t))
                button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
            return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_progress_dragging:
        total_frames = getattr(button_state, "_gallery_video_total_frames", 0)
        if total_frames > 0 and "gallery_progress" in menu_buttons:
            prog = menu_buttons["gallery_progress"]
            t = (x - prog.x) / prog.w
            t = max(0.0, min(1.0, t))
            button_state.gallery_video_frame_idx = int(t * (total_frames - 1))
        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_progress_dragging:
        button_state.gallery_progress_dragging = False
        return True

    # LBUTTONUP: clear drag state and treat as click if we didn't move (nav/back/delete get priority)
    if event == cv2.EVENT_LBUTTONUP:
        button_state.gallery_viewer_swipe_dragging = False
        if not button_state.gallery_viewer_swipe_drag_moved:
            return handle_gallery_click(x, y, output_dir)
        off = button_state.gallery_viewer_swipe_offset
        vel = button_state.gallery_viewer_swipe_velocity
        th = getattr(button_state, "_gallery_swipe_threshold_px", ui_cache.VIEWER_SWIPE_THRESHOLD_PX)
        vth = ui_cache.VIEWER_SWIPE_VELOCITY_THRESHOLD
        # Swipe right (positive off/vel) -> prev (newer). Swipe left (negative) -> next (older).
        if (off >= th or vel >= vth) and idx > 0:
            button_state.gallery_selected_item = idx - 1
            button_state.gallery_viewer_mode = items[idx - 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if (off <= -th or vel <= -vth) and idx < n - 1:
            button_state.gallery_selected_item = idx + 1
            button_state.gallery_viewer_mode = items[idx + 1][1]
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
            if button_state.gallery_viewer_mode == "video":
                button_state.gallery_video_playing = False
                button_state.gallery_video_frame_idx = 0
            return True
        if abs(vel) > ui_cache.VIEWER_SWIPE_STOP_VELOCITY:
            button_state.gallery_viewer_swipe_inertia_active = True
            button_state.gallery_viewer_swipe_last_inertia_t = now
        else:
            button_state.gallery_viewer_swipe_offset = 0.0
            button_state.gallery_viewer_swipe_velocity = 0.0
        return True

    # LBUTTONDOWN: don't start swipe if on a button so nav/back/delete take priority
    if event == cv2.EVENT_LBUTTONDOWN:
        on_button = False
        for key in ("gallery_back", "gallery_prev", "gallery_next", "gallery_delete", "gallery_play", "gallery_progress"):
            if key in menu_buttons and menu_buttons[key].contains(x, y):
                on_button = True
                break
        if on_button:
            button_state.gallery_viewer_swipe_drag_moved = False
            return True
        button_state.gallery_viewer_swipe_dragging = True
        button_state.gallery_viewer_swipe_start_x = x
        button_state.gallery_viewer_swipe_offset = 0.0
        button_state.gallery_viewer_swipe_velocity = 0.0
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        button_state.gallery_viewer_swipe_drag_moved = False
        button_state.gallery_viewer_swipe_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_viewer_swipe_dragging:
        dx = x - button_state.gallery_viewer_swipe_start_x
        if abs(dx) > ui_cache.DRAG_PX:
            button_state.gallery_viewer_swipe_drag_moved = True
        # Clamp and apply rubber band at first/last item
        max_offset = 120.0
        raw = max(-max_offset, min(max_offset, float(dx)))
        button_state.gallery_viewer_swipe_offset = _viewer_rubber_band_offset(raw, idx, n)
        dt = max(1e-6, now - button_state.gallery_viewer_swipe_last_t)
        inst_v = (x - button_state.gallery_viewer_swipe_last_x) / dt * ui_cache.VIEWER_SWIPE_FLING_GAIN
        button_state.gallery_viewer_swipe_velocity = 0.5 * button_state.gallery_viewer_swipe_velocity + 0.5 * inst_v
        button_state.gallery_viewer_swipe_last_t = now
        button_state.gallery_viewer_swipe_last_x = x
        return True

    return False


def handle_gallery_mouse(event, x, y, flags, output_dir) -> bool:
    if not button_state.gallery_open or button_state.gallery_viewer_mode != "grid":
        return False

    now = time.perf_counter()

    if event == cv2.EVENT_LBUTTONDOWN:
        button_state.gallery_dragging = True
        button_state.gallery_drag_start_y = y
        button_state.gallery_drag_start_x = x
        button_state.gallery_drag_start_scroll = button_state.gallery_scroll_offset
        button_state.gallery_drag_moved = False

        button_state.gallery_scroll_velocity = 0.0
        button_state.gallery_last_drag_t = now
        button_state.gallery_last_drag_y = y
        button_state.gallery_inertia_active = False
        return True

    if event == cv2.EVENT_MOUSEMOVE and button_state.gallery_dragging:
        dy = y - button_state.gallery_drag_start_y
        dx = x - button_state.gallery_drag_start_x

        if (abs(dy) > ui_cache.DRAG_PX) or (abs(dx) > ui_cache.DRAG_PX):
            button_state.gallery_drag_moved = True

        if button_state.gallery_drag_moved:
            new_scroll = button_state.gallery_drag_start_scroll - dy
            max_scroll = int(getattr(button_state, "gallery_max_scroll", 0))
            button_state.gallery_scroll_offset = max(0, min(int(new_scroll), max_scroll))

            dt = max(1e-6, now - button_state.gallery_last_drag_t)
            FLING_GAIN = 2
            EMA_ALPHA = 0.6

            inst_v = (button_state.gallery_last_drag_y - y) / dt
            inst_v *= FLING_GAIN

            button_state.gallery_scroll_velocity = (1.0 - EMA_ALPHA) * button_state.gallery_scroll_velocity + EMA_ALPHA * inst_v
            button_state.gallery_last_drag_t = now
            button_state.gallery_last_drag_y = y

        return True

    if event == cv2.EVENT_LBUTTONUP and button_state.gallery_dragging:
        button_state.gallery_dragging = False

        if not button_state.gallery_drag_moved:
            return handle_gallery_click(x, y, output_dir)

        if abs(button_state.gallery_scroll_velocity) > 50.0:
            button_state.gallery_inertia_active = True
            button_state.gallery_last_inertia_t = now
        else:
            button_state.gallery_inertia_active = False
            button_state.gallery_scroll_velocity = 0.0

        return True

    return False

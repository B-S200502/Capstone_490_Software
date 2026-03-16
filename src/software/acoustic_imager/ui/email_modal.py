"""
Email Settings modal: provider selection (Gmail, Outlook, Yahoo, Other) then form with keyboard.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from ..state import button_state
from ..config import HUD_MENU_OPACITY, MENU_ACTIVE_BLUE
from ..io.email_config import load_config, SMTP_PRESETS
from .standard_keyboard import draw_standard_alpha_keyboard, draw_standard_symbol_keyboard

PROVIDERS = [
    ("Gmail", "gmail"),
    ("Outlook", "outlook"),
    ("Yahoo", "yahoo"),
    ("Other", "other"),
]


def draw_email_modal(frame: np.ndarray, output_dir: Optional[Path]) -> None:
    """Draw Email Settings modal: provider tiles or form with keyboard."""
    if not button_state.email_settings_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)

    # Dark overlay (cached black buffer, no frame copy)
    ui_cache.apply_modal_dim(frame, 0.5)

    if button_state.email_modal_screen == "provider":
        _draw_provider_screen(frame, fw, fh, font, text_color, border_color)
    else:
        _draw_form_screen(frame, fw, fh, font, text_color, border_color, output_dir)


def _draw_provider_screen(
    frame: np.ndarray, fw: int, fh: int, font, text_color, border_color
) -> None:
    """Draw 4 provider tiles (Gmail, Outlook, Yahoo, Other)."""
    modal_w = 380
    modal_h = 220
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), border_color, 3, cv2.LINE_AA)

    cv2.putText(frame, "EMAIL SETTINGS", (modal_x + 20, modal_y + 36), font, 0.65, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "Choose provider:", (modal_x + 20, modal_y + 58), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    pad = 16
    tile_w = (modal_w - 2 * pad - 24) // 2
    tile_h = 44
    y0 = modal_y + 78
    for i, (label, key) in enumerate(PROVIDERS):
        row, col = i // 2, i % 2
        tx = modal_x + pad + col * (tile_w + 24)
        ty = y0 + row * (tile_h + 12)
        key_name = f"email_provider_{key}"
        if key_name not in menu_buttons:
            menu_buttons[key_name] = Button(tx, ty, tile_w, tile_h, label)
        else:
            b = menu_buttons[key_name]
            b.x, b.y, b.w, b.h = tx, ty, tile_w, tile_h
            b.text = label
        menu_buttons[key_name].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    if "email_modal_panel" not in menu_buttons:
        menu_buttons["email_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["email_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h


def _cursor_x_for_text(font, font_scale: float, text_prefix: str) -> int:
    """Return x offset (pixels) of cursor position given the text before the cursor."""
    if not text_prefix:
        return 0
    (w, _), _ = cv2.getTextSize(text_prefix, font, font_scale, 1)
    return w


def _draw_eye_icon(frame: np.ndarray, cx: int, cy: int, r: int, visible: bool, color: tuple) -> None:
    """Draw eye icon: open eye (pupil visible) when visible=True, eye with slash when False."""
    # Eye outline (ellipse)
    cv2.ellipse(frame, (cx, cy), (r, r // 2 + 2), 0, 0, 360, color, 1, cv2.LINE_AA)
    if visible:
        # Pupil (small circle) for "show" state
        cv2.circle(frame, (cx, cy), max(1, r // 4), color, -1, cv2.LINE_AA)
    else:
        # Diagonal slash through eye for "hidden" state
        cv2.line(frame, (cx - r, cy - r // 2), (cx + r, cy + r // 2), color, 1, cv2.LINE_AA)


def _draw_form_screen(
    frame: np.ndarray,
    fw: int,
    fh: int,
    font,
    text_color,
    border_color,
    output_dir: Optional[Path],
) -> None:
    """Draw form (narrower, lower); keyboard full width, ~40% height."""
    provider = button_state.email_modal_provider or "gmail"
    title = f"EMAIL SETTINGS - {provider.capitalize()}"

    # Form stays centered with side margins for layout; keyboard uses full screen width
    side_margin = max(24, int(fw * 0.15))
    content_w = fw - 2 * side_margin
    content_x = side_margin

    # Keyboard: bottom 40% of screen, full width of screen (wider keys)
    keyboard_h = int(fh * 0.40)
    keyboard_y0 = fh - keyboard_h
    key_gap = 6
    is_symbol = getattr(button_state, "email_keyboard_mode", "alpha") == "symbol"
    n_rows = 5
    key_h = (keyboard_h - key_gap * (n_rows + 1)) // n_rows
    key_w = (fw - 9 * key_gap) // 10  # full width: keys span 0 to fw
    sp_w = (fw - 2 * key_gap) // 3

    # Form panel: above keyboard, centered
    form_y0 = 56
    form_bottom = keyboard_y0 - 16
    form_w = content_w
    form_inner_x = content_x + 20
    # Inputs 15% narrower
    label_w = 140
    input_w_raw = form_w - label_w - 56
    input_w = int(input_w_raw * 0.85)
    line_h = 50
    input_h = 42
    label_font = 0.56
    input_font = 0.52

    # Form panel background (content strip only)
    cv2.rectangle(
        frame,
        (content_x, form_y0),
        (content_x + form_w, form_bottom),
        (45, 45, 45),
        -1,
    )
    cv2.rectangle(
        frame,
        (content_x, form_y0),
        (content_x + form_w, form_bottom),
        border_color,
        3,
        cv2.LINE_AA,
    )

    focused_field = getattr(button_state, "email_focused_field", "") or "email"
    title_y = form_y0 + 38
    cv2.putText(
        frame, title,
        (form_inner_x, title_y),
        font, 0.64, text_color, 1, cv2.LINE_AA,
    )

    # Send Test button: top right, enabled only when all required fields are filled
    send_test_btn_w = 92
    send_test_btn_h = 28
    send_test_x = content_x + form_w - send_test_btn_w - 12
    send_test_y = form_y0 + 10
    email_ok = (button_state.email_form_email or "").strip()
    password_ok = (button_state.email_form_password or "").strip()
    smtp_ok = provider != "other" or bool((button_state.email_form_smtp_host or "").strip())
    send_test_enabled = bool(email_ok and password_ok and smtp_ok)
    if "email_send_test" not in menu_buttons:
        menu_buttons["email_send_test"] = Button(send_test_x, send_test_y, send_test_btn_w, send_test_btn_h, "Send Test")
    else:
        menu_buttons["email_send_test"].x, menu_buttons["email_send_test"].y = send_test_x, send_test_y
        menu_buttons["email_send_test"].w, menu_buttons["email_send_test"].h = send_test_btn_w, send_test_btn_h
    if send_test_enabled:
        menu_buttons["email_send_test"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)
    else:
        cv2.rectangle(frame, (send_test_x, send_test_y), (send_test_x + send_test_btn_w, send_test_y + send_test_btn_h), (60, 60, 60), -1)
        cv2.rectangle(frame, (send_test_x, send_test_y), (send_test_x + send_test_btn_w, send_test_y + send_test_btn_h), (100, 100, 100), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize("Send Test", font, 0.44, 1)
        cv2.putText(frame, "Send Test", (send_test_x + (send_test_btn_w - tw) // 2, send_test_y + (send_test_btn_h + th) // 2), font, 0.44, (120, 120, 120), 1, cv2.LINE_AA)
    # Feedback text immediately to the left of Send Test (right-aligned, small gap)
    test_status = getattr(button_state, "email_test_status", "")
    test_msg = getattr(button_state, "email_test_message", "")
    feedback_y = send_test_y + send_test_btn_h // 2 + 6
    gap = 12
    if test_status == "ok":
        s, scale, color = "Sent!", 0.46, (0, 200, 100)
        (tw, _), _ = cv2.getTextSize(s, font, scale, 1)
        cv2.putText(frame, s, (send_test_x - gap - tw, feedback_y), font, scale, color, 1, cv2.LINE_AA)
    elif test_status == "error" and test_msg:
        s, scale, color = test_msg[:32], 0.40, (0, 0, 220)
        (tw, _), _ = cv2.getTextSize(s, font, scale, 1)
        cv2.putText(frame, s, (send_test_x - gap - tw, feedback_y), font, scale, color, 1, cv2.LINE_AA)
    elif test_status == "sending":
        s, scale, color = "Sending...", 0.44, (200, 200, 0)
        (tw, _), _ = cv2.getTextSize(s, font, scale, 1)
        cv2.putText(frame, s, (send_test_x - gap - tw, feedback_y), font, scale, color, 1, cv2.LINE_AA)

    # For "other": SMTP Host and Port on same row as title (right side), larger small fields
    if provider == "other":
        smtp_label_w = 44
        smtp_input_w = 140
        port_label_w = 28
        port_input_w = 56
        row1_x = content_x + form_w - (smtp_label_w + smtp_input_w + 8 + port_label_w + port_input_w + 12)
        small_h = 32
        small_y = form_y0 + (50 - small_h) // 2
        cv2.putText(frame, "SMTP:", (row1_x, small_y + small_h // 2 + 6), font, 0.42, text_color, 1, cv2.LINE_AA)
        smtp_ix = row1_x + smtp_label_w + 4
        cv2.rectangle(frame, (smtp_ix, small_y), (smtp_ix + smtp_input_w, small_y + small_h), (28, 28, 34), -1)
        cv2.rectangle(frame, (smtp_ix, small_y), (smtp_ix + smtp_input_w, small_y + small_h), (160, 160, 160), 1, cv2.LINE_AA)
        smtp_val = (button_state.email_form_smtp_host or "")[:24]
        cv2.putText(frame, smtp_val, (smtp_ix + 4, small_y + small_h // 2 + 6), font, 0.44, text_color, 1, cv2.LINE_AA)
        if "email_field_smtp_host" not in menu_buttons:
            menu_buttons["email_field_smtp_host"] = Button(smtp_ix, small_y, smtp_input_w, small_h, "")
        else:
            menu_buttons["email_field_smtp_host"].x, menu_buttons["email_field_smtp_host"].y = smtp_ix, small_y
            menu_buttons["email_field_smtp_host"].w, menu_buttons["email_field_smtp_host"].h = smtp_input_w, small_h
        port_x = smtp_ix + smtp_input_w + 8
        cv2.putText(frame, "Port:", (port_x, small_y + small_h // 2 + 6), font, 0.42, text_color, 1, cv2.LINE_AA)
        port_ix = port_x + port_label_w + 4
        cv2.rectangle(frame, (port_ix, small_y), (port_ix + port_input_w, small_y + small_h), (28, 28, 34), -1)
        cv2.rectangle(frame, (port_ix, small_y), (port_ix + port_input_w, small_y + small_h), (160, 160, 160), 1, cv2.LINE_AA)
        port_val = (button_state.email_form_smtp_port or "587")[:6]
        cv2.putText(frame, port_val, (port_ix + 4, small_y + small_h // 2 + 6), font, 0.44, text_color, 1, cv2.LINE_AA)
        if "email_field_smtp_port" not in menu_buttons:
            menu_buttons["email_field_smtp_port"] = Button(port_ix, small_y, port_input_w, small_h, "")
        else:
            menu_buttons["email_field_smtp_port"].x, menu_buttons["email_field_smtp_port"].y = port_ix, small_y
            menu_buttons["email_field_smtp_port"].w, menu_buttons["email_field_smtp_port"].h = port_input_w, small_h
        if focused_field == "smtp_host":
            cv2.rectangle(frame, (smtp_ix, small_y), (smtp_ix + smtp_input_w, small_y + small_h), MENU_ACTIVE_BLUE, 2, cv2.LINE_AA)
            ci = max(0, min(getattr(button_state, "email_cursor_index", 0), len(button_state.email_form_smtp_host or "")))
            cx = _cursor_x_for_text(font, 0.44, smtp_val[:ci])
            if int(time.time() * 2) % 2 == 0:
                cv2.line(frame, (smtp_ix + 4 + cx, small_y + 4), (smtp_ix + 4 + cx, small_y + small_h - 4), (255, 255, 255), 2, cv2.LINE_AA)
        elif focused_field == "smtp_port":
            cv2.rectangle(frame, (port_ix, small_y), (port_ix + port_input_w, small_y + small_h), MENU_ACTIVE_BLUE, 2, cv2.LINE_AA)
            ci = max(0, min(getattr(button_state, "email_cursor_index", 0), len(button_state.email_form_smtp_port or "")))
            cx = _cursor_x_for_text(font, 0.44, port_val[:ci])
            if int(time.time() * 2) % 2 == 0:
                cv2.line(frame, (port_ix + 4 + cx, small_y + 4), (port_ix + 4 + cx, small_y + small_h - 4), (255, 255, 255), 2, cv2.LINE_AA)
    else:
        # When not "other", clear SMTP/Port hit areas so they don't capture clicks
        for k in ("email_field_smtp_host", "email_field_smtp_port"):
            if k in menu_buttons:
                menu_buttons[k].x, menu_buttons[k].y = 0, 0
                menu_buttons[k].w, menu_buttons[k].h = 0, 0

    field_y = form_y0 + 68
    eye_btn_w = 36
    eye_btn_gap = 6
    password_visible = getattr(button_state, "email_password_visible", False)

    def row(
        label: str,
        value: str,
        key: str,
        mask: bool = False,
        show_eye_toggle: bool = False,
        label_line2: Optional[str] = None,
    ) -> None:
        nonlocal field_y
        field_id = key.replace("email_field_", "")
        is_focused = focused_field == field_id
        if label_line2 is not None:
            # Two-line label (e.g. "Default To" / "(recipient):") so it doesn't run into the input
            (_, line_height), _ = cv2.getTextSize("Ay", font, label_font, 1)
            line1_y = field_y + 12 + line_height
            line2_y = field_y + 12 + line_height + 4 + line_height
            cv2.putText(frame, label, (form_inner_x, line1_y), font, label_font, text_color, 1, cv2.LINE_AA)
            cv2.putText(frame, label_line2, (form_inner_x, line2_y), font, label_font, text_color, 1, cv2.LINE_AA)
        else:
            cv2.putText(
                frame, label + ":",
                (form_inner_x, field_y + input_h // 2 + 6),
                font, label_font, text_color, 1, cv2.LINE_AA,
            )
        ix = form_inner_x + label_w + 10
        actual_input_w = (input_w - eye_btn_w - eye_btn_gap) if show_eye_toggle else input_w
        cv2.rectangle(frame, (ix, field_y), (ix + actual_input_w, field_y + input_h), (28, 28, 34), -1)
        if is_focused:
            cv2.rectangle(frame, (ix, field_y), (ix + actual_input_w, field_y + input_h), MENU_ACTIVE_BLUE, 3, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (ix, field_y), (ix + actual_input_w, field_y + input_h), (160, 160, 160), 2, cv2.LINE_AA)
        disp = ("*" * len(value)) if mask and value else (value or "")
        cv2.putText(
            frame, disp[:48],
            (ix + 8, field_y + input_h // 2 + 6),
            font, input_font, text_color, 1, cv2.LINE_AA,
        )
        if is_focused:
            cursor_index = max(0, min(getattr(button_state, "email_cursor_index", 0), len(value)))
            cursor_x = _cursor_x_for_text(font, input_font, disp[:cursor_index])
            if int(time.time() * 2) % 2 == 0:  # blink ~1 Hz
                cy1, cy2 = field_y + 6, field_y + input_h - 6
                cv2.line(frame, (ix + 8 + cursor_x, cy1), (ix + 8 + cursor_x, cy2), (255, 255, 255), 2, cv2.LINE_AA)
        if key not in menu_buttons:
            menu_buttons[key] = Button(ix, field_y, actual_input_w, input_h, "")
        else:
            menu_buttons[key].x, menu_buttons[key].y = ix, field_y
            menu_buttons[key].w, menu_buttons[key].h = actual_input_w, input_h
        if show_eye_toggle:
            eye_x = ix + actual_input_w + eye_btn_gap
            eye_y = field_y + (input_h - eye_btn_w) // 2
            if eye_y < field_y:
                eye_y = field_y
            eye_btn_h = min(eye_btn_w, input_h)
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (50, 50, 55), -1)
            cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (160, 160, 160), 1, cv2.LINE_AA)
            eye_cx = eye_x + eye_btn_w // 2
            eye_cy = eye_y + eye_btn_h // 2
            eye_r = min(eye_btn_w, eye_btn_h) // 3
            _draw_eye_icon(frame, eye_cx, eye_cy, eye_r, password_visible, (220, 220, 220))
            if "email_password_toggle_visible" not in menu_buttons:
                menu_buttons["email_password_toggle_visible"] = Button(eye_x, eye_y, eye_btn_w, eye_btn_h, "")
            else:
                menu_buttons["email_password_toggle_visible"].x = eye_x
                menu_buttons["email_password_toggle_visible"].y = eye_y
                menu_buttons["email_password_toggle_visible"].w = eye_btn_w
                menu_buttons["email_password_toggle_visible"].h = eye_btn_h
        field_y += line_h

    row("Email", button_state.email_form_email, "email_field_email")
    row("Password", button_state.email_form_password, "email_field_password", mask=not password_visible, show_eye_toggle=True)
    row("Default To", button_state.email_form_default_to, "email_field_default_to", label_line2="(recipient):")

    # Save and Cancel buttons: centered at bottom of form
    btn_y = field_y + 14
    btn_h = 42
    btn_w = 110
    btn_gap = 16
    total_btn_w = 2 * btn_w + btn_gap
    btn_start_x = content_x + (form_w - total_btn_w) // 2
    if "email_save" not in menu_buttons:
        menu_buttons["email_save"] = Button(btn_start_x, btn_y, btn_w, btn_h, "Save")
    else:
        menu_buttons["email_save"].x, menu_buttons["email_save"].y = btn_start_x, btn_y
        menu_buttons["email_save"].w, menu_buttons["email_save"].h = btn_w, btn_h
    menu_buttons["email_save"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)
    if "email_cancel" not in menu_buttons:
        menu_buttons["email_cancel"] = Button(btn_start_x + btn_w + btn_gap, btn_y, btn_w, btn_h, "Cancel")
    else:
        menu_buttons["email_cancel"].x, menu_buttons["email_cancel"].y = btn_start_x + btn_w + btn_gap, btn_y
        menu_buttons["email_cancel"].w, menu_buttons["email_cancel"].h = btn_w, btn_h
    menu_buttons["email_cancel"].draw(frame, transparent=True)

    # Keyboard: full width (0 to fw), wider key surfaces
    key_y = keyboard_y0 + key_gap
    key_x_base = 0
    key_font = 0.66 if key_h > 32 else 0.56
    key_font_special = 0.62 if key_h > 32 else 0.52

    if is_symbol:
        draw_standard_symbol_keyboard(
            frame, key_x_base, key_y, fw, key_w, key_h, sp_w, key_gap,
            "email_key_", font, key_font, key_font_special,
        )
    else:
        draw_standard_alpha_keyboard(
            frame, key_x_base, key_y, fw, key_w, key_h, sp_w, key_gap,
            "email_key_", font, key_font, key_font_special,
            getattr(button_state, "email_shift_next", False),
            shift_highlight_color=MENU_ACTIVE_BLUE,
        )

    # Hit-test: only form rect and keyboard region consume clicks; negative space (left/right of form above keyboard) closes modal
    if "email_modal_panel" not in menu_buttons:
        menu_buttons["email_modal_panel"] = Button(content_x, form_y0, form_w, form_bottom - form_y0, "")
    else:
        menu_buttons["email_modal_panel"].x = content_x
        menu_buttons["email_modal_panel"].y = form_y0
        menu_buttons["email_modal_panel"].w = form_w
        menu_buttons["email_modal_panel"].h = form_bottom - form_y0
    if "email_modal_keyboard_region" not in menu_buttons:
        menu_buttons["email_modal_keyboard_region"] = Button(0, keyboard_y0, fw, keyboard_h, "")
    else:
        menu_buttons["email_modal_keyboard_region"].x = 0
        menu_buttons["email_modal_keyboard_region"].y = keyboard_y0
        menu_buttons["email_modal_keyboard_region"].w = fw
        menu_buttons["email_modal_keyboard_region"].h = keyboard_h

"""
WiFi modal: discover networks, connect with password (keyboard for secured networks).
"""

from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

from . import ui_cache
from .button import menu_buttons, Button
from .standard_keyboard import (
    draw_standard_alpha_keyboard,
    draw_standard_symbol_keyboard,
    compute_standard_keyboard_dimensions,
)
from ..state import HUD
from ..config import MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT
from ..io.wifi_scan import scan_wifi_networks, connect_wifi, disconnect_wifi
from ..system_info import get_system_network_info
from .email_modal import _draw_eye_icon

MODAL_W = 520
MODAL_H_LIST = 380
MODAL_H_PASSWORD = 420
SCROLLBAR_W = 24
SCROLL_BTN_H = 28
ROW_H = 36
DRAG_THRESHOLD_PX = 10   # only start list scroll drag after finger moves this many px


def _run_connect(ssid: str, password: str, bssid: str = "") -> None:
    """Background thread: connect and set status."""
    HUD.wifi_connect_status = "connecting"
    ok, msg = connect_wifi(ssid, password, bssid=bssid or None)
    HUD.wifi_connect_status = "ok" if ok else "error"
    HUD.wifi_connect_message = msg


def _run_scan() -> None:
    """Background thread: scan and update networks."""
    HUD.wifi_scanning = True
    try:
        HUD.wifi_networks = scan_wifi_networks()
    finally:
        HUD.wifi_scanning = False


def _get_wifi_list_layout(fw: int, fh: int) -> dict:
    """Return layout for list screen: list rect, scrollbar rect, row_h, max_scroll, other_networks."""
    modal_w = MODAL_W
    modal_h = MODAL_H_LIST
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2
    pad = 16
    row_h = ROW_H
    section_gap = 8
    current_y = modal_y + 52
    row_x = modal_x + pad
    row_w = modal_w - 2 * pad
    list_content_w = row_w - SCROLLBAR_W
    expanded = HUD.wifi_current_expanded
    box_h = (row_h - 4) + (72 if expanded else 0)
    current_row_y = current_y + 6
    list_y0 = current_row_y + box_h + section_gap + 16 + 6
    list_visible_h = modal_y + modal_h - pad - list_y0
    list_visible_h = max(0, list_visible_h)
    connected_ssid, _, _ = get_system_network_info(0)
    networks = HUD.wifi_networks or []
    other_networks = [n for n in networks if (n.get("ssid") or "").strip() != connected_ssid]
    total_content_h = len(other_networks) * row_h
    max_scroll = max(0, total_content_h - list_visible_h)
    scroll_offset = min(HUD.wifi_list_scroll_offset, max_scroll) if max_scroll > 0 else 0
    HUD.wifi_list_scroll_offset = scroll_offset
    sb_x = row_x + list_content_w
    sb_y = list_y0
    sb_h = list_visible_h
    return {
        "row_x": row_x,
        "list_y0": list_y0,
        "list_content_w": list_content_w,
        "list_visible_h": list_visible_h,
        "row_h": row_h,
        "total_content_h": total_content_h,
        "max_scroll": max_scroll,
        "scroll_offset": scroll_offset,
        "sb_x": sb_x,
        "sb_y": sb_y,
        "sb_h": sb_h,
        "other_networks": other_networks,
    }


def draw_wifi_modal(frame: np.ndarray) -> None:
    """Draw WiFi modal: list screen or password screen with keyboard."""
    if not HUD.wifi_modal_open:
        return

    fh, fw = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    border_color = (100, 100, 100)

    # Dark overlay (cached black buffer, no frame copy)
    ui_cache.apply_modal_dim(frame, 0.5)

    if HUD.wifi_modal_screen == "list":
        _draw_list_screen(frame, fw, fh, font, text_color, border_color)
    else:
        _draw_password_screen(frame, fw, fh, font, text_color, border_color)


def _draw_list_screen(
    frame: np.ndarray, fw: int, fh: int, font, text_color, border_color
) -> None:
    """Draw network list: title, Scan, current connection (blue), Other Networks (scrollable)."""
    modal_w = MODAL_W
    modal_h = MODAL_H_LIST
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), border_color, 3, cv2.LINE_AA)

    pad = 16
    title_y = modal_y + 32
    cv2.putText(frame, "WiFi", (modal_x + pad, title_y), font, 0.7, text_color, 1, cv2.LINE_AA)

    # Scan button
    scan_w, scan_h = 80, 32
    scan_x = modal_x + modal_w - scan_w - pad
    scan_y = modal_y + 12
    if "wifi_scan" not in menu_buttons:
        menu_buttons["wifi_scan"] = Button(scan_x, scan_y, scan_w, scan_h, "Scan")
    else:
        menu_buttons["wifi_scan"].x, menu_buttons["wifi_scan"].y = scan_x, scan_y
        menu_buttons["wifi_scan"].w, menu_buttons["wifi_scan"].h = scan_w, scan_h
    menu_buttons["wifi_scan"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    layout = _get_wifi_list_layout(fw, fh)
    row_x = layout["row_x"]
    list_y0 = layout["list_y0"]
    list_content_w = layout["list_content_w"]
    list_visible_h = layout["list_visible_h"]
    row_h = layout["row_h"]
    scroll_offset = layout["scroll_offset"]
    max_scroll = layout["max_scroll"]
    other_networks = layout["other_networks"]
    sb_x, sb_y, sb_h = layout["sb_x"], layout["sb_y"], layout["sb_h"]
    connected_ssid, _, _ = get_system_network_info(0)
    current_y = modal_y + 52
    row_w = modal_w - 2 * pad
    section_gap = 8
    current_row_y = current_y + 6

    # "Current connection" section header
    cv2.putText(frame, "Current connection", (row_x, current_y - 4), font, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    current_row_y = current_y + 6
    expanded = HUD.wifi_current_expanded
    box_h = (row_h - 4) + (72 if expanded else 0)  # extra height when expanded
    if "wifi_current" not in menu_buttons:
        menu_buttons["wifi_current"] = Button(row_x, current_row_y, row_w, box_h, "")
    else:
        menu_buttons["wifi_current"].x, menu_buttons["wifi_current"].y = row_x, current_row_y
        menu_buttons["wifi_current"].w, menu_buttons["wifi_current"].h = row_w, box_h
    # Blue highlight for current connection box (expanded when clicked)
    cv2.rectangle(frame, (row_x, current_row_y), (row_x + row_w, current_row_y + box_h), MENU_ACTIVE_BLUE, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (row_x, current_row_y), (row_x + row_w, current_row_y + box_h), MENU_ACTIVE_BLUE_LIGHT, 1, cv2.LINE_AA)
    current_label = connected_ssid[:24] if connected_ssid else "Not connected"
    cv2.putText(frame, current_label, (row_x + 8, current_row_y + (row_h - 4) // 2 + 6), font, 0.48, text_color, 1, cv2.LINE_AA)
    if expanded:
        info_y = current_row_y + row_h + 4
        ssid_full, ip, hostname = get_system_network_info(0)
        line_h = 18
        info_color = (220, 220, 220)
        if ssid_full:
            cv2.putText(frame, f"SSID: {ssid_full[:28]}", (row_x + 8, info_y), font, 0.40, info_color, 1, cv2.LINE_AA)
        if ip:
            cv2.putText(frame, f"IP: {ip}", (row_x + 8, info_y + line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        if hostname:
            cv2.putText(frame, f"Device: {hostname[:24]}", (row_x + 8, info_y + 2 * line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        # Signal/security from scan if available
        networks = HUD.wifi_networks or []
        match = next((n for n in networks if (n.get("ssid") or "").strip() == ssid_full), None)
        if match:
            sig = match.get("signal", "")
            sec = match.get("security", "Open")
            cv2.putText(frame, f"Signal: {sig}%  Security: {sec}", (row_x + 8, info_y + 3 * line_h), font, 0.40, info_color, 1, cv2.LINE_AA)
        # Disconnect button: only when expanded, at bottom right of box
        if connected_ssid:
            disconnect_btn_w, disconnect_btn_h = 90, 28
            disc_x = row_x + row_w - disconnect_btn_w - 8
            disc_y = current_row_y + box_h - disconnect_btn_h - 6
            if "wifi_disconnect" not in menu_buttons:
                menu_buttons["wifi_disconnect"] = Button(disc_x, disc_y, disconnect_btn_w, disconnect_btn_h, "Disconnect")
            else:
                menu_buttons["wifi_disconnect"].x, menu_buttons["wifi_disconnect"].y = disc_x, disc_y
                menu_buttons["wifi_disconnect"].w, menu_buttons["wifi_disconnect"].h = disconnect_btn_w, disconnect_btn_h
            cv2.rectangle(frame, (disc_x, disc_y), (disc_x + disconnect_btn_w, disc_y + disconnect_btn_h), (60, 60, 60), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (disc_x, disc_y), (disc_x + disconnect_btn_w, disc_y + disconnect_btn_h), (255, 255, 255), 1, cv2.LINE_AA)
            (tw, th), _ = cv2.getTextSize("Disconnect", font, 0.4, 1)
            cv2.putText(frame, "Disconnect", (disc_x + (disconnect_btn_w - tw) // 2, disc_y + (disconnect_btn_h + th) // 2), font, 0.4, text_color, 1, cv2.LINE_AA)
    if not (expanded and connected_ssid) and "wifi_disconnect" in menu_buttons:
        menu_buttons["wifi_disconnect"].w = 0
        menu_buttons["wifi_disconnect"].h = 0

    # "Other Networks" section (scrollable list + scrollbar)
    other_label = "Other Networks"
    if HUD.wifi_scanning:
        other_label += "  Scanning..."
    cv2.putText(frame, other_label, (row_x, list_y0 - 10), font, 0.44, (180, 180, 180), 1, cv2.LINE_AA)
    for i, net in enumerate(other_networks):
        row_y = list_y0 + i * row_h - scroll_offset
        if row_y < list_y0 or row_y + row_h > list_y0 + list_visible_h:
            continue
        ssid = net.get("ssid", "")
        signal = net.get("signal", "")
        security = net.get("security", "Open")
        cv2.rectangle(frame, (row_x, row_y), (row_x + list_content_w, row_y + row_h - 4), (50, 50, 50), -1)
        cv2.rectangle(frame, (row_x, row_y), (row_x + list_content_w, row_y + row_h - 4), (80, 80, 80), 1, cv2.LINE_AA)
        label = f"{ssid[:24]}  {signal}%  {security}" if signal else f"{ssid[:28]}  {security}"
        cv2.putText(frame, label, (row_x + 8, row_y + (row_h - 4) // 2 + 6), font, 0.48, text_color, 1, cv2.LINE_AA)
    # Scroll up/down buttons (when scrollable)
    if max_scroll > 0:
        up_x = sb_x
        up_y = sb_y - SCROLL_BTN_H - 4
        down_x = sb_x
        down_y = sb_y + sb_h + 4
        if up_y >= modal_y + 50:
            if "wifi_list_up" not in menu_buttons:
                menu_buttons["wifi_list_up"] = Button(up_x, up_y, SCROLLBAR_W, SCROLL_BTN_H, "")
            else:
                menu_buttons["wifi_list_up"].x, menu_buttons["wifi_list_up"].y = up_x, up_y
                menu_buttons["wifi_list_up"].w, menu_buttons["wifi_list_up"].h = SCROLLBAR_W, SCROLL_BTN_H
            cv2.rectangle(frame, (up_x, up_y), (up_x + SCROLLBAR_W, up_y + SCROLL_BTN_H), (70, 70, 70), -1)
            cv2.putText(frame, "^", (up_x + 4, up_y + SCROLL_BTN_H - 6), font, 0.4, text_color, 1, cv2.LINE_AA)
        if down_y + SCROLL_BTN_H <= modal_y + modal_h - pad:
            if "wifi_list_down" not in menu_buttons:
                menu_buttons["wifi_list_down"] = Button(down_x, down_y, SCROLLBAR_W, SCROLL_BTN_H, "")
            else:
                menu_buttons["wifi_list_down"].x, menu_buttons["wifi_list_down"].y = down_x, down_y
                menu_buttons["wifi_list_down"].w, menu_buttons["wifi_list_down"].h = SCROLLBAR_W, SCROLL_BTN_H
            cv2.rectangle(frame, (down_x, down_y), (down_x + SCROLLBAR_W, down_y + SCROLL_BTN_H), (70, 70, 70), -1)
            cv2.putText(frame, "v", (down_x + 6, down_y + SCROLL_BTN_H - 4), font, 0.4, text_color, 1, cv2.LINE_AA)
    # Scrollbar track and thumb
    if layout["total_content_h"] > list_visible_h:
        cv2.rectangle(frame, (sb_x, sb_y), (sb_x + SCROLLBAR_W, sb_y + sb_h), (45, 45, 45), -1)
        thumb_h = max(28, int(sb_h * list_visible_h / layout["total_content_h"]))
        thumb_range = sb_h - thumb_h
        thumb_y = sb_y + (int(thumb_range * scroll_offset / max_scroll) if max_scroll > 0 else 0)
        cv2.rectangle(frame, (sb_x + 3, thumb_y), (sb_x + SCROLLBAR_W - 3, thumb_y + thumb_h), (120, 120, 120), -1)

    if "wifi_modal_panel" not in menu_buttons:
        menu_buttons["wifi_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["wifi_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h


def _draw_password_screen(
    frame: np.ndarray, fw: int, fh: int, font, text_color, border_color
) -> None:
    """Draw password form + standard keyboard."""
    modal_w = MODAL_W
    modal_h = MODAL_H_PASSWORD
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), border_color, 3, cv2.LINE_AA)

    pad = 16
    ssid = HUD.wifi_connect_ssid or ""
    title_y = modal_y + 32
    cv2.putText(frame, f"Connect to: {ssid[:28]}", (modal_x + pad, title_y), font, 0.56, text_color, 1, cv2.LINE_AA)

    # Password field with show/hide toggle
    field_y = modal_y + 56
    field_h = 40
    eye_btn_w = 36
    eye_btn_gap = 6
    field_w = modal_w - 2 * pad - eye_btn_w - eye_btn_gap
    field_x = modal_x + pad
    cv2.putText(frame, "Password:", (field_x, field_y - 8), font, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (field_x, field_y), (field_x + field_w, field_y + field_h), (28, 28, 34), -1)
    cv2.rectangle(frame, (field_x, field_y), (field_x + field_w, field_y + field_h), (120, 120, 120), 1, cv2.LINE_AA)
    pw_visible = HUD.wifi_password_visible
    pw_display = HUD.wifi_password if pw_visible else ("*" * len(HUD.wifi_password))
    cv2.putText(frame, pw_display or " ", (field_x + 8, field_y + field_h // 2 + 6), font, 0.52, text_color, 1, cv2.LINE_AA)
    # Show password (eye) button
    eye_x = field_x + field_w + eye_btn_gap
    eye_y = field_y + (field_h - eye_btn_w) // 2
    eye_btn_h = min(eye_btn_w, field_h)
    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (50, 50, 55), -1)
    cv2.rectangle(frame, (eye_x, eye_y), (eye_x + eye_btn_w, eye_y + eye_btn_h), (160, 160, 160), 1, cv2.LINE_AA)
    eye_cx, eye_cy = eye_x + eye_btn_w // 2, eye_y + eye_btn_h // 2
    _draw_eye_icon(frame, eye_cx, eye_cy, min(eye_btn_w, eye_btn_h) // 3, pw_visible, (220, 220, 220))
    if "wifi_password_field" not in menu_buttons:
        menu_buttons["wifi_password_field"] = Button(field_x, field_y, field_w, field_h, "")
    else:
        menu_buttons["wifi_password_field"].x, menu_buttons["wifi_password_field"].y = field_x, field_y
        menu_buttons["wifi_password_field"].w, menu_buttons["wifi_password_field"].h = field_w, field_h
    if "wifi_password_toggle" not in menu_buttons:
        menu_buttons["wifi_password_toggle"] = Button(eye_x, eye_y, eye_btn_w, eye_btn_h, "")
    else:
        menu_buttons["wifi_password_toggle"].x, menu_buttons["wifi_password_toggle"].y = eye_x, eye_y
        menu_buttons["wifi_password_toggle"].w, menu_buttons["wifi_password_toggle"].h = eye_btn_w, eye_btn_h

    # Connect button
    conn_w, conn_h = 100, 36
    conn_x = modal_x + modal_w - conn_w - pad
    conn_y = field_y + field_h + 12
    if "wifi_connect_btn" not in menu_buttons:
        menu_buttons["wifi_connect_btn"] = Button(conn_x, conn_y, conn_w, conn_h, "Connect")
    else:
        menu_buttons["wifi_connect_btn"].x, menu_buttons["wifi_connect_btn"].y = conn_x, conn_y
        menu_buttons["wifi_connect_btn"].w, menu_buttons["wifi_connect_btn"].h = conn_w, conn_h
    menu_buttons["wifi_connect_btn"].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    # Status message
    status = HUD.wifi_connect_status
    msg = HUD.wifi_connect_message
    if status == "connecting":
        cv2.putText(frame, "Connecting...", (modal_x + pad, conn_y + conn_h + 24), font, 0.44, (200, 200, 0), 1, cv2.LINE_AA)
    elif status == "ok":
        cv2.putText(frame, "Connected!", (modal_x + pad, conn_y + conn_h + 24), font, 0.44, (0, 200, 100), 1, cv2.LINE_AA)
    elif status == "error" and msg:
        cv2.putText(frame, msg[:40], (modal_x + pad, conn_y + conn_h + 24), font, 0.40, (0, 0, 220), 1, cv2.LINE_AA)

    # Keyboard: standard layout (matches app style)
    keyboard_y0 = modal_y + modal_h - 200
    keyboard_h = 200
    content_w = modal_w - 2 * pad
    key_gap = 6
    n_rows = 5
    dims = compute_standard_keyboard_dimensions(content_w, keyboard_h, key_gap=key_gap, n_rows=n_rows)
    key_w, key_h, sp_w = dims["key_w"], dims["key_h"], dims["sp_w"]
    key_x_base = modal_x + pad
    key_y = keyboard_y0 + key_gap
    key_font = 0.56
    key_font_special = 0.50
    is_symbol = HUD.wifi_keyboard_mode == "symbol"
    shift_highlight = HUD.wifi_shift_next

    if is_symbol:
        draw_standard_symbol_keyboard(
            frame, key_x_base, key_y, content_w, key_w, key_h, sp_w, key_gap,
            "wifi_key_", font, key_font, key_font_special,
        )
    else:
        draw_standard_alpha_keyboard(
            frame, key_x_base, key_y, content_w, key_w, key_h, sp_w, key_gap,
            "wifi_key_", font, key_font, key_font_special,
            shift_highlight,
            shift_highlight_color=MENU_ACTIVE_BLUE,
        )

    if "wifi_modal_panel" not in menu_buttons:
        menu_buttons["wifi_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["wifi_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h

    if "wifi_keyboard_region" not in menu_buttons:
        menu_buttons["wifi_keyboard_region"] = Button(modal_x, keyboard_y0, modal_w, keyboard_h, "")
    else:
        menu_buttons["wifi_keyboard_region"].x = modal_x
        menu_buttons["wifi_keyboard_region"].y = keyboard_y0
        menu_buttons["wifi_keyboard_region"].w = modal_w
        menu_buttons["wifi_keyboard_region"].h = keyboard_h


def _is_in_wifi_list_content(x: int, y: int, fw: int, fh: int) -> bool:
    """True if (x,y) is in the list content area (not scrollbar)."""
    if fw <= 0 or fh <= 0 or HUD.wifi_modal_screen != "list":
        return False
    layout = _get_wifi_list_layout(fw, fh)
    rx, ly0, lw, lh = layout["row_x"], layout["list_y0"], layout["list_content_w"], layout["list_visible_h"]
    if not (rx <= x < rx + lw and ly0 <= y < ly0 + lh):
        return False
    return True


def _is_in_wifi_scrollbar(x: int, y: int, fw: int, fh: int) -> bool:
    """True if (x,y) is in the scrollbar track/thumb area."""
    if fw <= 0 or fh <= 0 or HUD.wifi_modal_screen != "list":
        return False
    layout = _get_wifi_list_layout(fw, fh)
    sx, sy, sh = layout["sb_x"], layout["sb_y"], layout["sb_h"]
    return sx <= x < sx + SCROLLBAR_W and sy <= y < sy + sh


def _wifi_select_network_at_position(x: int, y: int, fw: int, fh: int) -> bool:
    """If (x,y) is on a network row, select it (open password or connect). Returns True if selected."""
    if fw <= 0 or fh <= 0:
        return False
    layout = _get_wifi_list_layout(fw, fh)
    row_x = layout["row_x"]
    list_y0 = layout["list_y0"]
    list_content_w = layout["list_content_w"]
    list_visible_h = layout["list_visible_h"]
    if not (row_x <= x < row_x + list_content_w and list_y0 <= y < list_y0 + list_visible_h):
        return False
    row_h = layout["row_h"]
    scroll_offset = layout["scroll_offset"]
    other_networks = layout["other_networks"]
    row_i = (y - list_y0 + scroll_offset) // row_h
    if row_i < 0 or row_i >= len(other_networks):
        return False
    net = other_networks[row_i]
    ssid = net.get("ssid", "")
    security = net.get("security", "Open")
    if security == "Open":
        HUD.wifi_connect_ssid = ssid
        HUD.wifi_password = ""
        HUD.wifi_connect_status = ""
        HUD.wifi_connect_message = ""
        thread = threading.Thread(target=_run_connect, args=(ssid, ""), daemon=True)
        thread.start()
        return True
    HUD.wifi_connect_ssid = ssid
    HUD.wifi_connect_bssid = (net.get("bssid") or "").strip()
    HUD.wifi_password = ""
    HUD.wifi_modal_screen = "password"
    HUD.wifi_keyboard_mode = "alpha"
    HUD.wifi_shift_next = False
    HUD.wifi_password_visible = False
    HUD.wifi_connect_status = ""
    HUD.wifi_connect_message = ""
    return True


def handle_wifi_modal_touch_drag(event: int, x: int, y: int, fw: int, fh: int) -> bool:
    """Handle touch-drag for list scroll. Tap = select network; drag = scroll. Returns True if consumed."""
    if not HUD.wifi_modal_open or HUD.wifi_modal_screen != "list" or fw <= 0 or fh <= 0:
        return False
    layout = _get_wifi_list_layout(fw, fh)
    max_scroll = layout["max_scroll"]
    list_visible_h = layout["list_visible_h"]
    sb_y, sb_h = layout["sb_y"], layout["sb_h"]

    if event == cv2.EVENT_LBUTTONDOWN:
        if _is_in_wifi_list_content(x, y, fw, fh) and not _is_in_wifi_scrollbar(x, y, fw, fh):
            HUD.wifi_list_touch_started = True
            HUD.wifi_list_touch_start_x = x
            HUD.wifi_list_touch_start_y = y
            return True
        return False

    if event == cv2.EVENT_MOUSEMOVE:
        if HUD.wifi_list_touch_started:
            if abs(y - HUD.wifi_list_touch_start_y) > DRAG_THRESHOLD_PX:
                HUD.wifi_list_content_dragging = True
                HUD.wifi_list_content_drag_start_y = y
                HUD.wifi_list_content_drag_start_scroll = HUD.wifi_list_scroll_offset
                HUD.wifi_list_touch_started = False
        if HUD.wifi_list_content_dragging:
            dy = y - HUD.wifi_list_content_drag_start_y
            new_scroll = HUD.wifi_list_content_drag_start_scroll - dy
            HUD.wifi_list_scroll_offset = max(0, min(max_scroll, int(new_scroll)))
            return True
        if HUD.wifi_list_scroll_dragging:
            thumb_range = sb_h - max(28, int(sb_h * list_visible_h / max(layout["total_content_h"], 1)))
            if thumb_range > 0:
                dy = y - HUD.wifi_list_scrollbar_drag_start_y
                scroll_delta = int(dy * max_scroll / thumb_range)
                HUD.wifi_list_scroll_offset = max(0, min(max_scroll, HUD.wifi_list_scrollbar_drag_start_scroll + scroll_delta))
            return True
        return False

    if event == cv2.EVENT_LBUTTONUP:
        if HUD.wifi_list_touch_started:
            _wifi_select_network_at_position(
                HUD.wifi_list_touch_start_x, HUD.wifi_list_touch_start_y, fw, fh
            )
            HUD.wifi_list_touch_started = False
            return True
        if HUD.wifi_list_content_dragging:
            HUD.wifi_list_content_dragging = False
            return True
        if HUD.wifi_list_scroll_dragging:
            HUD.wifi_list_scroll_dragging = False
            return True
        return False

    return False


def handle_wifi_modal_click(x: int, y: int, fw: int = 0, fh: int = 0) -> bool:
    """Handle click. Returns True if handled. Pass fw, fh for list scroll/row hit-test."""
    if not HUD.wifi_modal_open:
        return False

    if HUD.wifi_modal_screen == "list":
        return _handle_list_click(x, y, fw, fh)
    return _handle_password_click(x, y)


def _handle_list_click(x: int, y: int, fw: int, fh: int) -> bool:
    if "wifi_scan" in menu_buttons and menu_buttons["wifi_scan"].contains(x, y):
        if not HUD.wifi_scanning:
            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
        return True
    if "wifi_disconnect" in menu_buttons and menu_buttons["wifi_disconnect"].w > 0 and menu_buttons["wifi_disconnect"].contains(x, y):
        ok, _ = disconnect_wifi()
        if ok and not HUD.wifi_scanning:
            thread = threading.Thread(target=_run_scan, daemon=True)
            thread.start()
        return True
    if "wifi_current" in menu_buttons and menu_buttons["wifi_current"].contains(x, y):
        HUD.wifi_current_expanded = not HUD.wifi_current_expanded
        return True
    if fw > 0 and fh > 0:
        layout = _get_wifi_list_layout(fw, fh)
        max_scroll = layout["max_scroll"]
        if "wifi_list_up" in menu_buttons and menu_buttons["wifi_list_up"].w > 0 and menu_buttons["wifi_list_up"].contains(x, y):
            HUD.wifi_list_scroll_offset = max(0, HUD.wifi_list_scroll_offset - 50)
            return True
        if "wifi_list_down" in menu_buttons and menu_buttons["wifi_list_down"].w > 0 and menu_buttons["wifi_list_down"].contains(x, y):
            HUD.wifi_list_scroll_offset = min(max_scroll, HUD.wifi_list_scroll_offset + 50)
            return True
        if _is_in_wifi_scrollbar(x, y, fw, fh) and max_scroll > 0:
            HUD.wifi_list_scroll_dragging = True
            HUD.wifi_list_scrollbar_drag_start_y = y
            HUD.wifi_list_scrollbar_drag_start_scroll = HUD.wifi_list_scroll_offset
            return True
        if _is_in_wifi_list_content(x, y, fw, fh) and not _is_in_wifi_scrollbar(x, y, fw, fh):
            if _wifi_select_network_at_position(x, y, fw, fh):
                return True
    if "wifi_modal_panel" in menu_buttons and menu_buttons["wifi_modal_panel"].contains(x, y):
        return True
    HUD.wifi_modal_open = False
    HUD.wifi_current_expanded = False
    return True


def _handle_password_click(x: int, y: int) -> bool:
    # Show/hide password toggle
    if "wifi_password_toggle" in menu_buttons and menu_buttons["wifi_password_toggle"].contains(x, y):
        HUD.wifi_password_visible = not HUD.wifi_password_visible
        return True
    # Back, Clear, Done (check before keyboard region so they always work)
    for key_id, action in [
        ("wifi_key_backspace", "backspace"),
        ("wifi_key_clear", "clear"),
        ("wifi_key_done", "done"),
    ]:
        if key_id in menu_buttons and menu_buttons[key_id].contains(x, y):
            if action == "backspace":
                HUD.wifi_password = HUD.wifi_password[:-1]
            elif action == "clear":
                HUD.wifi_password = ""
            else:
                HUD.wifi_modal_screen = "list"
                HUD.wifi_connect_ssid = ""
                HUD.wifi_password = ""
                HUD.wifi_password_visible = False
            return True
    # Connect button
    if "wifi_connect_btn" in menu_buttons and menu_buttons["wifi_connect_btn"].contains(x, y):
        if HUD.wifi_connect_status != "connecting":
            HUD.wifi_connect_status = ""
            HUD.wifi_connect_message = ""
            thread = threading.Thread(
                target=_run_connect,
                args=(HUD.wifi_connect_ssid, HUD.wifi_password, HUD.wifi_connect_bssid),
                daemon=True,
            )
            thread.start()
        return True

    # Keyboard
    in_keyboard = "wifi_keyboard_region" in menu_buttons and menu_buttons["wifi_keyboard_region"].contains(x, y)
    if in_keyboard:
        _handle_wifi_keyboard_click(x, y)
        return True

    if "wifi_modal_panel" in menu_buttons and menu_buttons["wifi_modal_panel"].contains(x, y):
        return True
    # Click outside: go back to list
    HUD.wifi_modal_screen = "list"
    HUD.wifi_connect_ssid = ""
    HUD.wifi_password = ""
    HUD.wifi_password_visible = False
    return True


def _handle_wifi_keyboard_click(x: int, y: int) -> None:
    """Process keyboard key presses for wifi password."""
    from .keyboard import ROWS_QWERTY
    from .standard_keyboard import STANDARD_ALPHA_SHORTCUT_ROW, STANDARD_SYMBOL_ROW1, STANDARD_SYMBOL_ROW2, STANDARD_SYMBOL_ROW3, STANDARD_SYMBOL_ROW4
    from .keyboard import SPECIAL_KEYS_COMPACT

    is_alpha = HUD.wifi_keyboard_mode == "alpha"
    shift = HUD.wifi_shift_next
    pw = HUD.wifi_password

    # Mode switch
    if "wifi_key_switch_sym" in menu_buttons and menu_buttons["wifi_key_switch_sym"].contains(x, y):
        HUD.wifi_keyboard_mode = "symbol"
        return
    if "wifi_key_switch_alpha" in menu_buttons and menu_buttons["wifi_key_switch_alpha"].contains(x, y):
        HUD.wifi_keyboard_mode = "alpha"
        return
    if "wifi_key_shift" in menu_buttons and menu_buttons["wifi_key_shift"].contains(x, y):
        HUD.wifi_shift_next = not HUD.wifi_shift_next
        return

    # Special keys (Back, Clear, Done) - check first
    for key_id in ("wifi_key_backspace", "wifi_key_clear", "wifi_key_done"):
        if key_id in menu_buttons:
            b = menu_buttons[key_id]
            if b.w > 0 and b.h > 0 and b.contains(x, y):
                if key_id == "wifi_key_backspace":
                    HUD.wifi_password = pw[:-1]
                elif key_id == "wifi_key_clear":
                    HUD.wifi_password = ""
                else:
                    HUD.wifi_modal_screen = "list"
                    HUD.wifi_connect_ssid = ""
                    HUD.wifi_password = ""
                return

    # Alpha shortcut row
    for label, val, _ in STANDARD_ALPHA_SHORTCUT_ROW:
        k = f"wifi_key_{val}"
        if k in menu_buttons and menu_buttons[k].contains(x, y):
            if val == "space":
                HUD.wifi_password = pw + " "
            elif val == "dot":
                HUD.wifi_password = pw + "."
            elif val == "at":
                HUD.wifi_password = pw + "@"
            elif val == "dash":
                HUD.wifi_password = pw + "-"
            elif val == "underscore":
                HUD.wifi_password = pw + "_"
            elif val == "snippet_com":
                HUD.wifi_password = pw + ".com"
            return

    # Letter keys
    if is_alpha:
        for row in ROWS_QWERTY:
            for c in row:
                k = f"wifi_key_{c}"
                if k in menu_buttons and menu_buttons[k].contains(x, y):
                    char = c.upper() if shift else c.lower()
                    HUD.wifi_password = pw + char
                    if shift:
                        HUD.wifi_shift_next = False
                    return
    else:
        for row in [STANDARD_SYMBOL_ROW1, STANDARD_SYMBOL_ROW2, STANDARD_SYMBOL_ROW3, STANDARD_SYMBOL_ROW4]:
            for c in row:
                from .standard_keyboard import _symbol_key_suffix
                suf = _symbol_key_suffix(c)
                k = f"wifi_key_{suf}"
                if k in menu_buttons and menu_buttons[k].contains(x, y):
                    HUD.wifi_password = pw + c
                    return

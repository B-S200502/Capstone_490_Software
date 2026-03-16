"""
Top HUD: network, FPS, and time pills at the top of the main view.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional  # noqa: F401 Optional used in _draw_pill
import time
import cv2
import numpy as np

from ..config import (
    HUD_MENU_OPACITY,
    MENU_ACTIVE_BLUE,
    MENU_ACTIVE_BLUE_LIGHT,
    UI_PILL_H,
    UI_PILL_W,
    DB_BAR_WIDTH,
    FREQ_BAR_WIDTH,
    BATTERY_CAPACITY_MAH,
    BATTERY_DISCHARGE_MA,
)
from ..state import button_state, HUD
from . import ui_cache
from .menu import _blue_gradient_overlay
from .battery_icon import draw_battery_icon, BATTERY_BODY_W, BATTERY_TIP_W, BATTERY_BODY_H
from .button import menu_buttons, Button

@dataclass
class HudRects:
    net:     Tuple[int,int,int,int]
    fps:     Tuple[int,int,int,int]
    battery: Tuple[int,int,int,int]  # merged battery + wifi pill
    time:    Tuple[int,int,int,int]


def _draw_pill(frame: np.ndarray, x: int, y: int, w: int, h: int,
               bg=(0,0,0), alpha: Optional[float] = None, border=(255,255,255), is_active: bool = False) -> None:
    H, W = frame.shape[:2]
    x0 = max(0, x); y0 = max(0, y)
    x1 = min(W, x + w); y1 = min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return

    if alpha is None:
        alpha = HUD_MENU_OPACITY
    if is_active:
        alpha = HUD_MENU_OPACITY
        border = (255, 255, 255)

    roi = frame[y0:y1, x0:x1]
    pill_h, pill_w = roi.shape[:2]
    if is_active:
        overlay = _blue_gradient_overlay(pill_h, pill_w, MENU_ACTIVE_BLUE, MENU_ACTIVE_BLUE_LIGHT)
    else:
        overlay = np.empty_like(roi)
        overlay[:] = bg
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), border, 1, cv2.LINE_AA)

def _in_rect(x: int, y: int, r: Tuple[int,int,int,int]) -> bool:
    rx, ry, rw, rh = r
    return (rx <= x < rx+rw) and (ry <= y < ry+rh)

# Merged battery+wifi pill: wifi icon (left) + battery icon + %
BAT_WIFI_PILL_GAP = 10  # between wifi icon and battery

from .icons import draw_wifi_icon


def _time_remaining_display(percent: Optional[int], time_remaining_sec: Optional[float] = None) -> str:
    """
    Format battery time remaining. Uses time_remaining_sec if provided (from hardware);
    otherwise estimates from percent using capacity/discharge rate.
    Formula: time_remaining = (remaining_capacity_mAh / current_draw_mA) in hours.
    """
    if time_remaining_sec is not None and time_remaining_sec >= 0:
        sec = int(time_remaining_sec)
        if sec >= 3600:
            h, m = divmod(sec, 3600)
            m //= 60
            return f"{h}h {m}m" if m else f"{h}h"
        if sec >= 60:
            return f"{sec // 60}m"
        return f"{sec}s" if sec > 0 else "0m"
    if percent is None:
        return "—"
    pct = max(0, min(100, percent))
    if pct <= 0:
        return "0m"
    # Estimate: remaining_mAh = capacity * (pct/100), time_h = remaining_mAh / discharge_mA
    remaining_mah = BATTERY_CAPACITY_MAH * (pct / 100.0)
    time_h = remaining_mah / BATTERY_DISCHARGE_MA
    time_sec = time_h * 3600
    sec = int(time_sec)
    if sec >= 3600:
        h, rest = divmod(sec, 3600)
        m = rest // 60
        return f"{h}h {m}m" if m else f"{h}h"
    if sec >= 60:
        return f"{sec // 60}m"
    return f"{sec}s" if sec > 0 else "0m"

def draw_hud(
    frame: np.ndarray,
    *,
    details_level: str,
    open_panel: str,
    fps_ema: float,
    elapsed_s: float,
    frame_count: int,
    source_label: str,
    source_stats,
    fps_mode: str,
    frame_bytes: int,
    offset_y: float = 0.0,
    battery_percent: Optional[int] = None,
    time_remaining_sec: Optional[float] = None,
    wifi_connection_name: Optional[str] = None,
    ip_address: Optional[str] = None,
    device_name: Optional[str] = None,
) -> HudRects:
    """
    Draw compact HUD top-left. offset_y moves the HUD vertically (0=visible, negative=retracted up).
    Returns hit rects for mouse click handling (with offset applied).
    """
    if details_level == "OFF":
        return HudRects(net=(0,0,0,0), fps=(0,0,0,0), battery=(0,0,0,0), time=(0,0,0,0))

    h, w, _ = frame.shape

    # Layout constants; y with visibility offset (retract up when offset_y < 0); same pill size as bottom HUD
    pad = 110
    y = 1 + int(offset_y)
    pill_h = UI_PILL_H
    icon_w = 34
    gap = 8

    # Compute throughput from fps_ema
    bytes_per_s = frame_bytes * fps_ema
    mbps_bits = (bytes_per_s * 8) / 1e6

    time_txt = time.strftime("%I:%M:%S%P")
    fps_txt  = f"{fps_ema:4.1f}"
    net_txt  = f"{mbps_bits:4.1f}"

    # "Icons" (no external assets): simple glyphs inside circles
    # If you later want PNG icons, we can swap these with image blits.
    def icon_circle(cx, cy, label: str):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)  # filled white
        cv2.putText(frame, label, (cx-8, cy+6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)  # black letter

    def draw_clock_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        # hands
        cv2.line(frame, (cx, cy), (cx, cy-6), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx+5, cy+2), (0,0,0), 2, cv2.LINE_AA)

    def draw_fps_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        # gauge arc + needle
        cv2.ellipse(frame, (cx, cy+2), (7,7), 0, 200, 340, (0,0,0), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy+2), (cx+6, cy-2), (0,0,0), 2, cv2.LINE_AA)

    def draw_net_icon(cx, cy):
        cv2.circle(frame, (cx, cy), 12, (255,255,255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, (0,0,0), 1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-7, cy+4), (cx-5, cy+8), (0,0,0), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx-2, cy+2), (cx,   cy+8), (0,0,0), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (cx+3, cy-1), (cx+5, cy+8), (0,0,0), -1, cv2.LINE_AA)

    # Basic measurement (rough, but stable)
    def tw(s: str, scale=0.55, thick=1):
        (ww, hh), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        return ww

    net_w   = UI_PILL_W
    fps_w   = UI_PILL_W
    time_w  = UI_PILL_W
    # Merged battery+wifi pill: wifi icon (left) + gap + battery icon + %
    bat_w   = UI_PILL_W + 12  # room for wifi icon + battery icon + %

    # Center HUD within the camera feed segment (between dB bar and freq bar)
    camera_feed_left = DB_BAR_WIDTH
    camera_feed_width = w - DB_BAR_WIDTH - FREQ_BAR_WIDTH
    total_width = net_w + gap + fps_w + gap + bat_w + gap + time_w
    start_x = camera_feed_left + (camera_feed_width - total_width) // 2

    x_net  = start_x
    x_fps  = x_net + net_w + gap
    x_bat  = x_fps + fps_w + gap
    x_time = x_bat + bat_w + gap

    # Draw pills (highlight when active/open)
    _draw_pill(frame, x_net,  y, net_w,  pill_h, is_active=(open_panel == "net"))
    _draw_pill(frame, x_fps,  y, fps_w,  pill_h, is_active=(open_panel == "fps"))
    _draw_pill(frame, x_bat,  y, bat_w,  pill_h, is_active=(open_panel == "battery"))
    _draw_pill(frame, x_time, y, time_w, pill_h, is_active=(open_panel == "time"))

    cy = y + pill_h // 2
    text_y = y + pill_h // 2 + 6  # baseline for text
    icon_radius = 12
    icon_w = icon_radius * 2
    gap_icon_text = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55

    def draw_pill_icon_text(pill_x: int, pill_w: int, label: str, draw_icon_fn) -> None:
        (label_w, _), _ = cv2.getTextSize(label, font, scale, 1)
        content_w = icon_w + gap_icon_text + label_w
        start_x = pill_x + (pill_w - content_w) // 2
        icon_cx = start_x + icon_radius
        draw_icon_fn(icon_cx, cy)
        cv2.putText(frame, label, (start_x + icon_w + gap_icon_text, text_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # --- NETWORK ---
    draw_pill_icon_text(x_net, net_w, f"{net_txt} Mb/s", draw_net_icon)

    # --- FPS ---
    draw_pill_icon_text(x_fps, fps_w, f"{fps_txt} FPS", draw_fps_icon)

    # --- BATTERY + WIFI (merged) --- wifi icon first (left), then battery icon + %
    bat_pct = battery_percent if battery_percent is not None else 100
    pct_txt = f"{bat_pct}%"
    (pct_w, _), _ = cv2.getTextSize(pct_txt, font, scale, 1)
    bat_icon_w = BATTERY_BODY_W + BATTERY_TIP_W
    bat_icon_h = BATTERY_BODY_H
    # content: wifi_icon | gap | battery_icon | gap | %
    bat_content_w = icon_w + BAT_WIFI_PILL_GAP + bat_icon_w + gap_icon_text + pct_w
    bat_content_start = x_bat + (bat_w - bat_content_w) // 2
    bat_y = y + (pill_h - bat_icon_h) // 2
    if not button_state.gallery_open:
        wifi_cx = bat_content_start + icon_radius
        # HUD style: white background, black icon (inverted from dark-pill style)
        draw_wifi_icon(frame, wifi_cx, cy, color=(0, 0, 0), bg_color=(255, 255, 255), size=12, circular=True, clip_radius=12)
        cv2.circle(frame, (wifi_cx, cy), 12, (0, 0, 0), 1, cv2.LINE_AA)
        draw_battery_icon(frame, x=bat_content_start + icon_w + BAT_WIFI_PILL_GAP, y=bat_y, percent=battery_percent)
        cv2.putText(frame, pct_txt, (bat_content_start + icon_w + BAT_WIFI_PILL_GAP + bat_icon_w + gap_icon_text, text_y), font, scale, (255, 255, 255), 1, cv2.LINE_AA)

    # --- TIME --- (clock + time only)
    draw_pill_icon_text(x_time, time_w, time_txt, draw_clock_icon)

    rects = HudRects(
        net=(x_net,  y, net_w,   pill_h),
        fps=(x_fps,  y, fps_w,   pill_h),
        battery=(x_bat, y, bat_w, pill_h),
        time=(x_time, y, time_w, pill_h),
    )

    # Expanded panel (only when icon clicked); keep width within camera feed (don't cross into freq bar)
    panel_y = y + pill_h + 6
    panel_right = camera_feed_left + camera_feed_width

    def panel(lines, anchor_x):
        line_h = 18
        content_w = max(tw(s, 0.5, 1) for s in lines) + 20
        panel_w = min(max(165, content_w), max(1, panel_right - anchor_x))
        panel_h = 10 + line_h * len(lines)
        _draw_pill(frame, anchor_x, panel_y, panel_w, panel_h)  # same opacity as menu/bottom HUD (HUD_MENU_OPACITY)
        yy = panel_y + 22
        for s in lines:
            cv2.putText(frame, s, (anchor_x + 10, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
            yy += line_h

    # Build extra info
    if open_panel == "time":
        panel([
            f"Uptime: {elapsed_s:0.1f}s",
            f"Frame: {frame_count}",
            f"Source: {source_label}",
        ], rects.time[0])

    elif open_panel == "battery":
        bat_pct = battery_percent if battery_percent is not None else 100
        wifi_name = (wifi_connection_name or "").strip() or "[Not Connected]"
        ip_str = (ip_address or "").strip() or "—"
        dev_str = (device_name or "").strip() or "—"
        time_str = _time_remaining_display(bat_pct, time_remaining_sec)
        # Dropdown: Wi-Fi, IP, Device, then Battery: % (time) on one line as 4th
        panel([
            f"Wi-Fi: {wifi_name}",
            f"IP: {ip_str}",
            f"Device: {dev_str}",
            f"Battery: {bat_pct}% ({time_str})",
        ], rects.battery[0])

    elif open_panel == "fps":
        panel([
            f"Mode: {fps_mode}",
            f"dt: {1000.0/max(1e-6, (1.0/max(1e-6, fps_ema))):0.1f} ms (approx)",
            "Tip: MAX mode removes throttle",
        ], rects.fps[0])

    elif open_panel == "net":
        mbps_bytes = bytes_per_s / 1e6
        lines = [
            f"{mbps_bytes:0.2f} MB/s  ({mbps_bits:0.1f} Mb/s)",
        ]
        if source_label.startswith("SPI"):
            mhz = (getattr(source_stats, "sclk_hz_rep", 0.0) / 1e6) if getattr(source_stats, "sclk_hz_rep", 0) else 0.0
            lines += [
                f"SPI: {mhz:0.0f} MHz",
                f"ok: {getattr(source_stats,'frames_ok',0)}  badParse: {getattr(source_stats,'bad_parse',0)}  badCRC: {getattr(source_stats,'bad_crc',0)}",
            ]
            last_err = getattr(source_stats, "last_err", "")
            if last_err:
                lines.append(f"lastErr: {last_err[:40]}")
        panel(lines, rects.net[0])

    return rects

def handle_hud_click(mx: int, my: int, rects: HudRects, open_panel: str) -> str:
    """
    Toggle which panel is open based on click.
    """
    if _in_rect(mx, my, rects.net):
        return "" if open_panel == "net" else "net"
    if _in_rect(mx, my, rects.fps):
        return "" if open_panel == "fps" else "fps"
    if _in_rect(mx, my, rects.battery):
        return "" if open_panel == "battery" else "battery"
    if _in_rect(mx, my, rects.time):
        return "" if open_panel == "time" else "time"
    return open_panel


def draw_wifi_connections_modal(frame: np.ndarray) -> None:
    """Draw the WiFi connections modal: list of nearby networks, current connection highlighted."""
    if not HUD.wifi_modal_open:
        return

    networks = HUD.wifi_networks if HUD.wifi_networks else ["Home WiFi", "Network2", "Guest"]  # placeholder
    connected = (HUD.connected_ssid or "").strip()

    row_h = 40
    modal_w = 320
    modal_h = 60 + len(networks) * row_h + 50
    fh, fw = frame.shape[:2]
    modal_x = (fw - modal_w) // 2
    modal_y = (fh - modal_h) // 2

    ui_cache.apply_modal_dim(frame, 0.55)

    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (40, 40, 40), -1)
    cv2.rectangle(frame, (modal_x, modal_y), (modal_x + modal_w, modal_y + modal_h), (100, 100, 100), 3, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "Nearby connections", (modal_x + 16, modal_y + 36), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    for i, ssid in enumerate(networks):
        btn_y = modal_y + 52 + i * row_h
        btn_h = row_h - 6
        is_connected = (ssid == connected)
        label = f"{ssid}  (connected)" if is_connected else ssid
        key = f"hud_wifi_net_{i}"
        if key not in menu_buttons:
            menu_buttons[key] = Button(modal_x + 12, btn_y, modal_w - 24, btn_h, label)
        else:
            b = menu_buttons[key]
            b.x, b.y, b.w, b.h = modal_x + 12, btn_y, modal_w - 24, btn_h
            b.text = label
        menu_buttons[key].is_active = is_connected
        menu_buttons[key].draw(frame, transparent=True, active_color=MENU_ACTIVE_BLUE)

    close_y = modal_y + modal_h - 44
    if "hud_wifi_modal_close" not in menu_buttons:
        menu_buttons["hud_wifi_modal_close"] = Button(modal_x + 12, close_y, modal_w - 24, 34, "Close")
    else:
        b = menu_buttons["hud_wifi_modal_close"]
        b.x, b.y, b.w, b.h = modal_x + 12, close_y, modal_w - 24, 34
    menu_buttons["hud_wifi_modal_close"].draw(frame, transparent=True)

    if "hud_wifi_modal_panel" not in menu_buttons:
        menu_buttons["hud_wifi_modal_panel"] = Button(modal_x, modal_y, modal_w, modal_h, "")
    else:
        b = menu_buttons["hud_wifi_modal_panel"]
        b.x, b.y, b.w, b.h = modal_x, modal_y, modal_w, modal_h

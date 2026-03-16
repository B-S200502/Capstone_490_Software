"""
Button widget, icon drawing, and button/menu registries.
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import cv2
import numpy as np


def _vertical_gradient_uint8(h: int, w: int, top_bgr: Tuple[int, int, int], bot_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Vertical gradient (top -> bottom) as (h, w, 3) BGR uint8."""
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        out[:, :, c] = np.linspace(top_bgr[c], bot_bgr[c], h, dtype=np.uint8).reshape(-1, 1)
    return out

from . import ui_cache
from .storage_bar import feathered_composite
from .icons import draw_wifi_icon as _draw_wifi_icon_shared, draw_settings_icon as _draw_settings_icon_shared
from ..state import button_state

try:
    from ..config import (
        BUTTON_HITPAD_PX,
        HUD_MENU_OPACITY,
        ACTION_BTN_GLOW,
        ACTION_BTN_BORDER_THICKNESS,
        ACTION_BTN_NEON_BORDER_BGR,
        ACTION_BTN_NEON_GLOW,
        ACTION_BTN_FILL_ALPHA,
        ACTION_BTN_SHINE_ALPHA,
        GALLERY_ACTION_STYLE,
        CLASSIC_ACTION_TEXT_BGR,
        CLASSIC_ACTION_GLOW,
    )
    BUTTON_ALPHA = HUD_MENU_OPACITY  # menu/HUD use single knob
except Exception:
    BUTTON_HITPAD_PX = ui_cache.BUTTON_HITPAD_PX
    HUD_MENU_OPACITY = 0.92
    BUTTON_ALPHA = HUD_MENU_OPACITY
    ACTION_BTN_GLOW = 0.45
    ACTION_BTN_BORDER_THICKNESS = 1
    ACTION_BTN_NEON_BORDER_BGR = (255, 210, 100)
    ACTION_BTN_NEON_GLOW = 0.2
    ACTION_BTN_FILL_ALPHA = 0.98
    ACTION_BTN_SHINE_ALPHA = 0.11
    GALLERY_ACTION_STYLE = "neon"
    CLASSIC_ACTION_TEXT_BGR = (255, 255, 255)
    CLASSIC_ACTION_GLOW = 0.06

buttons: Dict[str, "Button"] = {}
menu_buttons: Dict[str, "Button"] = {}

FPS_MODE_TO_TARGET = {"30": 30, "60": 60}


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    """Draw rectangle (r unused; kept for API compatibility)."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness, lineType=cv2.LINE_AA)


def _add_glassy_shine(roi: np.ndarray, alpha: float = 0.11) -> None:
    """Add a subtle glassy shine to the ROI in-place: corner-focused highlights (four corners)."""
    if roi.size == 0 or alpha <= 0:
        return
    h, w = roi.shape[:2]
    if w < 4 or h < 4:
        return
    shine = np.zeros_like(roi)
    white = (255, 255, 255)
    ax, ay = max(2, int(w * 0.22)), max(2, int(h * 0.22))
    # Four corner highlights (tight in corners)
    cv2.ellipse(shine, (int(w * 0.12), int(h * 0.12)), (ax, ay), 0, 0, 360, white, -1, cv2.LINE_AA)
    cv2.ellipse(shine, (int(w * 0.88), int(h * 0.12)), (ax, ay), 0, 0, 360, white, -1, cv2.LINE_AA)
    cv2.ellipse(shine, (int(w * 0.92), int(h * 0.92)), (ax, ay), 0, 0, 360, white, -1, cv2.LINE_AA)
    cv2.ellipse(shine, (int(w * 0.12), int(h * 0.92)), (ax, ay), 0, 0, 360, white, -1, cv2.LINE_AA)
    sigma = max(1.0, min(w, h) * 0.12)
    k = max(3, int(sigma * 2.5) | 1)
    shine = cv2.GaussianBlur(shine, (k, k), sigma)
    np.clip(shine, 0, 255, out=shine)
    cv2.addWeighted(shine, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


def _draw_camera_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a camera lens icon."""
    cv2.rectangle(frame, (cx - size, cy - size//2), (cx + size, cy + size//2), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size//2, (200, 200, 200), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), size//2, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (cx - size, cy - size//2 - 3), (cx - size + 4, cy - size//2), (255, 255, 255), -1, cv2.LINE_AA)


def _draw_rec_icon(frame: np.ndarray, cx: int, cy: int, size: int = 8, is_active: bool = False):
    """Draw a red recording dot icon."""
    color = (0, 0, 255) if is_active else (255, 255, 255)
    cv2.circle(frame, (cx, cy), size, color, -1, cv2.LINE_AA)
    border_color = (255, 255, 255) if is_active else (0, 0, 0)
    cv2.circle(frame, (cx, cy), size, border_color, 1, cv2.LINE_AA)


def _draw_pause_icon(frame: np.ndarray, cx: int, cy: int, size: int = 10, color: Tuple[int, int, int] = (255, 255, 255)):
    """Draw pause bars icon (||)."""
    bar_w = size // 3
    bar_h = size
    gap = size // 3
    cv2.rectangle(frame,
                 (cx - gap - bar_w, cy - bar_h//2),
                 (cx - gap, cy + bar_h//2),
                 color, -1, cv2.LINE_AA)
    cv2.rectangle(frame,
                 (cx + gap, cy - bar_h//2),
                 (cx + gap + bar_w, cy + bar_h//2),
                 color, -1, cv2.LINE_AA)


def _draw_back_arrow_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a back chevron (<<) so it's clearly not a play triangle."""
    # Left-pointing chevron: two angled segments meeting at the left
    tip_x = cx - size // 2
    top_pt = (cx + size // 2, cy - size)
    bot_pt = (cx + size // 2, cy + size)
    thickness = max(2, size // 5)
    cv2.line(frame, (tip_x, cy), top_pt, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.line(frame, (tip_x, cy), bot_pt, (255, 255, 255), thickness, cv2.LINE_AA)


def _draw_gallery_icon(frame: np.ndarray, cx: int, cy: int, size: int = 10):
    """Draw a 2x2 grid of squares (gallery/thumbnails)."""
    half = size // 2
    gap = max(1, size // 6)
    for i in range(2):
        for j in range(2):
            x1 = cx - half - gap // 2 + i * (half + gap)
            y1 = cy - half - gap // 2 + j * (half + gap)
            x2 = x1 + half
            y2 = y1 + half
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1, cv2.LINE_AA)


def _draw_trash_icon(frame: np.ndarray, cx: int, cy: int, size: int = 12):
    """Draw a clean trash can icon: rectangular body with lid overhang, bail handle, and interior lines."""
    col = (255, 255, 255)
    dim = (170, 170, 170)

    body_w = int(size * 1.1)
    body_h = int(size * 1.3)
    lid_h = max(2, int(size * 0.18))
    bail_h = max(2, int(size * 0.22))

    # Vertical centre of entire icon (body + lid + bail)
    total_h = bail_h + 2 + lid_h + body_h
    icon_top = cy - total_h // 2

    bail_cy = icon_top + bail_h // 2
    lid_y = icon_top + bail_h + 2
    body_y = lid_y + lid_h

    # Bail (handle arc) above lid
    bail_rx = max(2, body_w // 4)
    cv2.ellipse(frame, (cx, bail_cy + bail_h // 2), (bail_rx, bail_h), 0, 180, 360, col, 2, cv2.LINE_AA)

    # Lid: slightly wider than body with overhang
    lid_w = body_w + 4
    cv2.rectangle(frame,
                  (cx - lid_w // 2, lid_y),
                  (cx + lid_w // 2, lid_y + lid_h),
                  col, -1, cv2.LINE_AA)

    # Body: clean rectangle
    bx0 = cx - body_w // 2
    bx1 = cx + body_w // 2
    by0 = body_y
    by1 = body_y + body_h
    cv2.rectangle(frame, (bx0, by0), (bx1, by1), col, 2, cv2.LINE_AA)

    # Interior vertical lines (3 evenly-spaced lines suggesting slots/ribs)
    line_top = by0 + max(2, body_h // 6)
    line_bot = by1 - max(2, body_h // 6)
    for offset in (-body_w // 4, 0, body_w // 4):
        lx = cx + offset
        cv2.line(frame, (lx, line_top), (lx, line_bot), dim, 1, cv2.LINE_AA)


class Button:
    def __init__(self, x, y, w, h, text):
        self.x, self.y, self.w, self.h = int(x), int(y), int(w), int(h)
        self.text = text
        self.is_hovered = False
        self.is_active = False

    def contains(self, mx, my) -> bool:
        pad = BUTTON_HITPAD_PX
        return (
            (self.x - pad) <= mx <= (self.x + self.w + pad) and
            (self.y - pad) <= my <= (self.y + self.h + pad)
        )

    def draw(
        self,
        frame: np.ndarray,
        transparent: bool = False,
        active_color: Optional[tuple] = None,
        active_border_color: Optional[tuple] = None,
        icon_type: Optional[str] = None,
        gradient_colors: Optional[Tuple[tuple, tuple]] = None,
        neon_glow: bool = False,
        fill_alpha: Optional[float] = None,
        neon_border_color: Optional[Tuple[int, int, int]] = None,
        inactive_bg: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        base = (60, 60, 60)
        hover = (85, 85, 85)
        active = active_color if active_color is not None else (40, 200, 60)

        color = active if self.is_active else (hover if self.is_hovered else base)

        x, y, w, h = self.x, self.y, self.w, self.h
        cx, cy = x + w // 2, y + h // 2
        is_back = icon_type == "back"

        if is_back:
            # Circular back button: gradient fill and border as circle
            r = min(w, h) // 2 - 1
            x0 = max(0, x)
            y0 = max(0, y)
            x1 = min(frame.shape[1], x + w)
            y1 = min(frame.shape[0], y + h)
            if x1 > x0 and y1 > y0:
                roi = frame[y0:y1, x0:x1]
                roi_h, roi_w = roi.shape[:2]
                mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                lcx, lcy = cx - x0, cy - y0
                cv2.circle(mask, (lcx, lcy), r, 255, -1)
                # Gradient: lighter top, darker bottom (gray tones)
                back_top = (95, 95, 95)
                back_bot = (45, 45, 45)
                grad = _vertical_gradient_uint8(roi_h, roi_w, back_top, back_bot)
                alpha = BUTTON_ALPHA
                for c in range(3):
                    blended = (roi[:, :, c].astype(np.float32) * (1 - alpha) + grad[:, :, c].astype(np.float32) * alpha).astype(np.uint8)
                    roi[:, :, c] = np.where(mask > 0, blended, roi[:, :, c])
            border_color = (255, 255, 255)
            if self.is_active and active_border_color is not None:
                border_color = active_border_color
            cv2.circle(frame, (cx, cy), r, border_color, 2, cv2.LINE_AA)
        else:
            if transparent:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame.shape[1], x + w)
                y1 = min(frame.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    roi = frame[y0:y1, x0:x1]
                    roi_h, roi_w = roi.shape[:2]
                    if gradient_colors is not None:
                        top_bgr, bot_bgr = gradient_colors
                        if neon_glow:
                            pad = 14
                            gx0 = max(0, x - pad)
                            gy0 = max(0, y - pad)
                            gx1 = min(frame.shape[1], x + w + pad)
                            gy1 = min(frame.shape[0], y + h + pad)
                            if gx1 > gx0 and gy1 > gy0:
                                glow_patch = frame[gy0:gy1, gx0:gx1].copy()
                                gh, gw = glow_patch.shape[:2]
                                ly0, lx0 = y - gy0, x - gx0
                                ly1, lx1 = min(gh, ly0 + h), min(gw, lx0 + w)
                                ly0, lx0 = max(0, ly0), max(0, lx0)
                                if ly1 > ly0 and lx1 > lx0:
                                    glow_patch[ly0:ly1, lx0:lx1] = _vertical_gradient_uint8(
                                        ly1 - ly0, lx1 - lx0, top_bgr, bot_bgr
                                    )
                                glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 10.0)
                                feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, ACTION_BTN_GLOW, feather_px=18)
                        overlay = _vertical_gradient_uint8(roi_h, roi_w, top_bgr, bot_bgr)
                    else:
                        overlay = np.empty_like(roi)
                        fill_color = (inactive_bg if (inactive_bg is not None and not self.is_active) else color)
                        overlay[:] = fill_color
                    alpha = fill_alpha if fill_alpha is not None else BUTTON_ALPHA
                    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
                    if neon_border_color is not None and GALLERY_ACTION_STYLE != "classic":
                        _add_glassy_shine(roi, alpha=ACTION_BTN_SHINE_ALPHA)
            else:
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame.shape[1], x + w)
                y1 = min(frame.shape[0], y + h)
                if x1 > x0 and y1 > y0:
                    roi = frame[y0:y1, x0:x1]
                    roi[:] = ui_cache.get_grad(roi.shape[1], roi.shape[0], color)
            if neon_border_color is not None:
                if ACTION_BTN_NEON_GLOW > 0 and GALLERY_ACTION_STYLE != "classic":
                    # Neon edge: glow layer then crisp border
                    pad = 16
                    gx0 = max(0, x - pad)
                    gy0 = max(0, y - pad)
                    gx1 = min(frame.shape[1], x + w + pad)
                    gy1 = min(frame.shape[0], y + h + pad)
                    if gx1 > gx0 and gy1 > gy0:
                        glow_patch = np.zeros((gy1 - gy0, gx1 - gx0, 3), dtype=np.uint8)
                        lx, ly = x - gx0, y - gy0
                        cv2.rectangle(glow_patch, (lx, ly), (lx + w, ly + h), neon_border_color, 8, cv2.LINE_AA)
                        glow_patch = cv2.GaussianBlur(glow_patch, (0, 0), 6.0)
                        feathered_composite(frame, gy0, gy1, gx0, gx1, glow_patch, ACTION_BTN_NEON_GLOW, feather_px=14)
                border_color = neon_border_color
                _rounded_rect(frame, x, y, w, h, 10, color=border_color, thickness=max(1, ACTION_BTN_BORDER_THICKNESS))
            else:
                if self.is_active:
                    if active_border_color is not None:
                        border_color = active_border_color
                    elif active_color is not None:
                        border_color = tuple(min(255, int(c * 1.3)) for c in active_color)
                    else:
                        border_color = (80, 255, 100)
                else:
                    border_color = (255, 255, 255)
                _rounded_rect(frame, x, y, w, h, 10, color=border_color, thickness=ACTION_BTN_BORDER_THICKNESS)

        if icon_type:
            cx = x + w // 2
            cy = y + h // 2

            if icon_type == "camera":
                _draw_camera_icon(frame, cx, cy, size=10)
            elif icon_type == "rec":
                _draw_rec_icon(frame, cx, cy, size=7, is_active=self.is_active)
            elif icon_type == "pause":
                _draw_pause_icon(frame, cx, cy, size=9)
            elif icon_type == "back":
                _draw_back_arrow_icon(frame, cx, cy, size=12)
            elif icon_type == "trash":
                _draw_trash_icon(frame, cx, cy, size=14)
            elif icon_type == "wifi":
                _draw_wifi_icon_shared(frame, cx, cy, color=(255, 255, 255), bg_color=None, size=14)
            elif icon_type == "settings":
                _draw_settings_icon_shared(frame, cx, cy, color=(255, 255, 255), size=12)
        else:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.42 if "LOOP" in self.text else 0.52
            thick = 1
            is_gallery_action = gradient_colors is not None or neon_border_color is not None
            text_color = (CLASSIC_ACTION_TEXT_BGR if (GALLERY_ACTION_STYLE == "classic" and is_gallery_action) else (255, 255, 255))
            # Word-wrap: "SELECT ALL" -> "SELECT"/"ALL", "DESELECT ALL" -> "DESELECT"/"ALL", "DELETE (N)" -> "DELETE"/"(N)"
            if self.text in ("SELECT ALL", "DESELECT ALL"):
                line1 = self.text.split()[0]   # "SELECT" or "DESELECT"
                line2 = "ALL"
            elif self.text.startswith("DELETE (") and self.text.endswith(")"):
                line1 = "DELETE"
                line2 = self.text[7:]  # "(1)" or "(12)" etc.
            else:
                line1, line2 = None, None
            if line1 is not None and line2 is not None:
                (tw1, th1), _ = cv2.getTextSize(line1, font, scale, thick)
                (tw2, th2), _ = cv2.getTextSize(line2, font, scale, thick)
                gap = 2
                # DELETE (N): push count line a tad more south
                line2_south = 4 if self.text.startswith("DELETE (") else 0
                total_h = th1 + gap + th2 + line2_south
                ty1 = y + (h - total_h) // 2 + th1
                ty2 = ty1 + gap + th2 + line2_south
                tx1 = x + (w - tw1) // 2
                tx2 = x + (w - tw2) // 2
                text_glow = (CLASSIC_ACTION_GLOW if (GALLERY_ACTION_STYLE == "classic" and is_gallery_action) else None) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_GLOW if (neon_glow and gradient_colors is not None) else None)
                if text_glow is not None and text_glow > 0:
                    pad_t = 14
                    fh, fw = frame.shape[:2]
                    ty0 = max(0, y + (h - total_h) // 2 - pad_t)
                    ty1_end = min(fh, ty2 + th2 + pad_t)
                    tx0 = max(0, min(tx1, tx2) - pad_t)
                    tx1_end = min(fw, max(tx1 + tw1, tx2 + tw2) + pad_t)
                    if tx1_end > tx0 and ty1_end > ty0:
                        tpatch = frame[ty0:ty1_end, tx0:tx1_end].copy()
                        ltx1, lty1 = tx1 - tx0, ty1 - ty0
                        ltx2, lty2 = tx2 - tx0, ty2 - ty0
                        cv2.putText(tpatch, line1, (ltx1, lty1), font, scale, text_color, 2, cv2.LINE_AA)
                        cv2.putText(tpatch, line1, (ltx1, lty1), font, scale, text_color, 1, cv2.LINE_AA)
                        cv2.putText(tpatch, line2, (ltx2, lty2), font, scale, text_color, 2, cv2.LINE_AA)
                        cv2.putText(tpatch, line2, (ltx2, lty2), font, scale, text_color, 1, cv2.LINE_AA)
                        tpatch = cv2.GaussianBlur(tpatch, (0, 0), 2.5)
                        feathered_composite(frame, ty0, ty1_end, tx0, tx1_end, tpatch, text_glow, feather_px=12)
                cv2.putText(frame, line1, (tx1, ty1), font, scale, text_color, thick, cv2.LINE_AA)
                cv2.putText(frame, line2, (tx2, ty2), font, scale, text_color, thick, cv2.LINE_AA)
            else:
                tw, th = cv2.getTextSize(self.text, font, scale, thick)[0]
                tx = x + (w - tw) // 2
                ty = y + (h + th) // 2
                text_glow_single = (CLASSIC_ACTION_GLOW if (GALLERY_ACTION_STYLE == "classic" and is_gallery_action) else None) if GALLERY_ACTION_STYLE == "classic" else (ACTION_BTN_GLOW if (neon_glow and gradient_colors is not None) else None)
                if text_glow_single is not None and text_glow_single > 0:
                    pad_t = 14
                    fh, fw = frame.shape[:2]
                    tx0 = max(0, tx - pad_t)
                    ty0 = max(0, ty - th - pad_t)
                    tx1 = min(fw, tx + tw + pad_t)
                    ty1 = min(fh, ty + pad_t)
                    if tx1 > tx0 and ty1 > ty0:
                        tpatch = frame[ty0:ty1, tx0:tx1].copy()
                        ltx, lty = tx - tx0, ty - ty0
                        cv2.putText(tpatch, self.text, (ltx, lty), font, scale, text_color, 2, cv2.LINE_AA)
                        cv2.putText(tpatch, self.text, (ltx, lty), font, scale, text_color, 1, cv2.LINE_AA)
                        tpatch = cv2.GaussianBlur(tpatch, (0, 0), 2.5)
                        feathered_composite(frame, ty0, ty1, tx0, tx1, tpatch, text_glow_single, feather_px=12)
                cv2.putText(frame, self.text, (tx, ty),
                            font, scale, text_color,
                            thick, cv2.LINE_AA)


def init_buttons(left_width: int, camera_available: bool) -> None:
    buttons.clear()


def init_menu_buttons(left_width: int, frame_height: Optional[int] = None) -> None:
    menu_buttons.clear()

    actual_height = frame_height if frame_height is not None else ui_cache.HEIGHT

    menu_w = 180
    menu_h = 50
    menu_margin_x = 15
    menu_margin_bottom = 5
    menu_x = left_width - menu_w - menu_margin_x
    menu_y = actual_height - menu_h - menu_margin_bottom

    menu_buttons["menu"] = Button(menu_x, menu_y, menu_w, menu_h, "MENU")

    item_h = 40
    gap = 8

    total_items = 5  # wifi|settings, fps, gain, source, spectrum_analyzer
    dropdown_h = total_items * (item_h + gap) + gap
    dropdown_y = menu_y - dropdown_h - gap

    # Top row: wifi | settings (2 segments)
    wifi_y = dropdown_y
    seg_gap_top = 6
    seg_w_top = (menu_w - 1 * seg_gap_top) // 2
    menu_buttons["wifi"] = Button(menu_x + 0 * (seg_w_top + seg_gap_top), wifi_y, seg_w_top, item_h, "")
    menu_buttons["main_menu_settings"] = Button(menu_x + 1 * (seg_w_top + seg_gap_top), wifi_y, seg_w_top, item_h, "")

    y0 = wifi_y + (item_h + gap)
    seg_gap = 6
    seg_w = (menu_w - 1 * seg_gap) // 2  # 30FPS | MAX
    menu_buttons["fps30"]  = Button(menu_x + 0 * (seg_w + seg_gap), y0, seg_w, item_h, "30FPS")
    menu_buttons["fpsmax"] = Button(menu_x + 1 * (seg_w + seg_gap), y0, seg_w, item_h, "MAX")

    gain_y = y0 + (item_h + gap)
    menu_buttons["gain"] = Button(menu_x, gain_y, menu_w, item_h, f"GAIN: {button_state.gain_mode}")

    source_y = gain_y + (item_h + gap)
    menu_buttons["source"] = Button(menu_x, source_y, menu_w, item_h, f"SRC: {button_state.source_mode}")

    spectrum_y = source_y + (item_h + gap)
    menu_buttons["spectrum_analyzer"] = Button(menu_x, spectrum_y, menu_w, item_h, f"SPECTRUM: {button_state.spectrum_analyzer_mode}")

    # SHOT, Gallery, REC live in bottom HUD (bottom_hud creates/positions them each frame)
    menu_buttons["shot"] = Button(0, 0, 0, 0, "SHOT")
    menu_buttons["gallery"] = Button(0, 0, 0, 0, "GALLERY")
    menu_buttons["rec"] = Button(0, 0, 165, 34, "REC")


def update_button_states(mx: int, my: int) -> None:
    for b in buttons.values():
        b.is_hovered = b.contains(mx, my)

    if "menu" in menu_buttons:
        menu_buttons["menu"].is_hovered = menu_buttons["menu"].contains(mx, my)

    # Bottom HUD pills: always use screen coords; independent of menu state
    for k in ("shot", "rec", "rec_resume", "rec_stop", "gallery"):
        if k in menu_buttons:
            b = menu_buttons[k]
            menu_buttons[k].is_hovered = b.w > 0 and b.contains(mx, my)

    # Dropdown items: only when menu is open
    if button_state.menu_open:
        for k in ("wifi", "main_menu_settings", "fps30", "fpsmax", "gain", "source", "spectrum_analyzer"):
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if button_state.gallery_open:
        gallery_keys = ("gallery_back", "gallery_select_mode", "gallery_select_all", "gallery_delete_selected",
                       "gallery_share_selected", "gallery_delete", "gallery_prev", "gallery_next", "gallery_play", "gallery_progress")
        for k in gallery_keys:
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if getattr(button_state, "share_confirm_modal_open", False):
        for k in ("share_confirm_send", "share_confirm_cancel"):
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if getattr(button_state, "share_modal_open", False):
        for k in ("share_modal_ok",):
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if button_state.gallery_delete_modal_open:
        for k in ("modal_yes", "modal_no"):
            if k in menu_buttons:
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)

    if button_state.email_settings_modal_open:
        for k in list(menu_buttons.keys()):
            if k.startswith("email_"):
                menu_buttons[k].is_hovered = menu_buttons[k].contains(mx, my)


def draw_buttons(frame: np.ndarray) -> None:
    for b in buttons.values():
        b.draw(frame)

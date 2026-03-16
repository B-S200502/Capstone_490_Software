# state.py
"""
Runtime state holders for the Acoustic Imager software.

This keeps "mutable globals" out of config.py and mirrors the defaults
from the original monolithic script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Any, List

from .config import (
    USE_CAMERA,
    F_MIN_HZ_DEFAULT,
    F_MAX_HZ_DEFAULT,
    SOURCE_DEFAULT,
    RADAR_UI_DEFAULT,
    POSITION_SERVICES_DEFAULT,
    RADAR_MAP_TILE_STYLE_DEFAULT,
    DIRECTIONAL_HISTORY_RECORD_DEFAULT,
    RADAR_DEBUG_OVERLAY_DEFAULT,
)

# ===============================================================
# HUD state
# ===============================================================
@dataclass
class RadarDetection:
    t: float
    rel_angle_deg: float
    world_bearing_deg: float
    db_value: float


@dataclass
class UnifiedPosition:
    lat: Optional[float] = None
    lon: Optional[float] = None
    accuracy_m: Optional[float] = None
    source: str = "none"   # "none" | "wifi" | "gps"
    timestamp_s: float = 0.0


@dataclass
class HudState:
    details_level: str = "MIN"   # "OFF" | "MIN" | "MAX"
    open_panel: str = ""         # "" | "time" | "fps" | "net" | "battery"
    wifi_modal_open: bool = False
    settings_modal_open: bool = False
    wifi_modal_screen: str = "list"   # "list" | "password"
    wifi_networks: List[dict] = field(default_factory=list)  # [{"ssid", "signal", "security"}, ...]
    wifi_connect_ssid: str = ""
    wifi_connect_bssid: str = ""   # optional BSSID of selected AP for more reliable connect
    wifi_password: str = ""
    wifi_keyboard_mode: str = "alpha"   # "alpha" | "symbol"
    wifi_shift_next: bool = False
    wifi_connect_status: str = ""   # "" | "connecting" | "ok" | "error"
    wifi_connect_message: str = ""
    connected_ssid: str = ""      # currently connected SSID or ""
    wifi_current_expanded: bool = False  # when True, show expanded network info
    wifi_scanning: bool = False   # True while scan is in progress (background thread)
    wifi_password_visible: bool = False  # show password characters (eye toggle)
    wifi_list_scroll_offset: int = 0   # vertical scroll (px) for "Other Networks" list
    wifi_list_content_dragging: bool = False
    wifi_list_content_drag_start_y: int = 0
    wifi_list_content_drag_start_scroll: int = 0
    wifi_list_touch_started: bool = False   # tap in list: only start drag after move past threshold
    wifi_list_touch_start_x: int = 0
    wifi_list_touch_start_y: int = 0
    wifi_list_scroll_dragging: bool = False   # scrollbar thumb drag
    wifi_list_scrollbar_drag_start_y: int = 0
    wifi_list_scrollbar_drag_start_scroll: int = 0
    settings_modal_scroll_offset: int = 0   # vertical scroll (px) for settings modal content
    settings_modal_scroll_dragging: bool = False  # True while dragging scrollbar
    settings_modal_drag_start_y: int = 0   # y at drag start
    settings_modal_drag_start_scroll: int = 0   # scroll offset at drag start
    # Touch/drag scroll (like gallery)
    settings_modal_content_dragging: bool = False
    settings_modal_content_drag_start_y: int = 0
    settings_modal_content_drag_start_scroll: int = 0
    settings_modal_content_drag_moved: bool = False
    settings_modal_scroll_velocity: float = 0.0   # px/s for inertia
    settings_modal_last_drag_t: float = 0.0
    settings_modal_last_drag_y: int = 0
    settings_modal_inertia_active: bool = False
    # Compass (BN-880 magnetometer)
    compass_heading_deg: float = 0.0   # 0 = North, clockwise positive
    compass_heading_valid: bool = False  # True when reader has received at least one valid heading
    mag_x_raw: int = 0
    mag_y_raw: int = 0
    mag_z_raw: int = 0
    mag_heading_dbg: float = 0.0
    mag_heading_cal_dbg: float = 0.0
    mag_cal_active: bool = False
    mag_pair_dbg: str = "XY"
    mag_span_x: int = 0
    mag_span_y: int = 0
    mag_span_z: int = 0
    mag_x_min: Optional[int] = None
    mag_x_max: Optional[int] = None
    mag_y_min: Optional[int] = None
    mag_y_max: Optional[int] = None
    mag_z_min: Optional[int] = None
    mag_z_max: Optional[int] = None
    # GPS (BN-880 UART)
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    gps_fix_valid: bool = False
    gps_sat_count: int = 0
    gps_last_update_s: float = 0.0
    gps_course_deg: Optional[float] = None
    gps_accuracy_m: Optional[float] = None
    # Wi-Fi geolocation estimate
    wifi_lat: Optional[float] = None
    wifi_lon: Optional[float] = None
    wifi_accuracy_m: Optional[float] = None
    wifi_last_update_s: float = 0.0
    # Unified position used by radar/map rendering
    position: UnifiedPosition = field(default_factory=UnifiedPosition)
    directional_log_last_write_s: float = 0.0
    directional_log_file: str = ""
    directional_log_date: str = ""
    directional_log_error: str = ""
    # Acoustic radar history
    radar_detections: List[RadarDetection] = field(default_factory=list)

HUD = HudState()


# ===============================================================
# UI / App state
# ===============================================================
@dataclass
class ButtonState:
    is_recording: bool = False
    is_paused: bool = False
    camera_enabled: bool = USE_CAMERA
    source_mode: str = SOURCE_DEFAULT  # "SIM" | "SIM_2" | "LOOP" | "HW" | "REF"

    # MENU states
    menu_open: bool = False
    fps_mode: str = "MAX"       # "30" | "60" | "MAX"
    email_settings_modal_open: bool = False
    email_modal_screen: str = "provider"   # "provider" | "form"
    email_modal_provider: str = ""         # "gmail" | "outlook" | "yahoo" | "other"
    email_form_email: str = ""
    email_form_password: str = ""
    email_form_default_to: str = ""
    email_form_smtp_host: str = ""
    email_form_smtp_port: str = "587"
    email_form_use_tls: bool = True
    email_focused_field: str = ""          # "email" | "password" | "default_to" | "smtp_host" | "smtp_port"
    email_cursor_index: int = 0            # insertion point in the focused field (0 to len(text))
    email_keyboard_mode: str = "alpha"     # "alpha" | "symbol" for form keyboard
    email_shift_next: bool = False        # next letter key inserts uppercase (for password etc.)
    email_password_visible: bool = False  # show password characters (eye toggle)
    email_test_status: str = ""          # "" | "sending" | "ok" | "error"
    email_test_message: str = ""         # short message for UI (e.g. "Sent!" or error)
    firmware_flash_modal_open: bool = False
    firmware_flash_version: str = "v1.0.0"   # placeholder; set when flashing starts
    firmware_flash_status: str = ""   # "" | "flashing" | "success" | "error"
    calibration_suite_modal_open: bool = False
    gain_mode: str = "HIGH"    # LOW or HIGH; drives GAIN_CONTROL (and --gain when launching calibration standalone)
    debug_enabled: bool = True
    radar_ui_enabled: bool = RADAR_UI_DEFAULT
    position_services_enabled: bool = POSITION_SERVICES_DEFAULT
    record_compass_history: bool = DIRECTIONAL_HISTORY_RECORD_DEFAULT
    show_radar_debug: bool = RADAR_DEBUG_OVERLAY_DEFAULT
    map_tile_style: str = RADAR_MAP_TILE_STYLE_DEFAULT  # "dark" | "light"
    colormap_mode: str = "MAGMA"  # "MAGMA" | "JET" | "TURBO" | "INFERNO"
    spectrum_analyzer_mode: str = "dB"  # "dB" | "NORM" | "dBA"
    crosshairs_enabled: bool = True     # heatmap crosshairs with freq/dB tooltip (menu: ON/OFF)
    crosshair_visible: bool = False     # True after click on heatmap; click again near crosshair to dismiss
    crosshair_x: float = 0.0            # position (updated each frame to local max when visible)
    crosshair_y: float = 0.0
    # 3 s trend / 12 s acceleration (smart tracking)
    crosshair_level_history: List[Tuple[float, float]] = field(default_factory=list)  # (t, db), last 60 s
    crosshair_trend_history: List[float] = field(default_factory=list)   # trend dB at each 3 s boundary, keep 4
    crosshair_prev_baseline_db: Optional[float] = None
    crosshair_next_boundary_time: float = 0.0

    # Gallery state
    gallery_open: bool = False
    gallery_pill_pressed: bool = False  # True while finger is down on Gallery pill (trigger on release)
    gallery_scroll_offset: int = 0
    gallery_selected_item: Optional[int] = None
    gallery_viewer_mode: str = "grid"  # "grid" | "image" | "video"
    gallery_video_playing: bool = False
    gallery_video_frame_idx: int = 0
    gallery_dragging = False
    gallery_drag_start_y: int = 0
    gallery_drag_start_scroll: int = 0
    gallery_drag_moved = False
    gallery_drag_start_x = 0
    gallery_select_mode: bool = False  # Whether we're in select mode for multi-select
    gallery_selected_items: set[int] = field(default_factory=set)  # Multi-select for deletion
    gallery_delete_confirm: bool = False  # Confirmation state for delete
    gallery_delete_modal_open: bool = False  # Whether delete confirmation modal is open
    gallery_scroll_velocity: float = 0.0   # px/s
    gallery_last_drag_t: float = 0.0
    gallery_last_drag_y: int = 0
    gallery_inertia_active: bool = False
    gallery_last_inertia_t: float = 0.0

    # Viewer (image/video) horizontal swipe with inertia
    gallery_viewer_swipe_dragging: bool = False
    gallery_viewer_swipe_start_x: int = 0
    gallery_viewer_swipe_offset: float = 0.0   # px, positive = dragged left (toward next)
    gallery_viewer_swipe_velocity: float = 0.0  # px/s
    gallery_viewer_swipe_last_t: float = 0.0
    gallery_viewer_swipe_last_x: int = 0
    gallery_viewer_swipe_drag_moved: bool = False
    gallery_viewer_swipe_inertia_active: bool = False
    gallery_viewer_swipe_last_inertia_t: float = 0.0

    # Video progress bar scrub (set by draw, read by handler)
    gallery_progress_dragging: bool = False

    # Viewer dock: green click feedback (button key + time)
    viewer_dock_feedback_button: str = ""
    viewer_dock_feedback_time: float = 0.0

    gallery_delete_modal_open: bool = False
    gallery_delete_modal_kind: str = "single"
    gallery_delete_modal_title: str = "Delete this item?"
    gallery_delete_modal_subtitle: str = "This action cannot be undone."
    gallery_storage_dirty: bool = False  # set when file(s) deleted so storage bar syncs and updates

    # Share (email) result/warning modal: message + OK
    share_modal_open: bool = False
    share_modal_sending: bool = False  # True while send is in progress (threaded)
    share_modal_title: str = "Share"
    share_modal_message: str = ""  # multi-line shown in modal
    share_modal_progress: float = 0.0  # 0.0–1.0 while sending

    # Share confirm modal (before send): details + Send / Cancel
    share_confirm_modal_open: bool = False
    share_confirm_to_email: str = ""
    share_confirm_n_images: int = 0
    share_confirm_n_videos: int = 0
    share_confirm_size_str: str = ""
    share_confirm_file_count: int = 0
    share_confirm_paths: List[str] = field(default_factory=list)  # paths to send when user confirms

    # Gallery dock: filter (type), sort, search
    gallery_filter_type: str = "all"  # "all" | "image" | "video"
    gallery_sort_by: str = "date"  # "date" | "name" | "size"
    gallery_search_query: str = ""
    gallery_filter_modal_open: bool = False
    gallery_sort_modal_open: bool = False
    gallery_search_keyboard_open: bool = False

    # Select-mode tool panels (Tags / Priority / Rename)
    gallery_priority_modal_open: bool = False
    gallery_tags_modal_open: bool = False
    gallery_rename_modal_open: bool = False
    # When > 0, show "Select one or more items first" until time.time() > this (set on Tags/Priority/Rename click with no selection)
    gallery_select_first_hint_until: float = 0.0

    # Modal slide animation: progress 0=hidden, 1=visible; track which modal so we always animate on switch
    gallery_modal_anim_progress: float = 0.0
    gallery_modal_anim_target: float = 0.0
    gallery_modal_anim_start_t: float = 0.0
    gallery_modal_anim_progress_at_flip: float = 0.0
    gallery_modal_anim_active_key: str = ""  # "" when none; forces re-animate when modal changes
    gallery_rename_query: str = ""
    gallery_file_priorities: dict = field(default_factory=dict)  # filename → "high"|"medium"|"low"
    gallery_file_tags: dict = field(default_factory=dict)        # filename → list[str]

    # Tag edit modal (grid select-mode Tags button)
    gallery_tag_modal_open: bool = False
    gallery_tag_active_field: str = ""        # "asset_name" | "asset_type" | "leak_type"
    gallery_tag_keyboard_open: bool = False
    gallery_tag_keyboard_query: str = ""
    gallery_tag_cursor_index: int = 0         # cursor position in active field for blinking caret
    gallery_keyboard_mode: str = "alpha"      # "alpha" | "symbol" for tag/rename/search/archive_rename
    gallery_keyboard_shift_next: bool = False
    gallery_tag_field_values: dict = field(default_factory=dict)  # live values during edit session
    gallery_tag_data: dict = field(default_factory=dict)          # filename → {"asset_type": str, "leak_type": str}
    gallery_tag_info_open: bool = False                           # viewer: read-only info panel

    # Screenshot feedback
    screenshot_flash_time: Optional[float] = None

    # Archive panel (folders for organizing media)
    gallery_archive_folders: list = field(default_factory=list)  # [{"id", "name", "files"}, ...]
    gallery_archive_move_modal_open: bool = False  # Move to folder modal
    gallery_archive_move_hint_until: float = 0.0  # Show "Add folders in Archive panel first"
    gallery_archive_folder_view_id: Optional[str] = None  # folder id when viewing contents (full page)
    gallery_archive_folder_action_id: Optional[str] = None  # folder id when in select mode (rename/delete modal)
    gallery_archive_rename_folder_id: Optional[str] = None  # folder id when renaming
    gallery_archive_rename_query: str = ""  # current rename text for folder
    gallery_archive_delete_confirm_folder_id: Optional[str] = None  # folder id for delete confirmation

    def __post_init__(self):
        if self.gallery_selected_items is None:
            self.gallery_selected_items = set()
        if self.gallery_file_priorities is None:
            self.gallery_file_priorities = {}
        if self.gallery_file_tags is None:
            self.gallery_file_tags = {}
        if self.gallery_tag_field_values is None:
            self.gallery_tag_field_values = {}
        if self.gallery_tag_data is None:
            self.gallery_tag_data = {}
        if self.gallery_archive_folders is None:
            self.gallery_archive_folders = []


# A single shared instance (matches original `button_state = ButtonState()`)
button_state = ButtonState()


# ===============================================================
# Bandpass drag state
# ===============================================================
DRAG_ACTIVE: bool = False
DRAG_TARGET: Optional[str] = None  # "min", "max", or "box"
DRAG_START_Y: int = 0  # Initial y position when box drag starts
DRAG_START_F_MIN: float = 0.0  # Initial F_MIN_HZ when box drag starts
DRAG_START_F_MAX: float = 0.0  # Initial F_MAX_HZ when box drag starts

F_MIN_HZ: float = float(F_MIN_HZ_DEFAULT)
F_MAX_HZ: float = float(F_MAX_HZ_DEFAULT)

# Spectrum analyzer: draggable red vertical cursor for measuring freq/dB (None = hidden)
SPECTRUM_CURSOR_X: Optional[float] = None
# Dot + label on curve: only shown after user taps once on the red line
SPECTRUM_CURSOR_DOT_ACTIVE: bool = False
# Frequency (Hz) the dot is pinned to (from tap y); stops dot bouncing between bins
SPECTRUM_CURSOR_DOT_FREQ: Optional[float] = None
# Double-tap detection for spectrum bar (toggle cursor on/off)
SPECTRUM_CURSOR_LAST_TAP_TIME: float = 0.0
SPECTRUM_CURSOR_LAST_TAP_X: int = 0
SPECTRUM_CURSOR_LAST_TAP_Y: int = 0
# Tap on line or drag-dot release: main loop snaps to closest curve (set in handler, consumed in main)
SPECTRUM_CURSOR_PENDING_TAP_X: Optional[float] = None
SPECTRUM_CURSOR_PENDING_TAP_Y: Optional[int] = None
# Dot position in bar coords (from last draw) for hit-test; None until dot has been drawn
SPECTRUM_CURSOR_DOT_BAR_X: Optional[float] = None
SPECTRUM_CURSOR_DOT_BAR_Y: Optional[float] = None
# True while user is dragging the dot along the curve
SPECTRUM_CURSOR_DOT_DRAG_ACTIVE: bool = False


# ===============================================================
# Mouse + shared frame/output handles (HUD_RECTS set by main loop)
# ===============================================================
CURSOR_POS: Tuple[int, int] = (0, 0)
HUD_RECTS: Optional[Any] = None  # HudRects from top_hud.draw_hud; set each frame when drawing

CURRENT_FRAME: Optional[Any] = None  # typically a numpy ndarray (H,W,3) uint8
CALIBRATION_SUITE_BACKGROUND_FRAME: Optional[Any] = None  # cached frame when calibration suite modal closed (for launcher background)
OUTPUT_DIR: Optional[Path] = None

CAMERA_AVAILABLE: bool = False
MAGNETOMETER_AVAILABLE: bool = False

# ===============================================================
# Recording timestamp tracking
# ===============================================================
RECORDING_START_TIME: Optional[float] = None  # time.time() when recording started
RECORDING_PAUSED_TIME: Optional[float] = None  # time.time() when paused
RECORDING_TOTAL_PAUSED: float = 0.0  # total time spent paused

# ===============================================================
# UI visibility (top HUD, bottom HUD, menu) – swipe/double-tap to hide/show with animation
# ===============================================================
# Current animated offset (px). Top: 0=visible, negative=hidden up. Bottom: 0=visible, positive=hidden down. Menu: x=right, y=down.
ui_top_hud_offset: float = 0.0
ui_bottom_hud_offset: float = 0.0
ui_menu_offset: float = 0.0
ui_menu_offset_y: float = 0.0
# Target offsets (we lerp toward these each frame)
ui_top_hud_offset_target: float = 0.0
ui_bottom_hud_offset_target: float = 0.0
ui_menu_offset_target: float = 0.0
ui_menu_offset_y_target: float = 0.0
# Gesture state for tap/double-tap and swipe
ui_last_tap_time: float = 0.0
ui_last_tap_x: int = 0
ui_last_tap_y: int = 0
ui_drag_start_x: int = 0
ui_drag_start_y: int = 0
ui_drag_start_time: float = 0.0
ui_drag_handled: bool = False  # True once we've treated this pointer down as a drag
ui_click_was_on_ui: bool = False  # True if LBUTTONDOWN hit menu/HUD/button (gestures ignored on this pointer)
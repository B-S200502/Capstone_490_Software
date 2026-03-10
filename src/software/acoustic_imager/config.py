# config.py
"""
Central configuration/constants for the Acoustic Imager software.

This file mirrors the constants from the original monolithic script
so that splitting into modules does not change behavior.
"""

from __future__ import annotations

import struct
import os
import numpy as np

# ===============================================================
# 1. Configuration
# ===============================================================
WINDOW_NAME = "Fermat Heatmap (SIM + SPI LOOPBACK) + Bandpass (MUSIC)"
N_MICS = 16
SAMPLES_PER_CHANNEL = 512

SAMPLE_RATE_HZ = 100000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005

WIDTH = 1024
HEIGHT = 600
ALPHA = 0.7
# Single knob: menu button, dropdown items, top HUD pills, bottom HUD pills (0–1; higher = more opaque)
HUD_MENU_OPACITY = 0.92
BUTTON_ALPHA = HUD_MENU_OPACITY  # used by Button.draw for menu dropdown and back button
BUTTON_HITPAD_PX = 14

# Shared UI blue (BGR) for menu, HUD
MENU_ACTIVE_BLUE = (130, 82, 32)       # slightly darker blue
MENU_ACTIVE_BLUE_LIGHT = (165, 110, 48)  # gradient end

# Gallery Action surfaces style: "neon" (dark fill + neon edges) or "classic" (white on blue, pre-neon)
GALLERY_ACTION_STYLE = "classic"

# Single point of control for action button neon glow (0=off, ~0.32=default, 1=full)
ACTION_BTN_GLOW = 0.1

# Storage bar: one glow amount for all subelements (label, ring glow, % text, Used/Free); 0=off
STORAGE_BAR_GLOW = 0.28
STORAGE_BAR_BRIGHTNESS = 1.15
ACTION_BTN_BORDER_THICKNESS = 1

# Gallery action buttons: dark fill + neon edge (BGR) — used when GALLERY_ACTION_STYLE == "neon"
ACTION_BTN_FILL_DARK_TOP = (58, 58, 52)   # dark grey (slightly towards black)
ACTION_BTN_FILL_DARK_BOT = (32, 32, 28)   # darker grey
# Neon border color (electric blue) and glow strength (0=off, ~0.4=visible)
ACTION_BTN_NEON_BORDER_BGR = (255, 210, 100)  # BGR bright cyan-blue
ACTION_BTN_NEON_GLOW = 0.1                # border glow strength
ACTION_BTN_FILL_ALPHA = 1
ACTION_BTN_MODAL_FILL_ALPHA = 1
# Glassy shine on action buttons (0=off, ~0.11=subtle)
ACTION_BTN_SHINE_ALPHA = 0.05

# Classic style (when GALLERY_ACTION_STYLE == "classic"): white on blue, same as pre-neon
# Closed buttons (dock rows): darker blue gradient. Expanded (strip + panel after growth): lighter blue.
CLASSIC_ACTION_FILL_DARK_TOP = (90, 56, 22)   # BGR darker blue — closed dock rows
CLASSIC_ACTION_FILL_DARK_BOT = (115, 77, 34)   # darker blue bottom
CLASSIC_ACTION_FILL_TOP = (130, 82, 32)        # BGR lighter blue — expanded strip/panel (MENU_ACTIVE_BLUE)
CLASSIC_ACTION_FILL_BOT = (165, 110, 48)      # lighter blue bottom (MENU_ACTIVE_BLUE_LIGHT)
CLASSIC_ACTION_TEXT_BGR = (255, 255, 255)     # white text on blue
CLASSIC_ACTION_BORDER_BGR = (255, 255, 255)  # white border
CLASSIC_ACTION_GLOW = 0.06                 # very minor text/icon glow (0=off)

# Freq bar / bandpass: lighter neon blue so it stands out on dark bar
FREQ_BAR_BLUE = (230, 170, 80)
# Gallery filter/sort modal: goldish yellow for selected option (bandpass-style accent)
MODAL_ACTIVE_GOLD = (0, 200, 255)  # BGR goldish yellow

# Dock and content area gradients (BGR grey to black)
DOCK_GRADIENT_TOP = (55, 55, 55)
DOCK_GRADIENT_BOT = (20, 20, 20)
# Slightly darker for grid view and individual viewer background
BG_GRADIENT_TOP = (42, 42, 42)
BG_GRADIENT_BOT = (10, 10, 10)

FREQ_BAR_WIDTH = 150
DB_BAR_WIDTH = 80

# Magnetometer (BN-880) for compass
MAG_UART_DEVICE = "/dev/ttyAMA0"
MAG_UART_BAUD = 9600
MAG_COMPASS_DEMO = False  # If True, drive heading from time when no device (for testing)
MAG_USE_I2C = True        # If True, try I2C (HMC5883L) first; fall back to UART NMEA
MAG_I2C_BUS = 1
MAG_I2C_ADDR = 0x1e
# HMC5883L gain register (0x20=default, 0xA0=lower sensitivity / less saturation)
MAG_I2C_GAIN_REG = 0xA0
# Enable hard-iron offset calibration from running X/Y extrema.
MAG_APPLY_HARD_IRON_CAL = True
# Minimum observed span on both axes before using calibrated heading.
MAG_CAL_MIN_SPAN = 100

# GPS (BN-880) over UART
GPS_USE_UART = True
GPS_UART_DEVICE = "/dev/ttyAMA0"
GPS_UART_BAUD = 9600

# Radar mini-map (under Mb/s pill)
RADAR_MAP_ENABLED = True
RADAR_MAP_DIAMETER_PX = 200
RADAR_MAP_GAP_PX = 8
RADAR_HISTORY_SEC = 6.0
RADAR_MAX_POINTS = 180
RADAR_MIN_DB = -45.0
RADAR_MAP_ZOOM = 17
RADAR_MAP_TILE_SIZE = 256
RADAR_MAP_CACHE_DIR = "data/map_tiles"
RADAR_MAP_TILE_STYLE_DEFAULT = "dark"  # "dark" | "light"
RADAR_MAP_TILE_URL_DARK = "https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
RADAR_MAP_TILE_URL_LIGHT = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
RADAR_MAP_TILE_URL = RADAR_MAP_TILE_URL_DARK
# Tile brightness multipliers by style (1.0 = unchanged).
RADAR_MAP_BRIGHTNESS_DARK = 2.2
RADAR_MAP_BRIGHTNESS_LIGHT = 1.0
# Fallback-only radar brightness (used when no map tiles are shown).
# Kept separate so pure radar darkness does not affect tile overlays.
RADAR_MAP_FALLBACK_BRIGHTNESS = 0.65
RADAR_MAP_USER_AGENT = "acoustic-imager-radar/1.0"
RADAR_MAP_FETCH_TIMEOUT_SEC = 1.5
RADAR_MAP_RETRY_SEC = 3.0
RADAR_MAP_UPDATE_HZ = 15.0
RADAR_MAP_POS_BIN_PX = 3
RADAR_MAP_PATCH_CACHE_SIZE = 12
RADAR_MAP_FETCH_WORKERS = 2
# Optional radar controls
RADAR_UI_DEFAULT = True
POSITION_SERVICES_DEFAULT = True
RADAR_DEBUG_OVERLAY_DEFAULT = False
RADAR_DEBUG_BOX_MAX_WIDTH = 360

# Wi-Fi geolocation (Google Geolocation API)
WIFI_GEO_API_KEY = os.getenv("GOOGLE_GEOLOCATION_API_KEY", "")
WIFI_GEO_REQUEST_TIMEOUT_SEC = 3.0
WIFI_GEO_INTERVAL_SEC = 12.0

# Position fusion behavior
POSITION_STALE_SEC = 30.0
POSITION_SOURCE_HYSTERESIS_M = 8.0
GPS_DEFAULT_ACCURACY_M = 30.0
WIFI_DEFAULT_ACCURACY_M = 80.0

# FPS defaults (MENU controls 30/60/MAX at runtime)
FPS_TARGET = 60
FPS_MODE_TO_TARGET = {"30": 30, "60": 60}

# Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = WIDTH
CAMERA_HEIGHT = HEIGHT
USE_CAMERA = True
USE_LIBCAMERA = True

# Camera post-processing
CAMERA_BRIGHTNESS_ALPHA = 1.20   # contrast multiplier (1.0 = no change)
CAMERA_BRIGHTNESS_BETA  = 15      # brightness offset (-100 to +100 typical)
DETECTION_BOX_COLOR = (0, 215, 255)
#DETECTION_BOX_COLOR = (150, 180, 0) #teal
#DETECTION_BOX_COLOR = (0, 200, 200) # yellow

# Bandpass
F_DISPLAY_MAX = 45000.0
F_MIN_HZ_DEFAULT = 24000.0
F_MAX_HZ_DEFAULT = 35000.0

DRAG_MARGIN_PX = 18

# UI visibility (swipe/double-tap): hide offsets in px, animation speed, gesture thresholds
UI_TOP_HUD_HIDE_OFFSET = -150      # top HUD + dropdown panel move fully up off screen
UI_BOTTOM_HUD_HIDE_OFFSET = 60     # bottom HUD moves down off screen
# Extra bottom hide distance when original debug panel is visible.
UI_BOTTOM_HUD_DEBUG_EXTRA_HIDE_OFFSET = 20
UI_MENU_HIDE_OFFSET = 220          # menu moves right off screen
UI_MENU_HIDE_OFFSET_Y = 80         # menu moves down off screen (with bottom HUD on swipe down)
UI_VISIBILITY_ANIM_SPEED = 0.18    # lerp per frame (~60fps: ~0.25s to settle)
UI_SWIPE_THRESHOLD_PX = 50         # min distance for swipe
UI_DOUBLE_TAP_MS = 400             # max ms between taps for double-tap
UI_DOUBLE_TAP_RADIUS_PX = 40       # max distance between taps
UI_TAP_MAX_MOVE_PX = 15            # max movement to count as tap (not drag)
# Shared pill size for top and bottom HUD (same size, larger for clickability)
UI_PILL_H = 48
UI_PILL_W = 170
# Battery time-remaining estimate when no hardware data (typical Pi portable pack)
BATTERY_CAPACITY_MAH = 10000   # mAh
BATTERY_DISCHARGE_MA = 2000    # mA typical draw
# Gestures (swipe/double-tap) only active in content area: between dB bar and freq bar, excluding top/bottom strips
UI_CONTENT_TOP_MARGIN = 58         # below top HUD
UI_CONTENT_BOTTOM_MARGIN = 62      # above bottom HUD
# Extra hit padding for bottom HUD pills (better clickability)
UI_BOTTOM_HUD_HIT_PAD = 8

# Angle grid
ANGLES = np.linspace(-90, 90, 181)

# ===============================================================
# FFT frequency axis
# ===============================================================
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)
N_BINS = SAMPLES_PER_CHANNEL // 2 + 1  # 257

# ===============================================================
# SIM sources
# ===============================================================
# SIM_1 (existing SIM mode): fixed source set
SIM_SOURCE_FREQS = [9000, 11000, 30000]
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]
# Display distance (m) per sim source for crosshair tooltip; sim uses far-field so this is for UI only
SIM_SOURCE_DISTANCES_M = [1.0, 1.5, 2.0]  # same length as SIM_SOURCE_ANGLES
SIM_N_SOURCES = len(SIM_SOURCE_ANGLES)

# SIM_2: directional event model (persistent + transient), no true range simulation
SIM2_PERSISTENT_BEARINGS = [-55.0, -15.0, 30.0]
SIM2_PERSISTENT_FREQS = [9500.0, 12000.0, 27000.0]
SIM2_PERSISTENT_AMPLS = [1.2, 0.9, 1.1]
SIM2_PERSISTENT_JITTER_DEG = 3.0
SIM2_TRANSIENT_EVENT_RATE_HZ = 0.20
SIM2_TRANSIENT_MIN_DURATION_S = 0.8
SIM2_TRANSIENT_MAX_DURATION_S = 2.4
SIM2_TRANSIENT_FREQ_RANGE = (8000.0, 32000.0)
SIM2_TRANSIENT_AMPL_RANGE = (0.6, 1.6)
SIM2_NOISE_POWER_SCALE = 1.2

# dB mapping for heatmap intensity
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0

# ===============================================================
# Blend acceleration (LUT + integer math)
# ===============================================================
# weight = uint8 0..255 derived from heatmap intensity (0..255)
# We fold in sqrt-gamma and ALPHA here so per-frame work is tiny.
BLEND_GAMMA = 0.5  # same as np.power(mask, 0.5)

_w_lut = np.arange(256, dtype=np.float32) / 255.0
_w_lut = np.power(_w_lut, BLEND_GAMMA) * float(ALPHA)  # 0..ALPHA
W_LUT_U8 = np.clip(np.round(_w_lut * 255.0), 0, 255).astype(np.uint8)

# optional: reuse these buffers to reduce allocs (left here for compatibility)
_tmp_w = None
_tmp_out = None

# ===============================================================
# 2. SPI frame format (float32 MAG + float32 PHASE)
# ===============================================================
MAGIC_START = 0x46524654  # 'TFRF' (example)
MAGIC_END = 0x454E4421    # 'END!' (example)
VERSION = 1

HEADER_FMT = "<IHH I HH I HH I"
HEADER_LEN = struct.calcsize(HEADER_FMT)

TRAILER_FMT = "<II"  # crc32(u32), end_magic(u32)
TRAILER_LEN = struct.calcsize(TRAILER_FMT)

PAYLOAD_LEN = N_MICS * N_BINS * 2 * 4  # 8 bytes per bin per mic
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN

# --- Per-mic packet format (firmware SPI_FrameHeader_t + payload + checksum) ---
SPI_MAGIC_FW = 0xAABBCCDD
# Packed struct matching SPI_FrameHeader_t: magic(I) version(H) header_len(H) frame_counter(I)
# batch_id(H) mic_index(B) fft_size(H) sample_rate(I) flags(H) payload_len(H) battery_mv(H) reserved0(H) reserved1(H)
SPI_MIC_HEADER_FMT = "<IHHIHBHIHHHHH"
SPI_MIC_HEADER_BYTES = struct.calcsize(SPI_MIC_HEADER_FMT)  # 31
SPI_MIC_PAYLOAD_BYTES = 2048   # 512 * 4 (packed RFFT floats per mic)
SPI_MIC_CHECKSUM_BYTES = 2
SPI_MIC_PACKET_BYTES = SPI_MIC_HEADER_BYTES + SPI_MIC_PAYLOAD_BYTES + SPI_MIC_CHECKSUM_BYTES  # 2081

# SPI config
SOURCE_DEFAULT = "SIM"   # or "LOOP" / "HW" / "REF"
SOURCE_MODES = ("SIM", "SIM_2", "LOOP", "HW", "REF")  # REF = 0 dB reference baseline (flat spectrum)

# Directional history persistence
DIRECTIONAL_HISTORY_ENABLED = True
DIRECTIONAL_HISTORY_DIR = "data/directional_history"
DIRECTIONAL_HISTORY_FLUSH_SEC = 1.0
DIRECTIONAL_HISTORY_RETENTION_DAYS = 7
DIRECTIONAL_HISTORY_RECORD_DEFAULT = False

# SIM_2 radar extraction tuning (multi-signal window behavior)
SIM2_RADAR_MIN_ROW_REL = 0.30
SIM2_RADAR_MIN_SEPARATION_DEG = 12.0

SPI_BUS = 0
SPI_DEV = 0
SPI_MODE = 0
SPI_BITS = 8

SPI_MAX_SPEED_HZ = 80_000_000
SPI_XFER_CHUNK = 8192

# For debugging if FPS is too slow, we can pre-generate a static frame and reuse it.
STATIC_TX_FRAME = None

# --- LOOPBACK SPI "virtual sources" ---
SPI_SIM_BINS = [35, 80, 160, 220]
SPI_SIM_AMPLS = [6.0, 3.0, 5.0, 4.0]  # boosted
SPI_SIM_ANGLES = [-25.0, 35.0, -5.0, 60.0]
SPI_SIM_DRIFT_DEG_PER_SEC = [1.2, -0.6, 0.4, -0.3]

CRC_EVERY_N = 30

# ===============================================================
# 3. Geometry setup (Fermat spiral)
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.010
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

pitch = (
    np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2)))))
    if N_MICS > 2
    else 0.0
)
# config.py
"""
Central configuration/constants for the Acoustic Imager software.

Layout: core/UI/heatmap display first; then SIM (SRC: SIM + SRC: LOOP);
then Real hardware (SRC: HW) — SPI transport and heatmap pipeline.
Default source on startup is HW.
"""

from __future__ import annotations

import struct
import numpy as np

# ===============================================================
# 1. Configuration
# ===============================================================
WINDOW_NAME = "Fermat Heatmap (SIM + SPI LOOPBACK) + Bandpass (MUSIC)"
N_MICS = 16
SAMPLES_PER_CHANNEL = 512

SAMPLE_RATE_HZ = 100000
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005   # SIM source only (synthetic noise floor)

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

# Radar / position (branch-specific: state.py imports these)
RADAR_UI_DEFAULT = True
POSITION_SERVICES_DEFAULT = False
RADAR_MAP_TILE_STYLE_DEFAULT = "dark"   # "dark" | "light"
DIRECTIONAL_HISTORY_RECORD_DEFAULT = False
RADAR_DEBUG_OVERLAY_DEFAULT = False
# Circular radar widget diameter (px)
RADAR_MAP_DIAMETER_PX = 200

# Magnetometer (compass) — main.py uses these for MagnetometerReader
MAG_UART_DEVICE = "/dev/ttyS0"
MAG_UART_BAUD = 9600

DRAG_MARGIN_PX = 18

# UI visibility (swipe/double-tap): hide offsets in px, animation speed, gesture thresholds
UI_TOP_HUD_HIDE_OFFSET = -150      # top HUD + dropdown panel move fully up off screen
UI_BOTTOM_HUD_HIDE_OFFSET = 60     # bottom HUD moves down off screen
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

# dB mapping for heatmap intensity
REL_DB_MIN = -60.0
REL_DB_MAX = 0.0
# Temporal smoothing for HW/LOOP heatmap: retain this fraction of previous frame (0=none, 1=no update)
HEATMAP_SMOOTH_ALPHA = 0.35
# Bandpass power above this gives full heatmap brightness; below scales down (with floor) so heatmap reacts to level (tune to room)
HEATMAP_LEVEL_REFERENCE = 1e6
# Minimum heatmap scale so blobs stay visible when quiet (0=can go black, 1=no level scaling)
HEATMAP_LEVEL_FLOOR = 0.18
# Per-frame contrast stretch: map this percentile to 255 (0=disable). Improves differentiation.
HEATMAP_CONTRAST_STRETCH_PERCENTILE = 98.0

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
# SIM (synthetic source + SRC: LOOP loopback)
# ===============================================================
# --- SRC: SIM (pure synthetic FFT) ---
SIM_SOURCE_FREQS = [9000, 11000, 30000]
SIM_SOURCE_ANGLES = [-35.0, 0.0, 40.0]
SIM_SOURCE_AMPLS = [0.6, 1.0, 2.0]
# Display distance (m) per sim source for crosshair tooltip; sim uses far-field so this is for UI only
SIM_SOURCE_DISTANCES_M = [1.0, 1.5, 2.0]  # same length as SIM_SOURCE_ANGLES
SIM_N_SOURCES = len(SIM_SOURCE_ANGLES)

# --- SRC: SIM_2 (event model: persistent + transient sources) ---
SIM2_PERSISTENT_BEARINGS = [-20.0, 10.0]   # degrees
SIM2_PERSISTENT_FREQS = [10000.0, 28000.0]  # Hz
SIM2_PERSISTENT_AMPLS = [0.8, 1.0]
SIM2_PERSISTENT_JITTER_DEG = 2.0
SIM2_TRANSIENT_EVENT_RATE_HZ = 0.5
SIM2_TRANSIENT_MIN_DURATION_S = 0.1
SIM2_TRANSIENT_MAX_DURATION_S = 2.0
SIM2_TRANSIENT_FREQ_RANGE = (5000.0, 35000.0)
SIM2_TRANSIENT_AMPL_RANGE = (0.2, 1.0)
SIM2_NOISE_POWER_SCALE = 1.0

# --- SRC: LOOP (loopback: simulated bins fed through same pipeline as HW) ---
SPI_SIM_BINS = [35, 80, 160, 220]       # which frequency bins to simulate in loopback
SPI_SIM_AMPLS = [6.0, 3.0, 5.0, 4.0]    # per-bin amplitude
SPI_SIM_ANGLES = [-25.0, 35.0, -5.0, 60.0]
SPI_SIM_DRIFT_DEG_PER_SEC = [1.2, -0.6, 0.4, -0.3]

# ===============================================================
# Real hardware (SRC: HW) — SPI transport & heatmap pipeline
# ===============================================================
# --- SPI frame format (matches STM32) ---
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

# Full-frame format: one header + all N_MICS payloads + checksum (matches STM32 SPI_FRAME_PACKET_SIZE_BYTES).
SPI_FRAME_PACKET_SIZE_BYTES = SPI_MIC_HEADER_BYTES + N_MICS * SPI_MIC_PAYLOAD_BYTES + SPI_MIC_CHECKSUM_BYTES  # 32801
SPI_USE_FULL_FRAME = True  # HW: one read of 32801 bytes per frame; no per-mic accumulator.

# --- SPI bus & GPIO (HW only) ---
# Single switch: 0 = SPI0 (CE0, primary header), 1 = SPI1 (CE2, secondary header). Pinouts in branch_merge_preservation_checklist.md §10.
SPI_INTERFACE = 1  # default SPI1 (current setup); set 0 for SPI0
# (SPI_BUS, SPI_DEV, FRAME_READY_BCM_PIN, GAIN_CTRL_BCM_PIN) per interface
_SPI_PIN_SETUPS = {
    0: (0, 0, 7, 25),   # SPI0: /dev/spidev0.0, frame-ready BCM7, gain BCM25
    1: (1, 2, 7, 25),   # SPI1: /dev/spidev1.2, frame-ready BCM7, gain BCM25
}
_SPI_BUS, _SPI_DEV, _FRAME_READY_BCM, _GAIN_CTRL_BCM = _SPI_PIN_SETUPS[SPI_INTERFACE]
SPI_BUS = _SPI_BUS
SPI_DEV = _SPI_DEV
SPI_MODE = 1
SPI_BITS = 8

SPI_MAX_SPEED_HZ = 20_000_000  # 20 MHz; stable for full-frame. See utilities/debug/SPI_settings.md for 21.25/30 MHz notes.
SPI_XFER_CHUNK = 8192

# Frame-ready GPIO: MCU_STATUS from STM32 -> Pi (physical pin 26 = BCM7 for both interfaces)
FRAME_READY_BCM_PIN = _FRAME_READY_BCM
FRAME_READY_PULL = "down"
FRAME_READY_TIMEOUT_S = 0.25

# Gain control output: Pi drives STM32 AUTO_GAIN_CNTL (physical pin 22 = BCM25). Set True to link GAIN menu to hardware.
GAIN_CTRL_ENABLED = True
GAIN_CTRL_BCM_PIN = _GAIN_CTRL_BCM

# Source modes (order = cycle order; default on startup)
SOURCE_DEFAULT = "HW"    # real hardware first
SOURCE_MODES = ("HW", "SIM", "LOOP", "REF")  # REF = 0 dB reference baseline (flat spectrum)

# For debugging if FPS is too slow, we can pre-generate a static frame and reuse it.
STATIC_TX_FRAME = None

# --- Camera–array calibration (debug test setup) ---
# See utilities/debug/calibration_test.md. For overlay alignment: distance and lateral offset.
CALIBRATION_DISTANCE_INCHES = 5.0   # depth: camera to array center
CALIBRATION_OFFSET_INCHES = 7.0     # lateral: + = array to right of camera center (camera left, board right)
CALIBRATION_NOTE = "Camera left, board right, same heading"

# --- HW heatmap pipeline (gain, MUSIC, directivity, etc.) ---
# Per-mic gain correction (length N_MICS): boost weak mics; 1.0 = no change. Use metrics_debug.py --live --write-config to tune.
SPI_MIC_GAIN = (1.00, 3.03, 12.56, 5.18, 2.62, 9.78, 9.79, 12.94, 2.66, 10.76, 24.53, 10.12, 4.03, 9.67, 12.75, 11.90)
# Whole-array gain boost (linear): 2.0 = ~6 dB; use if mics seem low
SPI_ARRAY_GAIN = 2.0
# Number of bins to use for heatmap in HW/LOOP: top-K by power within bandpass (replaces fixed SPI_SIM_BINS for live display)
SPI_TOP_K_BINS = 4
# Only bins within this many dB of peak (in bandpass) are eligible for heatmap; lower = stricter, less noisy
SPI_NOISE_FLOOR_DB = 15.0
# Number of spatial sources MUSIC assumes per bin (1 = one dominant source e.g. one speaker, 2 = allow one reflection)
SPI_MUSIC_N_SOURCES = 1
# Covariance averaging: smooth R over this many frames (EMA) before MUSIC; 1 = no averaging, 3–5 = less noisy peaks.
SPI_COV_AVG_FRAMES = 4
# Only show bins that are directional: lambda_1/sum(eigvals) >= this (0=off). Stricter = less random noise.
SPI_DIRECTIVITY_MIN = 0.5
# Only show bin if its MUSIC peak angle is stable: change from last frame <= this deg (0=off). Suppresses jitter.
SPI_ANGLE_STABILITY_DEG = 25.0
# Per-mic normalization: scale each mic so L2 norm across bins is 1 (balances gain across mics; use if some mics are weak).
SPI_PER_MIC_NORMALIZE = True
# Power curve for blob brightness: 1.0=linear, >1=stronger bins dominate (more differentiation), <1=lift weak bins
SPI_HEATMAP_POWER_GAMMA = 1.15

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
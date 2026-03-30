"""
Magnetometer reader for BN-880 (or compatible): compass via UART (NMEA HDT/HDG) or I2C (HMC5883L).
Updates HUD.compass_heading_deg and HUD.compass_heading_valid. Stub/demo when device unavailable.

Default plane XY uses atan2(y,x): best when yaw is rotation about the sensor normal (e.g. device
flat or screen facing you, spin like a turntable). Use XZ or YZ if your mechanical mount puts
horizontal field rotation mostly in those axes instead.

Without an accelerometer, pitch/roll repartition the field into x,y,z — the 2D heading will jump
when you tilt. That is expected until tilt compensation is added (different sensor path).
"""

from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

try:
    import serial
except ImportError:
    serial = None

try:
    from smbus2 import SMBus
except ImportError:
    SMBus = None

from .. import config
from ..state import HUD
from .compass_calibration import cal_dir


def _median_int(samples: list[int]) -> int:
    return int(round(statistics.median(samples)))


def _mag_heading_debug_log_file() -> Optional[Path]:
    raw = str(getattr(config, "MAG_HEADING_DEBUG_LOG_PATH", "") or "").strip()
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else cal_dir() / p


def _nmea_checksum(s: str) -> bool:
    """Validate NMEA checksum. s is full sentence including $ and *XX."""
    if "*" not in s or len(s) < 10:
        return False
    payload, _, checksum_str = s.strip().rpartition("*")
    if not payload.startswith("$") or len(checksum_str) != 2:
        return False
    try:
        expected = int(checksum_str, 16)
    except ValueError:
        return False
    xor = 0
    for c in payload[1:]:
        xor ^= ord(c)
    return xor == expected


def _parse_heading_nmea(line: str) -> Optional[float]:
    """Parse NMEA HDT or HDG sentence; return heading in degrees or None."""
    line = line.strip()
    if not line.startswith("$") or not _nmea_checksum(line):
        return None
    # HDT: $GPHDT,123.45,T*hh  or  $GNHDT,...
    if ",HDT," in line or line.startswith("$GPHDT,") or line.startswith("$GNHDT,"):
        parts = line.split(",")
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                pass
    # HDG: $GPHDG,123.45,,,E*hh  (heading, dev, var, E/W)
    if ",HDG," in line or line.startswith("$GPHDG,") or line.startswith("$GNHDG,"):
        parts = line.split(",")
        if len(parts) >= 1:
            try:
                return float(parts[1])
            except ValueError:
                pass
    return None


def probe_i2c_magnetometer(bus: int, addr: int) -> bool:
    """Probe for HMC5883L (or compatible) at the given I2C bus and address. Returns True if present."""
    if SMBus is None:
        return False
    try:
        smb = SMBus(bus)
        try:
            smb.read_byte_data(addr, 0x00)  # config A
            return True
        finally:
            smb.close()
    except Exception:
        return False


class MagnetometerReader:
    """
    Background thread: read compass from I2C (HMC5883L) or UART (NMEA HDT/HDG), update
    HUD.compass_heading_deg and HUD.compass_heading_valid. Falls back to UART then demo if enabled.
    """

    def __init__(
        self,
        device: str,
        baud: int,
        demo: bool = False,
        use_i2c: bool = True,
        i2c_bus: int = 1,
        i2c_addr: int = 0x1E,
        i2c_gain_reg: int = 0xA0,
    ) -> None:
        self.device = device
        self.baud = baud
        self.demo = demo
        self.use_i2c = use_i2c
        self.i2c_bus = i2c_bus
        self.i2c_addr = i2c_addr
        self.i2c_gain_reg = i2c_gain_reg
        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._mag_heading_dbg_last_t = 0.0
        n_med = int(getattr(config, "MAG_HEADING_MEDIAN_LEN", 5))
        self._mag_med_n = max(1, min(n_med, 15))
        self._mag_x_hist: deque[int] = deque(maxlen=self._mag_med_n)
        self._mag_y_hist: deque[int] = deque(maxlen=self._mag_med_n)
        self._mag_z_hist: deque[int] = deque(maxlen=self._mag_med_n)

    def start(self) -> None:
        if self._thr is not None and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=2.0)
        self._thr = None

    def _worker(self) -> None:
        if self.use_i2c and SMBus is not None:
            self._run_i2c()
            return
        if serial is None:
            if self.demo:
                self._run_demo()
            return
        retry_sleep = 2.0
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.device, baudrate=self.baud, timeout=0.5)
                ser.reset_input_buffer()
            except Exception:
                HUD.compass_heading_valid = False
                if self.demo:
                    self._run_demo()
                    return
                time.sleep(retry_sleep)
                continue
            # Read lines and parse
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = ser.read(256)
                    if not chunk:
                        continue
                    buf += chunk
                except Exception:
                    HUD.compass_heading_valid = False
                    break
                while b"\r" in buf or b"\n" in buf:
                    line, sep, buf = buf.partition(b"\n")
                    if not sep:
                        line, sep, buf = buf.partition(b"\r")
                    line = (line + sep).decode("ascii", errors="ignore").strip()
                    if not line.startswith("$"):
                        continue
                    heading = _parse_heading_nmea(line)
                    if heading is not None:
                        hdg = heading % 360.0
                        if bool(getattr(config, "COMPASS_USER_CAL_VALID", False)):
                            hdg = (hdg + float(getattr(config, "COMPASS_USER_CAL_OFFSET_DEG", 0.0))) % 360.0
                        HUD.compass_heading_deg = hdg
                        HUD.compass_heading_valid = True
            try:
                ser.close()
            except Exception:
                pass
            HUD.compass_heading_valid = False
            if self._stop.is_set():
                break
            time.sleep(retry_sleep)

    def _run_i2c(self) -> None:
        """Read HMC5883L over I2C; heading from MAG_HEADING_PLANE (XY/XZ/YZ). Retry on failure."""
        retry_sleep = 2.0
        while not self._stop.is_set():
            try:
                bus = SMBus(self.i2c_bus)
            except Exception:
                HUD.compass_heading_valid = False
                time.sleep(retry_sleep)
                continue
            try:
                bus.write_byte_data(self.i2c_addr, 0x00, 0x70)  # config A
                bus.write_byte_data(self.i2c_addr, 0x01, int(self.i2c_gain_reg) & 0xFF)  # config B (gain)
                bus.write_byte_data(self.i2c_addr, 0x02, 0x00)  # continuous mode
            except Exception:
                HUD.compass_heading_valid = False
                try:
                    bus.close()
                except Exception:
                    pass
                time.sleep(retry_sleep)
                continue

            def read_word_2c(reg: int) -> int:
                hi = bus.read_byte_data(self.i2c_addr, reg)
                lo = bus.read_byte_data(self.i2c_addr, reg + 1)
                val = (hi << 8) | lo
                if val >= 32768:
                    val -= 65536
                return val

            self._mag_x_hist.clear()
            self._mag_y_hist.clear()
            self._mag_z_hist.clear()
            mag_journal_last = 0.0
            while not self._stop.is_set():
                try:
                    x = read_word_2c(0x03)
                    z = read_word_2c(0x05)
                    y = read_word_2c(0x07)
                    self._mag_x_hist.append(x)
                    self._mag_y_hist.append(y)
                    self._mag_z_hist.append(z)
                    if self._mag_med_n <= 1:
                        xm, ym, zm = x, y, z
                    else:
                        xm = _median_int(list(self._mag_x_hist))
                        ym = _median_int(list(self._mag_y_hist))
                        zm = _median_int(list(self._mag_z_hist))
                    # Hard-iron extrema + heading use median-smoothed samples to limit spike poisoning.
                    if HUD.mag_x_min is None or xm < HUD.mag_x_min:
                        HUD.mag_x_min = int(xm)
                    if HUD.mag_x_max is None or xm > HUD.mag_x_max:
                        HUD.mag_x_max = int(xm)
                    if HUD.mag_y_min is None or ym < HUD.mag_y_min:
                        HUD.mag_y_min = int(ym)
                    if HUD.mag_y_max is None or ym > HUD.mag_y_max:
                        HUD.mag_y_max = int(ym)
                    if HUD.mag_z_min is None or zm < HUD.mag_z_min:
                        HUD.mag_z_min = int(zm)
                    if HUD.mag_z_max is None or zm > HUD.mag_z_max:
                        HUD.mag_z_max = int(zm)

                    x_span = int(HUD.mag_x_max - HUD.mag_x_min) if (HUD.mag_x_max is not None and HUD.mag_x_min is not None) else 0
                    y_span = int(HUD.mag_y_max - HUD.mag_y_min) if (HUD.mag_y_max is not None and HUD.mag_y_min is not None) else 0
                    z_span = int(HUD.mag_z_max - HUD.mag_z_min) if (HUD.mag_z_max is not None and HUD.mag_z_min is not None) else 0

                    # Centered heading debug (hard-iron offset only, no soft-iron scale).
                    x_off = 0.5 * (HUD.mag_x_min + HUD.mag_x_max) if (HUD.mag_x_min is not None and HUD.mag_x_max is not None) else 0.0
                    y_off = 0.5 * (HUD.mag_y_min + HUD.mag_y_max) if (HUD.mag_y_min is not None and HUD.mag_y_max is not None) else 0.0
                    z_off = 0.5 * (HUD.mag_z_min + HUD.mag_z_max) if (HUD.mag_z_min is not None and HUD.mag_z_max is not None) else 0.0
                    x_cal = float(xm) - x_off
                    y_cal = float(ym) - y_off
                    z_cal = float(zm) - z_off

                    def heading_from(a: float, b: float) -> float:
                        h = math.degrees(math.atan2(b, a))
                        if h < 0:
                            h += 360.0
                        return h

                    min_span = int(getattr(config, "MAG_CAL_MIN_SPAN", 100))
                    apply_hi = bool(getattr(config, "MAG_APPLY_HARD_IRON_CAL", True))
                    raw_xy_heading = heading_from(float(xm), float(ym))
                    dbg_h_xz = heading_from(float(xm), float(zm))
                    dbg_h_yz = heading_from(float(ym), float(zm))

                    def fixed_plane_heading(plane: str) -> tuple[float, float, bool, str]:
                        pl = plane.upper()
                        if pl == "XZ":
                            return (
                                heading_from(float(xm), float(zm)),
                                heading_from(x_cal, z_cal),
                                x_span >= min_span and z_span >= min_span,
                                "XZ",
                            )
                        if pl == "YZ":
                            return (
                                heading_from(float(ym), float(zm)),
                                heading_from(y_cal, z_cal),
                                y_span >= min_span and z_span >= min_span,
                                "YZ",
                            )
                        return (
                            heading_from(float(xm), float(ym)),
                            heading_from(x_cal, y_cal),
                            x_span >= min_span and y_span >= min_span,
                            "XY",
                        )

                    user_cal = bool(getattr(config, "COMPASS_USER_CAL_VALID", False))
                    user_plane = getattr(config, "COMPASS_USER_CAL_PLANE", None)
                    user_off = float(getattr(config, "COMPASS_USER_CAL_OFFSET_DEG", 0.0))

                    if user_cal and user_plane in ("XY", "XZ", "YZ"):
                        heading_deg, heading_cal_deg, cal_ready, best_pair = fixed_plane_heading(str(user_plane))
                        use_cal = apply_hi and cal_ready
                        base = heading_cal_deg if use_cal else heading_deg
                        HUD.compass_heading_deg = (base + user_off) % 360.0
                    else:
                        plane_mode = str(getattr(config, "MAG_HEADING_PLANE", "xy")).strip().lower()
                        if plane_mode not in ("xy", "xz", "yz"):
                            plane_mode = "xy"
                        heading_deg, heading_cal_deg, cal_ready, best_pair = fixed_plane_heading(plane_mode)
                        use_cal = apply_hi and cal_ready
                        HUD.compass_heading_deg = heading_cal_deg if use_cal else heading_deg

                    HUD.mag_x_raw = int(x)
                    HUD.mag_y_raw = int(y)
                    HUD.mag_z_raw = int(z)
                    HUD.mag_span_x = int(x_span)
                    HUD.mag_span_y = int(y_span)
                    HUD.mag_span_z = int(z_span)
                    HUD.mag_pair_dbg = best_pair
                    HUD.mag_heading_dbg = float(heading_deg)
                    HUD.mag_heading_cal_dbg = float(heading_cal_deg)
                    HUD.mag_cal_active = bool(use_cal)
                    HUD.compass_heading_valid = True
                    log_iv = float(getattr(config, "MAG_JOURNAL_LOG_INTERVAL_S", 0.0))
                    if log_iv > 0:
                        tn = time.time()
                        if tn - mag_journal_last >= log_iv:
                            mag_journal_last = tn
                            line = (
                                f"[mag] x={HUD.mag_x_raw} y={HUD.mag_y_raw} z={HUD.mag_z_raw} "
                                f"heading={HUD.compass_heading_deg:.1f} plane={best_pair}"
                            )
                            if bool(getattr(config, "MAG_JOURNAL_LOG_VERBOSE", False)):
                                line += f" raw_xy={raw_xy_heading:.1f}"
                            print(line, flush=True)
                    dbg_file = _mag_heading_debug_log_file()
                    if dbg_file is not None:
                        dbg_iv = float(getattr(config, "MAG_HEADING_DEBUG_LOG_INTERVAL_S", 0.2))
                        tdbg = time.time()
                        if dbg_iv <= 0 or (tdbg - self._mag_heading_dbg_last_t) >= dbg_iv:
                            self._mag_heading_dbg_last_t = tdbg
                            try:
                                dbg_file.parent.mkdir(parents=True, exist_ok=True)
                                with dbg_file.open("a", encoding="utf-8") as _df:
                                    _df.write(
                                        f"{tdbg:.3f}\t{HUD.mag_x_raw}\t{HUD.mag_y_raw}\t{HUD.mag_z_raw}\t"
                                        f"{HUD.compass_heading_deg:.2f}\t{best_pair}\t{int(use_cal)}\t"
                                        f"{HUD.mag_heading_dbg:.2f}\t{HUD.mag_heading_cal_dbg:.2f}\t"
                                        f"{raw_xy_heading:.2f}\t{dbg_h_xz:.2f}\t{dbg_h_yz:.2f}\n"
                                    )
                            except OSError:
                                pass
                except Exception:
                    HUD.compass_heading_valid = False
                    break
                time.sleep(0.05)
            try:
                bus.close()
            except Exception:
                pass
            HUD.compass_heading_valid = False
            if self._stop.is_set():
                break
            time.sleep(retry_sleep)

    def _run_demo(self) -> None:
        """Time-based heading for testing when no device."""
        while not self._stop.is_set():
            HUD.compass_heading_deg = (time.time() * 20.0) % 360.0
            HUD.compass_heading_valid = True
            time.sleep(0.05)

"""
Magnetometer reader for BN-880 (or compatible): compass via UART (NMEA HDT/HDG) or I2C (HMC5883L).
Updates HUD.compass_heading_deg and HUD.compass_heading_valid. Stub/demo when device unavailable.
"""

from __future__ import annotations

import math
import threading
import time
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

# #region agent log
DEBUG_LOG_PATH = "/home/acousticgod/Capstone_490_Software/.cursor/debug-2bfbda.log"
def _debug_log(location: str, message: str, data: dict, hypothesis_id: str = "") -> None:
    import json
    try:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps({"sessionId": "2bfbda", "timestamp": int(time.time() * 1000), "location": location, "message": message, "data": data, "hypothesisId": hypothesis_id}) + "\n")
    except Exception:
        pass
# #endregion


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


def _i2c_scan_bus(bus_num: int) -> list[int]:
    """Scan I2C bus for devices that ACK (addresses 0x08--0x77). Returns list of addresses."""
    if SMBus is None:
        return []
    found = []
    try:
        smb = SMBus(bus_num)
        try:
            for a in range(0x08, 0x78):
                try:
                    smb.read_byte(a)
                    found.append(a)
                except Exception:
                    pass
        finally:
            smb.close()
    except Exception:
        pass
    return found


def probe_i2c_magnetometer(bus: int, addr: int) -> bool:
    """Probe for HMC5883L (or compatible) at the given I2C bus and address. Returns True if present."""
    # #region agent log
    _debug_log("magnetometer.py:probe_i2c_magnetometer", "I2C probe start", {"bus": bus, "expected_addr_hex": hex(addr), "expected_addr_dec": addr}, "A")
    # #endregion
    if SMBus is None:
        # #region agent log
        _debug_log("magnetometer.py:probe_i2c_magnetometer", "SMBus not available", {"reason": "smbus2 not installed"}, "D")
        # #endregion
        return False
    try:
        smb = SMBus(bus)
        try:
            # #region agent log
            addrs_found = _i2c_scan_bus(bus)
            _debug_log("magnetometer.py:probe_i2c_magnetometer", "I2C scan result", {"bus": bus, "addresses_hex": [hex(a) for a in addrs_found], "expected_addr_hex": hex(addr), "expected_in_scan": addr in addrs_found}, "B")
            # #endregion
            smb.read_byte_data(addr, 0x00)  # config A
            # #region agent log
            _debug_log("magnetometer.py:probe_i2c_magnetometer", "Probe success", {"addr_hex": hex(addr)}, "E")
            # #endregion
            return True
        finally:
            smb.close()
    except Exception as e:
        # #region agent log
        _debug_log("magnetometer.py:probe_i2c_magnetometer", "Probe failed", {"bus": bus, "addr_hex": hex(addr), "exception_type": type(e).__name__, "exception_msg": str(e)}, "C")
        # #endregion
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
                        HUD.compass_heading_deg = (heading % 360.0)
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
        """Read HMC5883L over I2C; compute heading from atan2(y, x). Retry on failure."""
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

            while not self._stop.is_set():
                try:
                    x = read_word_2c(0x03)
                    z = read_word_2c(0x05)
                    y = read_word_2c(0x07)
                    heading_deg = math.degrees(math.atan2(y, x))
                    if heading_deg < 0:
                        heading_deg += 360.0
                    # Track extrema for quick hard-iron bias diagnostics.
                    if HUD.mag_x_min is None or x < HUD.mag_x_min:
                        HUD.mag_x_min = int(x)
                    if HUD.mag_x_max is None or x > HUD.mag_x_max:
                        HUD.mag_x_max = int(x)
                    if HUD.mag_y_min is None or y < HUD.mag_y_min:
                        HUD.mag_y_min = int(y)
                    if HUD.mag_y_max is None or y > HUD.mag_y_max:
                        HUD.mag_y_max = int(y)
                    if HUD.mag_z_min is None or z < HUD.mag_z_min:
                        HUD.mag_z_min = int(z)
                    if HUD.mag_z_max is None or z > HUD.mag_z_max:
                        HUD.mag_z_max = int(z)

                    x_span = int(HUD.mag_x_max - HUD.mag_x_min) if (HUD.mag_x_max is not None and HUD.mag_x_min is not None) else 0
                    y_span = int(HUD.mag_y_max - HUD.mag_y_min) if (HUD.mag_y_max is not None and HUD.mag_y_min is not None) else 0
                    z_span = int(HUD.mag_z_max - HUD.mag_z_min) if (HUD.mag_z_max is not None and HUD.mag_z_min is not None) else 0

                    # Centered heading debug (hard-iron offset only, no soft-iron scale).
                    x_off = 0.5 * (HUD.mag_x_min + HUD.mag_x_max) if (HUD.mag_x_min is not None and HUD.mag_x_max is not None) else 0.0
                    y_off = 0.5 * (HUD.mag_y_min + HUD.mag_y_max) if (HUD.mag_y_min is not None and HUD.mag_y_max is not None) else 0.0
                    z_off = 0.5 * (HUD.mag_z_min + HUD.mag_z_max) if (HUD.mag_z_min is not None and HUD.mag_z_max is not None) else 0.0
                    x_cal = float(x) - x_off
                    y_cal = float(y) - y_off
                    z_cal = float(z) - z_off

                    def heading_from(a: float, b: float) -> float:
                        h = math.degrees(math.atan2(b, a))
                        if h < 0:
                            h += 360.0
                        return h

                    pair_scores = {
                        "XY": x_span + y_span,
                        "XZ": x_span + z_span,
                        "YZ": y_span + z_span,
                    }
                    best_pair = max(pair_scores, key=pair_scores.get)
                    if best_pair == "XZ":
                        heading_cal_deg = heading_from(x_cal, z_cal)
                    elif best_pair == "YZ":
                        heading_cal_deg = heading_from(y_cal, z_cal)
                    else:
                        heading_cal_deg = heading_from(x_cal, y_cal)

                    min_span = int(getattr(config, "MAG_CAL_MIN_SPAN", 100))
                    if best_pair == "XZ":
                        cal_ready = x_span >= min_span and z_span >= min_span
                    elif best_pair == "YZ":
                        cal_ready = y_span >= min_span and z_span >= min_span
                    else:
                        cal_ready = x_span >= min_span and y_span >= min_span
                    use_cal = bool(getattr(config, "MAG_APPLY_HARD_IRON_CAL", True)) and cal_ready
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
                    HUD.compass_heading_deg = heading_cal_deg if use_cal else heading_deg
                    HUD.compass_heading_valid = True
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

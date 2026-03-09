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

from ..state import HUD


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
    ) -> None:
        self.device = device
        self.baud = baud
        self.demo = demo
        self.use_i2c = use_i2c
        self.i2c_bus = i2c_bus
        self.i2c_addr = i2c_addr
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
                bus.write_byte_data(self.i2c_addr, 0x01, 0x20)  # config B (gain)
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
                    HUD.compass_heading_deg = heading_deg
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

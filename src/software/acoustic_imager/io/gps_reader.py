"""
GPS reader for BN-880 over UART.

Parses NMEA RMC/GGA lines to maintain cached GPS fields in HUD state.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

try:
    import serial
except ImportError:
    serial = None

from .. import config
from ..state import HUD


def _nmea_checksum_ok(line: str) -> bool:
    if "*" not in line or len(line) < 10:
        return False
    payload, _, checksum = line.strip().rpartition("*")
    if not payload.startswith("$") or len(checksum) != 2:
        return False
    try:
        expected = int(checksum, 16)
    except ValueError:
        return False
    x = 0
    for ch in payload[1:]:
        x ^= ord(ch)
    return x == expected


def _parse_deg_min(value: str, hemi: str, is_lon: bool) -> Optional[float]:
    if not value or not hemi:
        return None
    try:
        dlen = 3 if is_lon else 2
        deg = float(value[:dlen])
        mins = float(value[dlen:])
    except Exception:
        return None
    out = deg + mins / 60.0
    if hemi in ("S", "W"):
        out = -out
    return out


def _parse_rmc(parts: list[str]) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
    # $GPRMC,time,status,lat,N,lon,E,speed,course,date,...
    if len(parts) < 9:
        return None, None, None, False
    status = parts[2].strip().upper()
    lat = _parse_deg_min(parts[3], parts[4].strip().upper(), is_lon=False)
    lon = _parse_deg_min(parts[5], parts[6].strip().upper(), is_lon=True)
    course = None
    try:
        if parts[8]:
            course = float(parts[8]) % 360.0
    except ValueError:
        pass
    return lat, lon, course, status == "A"


def _parse_gga(parts: list[str]) -> Tuple[Optional[float], Optional[float], int, bool, Optional[float]]:
    # $GPGGA,time,lat,N,lon,E,fix,sats,...
    if len(parts) < 8:
        return None, None, 0, False, None
    lat = _parse_deg_min(parts[2], parts[3].strip().upper(), is_lon=False)
    lon = _parse_deg_min(parts[4], parts[5].strip().upper(), is_lon=True)
    try:
        fix_q = int(parts[6]) if parts[6] else 0
    except ValueError:
        fix_q = 0
    try:
        sats = int(parts[7]) if parts[7] else 0
    except ValueError:
        sats = 0
    hdop = None
    if len(parts) > 8 and parts[8]:
        try:
            hdop = float(parts[8])
        except ValueError:
            hdop = None
    return lat, lon, sats, fix_q > 0, hdop


class GPSReader:
    """Background UART reader for NMEA GPS fix state."""

    def __init__(self, device: str, baud: int, enabled: bool = True) -> None:
        self.device = device
        self.baud = baud
        self.enabled = enabled
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
        if not self.enabled or serial is None:
            return
        retry_sleep = 2.0
        while not self._stop.is_set():
            try:
                ser = serial.Serial(self.device, baudrate=self.baud, timeout=0.5)
                ser.reset_input_buffer()
            except Exception:
                HUD.gps_fix_valid = False
                time.sleep(retry_sleep)
                continue
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = ser.read(256)
                except Exception:
                    HUD.gps_fix_valid = False
                    break
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf or b"\r" in buf:
                    line, sep, buf = buf.partition(b"\n")
                    if not sep:
                        line, sep, buf = buf.partition(b"\r")
                    s = (line + sep).decode("ascii", errors="ignore").strip()
                    if not s.startswith("$") or not _nmea_checksum_ok(s):
                        continue
                    parts = s.split(",")
                    if not parts:
                        continue
                    talker = parts[0]
                    now_s = time.time()
                    if talker.endswith("RMC"):
                        lat, lon, course, ok = _parse_rmc(parts)
                        if lat is not None and lon is not None:
                            HUD.gps_lat = lat
                            HUD.gps_lon = lon
                            HUD.gps_last_update_s = now_s
                        if course is not None:
                            HUD.gps_course_deg = course
                        HUD.gps_fix_valid = bool(ok)
                        if ok and HUD.gps_accuracy_m is None:
                            HUD.gps_accuracy_m = float(getattr(config, "GPS_DEFAULT_ACCURACY_M", 30.0))
                    elif talker.endswith("GGA"):
                        lat, lon, sats, ok, hdop = _parse_gga(parts)
                        if lat is not None and lon is not None:
                            HUD.gps_lat = lat
                            HUD.gps_lon = lon
                            HUD.gps_last_update_s = now_s
                        HUD.gps_sat_count = sats
                        if hdop is not None:
                            # Simple HDOP->meters estimate (rule-of-thumb)
                            HUD.gps_accuracy_m = max(3.0, float(hdop) * 5.0)
                        elif ok and HUD.gps_accuracy_m is None:
                            HUD.gps_accuracy_m = float(getattr(config, "GPS_DEFAULT_ACCURACY_M", 30.0))
                        HUD.gps_fix_valid = bool(ok)
            try:
                ser.close()
            except Exception:
                pass
            HUD.gps_fix_valid = False
            if not self._stop.is_set():
                time.sleep(retry_sleep)

"""
Position fusion manager:
- Polls Wi-Fi geolocation in background.
- Fuses Wi-Fi and GPS estimates into one unified HUD.position object.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from .. import config
from ..state import HUD, button_state
from .wifi_scan import scan_wifi_networks
from .wifi_geolocation import geolocate_from_wifi

# #region agent log
_DEBUG_LOG_PATH = "/home/acousticgod/Capstone_490_Software/.cursor/debug-2bfbda.log"
_DEBUG_LOG_FALLBACK = "/tmp/debug-2bfbda.log"
_FUSE_LOG_LAST = 0.0
def _pos_dbg(m: str, d: dict, hid: str) -> None:
    import json
    line = json.dumps({"sessionId": "2bfbda", "timestamp": int(time.time() * 1000), "location": "position_manager", "message": m, "data": d, "hypothesisId": hid}) + "\n"
    for path in (_DEBUG_LOG_PATH, _DEBUG_LOG_FALLBACK):
        try:
            with open(path, "a") as f:
                f.write(line)
            break
        except Exception:
            continue
# #endregion


class PositionManager:
    def __init__(self) -> None:
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
        interval = max(2.0, float(getattr(config, "WIFI_GEO_INTERVAL_SEC", 12.0)))
        default_wifi_acc = float(getattr(config, "WIFI_DEFAULT_ACCURACY_M", 80.0))
        skip_log_count = 0
        while not self._stop.is_set():
            pos_enabled = getattr(button_state, "position_services_enabled", True)
            if not pos_enabled:
                skip_log_count += 1
                # #region agent log
                if skip_log_count == 1 or skip_log_count % 20 == 0:
                    _pos_dbg("worker_skip", {"position_services_enabled": False, "skip_count": skip_log_count}, "B")
                # #endregion
                time.sleep(0.5)
                continue
            skip_log_count = 0
            try:
                nets = scan_wifi_networks()
                geo = geolocate_from_wifi(nets)
                # #region agent log
                _pos_dbg("after_geo", {"geo_success": geo is not None}, "C")
                # #endregion
                if geo is not None:
                    lat, lon, acc, ts = geo
                    HUD.wifi_lat = float(lat)
                    HUD.wifi_lon = float(lon)
                    HUD.wifi_accuracy_m = max(1.0, float(acc))
                    HUD.wifi_last_update_s = float(ts)
                elif HUD.wifi_lat is not None and HUD.wifi_lon is not None and HUD.wifi_accuracy_m is None:
                    HUD.wifi_accuracy_m = default_wifi_acc
            except Exception:
                pass
            self._fuse_once()
            sleep_s = interval
            end_t = time.time() + sleep_s
            while not self._stop.is_set() and time.time() < end_t:
                time.sleep(0.2)
                self._fuse_once()

    def _fuse_once(self) -> None:
        stale_s = float(getattr(config, "POSITION_STALE_SEC", 30.0))
        hysteresis_m = float(getattr(config, "POSITION_SOURCE_HYSTERESIS_M", 8.0))
        now = time.time()

        # Candidate: Wi-Fi
        wifi_ok = (
            HUD.wifi_lat is not None
            and HUD.wifi_lon is not None
            and (now - float(HUD.wifi_last_update_s or 0.0)) <= stale_s
        )
        wifi_acc = float(HUD.wifi_accuracy_m) if HUD.wifi_accuracy_m is not None else float(getattr(config, "WIFI_DEFAULT_ACCURACY_M", 80.0))

        # Candidate: GPS
        gps_ok = (
            bool(HUD.gps_fix_valid)
            and HUD.gps_lat is not None
            and HUD.gps_lon is not None
            and (now - float(HUD.gps_last_update_s or 0.0)) <= stale_s
        )
        gps_acc = float(HUD.gps_accuracy_m) if HUD.gps_accuracy_m is not None else float(getattr(config, "GPS_DEFAULT_ACCURACY_M", 30.0))

        chosen = "none"
        if gps_ok and wifi_ok:
            current_source = HUD.position.source
            if current_source == "gps":
                # Stay on GPS unless Wi-Fi is clearly better
                chosen = "wifi" if (wifi_acc + hysteresis_m) < gps_acc else "gps"
            elif current_source == "wifi":
                # Switch to GPS only when clearly better
                chosen = "gps" if (gps_acc + hysteresis_m) < wifi_acc else "wifi"
            else:
                chosen = "gps" if gps_acc <= wifi_acc else "wifi"
        elif gps_ok:
            chosen = "gps"
        elif wifi_ok:
            chosen = "wifi"

        if chosen == "gps":
            HUD.position.lat = float(HUD.gps_lat) if HUD.gps_lat is not None else None
            HUD.position.lon = float(HUD.gps_lon) if HUD.gps_lon is not None else None
            HUD.position.accuracy_m = float(gps_acc)
            HUD.position.source = "gps"
            HUD.position.timestamp_s = float(HUD.gps_last_update_s or now)
        elif chosen == "wifi":
            HUD.position.lat = float(HUD.wifi_lat) if HUD.wifi_lat is not None else None
            HUD.position.lon = float(HUD.wifi_lon) if HUD.wifi_lon is not None else None
            HUD.position.accuracy_m = float(wifi_acc)
            HUD.position.source = "wifi"
            HUD.position.timestamp_s = float(HUD.wifi_last_update_s or now)
        else:
            HUD.position.source = "none"
            # #region agent log
            global _FUSE_LOG_LAST
            now_log = time.time()
            if now_log - _FUSE_LOG_LAST >= 5.0:
                _FUSE_LOG_LAST = now_log
                wifi_age = (now - float(HUD.wifi_last_update_s or 0.0)) if (HUD.wifi_lat is not None) else None
                _pos_dbg("fuse_none", {"wifi_ok": wifi_ok, "gps_ok": gps_ok, "wifi_lat_set": HUD.wifi_lat is not None, "wifi_age_s": round(wifi_age, 1) if wifi_age is not None else None}, "E")
            # #endregion

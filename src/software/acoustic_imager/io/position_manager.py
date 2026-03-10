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
        while not self._stop.is_set():
            if not getattr(button_state, "position_services_enabled", True):
                time.sleep(0.5)
                continue
            try:
                nets = scan_wifi_networks()
                geo = geolocate_from_wifi(nets)
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

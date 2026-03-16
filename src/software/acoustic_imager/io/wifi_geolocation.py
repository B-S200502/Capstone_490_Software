"""
Wi-Fi geolocation client using Google Geolocation API.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Optional, List, Dict, Any, Tuple

from .. import config


def _build_payload(networks: List[Dict[str, str]]) -> Dict[str, Any]:
    ap_list: List[Dict[str, Any]] = []
    for n in networks:
        bssid = (n.get("bssid") or "").strip().lower()
        if not bssid or len(bssid) != 17:
            continue
        try:
            rssi = int(float(n.get("rssi_dbm") or 0))
        except Exception:
            rssi = 0
        ap: Dict[str, Any] = {"macAddress": bssid, "signalStrength": rssi}
        freq = (n.get("frequency") or "").strip()
        if freq.isdigit():
            ap["frequency"] = int(freq)
        ap_list.append(ap)
    return {"considerIp": "true", "wifiAccessPoints": ap_list}


# #region agent log
_DEBUG_LOG_PATH = "/home/acousticgod/Capstone_490_Software/.cursor/debug-2bfbda.log"
_DEBUG_LOG_FALLBACK = "/tmp/debug-2bfbda.log"
def _pos_dbg(m: str, d: dict, hid: str) -> None:
    import json
    line = json.dumps({"sessionId": "2bfbda", "timestamp": int(time.time() * 1000), "location": "wifi_geolocation.py:geolocate_from_wifi", "message": m, "data": d, "hypothesisId": hid}) + "\n"
    for path in (_DEBUG_LOG_PATH, _DEBUG_LOG_FALLBACK):
        try:
            with open(path, "a") as f:
                f.write(line)
            break
        except Exception:
            continue
# #endregion


def geolocate_from_wifi(networks: List[Dict[str, str]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (lat, lon, accuracy_m, timestamp_s) or None.
    """
    api_key = (getattr(config, "WIFI_GEO_API_KEY", "") or "").strip()
    payload = _build_payload(networks)
    num_aps = len(payload["wifiAccessPoints"])
    # #region agent log
    _pos_dbg("geo_call", {"has_key": bool(api_key), "num_networks": len(networks), "num_aps": num_aps}, "A" if api_key else "C")
    # #endregion
    if not api_key:
        return None
    if not payload["wifiAccessPoints"]:
        # #region agent log
        _pos_dbg("geo_fail", {"reason": "no_aps"}, "D")
        # #endregion
        return None

    url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
    timeout = float(getattr(config, "WIFI_GEO_REQUEST_TIMEOUT_SEC", 3.0))
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        data = json.loads(raw.decode("utf-8", errors="ignore"))
        loc = data.get("location") or {}
        lat = float(loc.get("lat"))
        lon = float(loc.get("lng"))
        acc = float(data.get("accuracy"))
        # #region agent log
        _pos_dbg("geo_ok", {"lat": round(lat, 5), "lon": round(lon, 5)}, "C")
        # #endregion
        return lat, lon, acc, time.time()
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, json.JSONDecodeError, Exception) as e:
        # #region agent log
        _pos_dbg("geo_fail", {"reason": "api_error", "err_type": type(e).__name__}, "C")
        # #endregion
        return None

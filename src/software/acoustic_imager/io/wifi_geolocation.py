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


def geolocate_from_wifi(networks: List[Dict[str, str]]) -> Optional[Tuple[float, float, float, float]]:
    """
    Returns (lat, lon, accuracy_m, timestamp_s) or None.
    """
    api_key = (getattr(config, "WIFI_GEO_API_KEY", "") or "").strip()
    if not api_key:
        return None
    payload = _build_payload(networks)
    if not payload["wifiAccessPoints"]:
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
        return lat, lon, acc, time.time()
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, json.JSONDecodeError, Exception):
        return None

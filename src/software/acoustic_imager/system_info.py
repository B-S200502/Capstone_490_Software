"""
System network and device info for HUD (WiFi SSID, IP address, hostname).

Uses hostname for device name (e.g. Pi name like "acousticlord").
Caches results briefly to avoid subprocess on every frame.
"""

from __future__ import annotations

import socket
import subprocess
from typing import Tuple, Optional

_CACHE: Optional[Tuple[str, str, str]] = None
_CACHE_TICKS = 0
_CACHE_TTL = 30  # refresh every ~30 frames at 60fps ≈ 0.5s


def get_system_network_info(ticks: int = 0) -> Tuple[str, str, str]:
    """
    Return (wifi_ssid, ip_address, device_name).
    Device name is the system hostname (e.g. Pi name).
    Caches for a short time; pass a tick counter to invalidate periodically.
    """
    global _CACHE, _CACHE_TICKS
    if _CACHE is not None and (ticks - _CACHE_TICKS) < _CACHE_TTL:
        return _CACHE

    wifi = _get_wifi_ssid()
    ip = _get_primary_ip()
    hostname = _get_hostname()
    _CACHE = (wifi, ip, hostname)
    _CACHE_TICKS = ticks
    return _CACHE


def _get_hostname() -> str:
    """System hostname (e.g. 'acousticlord' on Pi)."""
    try:
        return socket.gethostname() or ""
    except Exception:
        return ""


def _get_primary_ip() -> str:
    """Primary IPv4 address (prefer non-loopback)."""
    try:
        # Prefer the IP used to reach the default route
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip or ""
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=1,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.split()[0].strip()
    except Exception:
        pass
    return ""


def _get_wifi_ssid() -> str:
    """Current WiFi SSID if connected (Linux/Raspberry Pi)."""
    # iwgetid -r returns just the SSID (common on Pi)
    for cmd in [["iwgetid", "-r"], ["iwgetid", "-r", "-s"]]:
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            if out.returncode == 0 and out.stdout:
                return out.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue
    # nmcli fallback (NetworkManager)
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if out.returncode == 0 and out.stdout:
            for line in out.stdout.strip().splitlines():
                if line.startswith("yes:"):
                    return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return ""

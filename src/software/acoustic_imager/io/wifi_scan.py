"""
WiFi network scanning for Raspberry Pi / Linux.
Uses nmcli (NetworkManager) for connect; scan uses nmcli then iw/iwlist.
Disconnect tries nmcli first, then ip link down/up (no nmcli) so it works on Pi.
SSIDs are normalized to ASCII apostrophe so "Basem's iPhone" works everywhere.
"""

from __future__ import annotations

import re
import subprocess
import time
from typing import List, Dict

# All common Unicode apostrophe/quote code points -> ASCII apostrophe (U+0027)
_APOSTROPHE_MAP = str.maketrans({
    "\u2019": "'",   # RIGHT SINGLE QUOTATION MARK (iOS, Word)
    "\u2018": "'",   # LEFT SINGLE QUOTATION MARK
    "\u02BC": "'",   # MODIFIER LETTER APOSTROPHE
    "\u02B9": "'",   # MODIFIER LETTER PRIME
    "\u0060": "'",   # GRAVE ACCENT (sometimes used as apostrophe)
    "\u00B4": "'",   # ACUTE ACCENT
    "\u2032": "'",   # PRIME
    "\u2039": "'",   # SINGLE LEFT-POINTING ANGLE QUOTATION MARK (abuse)
})
# Three-byte sequence when SSID was decoded wrong (UTF-8 bytes as Latin-1)
_THREE_BYTE_APOSTROPHE = "\xe2\x80\x99"
# Literal backslash-x hex as printed in errors (e.g. "Basem\xe2\x80\x99s iPhone" as 12 chars)
_LITERAL_APOSTROPHE_ESCAPE = "\\xe2\\x80\\x99"


def _scan_nmcli() -> List[Dict[str, str]]:
    """Scan using nmcli (NetworkManager)."""
    result: List[Dict[str, str]] = []
    try:
        out = subprocess.run(
            [
                "nmcli",
                "-t",
                "--separator", "|",
                "-f", "BSSID,SSID,SIGNAL,SECURITY,FREQ,CHAN",
                "device", "wifi", "list",
                "--rescan", "yes",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode != 0 or not out.stdout:
            return result

        seen = set()
        for line in out.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            bssid = (parts[0] or "").strip()
            ssid = (parts[1] or "").strip()
            signal = (parts[2] or "").strip()
            security = (parts[3] or "").strip()
            freq = (parts[4] or "").strip()
            channel = (parts[5] or "").strip()
            ssid_ascii = normalize_ssid(ssid)
            if not ssid_ascii or ssid_ascii in seen:
                continue
            seen.add(ssid_ascii)
            # Convert signal % (0..100) to rough dBm estimate for geolocation APIs.
            try:
                sig_pct = float(signal)
                rssi = int(round(-100.0 + (sig_pct / 100.0) * 70.0))
            except Exception:
                rssi = 0
            result.append({
                "ssid": ssid_ascii,
                "signal": signal,
                "security": security if security and security != "--" else "Open",
                "bssid": bssid,
                "rssi_dbm": str(rssi),
                "frequency": freq,
                "channel": channel,
            })
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return result


def _parse_iw(text: str) -> List[Dict[str, str]]:
    """Parse 'iw dev wlan0 scan' output into list of {ssid, signal, security}."""
    result: List[Dict[str, str]] = []
    seen: set[str] = set()
    current: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("BSS "):
            if current.get("ssid") and current["ssid"] not in seen:
                seen.add(current["ssid"])
                result.append(current)
            m = re.match(r"BSS\s+([0-9a-f:]{17})", line, flags=re.IGNORECASE)
            current = {
                "ssid": "",
                "signal": "",
                "security": "Open",
                "bssid": m.group(1).lower() if m else "",
                "rssi_dbm": "0",
                "frequency": "",
                "channel": "",
            }
            continue
        if not current:
            continue
        if line.startswith("SSID:"):
            current["ssid"] = normalize_ssid(line[5:].strip())
        elif line.startswith("freq:"):
            m = re.search(r"(\d+)", line)
            if m:
                current["frequency"] = m.group(1)
        elif line.startswith("DS Parameter set: channel"):
            m = re.search(r"channel\s+(\d+)", line)
            if m:
                current["channel"] = m.group(1)
        elif line.startswith("signal:"):
            m = re.search(r"(-?\d+(?:\.\d+)?)\s*dBm", line)
            if m:
                dbm = float(m.group(1))
                current["rssi_dbm"] = str(int(round(dbm)))
                pct = min(100, max(0, int(100 + (dbm + 30) * (100 / 60))))
                current["signal"] = str(pct)
        elif "RSN:" in line or "WPA" in line or "WPA2" in line:
            current["security"] = "WPA2"
        elif "WEP" in line:
            current["security"] = "WEP"
    if current.get("ssid") and current["ssid"] not in seen:
        seen.add(current["ssid"])
        result.append(current)
    return result


def _scan_iw() -> List[Dict[str, str]]:
    """Scan using 'iw dev wlan0 scan' - often returns more networks than nmcli on Pi."""
    for iface in ["wlan0", "wlan1"]:
        for cmd in [["iw", "dev", iface, "scan"], ["sudo", "iw", "dev", iface, "scan"]]:
            try:
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    encoding="utf-8",
                    errors="replace",
                )
                if out.returncode == 0 and out.stdout:
                    result = _parse_iw(out.stdout)
                    if result:
                        return result
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
    return []


def _scan_iwlist() -> List[Dict[str, str]]:
    """Scan using iwlist (fallback when nmcli unavailable or empty)."""
    result: List[Dict[str, str]] = []
    interfaces = ["wlan0", "wlan1"]
    for iface in interfaces:
        for cmd in [["iwlist", iface, "scan"], ["sudo", "iwlist", iface, "scan"]]:
            try:
                out = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    encoding="utf-8",
                    errors="replace",
                )
                if out.returncode != 0:
                    continue
                result = _parse_iwlist(out.stdout or "")
                if result:
                    return result
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                continue
    return []


def _parse_iwlist(text: str) -> List[Dict[str, str]]:
    """Parse iwlist scan output into list of {ssid, signal, security}."""
    result: List[Dict[str, str]] = []
    cells = re.split(r"Cell \d+ - Address:", text, flags=re.IGNORECASE)
    for cell in cells[1:]:
        ssid = ""
        signal = ""
        rssi = ""
        security = "Open"
        bssid = ""
        essid_match = re.search(r'ESSID:"([^"]*)"', cell)
        if essid_match:
            ssid = normalize_ssid(essid_match.group(1).strip())
        addr_match = re.search(r"Address:\s*([0-9A-Fa-f:]{17})", cell)
        if addr_match:
            bssid = addr_match.group(1).lower()
        quality_match = re.search(r"Quality=(\d+)/(\d+)", cell)
        if quality_match:
            num, den = int(quality_match.group(1)), int(quality_match.group(2))
            signal = str(int(100 * num / den)) if den else ""
        dbm_match = re.search(r"Signal level=(-?\d+)\s*dBm", cell)
        if dbm_match:
            rssi = dbm_match.group(1)
        if "Encryption key:on" in cell or "IE: WPA" in cell or "IE: IEEE 802.11i/WPA2" in cell:
            security = "WPA2"
        elif "Encryption key:on" in cell:
            security = "WEP"
        if ssid:
            result.append(
                {
                    "ssid": ssid,
                    "signal": signal,
                    "security": security,
                    "bssid": bssid,
                    "rssi_dbm": rssi or "0",
                }
            )
    return result


def scan_wifi_networks() -> List[Dict[str, str]]:
    """
    Scan for nearby WiFi networks.
    Tries nmcli first, then iw (often returns more networks on Pi), then iwlist.
    On Raspberry Pi, nmcli often returns only the connected network; iw/sudo iw
    typically returns all visible networks.
    Returns list of {"ssid", "signal", "security", ...} dicts. SSIDs are normalized
    (Unicode apostrophe -> ASCII) for reliable matching and connect.
    """
    result = _scan_nmcli()
    if not result:
        result = _scan_nmcli_iface()
    # nmcli often returns only the connected network on Pi; try iw for full scan
    if len(result) <= 1:
        iw_result = _scan_iw()
        if len(iw_result) > len(result):
            result = iw_result
    if not result:
        result = _scan_iwlist()
    # Normalize SSIDs so "Basem's iPhone" (Unicode apostrophe) matches and connects reliably
    for entry in result:
        if "ssid" in entry and entry["ssid"]:
            entry["ssid"] = normalize_ssid(entry["ssid"])
    return result


def _scan_nmcli_iface() -> List[Dict[str, str]]:
    """Try nmcli with explicit wlan0 interface (some Pi setups need this)."""
    try:
        out = subprocess.run(
            [
                "nmcli", "-t", "--separator", "|", "-f", "BSSID,SSID,SIGNAL,SECURITY,FREQ,CHAN",
                "device", "wifi", "list", "ifname", "wlan0", "--rescan", "yes",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode != 0 or not out.stdout:
            return []
        result = []
        seen = set()
        for line in (out.stdout or "").strip().splitlines():
            parts = line.split("|")
            if len(parts) < 6:
                continue
            bssid = (parts[0] or "").strip()
            ssid = (parts[1] or "").strip()
            signal = (parts[2] or "").strip()
            security = (parts[3] or "").strip()
            freq = (parts[4] or "").strip()
            channel = (parts[5] or "").strip()
            ssid_ascii = normalize_ssid(ssid)
            if not ssid_ascii or ssid_ascii in seen:
                continue
            seen.add(ssid_ascii)
            try:
                sig_pct = float(signal)
                rssi = int(round(-100.0 + (sig_pct / 100.0) * 70.0))
            except Exception:
                rssi = 0
            result.append(
                {
                    "ssid": ssid_ascii,
                    "signal": signal,
                    "security": security if security and security != "--" else "Open",
                    "bssid": bssid,
                    "rssi_dbm": str(rssi),
                    "frequency": freq,
                    "channel": channel,
                }
            )
        return result
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return []


def _is_full_bssid(bssid: str | None) -> bool:
    """True if bssid looks like a full 6-octet MAC (e.g. AA:BB:CC:DD:EE:FF)."""
    if not bssid or not bssid.strip():
        return False
    s = bssid.strip()
    return bool(re.match(r"^[0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5}$", s))


def normalize_ssid(ssid: str) -> str:
    """
    Normalize SSID to ASCII: all apostrophe-like/smart-quote chars -> ASCII apostrophe (').
    Handles: U+2019, three-byte 0xe2/0x80/0x99, and literal string "\\xe2\\x80\\x99".
    Use for display, comparison, and connect so "Basem's iPhone" works everywhere.
    """
    if not ssid:
        return ssid
    # Literal backslash-x hex (e.g. from nmcli error or wrong encoding)
    s = ssid.replace(_LITERAL_APOSTROPHE_ESCAPE, "'")
    s = s.translate(_APOSTROPHE_MAP)
    s = s.replace(_THREE_BYTE_APOSTROPHE, "'")
    return s


def _normalize_ssid_for_connect(ssid: str) -> str:
    """Alias for connect path; use normalize_ssid for consistency."""
    return normalize_ssid(ssid)


def _rescan_wifi() -> None:
    """Rescan so the AP list is fresh before connect. Tries with ifname for Pi."""
    for args in [
        ["nmcli", "device", "wifi", "rescan", "ifname", "wlan0"],
        ["nmcli", "device", "wifi", "rescan"],
    ]:
        try:
            out = subprocess.run(
                args,
                capture_output=True,
                timeout=10,
                encoding="utf-8",
                errors="replace",
            )
            if out.returncode == 0:
                break
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            pass


def _get_saved_connection_for_ssid(normalized_ssid: str) -> str | None:
    """
    Return the saved NetworkManager connection name (id) for the given SSID, or None.
    Used to reconnect via 'nmcli connection up' so previously saved credentials are used.
    """
    if not normalized_ssid:
        return None
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-g", "NAME", "connection", "show"],
            capture_output=True,
            text=True,
            timeout=5,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode != 0 or not out.stdout:
            return None
        for name in out.stdout.strip().splitlines():
            name = (name or "").strip()
            if not name:
                continue
            out2 = subprocess.run(
                ["nmcli", "-t", "-g", "802-11-wireless.ssid", "connection", "show", name],
                capture_output=True,
                text=True,
                timeout=2,
                encoding="utf-8",
                errors="replace",
            )
            if out2.returncode != 0:
                continue
            saved_ssid = (out2.stdout or "").strip()
            if normalize_ssid(saved_ssid) == normalized_ssid:
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def connect_wifi(ssid: str, password: str, bssid: str | None = None) -> tuple[bool, str]:
    """
    Connect to WiFi. For a network the user has connected to before (saved profile),
    uses only 'nmcli connection up' with retries so we don't hit "No network with SSID"
    from device wifi connect (nmcli device wifi list often omits networks on Pi).
    For first-time connect, uses nmcli device wifi connect with password.
    """
    connect_timeout = 35

    def run(cmd_args: list) -> tuple[bool, str]:
        out = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            timeout=connect_timeout,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode == 0:
            return True, "Connected"
        err = (out.stderr or out.stdout or "").strip()
        return False, err[:120] if err else "Connection failed"

    try:
        normalized = normalize_ssid(ssid)
        saved = _get_saved_connection_for_ssid(normalized)

        if saved:
            # Reconnect: use only connection up (no device wifi connect fallback).
            # Retry with rescan so the AP is visible; ifname wlan0 helps on Pi.
            last_msg = "Connection failed"
            for attempt in range(2):
                _rescan_wifi()
                time.sleep(2)
                ok, msg = run(["nmcli", "-w", "30", "connection", "up", saved, "ifname", "wlan0"])
                if ok:
                    return True, msg
                last_msg = msg
                # If first attempt failed, try without ifname (some NM versions)
                if not ok and attempt == 0:
                    ok, msg = run(["nmcli", "-w", "30", "connection", "up", saved])
                    if ok:
                        return True, msg
                    last_msg = msg
            return False, last_msg

        # First-time connect (e.g. hotspot): device wifi connect with password; ifname helps on Pi.
        _rescan_wifi()
        time.sleep(2)
        args = ["nmcli", "-w", "30", "device", "wifi", "connect", normalized, "ifname", "wlan0"]
        if password:
            args += ["password", password]
        if bssid and _is_full_bssid(bssid):
            args += ["bssid", bssid.strip()]
        ok, msg = run(args)
        if not ok:
            # Retry without ifname in case this NM version doesn't support it for device wifi connect
            args_retry = ["nmcli", "-w", "30", "device", "wifi", "connect", normalized]
            if password:
                args_retry += ["password", password]
            if bssid and _is_full_bssid(bssid):
                args_retry += ["bssid", bssid.strip()]
            ok2, msg2 = run(args_retry)
            if ok2:
                return True, msg2
            msg = msg2
        return (True, msg) if ok else (False, msg)
    except subprocess.TimeoutExpired:
        return False, "Connection timed out"
    except FileNotFoundError:
        return False, "nmcli not found"
    except Exception as e:
        return False, str(e)[:80]


def _get_active_wifi_connection_name() -> str | None:
    """Return the active connection name for wlan0, or None."""
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-g", "GENERAL.CONNECTION", "device", "show", "wlan0"],
            capture_output=True,
            text=True,
            timeout=3,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode == 0 and out.stdout:
            name = out.stdout.strip()
            if name and name != "--":
                return name
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return None


def _run_cmd(args: list, timeout: int = 8) -> bool:
    """Run a command; return True if returncode is 0."""
    try:
        out = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        return out.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def _is_wifi_connected() -> bool:
    """True if the system reports an active WiFi connection (so we can verify disconnect)."""
    for cmd in [["iwgetid", "-r"], ["iwgetid", "-r", "-s"]]:
        try:
            out = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2,
                encoding="utf-8",
                errors="replace",
            )
            if out.returncode == 0 and (out.stdout or "").strip():
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            continue
    try:
        out = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            timeout=2,
            encoding="utf-8",
            errors="replace",
        )
        if out.returncode == 0 and out.stdout:
            for line in out.stdout.strip().splitlines():
                if line.startswith("yes:"):
                    return True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return False


def disconnect_wifi() -> tuple[bool, str]:
    """
    Disconnect from current WiFi. Returns (success, message).
    Tries several methods and only reports success when the Pi is actually disconnected
    (verified with iwgetid/nmcli). So the app really controls WiFi state.
    Order: wpa_cli, nmcli device disconnect, nmcli connection down, ip link down/up.
    """
    if not _is_wifi_connected():
        return True, "Disconnected"

    def try_disconnect() -> bool:
        if not _is_wifi_connected():
            return True
        # 1) wpa_cli: tell wpa_supplicant to disconnect (no sudo, often works on Pi)
        for iface in ["wlan0", "wlan1"]:
            if _run_cmd(["wpa_cli", "-i", iface, "disconnect"], timeout=5):
                time.sleep(2)
                if not _is_wifi_connected():
                    return True
        # 2) nmcli device disconnect
        if _run_cmd(["nmcli", "device", "disconnect", "ifname", "wlan0"]):
            time.sleep(2)
            if not _is_wifi_connected():
                return True
        if _run_cmd(["nmcli", "device", "disconnect"]):
            time.sleep(2)
            if not _is_wifi_connected():
                return True
        # 3) nmcli connection down by name
        conn = _get_active_wifi_connection_name()
        if conn and _run_cmd(["nmcli", "connection", "down", conn]):
            time.sleep(2)
            if not _is_wifi_connected():
                return True
        # 4) ip link down then up (requires sudo)
        for iface in ["wlan0", "wlan1"]:
            if _run_cmd(["sudo", "ip", "link", "set", iface, "down"], timeout=5):
                time.sleep(2)
                _run_cmd(["sudo", "ip", "link", "set", iface, "up"], timeout=5)
                time.sleep(1)
                if not _is_wifi_connected():
                    return True
        return False

    if try_disconnect():
        return True, "Disconnected"
    return False, "Disconnect failed"

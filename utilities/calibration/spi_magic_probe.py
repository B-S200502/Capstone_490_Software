#!/usr/bin/env python3
"""
Probe SPI with different mode and bit-order settings; report which (if any) see magic.

Firmware (spi.c): CPOL=0, CPHA=0, FirstBit=MSB -> SPI mode 0, MSB first.
Pi (config.py): SPI_MODE=0; spidev lsb_first not set -> kernel default MSB first.
See utilities/debug/SPI_settings.md for full reference.

Usage (from repo root, acoustic app stopped):
  python3 utilities/debug/spi_magic_probe.py
  python3 utilities/debug/spi_magic_probe.py --no-sync
  python3 utilities/debug/spi_magic_probe.py --mode 0,1
"""
import argparse
import os
import subprocess
import sys
import time
from typing import Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_SRC_SOFTWARE = os.path.join(_REPO_ROOT, "src", "software")
if _SRC_SOFTWARE not in sys.path:
    sys.path.insert(0, _SRC_SOFTWARE)

try:
    from acoustic_imager import config
except ImportError:
    config = None

# 0xAABBCCDD: LE bytes dd cc bb aa; BE bytes aa bb cc dd
MAGIC_LE = bytes([0xDD, 0xCC, 0xBB, 0xAA])
MAGIC_BE = bytes([0xAA, 0xBB, 0xCC, 0xDD])


def _reverse_bits_byte(b: int) -> int:
    out = 0
    for _ in range(8):
        out = (out << 1) | (b & 1)
        b >>= 1
    return out & 0xFF


# Bit-reversed magic LE: reverse bits of each of 0xDD,0xCC,0xBB,0xAA
MAGIC_LE_BITREV = bytes(_reverse_bits_byte(b) for b in MAGIC_LE)


def get_frame_size() -> int:
    if config is not None and hasattr(config, "SPI_FRAME_PACKET_SIZE_BYTES"):
        return int(config.SPI_FRAME_PACKET_SIZE_BYTES)
    return 32801


def read_pin_pinctrl(bcm: int) -> Optional[str]:
    out = subprocess.run(
        ["pinctrl", "get", str(bcm)],
        capture_output=True,
        text=True,
        timeout=1,
    )
    if out.returncode != 0 or not out.stdout:
        return None
    for line in out.stdout.strip().splitlines():
        if "|" in line:
            part = line.split("|", 1)[1].strip().split()
            if part and part[0].lower() == "hi":
                return "hi"
            if part and part[0].lower() == "lo":
                return "lo"
    return None


def wait_for_mcu_status_high_pinctrl(bcm: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if read_pin_pinctrl(bcm) == "hi":
            return True
        time.sleep(0.01)
    return False


def find_all(data: bytes, pattern: bytes) -> list[int]:
    out = []
    start = 0
    while True:
        i = data.find(pattern, start)
        if i == -1:
            break
        out.append(i)
        start = i + 1
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe SPI mode/bit-order for magic header")
    parser.add_argument("--no-sync", action="store_true", help="Skip pinctrl wait (no MCU sync)")
    parser.add_argument("--mode", type=str, default="0,1,2,3", help="Comma-separated SPI modes (default 0,1,2,3)")
    args = parser.parse_args()

    try:
        import spidev
    except ImportError:
        print("spidev not installed")
        sys.exit(1)

    if config is None:
        print("acoustic_imager.config not available")
        sys.exit(1)

    modes = [int(m.strip()) for m in args.mode.split(",") if m.strip()]
    if not modes:
        modes = [0, 1, 2, 3]

    path = f"/dev/spidev{getattr(config, 'SPI_BUS', 1)}.{getattr(config, 'SPI_DEV', 2)}"
    frame_size = get_frame_size()
    bus = getattr(config, "SPI_BUS", 1)
    dev = getattr(config, "SPI_DEV", 2)
    speed = getattr(config, "SPI_MAX_SPEED_HZ", 20_000_000)
    bcm_pin = int(getattr(config, "FRAME_READY_BCM_PIN", 7))
    timeout_s = float(getattr(config, "FRAME_READY_TIMEOUT_S", 0.25))
    use_sync = not args.no_sync

    if not os.path.exists(path):
        print(f"SPI device {path} not found")
        sys.exit(1)

    print(f"Probing {path}, frame_size={frame_size}, sync={use_sync}")
    print("Magic LE: dd cc bb aa  |  BE: aa bb cc dd  |  bit_rev LE:", MAGIC_LE_BITREV.hex())
    print("-" * 60)

    results = []
    for mode in modes:
        for lsb_first in (False, True):
            s = None
            try:
                s = spidev.SpiDev()
                s.open(bus, dev)
                s.max_speed_hz = speed
                s.mode = mode
                try:
                    s.lsbfirst = lsb_first
                except (OSError, IOError) as e:
                    if getattr(e, "errno", None) == 22:
                        results.append((mode, lsb_first, None, "lsbfirst not supported"))
                        continue
                    raise
                if use_sync:
                    wait_for_mcu_status_high_pinctrl(bcm_pin, timeout_s)
                tx = bytes(frame_size)
                rx = bytes(s.xfer3(tx))
                off_le = find_all(rx, MAGIC_LE)
                off_be = find_all(rx, MAGIC_BE)
                off_rev = find_all(rx, MAGIC_LE_BITREV)
                results.append((mode, lsb_first, (off_le, off_be, off_rev), None))
            except Exception as e:
                results.append((mode, lsb_first, None, str(e)))
            finally:
                if s is not None:
                    try:
                        s.close()
                    except Exception:
                        pass

    for mode, lsb_first, hits, err in results:
        label = f"mode={mode} lsbfirst={lsb_first}"
        if err is not None:
            print(f"{label}: {err}")
            continue
        off_le, off_be, off_rev = hits
        parts = []
        if off_le:
            parts.append(f"magic@[{','.join(f'0x{o:x}' for o in off_le[:5])}{'...' if len(off_le) > 5 else ''}]")
        else:
            parts.append("magic no")
        if off_be:
            parts.append(f"BE@[{','.join(f'0x{o:x}' for o in off_be[:3])}]")
        else:
            parts.append("BE no")
        if off_rev:
            parts.append(f"bit_rev@[{','.join(f'0x{o:x}' for o in off_rev[:5])}]")
        else:
            parts.append("bit_rev no")
        print(f"{label}: {' '.join(parts)}")

    any_hit = any(
        r[2] is not None and (r[2][0] or r[2][1] or r[2][2])
        for r in results
    )
    print("-" * 60)
    if any_hit:
        print("At least one setting saw magic (LE/BE/bit_rev). Consider aligning config to that.")
    else:
        print("No setting saw magic or bit_rev. Check firmware phase/FirstBit or timing.")


if __name__ == "__main__":
    main()

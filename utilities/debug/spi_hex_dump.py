#!/usr/bin/env python3
"""
SPI read synced to MCU_STATUS (frame-ready): 3 frames, hex dump to file.
Use to confirm whether the magic header 0xAABBCCDD (dd cc bb aa in LE) appears.
Run from repo root with acoustic app stopped: python3 utilities/debug/spi_hex_dump.py
Uses pinctrl for MCU_STATUS when RPi.GPIO/gpiozero are unavailable (same as pin_monitor).
"""
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

BYTES_PER_LINE = 16  # 16 bytes -> 32 hex chars + spaces, fits in 80-col terminal
NUM_FRAMES = 3
MAGIC_LE = bytes([0xDD, 0xCC, 0xBB, 0xAA])  # 0xAABBCCDD little-endian
OUTPUT_FILENAME = "spi_hex_dump.txt"


def get_spi_path():
    if config is not None and hasattr(config, "SPI_BUS") and hasattr(config, "SPI_DEV"):
        return f"/dev/spidev{config.SPI_BUS}.{config.SPI_DEV}"
    return "/dev/spidev1.2"


def get_frame_size():
    if config is not None and hasattr(config, "SPI_FRAME_PACKET_SIZE_BYTES"):
        return int(config.SPI_FRAME_PACKET_SIZE_BYTES)
    return 32801


def read_pin_pinctrl(bcm: int) -> Optional[str]:
    """Read pin state via pinctrl (same as pin_monitor). Returns 'hi', 'lo', or None."""
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
    """Poll pinctrl until pin is 'hi' or timeout. Returns True if saw hi."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if read_pin_pinctrl(bcm) == "hi":
            return True
        time.sleep(0.01)
    return False


def detect_magic_in_buffer(data: bytes, bytes_per_line: int = BYTES_PER_LINE):
    """
    Find all occurrences of magic 0xAABBCCDD (LE: dd cc bb aa) in data.
    Returns list of byte offsets (int).
    """
    if len(data) < 4:
        return []
    out = []
    offset = 0
    while True:
        i = data.find(MAGIC_LE, offset)
        if i == -1:
            break
        out.append(i)
        offset = i + 1
    return out


def main():
    try:
        import spidev
    except ImportError:
        print("spidev not installed")
        sys.exit(1)

    if config is None:
        print("acoustic_imager.config not available")
        sys.exit(1)

    try:
        from acoustic_imager.spi.frame_ready import FrameReadyGPIO
    except Exception as e:
        print(f"FrameReadyGPIO import failed: {e}")
        sys.exit(1)

    path = get_spi_path()
    frame_size = get_frame_size()
    bus = getattr(config, "SPI_BUS", 1)
    dev = getattr(config, "SPI_DEV", 2)
    speed = getattr(config, "SPI_MAX_SPEED_HZ", 20_000_000)
    mode = getattr(config, "SPI_MODE", 0)
    bcm_pin = int(getattr(config, "FRAME_READY_BCM_PIN", 7))
    pull = getattr(config, "FRAME_READY_PULL", "down")
    timeout_s = float(getattr(config, "FRAME_READY_TIMEOUT_S", 0.25))

    if not os.path.exists(path):
        print(f"SPI device {path} not found")
        sys.exit(1)

    frame_ready = None
    use_pinctrl_sync = False
    try:
        frame_ready = FrameReadyGPIO(bcm_pin=bcm_pin, pull=pull)
    except Exception as e:
        print(f"MCU_STATUS (GPIO) unavailable: {e}")
        print("Using pinctrl for frame sync (same as pin_monitor).")
        use_pinctrl_sync = True

    try:
        s = spidev.SpiDev()
        s.open(bus, dev)
        s.max_speed_hz = speed
        s.mode = mode
        try:
            frames = []  # list of (bytes, timed_out: bool)
            for i in range(NUM_FRAMES):
                if frame_ready is not None:
                    frame_ready.clear()
                    got = frame_ready.wait(timeout=timeout_s)
                elif use_pinctrl_sync:
                    got = wait_for_mcu_status_high_pinctrl(bcm_pin, timeout_s)
                else:
                    got = False
                tx = bytes(frame_size)
                rx = s.xfer3(tx)
                frames.append((bytes(rx), not got))
        finally:
            s.close()
    finally:
        if frame_ready is not None:
            frame_ready.close()

    # Build output file: header + 3 sections
    if frame_ready is not None:
        sync_note = f"synced to MCU_STATUS (BCM{bcm_pin})"
    elif use_pinctrl_sync:
        sync_note = f"synced via pinctrl (BCM{bcm_pin})"
    else:
        sync_note = "NOT synced (MCU_STATUS unavailable)"
    lines = [
        f"{NUM_FRAMES} frames, {sync_note}, {frame_size} bytes each",
        "Magic 0xAABBCCDD in little-endian hex: dd cc bb aa",
        "",
    ]
    for idx, (rx, timed_out) in enumerate(frames, start=1):
        section_header = f"=== Frame {idx} ==="
        if not frame_ready and not use_pinctrl_sync:
            section_header += " (no sync)"
        elif timed_out:
            section_header += " (frame_ready timeout)"
        lines.append(section_header)
        for offset in range(0, len(rx), BYTES_PER_LINE):
            chunk = rx[offset : offset + BYTES_PER_LINE]
            hex_str = " ".join(f"{b:02x}" for b in chunk)
            lines.append(f"{offset:05x}  {hex_str}")
        lines.append("")

    out_path = os.path.join(_SCRIPT_DIR, OUTPUT_FILENAME)
    with open(out_path, "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    total_bytes = sum(len(rx) for rx, _ in frames)
    print(f"Wrote {total_bytes} bytes ({NUM_FRAMES} frames) to {out_path}")

    # Magic detection on concatenated buffer
    concatenated = b"".join(rx for rx, _ in frames)
    matches = detect_magic_in_buffer(concatenated, BYTES_PER_LINE)
    if not matches:
        print("Magic header (0xAABBCCDD): No — not found")
    else:
        print(f"Magic header (0xAABBCCDD): Yes — found {len(matches)} time(s)")
        for byte_off in matches:
            frame_no = byte_off // frame_size + 1
            offset_in_frame = byte_off % frame_size
            line_in_section = (offset_in_frame // BYTES_PER_LINE) + 1
            print(f"  frame {frame_no}, byte offset 0x{offset_in_frame:05x}, line {line_in_section}")


if __name__ == "__main__":
    main()

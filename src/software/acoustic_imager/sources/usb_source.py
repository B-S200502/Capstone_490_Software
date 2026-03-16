# acoustic_imager/sources/usb_source.py
from __future__ import annotations

import re
import struct
import threading
import time
from dataclasses import replace
from typing import Optional

import numpy as np

from acoustic_imager import config
from acoustic_imager.custom_types import LatestFrame, SourceStats

try:
    import serial  # pyserial
except Exception as e:
    serial = None


MAGIC = 0xAABBCCDD
HEADER_LEN = 28
HEX_RE = re.compile(r"\b[0-9A-Fa-f]{2}\b")


def _bytes_from_hex_lines(lines) -> bytes:
    out = bytearray()
    for ln in lines:
        out.extend(int(x, 16) for x in HEX_RE.findall(ln))
    return bytes(out)


def _parse_header(h: bytes):
    # SPI_FrameHeader_t packed, little-endian
    # <IHHIHHIHHI => 28 bytes
    return struct.unpack("<IHHIHHIHHI", h)


class USBSource:
    """
    USB CDC reader that watches the STM32 text output containing:
      "Hex dump (first 64 bytes):"
      followed by 4 lines of hex bytes

    It converts that into fft_data shaped (N_MICS, N_BINS) complex64
    and returns it via LatestFrame, same as your SPI sources.
    """

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> None:
        self.port = port
        self.baud = int(baud)

        self._thr: Optional[threading.Thread] = None
        self._stop = threading.Event()

        # mailbox (latest frame)
        z = np.zeros((config.N_MICS, config.N_BINS), dtype=np.complex64)
        self._latest: LatestFrame = LatestFrame(ok=False, fft_data=z, stats=SourceStats())

    def start(self) -> None:
        if self._thr and self._thr.is_alive():
            return
        self._stop.clear()
        self._thr = threading.Thread(target=self._worker, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thr:
            self._thr.join(timeout=1.0)
        self._thr = None

    def get_latest(self) -> LatestFrame:
        return self._latest

    def _worker(self) -> None:
        if serial is None:
            self._latest = replace(
                self._latest,
                ok=False,
                stats=replace(self._latest.stats, last_err="pyserial not available"),
            )
            return

        stats = self._latest.stats
        last_frame = self._latest

        try:
            ser = serial.Serial(self.port, baudrate=self.baud, timeout=0.25)
            ser.reset_input_buffer()
        except Exception as e:
            self._latest = replace(
                last_frame,
                ok=False,
                stats=replace(stats, last_err=f"open {self.port} failed: {e}"),
            )
            return

        collecting = False
        hex_lines = []

        # For demo: map the first bins we see into these FFT bin indices.
        # If your config has SPI_SIM_BINS, we’ll use those; else just 0..7.
        try:
            demo_bins = list(getattr(config, "SPI_SIM_BINS"))
        except Exception:
            demo_bins = list(range(8))
        if len(demo_bins) < 8:
            demo_bins = (demo_bins + list(range(8)))[:8]
        demo_bins = demo_bins[:8]

        try:
            while not self._stop.is_set():
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if "Hex dump (first 64 bytes)" in line:
                    collecting = True
                    hex_lines = []
                    continue

                if collecting:
                    if HEX_RE.search(line):
                        hex_lines.append(line)

                    if len(hex_lines) >= 4:
                        collecting = False
                        blob = _bytes_from_hex_lines(hex_lines[:4])
                        if len(blob) < 64:
                            stats = replace(stats, bad_parse=stats.bad_parse + 1, last_err="short hex dump")
                            continue

                        header = blob[:HEADER_LEN]
                        try:
                            (
                                magic, version, header_len, frame_counter, mic_count, fft_size,
                                sample_rate, bin_count, reserved_adc, payload_len
                            ) = _parse_header(header)
                        except Exception:
                            stats = replace(stats, bad_parse=stats.bad_parse + 1, last_err="header unpack failed")
                            continue

                        if magic != MAGIC or header_len != HEADER_LEN:
                            stats = replace(stats, bad_parse=stats.bad_parse + 1, last_err="bad magic/header_len")
                            continue

                        # We only have 64 bytes total, so payload is partial here (36 bytes max).
                        payload = blob[HEADER_LEN:64]

                        # Build an fft_data array of expected shape for main.py
                        fft = np.zeros((config.N_MICS, config.N_BINS), dtype=np.complex64)

                        # If unit test is mics=4 and bins=8, payload begins with (re,im) float pairs.
                        # We’ll decode as many complex bins as we can from payload.
                        n_complex_avail = len(payload) // 8  # each complex is 2 float32 = 8 bytes
                        n_complex = min(8, n_complex_avail)

                        # Fill mic 0 with the decoded bins (so UI has something coherent)
                        for k in range(n_complex):
                            re_f, im_f = struct.unpack("<ff", payload[k * 8:(k + 1) * 8])
                            b = demo_bins[k]
                            if 0 <= b < config.N_BINS:
                                fft[0, b] = np.complex64(re_f + 1j * im_f)

                        # Optional: copy mic0 to other mics so beamforming doesn’t go completely flat
                        # (for demo only; remove once real multi-mic data streams)
                        for m in range(1, min(config.N_MICS, 4)):
                            fft[m, :] = fft[0, :]

                        stats = replace(
                            stats,
                            frames_ok=stats.frames_ok + 1,
                            last_err="",
                        )

                        self._latest = LatestFrame(ok=True, fft_data=fft, stats=stats)

        except Exception as e:
            stats = replace(stats, last_err=f"usb worker error: {e}")
            self._latest = LatestFrame(ok=False, fft_data=last_frame.fft_data, stats=stats)
        finally:
            try:
                ser.close()
            except Exception:
                pass

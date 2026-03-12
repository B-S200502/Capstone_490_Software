#!/usr/bin/env python3
"""
spi_manager.py

Manages SPI acquisition for the Acoustic Imager (loopback or real device).

Responsibilities:
- Defines the SPI frame format (header/payload/trailer) and validation helpers.
- Generates loopback frames (mag/phase) when operating without an STM32.
- Provides a SpiFFTSource reader that produces FFTFrame-like data (complex64 bins).
- Runs SPI reading in a background worker thread and exposes the latest fft_data + stats.
"""

from __future__ import annotations

import logging
import time
import struct
import zlib
import threading
from dataclasses import replace
from typing import Optional

import numpy as np

from acoustic_imager.custom_types import LatestFrame, SourceStats
from acoustic_imager.spi.frame_ready import FrameReady, FrameReadyGPIO
from acoustic_imager.spi.spi_protocol import SPIProtocol

log = logging.getLogger(__name__)

try:
    import spidev
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None
    SPIDEV_AVAILABLE = False

from acoustic_imager import config

def _cfg(name: str):
    if not hasattr(config, name):
        raise AttributeError(
            f"config.{name} is missing. spi_manager expects it. "
            f"Add `{name} = ...` to config.py"
        )
    return getattr(config, name)


class SPIManager:
    """
    Threaded SPI reader. Stores "latest FFT frame" in a mailbox (LatestFrame).
    Supports your STM32 framing format: header + payload(mag/phase float32) + crc32 + end magic.

    Public API:
      - start()
      - stop()
      - get_latest() -> LatestFrame (safe copy)
    """

    def __init__(
        self,
        frame_ready: Optional[FrameReady] = None,
        use_frame_ready: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._latest = LatestFrame()

        self._stop = False
        self._thread: Optional[threading.Thread] = None

        self._spi = None
        self._seq_seen = 0

        # --- framing constants from config ---
        self.N_MICS = int(_cfg("N_MICS"))
        self.SAMPLES_PER_CHANNEL = int(_cfg("SAMPLES_PER_CHANNEL"))
        self.SAMPLE_RATE_HZ = int(_cfg("SAMPLE_RATE_HZ"))
        self.N_BINS = int(_cfg("N_BINS"))

        self.MAGIC_START = int(_cfg("MAGIC_START"))
        self.MAGIC_END = int(_cfg("MAGIC_END"))
        self.VERSION = int(_cfg("VERSION"))

        self.HEADER_FMT = str(_cfg("HEADER_FMT"))
        self.HEADER_LEN = int(_cfg("HEADER_LEN"))
        self.TRAILER_FMT = str(_cfg("TRAILER_FMT"))
        self.TRAILER_LEN = int(_cfg("TRAILER_LEN"))
        self.PAYLOAD_LEN = int(_cfg("PAYLOAD_LEN"))
        self.FRAME_BYTES = int(_cfg("FRAME_BYTES"))
        self.SPI_MIC_PACKET_BYTES = int(getattr(config, "SPI_MIC_PACKET_BYTES", 2081))
        self.SPI_FRAME_PACKET_SIZE_BYTES = int(getattr(config, "SPI_FRAME_PACKET_SIZE_BYTES", 32801))

        # --- spi config ---
        self.SPI_BUS = int(_cfg("SPI_BUS"))
        self.SPI_DEV = int(_cfg("SPI_DEV"))
        self.SPI_MODE = int(_cfg("SPI_MODE"))
        self.SPI_BITS = int(_cfg("SPI_BITS"))
        self.SPI_MAX_SPEED_HZ = int(_cfg("SPI_MAX_SPEED_HZ"))
        self.SPI_XFER_CHUNK = int(_cfg("SPI_XFER_CHUNK"))

        # how often to CRC check (0 = never)
        self.CRC_EVERY_N = int(getattr(config, "CRC_EVERY_N", 30))

        # --- frame-ready (GPIO IRQ) ---
        self.use_frame_ready = bool(use_frame_ready)
        self._frame_ready: Optional[FrameReady] = frame_ready

        # --- full-frame vs per-mic vs legacy ---
        self._use_full_frame = bool(getattr(config, "SPI_USE_FULL_FRAME", True))
        self._mic_proto = SPIProtocol()
        self._use_fw_mic_packets = False
        self._mode_probe_done = False
        self._mic_batch_id: Optional[int] = None
        self._mic_seen: set = set()
        self._mic_fft: Optional[np.ndarray] = None  # (N_MICS, N_BINS) accumulator

    def _reset_mic_accum(self) -> None:
        self._mic_batch_id = None
        self._mic_seen = set()
        self._mic_fft = None

    def _accum_mic_packet(self, batch_id: int, mic_index: int, fft_1mic: np.ndarray) -> bool:
        """Accumulate one mic's FFT. Returns True when we have all N_MICS for this batch."""
        if self._mic_batch_id is not None and self._mic_batch_id != batch_id:
            self._reset_mic_accum()
        self._mic_batch_id = batch_id
        if self._mic_fft is None:
            self._mic_fft = np.zeros((self.N_MICS, fft_1mic.shape[0]), dtype=np.complex64)
        self._mic_fft[mic_index, :] = fft_1mic
        self._mic_seen.add(mic_index)
        return len(self._mic_seen) >= self.N_MICS

    def _read_one_mic_packet(self) -> Optional[bytes]:
        """Read exactly SPI_MIC_PACKET_BYTES from SPI. Caller must ensure _spi is open."""
        if self._spi is None:
            return None
        tx = bytes(self.SPI_MIC_PACKET_BYTES)
        rx = self._spi_xfer_bytes(tx)
        return rx

    def _probe_modes_once(self) -> None:
        """Try SPI modes 0–3 with one 2081-byte read each; if any parses as per-mic, set _use_fw_mic_packets."""
        if self._mode_probe_done or self._spi is None:
            return
        self._mode_probe_done = True
        saved_mode = self._spi.mode
        for mode in range(4):
            try:
                self._spi.mode = mode
                rx = self._read_one_mic_packet()
                if rx is None:
                    continue
                ok, batch_id, mic_index, fft_1mic, why = self._mic_proto.parse_mic_packet(rx)
                if ok and fft_1mic is not None:
                    self._use_fw_mic_packets = True
                    log.info("spi_manager: probed mode %d -> fw per-mic packets", mode)
                    break
            except Exception as e:
                log.debug("spi_manager probe mode %d: %s", mode, e)
            finally:
                self._spi.mode = saved_mode

    # ------------------------------
    # Public API
    # ------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return

        # Create FrameReadyGPIO only when enabled (SPI_HW). Fail-open: if GPIO fails, run without wait.
        if self.use_frame_ready and self._frame_ready is None:
            try:
                bcm_pin = int(getattr(config, "FRAME_READY_BCM_PIN", 7))   # BCM7 == physical pin 26 (MCU_STATUS)
                pull = getattr(config, "FRAME_READY_PULL", "down")          # "up" or "down"
                self._frame_ready = FrameReadyGPIO(bcm_pin=bcm_pin, pull=pull)
            except Exception as e:
                log.warning("FrameReadyGPIO init failed (%s); continuing without GPIO wait", e)
                self._frame_ready = None
                self.use_frame_ready = False

        self._stop = False
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self._thread = None

        self._close_spi()

        if self._frame_ready is not None:
            try:
                self._frame_ready.close()
            except Exception:
                pass
            self._frame_ready = None


    def get_latest(self) -> LatestFrame:
        # Return a safe copy: copy fft array ref (ok), and copy stats values
        with self._lock:
            lf = self._latest
            stats_copy = replace(lf.stats)
            stats_copy.spi_path = f"/dev/spidev{self.SPI_BUS}.{self.SPI_DEV}"
            if self._use_full_frame:
                stats_copy.spi_read_size = self.SPI_FRAME_PACKET_SIZE_BYTES
            elif self._use_fw_mic_packets:
                stats_copy.spi_read_size = self.SPI_MIC_PACKET_BYTES
            else:
                stats_copy.spi_read_size = self.FRAME_BYTES
            out = LatestFrame(
                fft_data=lf.fft_data,
                frame_id=int(lf.frame_id),
                ok=bool(lf.ok),
                stats=stats_copy,
            )
        return out

    # ------------------------------
    # SPI device open/close
    # ------------------------------
    def _open_spi(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self._set_err("spidev not installed")
            return False
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(self.SPI_BUS, self.SPI_DEV)
            self._spi.mode = self.SPI_MODE
            self._spi.bits_per_word = self.SPI_BITS
            self._spi.max_speed_hz = int(self.SPI_MAX_SPEED_HZ)
            self._set_sclk(int(self._spi.max_speed_hz))
            self._set_err("")
            return True
        except Exception as e:
            self._spi = None
            self._set_err(f"SPI open failed: {e}")
            return False

    def _close_spi(self) -> None:
        if self._spi is not None:
            try:
                self._spi.close()
            except Exception:
                pass
        self._spi = None

    # ------------------------------
    # Worker loop
    # ------------------------------
    def _worker(self) -> None:
        # reset mailbox
        with self._lock:
            self._latest = LatestFrame()

        while not self._stop:
            if self._spi is None:
                if not self._open_spi():
                    time.sleep(0.1)
                    continue

            fr = self._read_one()
            if fr is not None:
                with self._lock:
                    self._latest = fr

    # ------------------------------
    # One frame read
    # ------------------------------
    # STM32 sends one DMA transfer of SPI_FRAME_PACKET_SIZE_BYTES (32801) per frame; PE0
    # clears only when that full transfer completes. Per-mic reads (16x2081) do not
    # complete that transfer; use one 32801-byte read or firmware sending 16x2081 with
    # one PE0 pulse per packet to get MCU_STATUS toggling.
    def _read_one(self) -> Optional[LatestFrame]:
        self._seq_seen += 1
        irq_timed_out = False

        # Wait for STM32 "frame ready" pulse/level. On timeout still do one SPI read (so we see ok/badParse/irqTimeout).
        if self.use_frame_ready and self._frame_ready is not None:
            timeout_s = float(getattr(config, "FRAME_READY_TIMEOUT_S", 0.25))
            self._frame_ready.clear()
            got = self._frame_ready.wait(timeout=timeout_s)
            if not got:
                irq_timed_out = True
                stats = self._get_stats()
                stats.irq_timeout += 1
                stats.last_err = "irq_timeout"
                self._set_stats(stats)
                # Do not return: proceed to one SPI read so HW shows packet activity

        try:
            if self._use_full_frame:
                return self._read_one_full_frame()
            if self._use_fw_mic_packets:
                return self._read_one_fw_mic()
            # Legacy path
            tx = bytes(self.FRAME_BYTES)
            rx = self._spi_xfer_bytes(tx)
            ok, why = self._framing_validate(rx)
            stats = self._get_stats()
            if not ok:
                stats.bad_parse += 1
                stats.last_err = f"parse:{why}"
                self._set_stats(stats)
                if why == "magic_start":
                    self._probe_modes_once()
                return None

            if self.CRC_EVERY_N and ((stats.frames_ok % self.CRC_EVERY_N) == 0):
                if not self._crc_validate(rx):
                    stats.bad_crc += 1
                    stats.last_err = "crc"
                    self._set_stats(stats)
                    return None

            payload = rx[self.HEADER_LEN : self.HEADER_LEN + self.PAYLOAD_LEN]
            mp = np.frombuffer(payload, dtype=np.float32).reshape(self.N_MICS, self.N_BINS, 2)
            mag = mp[:, :, 0]
            phase = mp[:, :, 1]
            fft_data = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)

            stats.frames_ok += 1
            stats.last_err = ""
            self._set_stats(stats)
            return LatestFrame(
                fft_data=fft_data,
                frame_id=self._seq_seen,
                ok=True,
                stats=stats,
            )

        except Exception as e:
            stats = self._get_stats()
            stats.bad_parse += 1
            stats.last_err = f"spi_exc:{e}"
            self._set_stats(stats)
            return None

    def _read_one_full_frame(self) -> Optional[LatestFrame]:
        """One read of SPI_FRAME_PACKET_SIZE_BYTES (32801), parse all 16 mics; no accumulator.
        Single xfer to match firmware: one NSS-low period for full frame (no SPI_XFER_CHUNK)."""
        size = self.SPI_FRAME_PACKET_SIZE_BYTES
        tx = bytes(size)
        r = self._spi.xfer3(tx)
        rx = bytes(r)
        ok, frame_counter, fft_data, why = self._mic_proto.parse_full_frame(rx)
        stats = self._get_stats()
        if not ok or fft_data is None:
            stats.bad_parse += 1
            stats.last_err = f"parse:{why}"
            self._set_stats(stats)
            return None
        stats.frames_ok += 1
        stats.last_err = ""
        self._set_stats(stats)
        return LatestFrame(
            fft_data=fft_data,
            frame_id=self._seq_seen,
            ok=True,
            stats=stats,
        )

    def _read_one_fw_mic(self) -> Optional[LatestFrame]:
        """Read one per-mic packet, accumulate; return LatestFrame when we have 16 mics for same batch."""
        rx = self._read_one_mic_packet()
        if rx is None:
            return None
        ok, batch_id, mic_index, fft_1mic, why = self._mic_proto.parse_mic_packet(rx)
        stats = self._get_stats()
        if not ok:
            stats.bad_parse += 1
            stats.last_err = f"parse:{why}"
            self._set_stats(stats)
            return None
        if fft_1mic is None:
            return None
        complete = self._accum_mic_packet(batch_id, mic_index, fft_1mic)
        if not complete:
            return None
        fft_data = self._mic_fft.copy()
        self._reset_mic_accum()
        stats.frames_ok += 1
        stats.last_err = ""
        self._set_stats(stats)
        return LatestFrame(
            fft_data=fft_data,
            frame_id=self._seq_seen,
            ok=True,
            stats=stats,
        )

    # ------------------------------
    # Transfer
    # ------------------------------
    def _spi_xfer_bytes(self, tx: bytes) -> bytes:
        assert self._spi is not None
        rx = bytearray(len(tx))
        mv = memoryview(tx)
        offset = 0

        while offset < len(tx):
            end = min(offset + self.SPI_XFER_CHUNK, len(tx))
            chunk = mv[offset:end]
            r = self._spi.xfer3(chunk)
            rx[offset:end] = bytes(r)
            offset = end

        return bytes(rx)

    # ------------------------------
    # Framing validation
    # ------------------------------
    def _framing_validate(self, buf: bytes) -> tuple[bool, str]:
        if len(buf) != self.FRAME_BYTES:
            return False, "len"
        try:
            (magic,) = struct.unpack_from("<I", buf, 0)
        except struct.error:
            return False, "hdr_unpack0"
        if magic != self.MAGIC_START:
            return False, "magic_start"

        try:
            (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
                self.HEADER_FMT, buf, 0
            )
        except struct.error:
            return False, "hdr_unpack"

        if ver != self.VERSION or hdr_len != self.HEADER_LEN:
            return False, "hdr_fields"
        if mic != self.N_MICS or fft_size != self.SAMPLES_PER_CHANNEL or fs != self.SAMPLE_RATE_HZ or bins != self.N_BINS:
            return False, "cfg_mismatch"
        if pay_len != self.PAYLOAD_LEN:
            return False, "pay_len"

        try:
            (_, magic_end) = struct.unpack_from(self.TRAILER_FMT, buf, self.HEADER_LEN + self.PAYLOAD_LEN)
        except struct.error:
            return False, "trl_unpack"
        if magic_end != self.MAGIC_END:
            return False, "magic_end"

        return True, "ok"

    def _crc_validate(self, buf: bytes) -> bool:
        if len(buf) != self.FRAME_BYTES:
            return False

        try:
            (magic, ver, hdr_len, seq, mic, fft_size, fs, bins, _res, pay_len) = struct.unpack_from(
                self.HEADER_FMT, buf, 0
            )
        except struct.error:
            return False

        if magic != self.MAGIC_START or ver != self.VERSION or hdr_len != self.HEADER_LEN:
            return False
        if mic != self.N_MICS or fft_size != self.SAMPLES_PER_CHANNEL or fs != self.SAMPLE_RATE_HZ or bins != self.N_BINS:
            return False
        if pay_len != self.PAYLOAD_LEN:
            return False

        crc_rx, magic_end = struct.unpack_from(self.TRAILER_FMT, buf, self.HEADER_LEN + self.PAYLOAD_LEN)
        if magic_end != self.MAGIC_END:
            return False

        header = buf[: self.HEADER_LEN]
        payload = buf[self.HEADER_LEN : self.HEADER_LEN + self.PAYLOAD_LEN]
        crc = zlib.crc32(header)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return crc == crc_rx

    # ------------------------------
    # Stats helpers (thread-safe)
    # ------------------------------
    def _get_stats(self) -> SourceStats:
        with self._lock:
            return replace(self._latest.stats)

    def _set_stats(self, stats: SourceStats) -> None:
        with self._lock:
            self._latest.stats = stats

    def _set_err(self, msg: str) -> None:
        stats = self._get_stats()
        stats.last_err = msg
        self._set_stats(stats)

    def _set_sclk(self, hz: int) -> None:
        stats = self._get_stats()
        stats.sclk_hz_rep = int(hz)
        self._set_stats(stats)
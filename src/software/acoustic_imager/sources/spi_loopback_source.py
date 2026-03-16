#!/usr/bin/env python3
from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from acoustic_imager import config
from acoustic_imager.custom_types import LatestFrame, SourceStats
from acoustic_imager.spi.spi_protocol import SPIProtocol

try:
    import spidev  # type: ignore
    SPIDEV_AVAILABLE = True
except Exception:
    spidev = None
    SPIDEV_AVAILABLE = False


@dataclass
class LoopbackStats(SourceStats):
    """Reuse SourceStats shape so main debug overlay works."""
    pass


class SPILoopbackSource:
    """
    REAL physical SPI loopback mode (no STM32).

    Requirements:
      - MOSI physically connected to MISO (Pi pin 19 -> pin 21).
      - SPI enabled on the Pi.

    Behavior:
      - Generates a synthetic FFT frame (software)
      - Packs it into your STM32-style framed packet (header+payload+crc+end)
      - Sends it over SPI (TX) and reads it back (RX) via MOSI->MISO loopback
      - Validates framing/CRC and parses payload back into complex FFT
      - Publishes LatestFrame via mailbox for the UI thread
    """

    def __init__(
        self,
        spi_bus: int = getattr(config, "SPI_BUS", 0),
        spi_dev: int = getattr(config, "SPI_DEV", 0),
        spi_mode: int = getattr(config, "SPI_MODE", 0),
        spi_bits: int = getattr(config, "SPI_BITS", 8),
        spi_max_speed_hz: int = getattr(config, "SPI_MAX_SPEED_HZ", 12_000_000),
        spi_xfer_chunk: int = getattr(config, "SPI_XFER_CHUNK", 4096),
        update_hz: float = 200.0,
        crc_every_n: int = int(getattr(config, "CRC_EVERY_N", 1)),  # loopback: default validate always
    ) -> None:
        self._lock = threading.Lock()
        self._latest = LatestFrame()
        self._stop = False
        self._th: Optional[threading.Thread] = None

        self._spi = None
        self._seq = 0

        self._bus = int(spi_bus)
        self._dev = int(spi_dev)
        self._bus   = int(spi_bus)
        self._dev   = int(spi_dev)
        self._mode  = int(spi_mode)
        self._bits  = int(spi_bits)
        self._max_hz = int(spi_max_speed_hz)

        def _env_int(name: str, default: int) -> int:
            v = os.getenv(name)
            return default if v is None else int(v)

        self._bus = _env_int("ACOUSTIC_SPI_BUS", self._bus)
        self._dev = _env_int("ACOUSTIC_SPI_DEV", self._dev)
        self._max_hz = _env_int("ACOUSTIC_SPI_HZ", self._max_hz)

        self._mode = int(spi_mode)
        self._bits = int(spi_bits)
        self._chunk = int(spi_xfer_chunk)
        self._dt = 1.0 / float(update_hz)
        self._crc_every_n = int(crc_every_n)

        self._proto = SPIProtocol()

        # You can tune these “synthetic sources” to look like your old demo
        self._bins = list(getattr(config, "SPI_SIM_BINS", [35, 80, 160, 220]))
        self._ampls = [6.0, 3.0, 5.0, 4.0]
        self._angles = [-25.0, 35.0, -5.0, 60.0]
        self._drift = [1.2, -0.6, 0.4, -0.3]
        self._spread_bins = 3
        self._sigma_bins = 1.2
        self._noise_mag = 0.006
        self._noise_phase = 0.015

    # -----------------------------
    # Public API
    # -----------------------------
    def start(self) -> None:
        if self._th is not None and self._th.is_alive():
            return
        self._stop = False
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    def stop(self) -> None:
        self._stop = True
        if self._th is not None:
            self._th.join(timeout=0.5)
        self._th = None
        self._close_spi()

    def get_latest(self) -> LatestFrame:
        with self._lock:
            lf = self._latest
            # LatestFrame contains numpy arrays; that’s fine to share read-only
            return LatestFrame(
                ok=bool(lf.ok),
                fft_data=lf.fft_data,
                stats=lf.stats,
                frame_id=int(lf.frame_id),
            )

    # -----------------------------
    # SPI open/close
    # -----------------------------
    def _open_spi(self) -> bool:
        if not SPIDEV_AVAILABLE:
            self._set_err("spidev not installed")
            return False
        try:
            self._spi = spidev.SpiDev()
            self._spi.open(self._bus, self._dev)
            self._spi.mode = self._mode
            self._spi.bits_per_word = self._bits
            self._spi.max_speed_hz = int(self._max_hz)

            st = self._get_stats()
            st.sclk_hz_rep = int(self._spi.max_speed_hz)
            st.last_err = ""
            self._set_stats(st)
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

    # -----------------------------
    # Worker
    # -----------------------------
    def _worker(self) -> None:
        with self._lock:
            self._latest = LatestFrame()

        t0 = time.time()

        while not self._stop:
            if self._spi is None:
                if not self._open_spi():
                    time.sleep(0.2)
                    continue

            self._seq += 1
            t = time.time() - t0

            try:
                # TODO: this is hardcoded synthetic FFT generation
                # fft = self._make_fft_frame(t)

                # TODO: this is random noise FFT generation (exercises framing/CRC/parser without any particular source structure)
                fft = (
                    np.random.randn(config.N_MICS, config.N_BINS) +
                    1j * np.random.randn(config.N_MICS, config.N_BINS)
                ).astype(np.complex64)


                # 2) Convert to mag/phase float32 payload to match current protocol
                payload = self._fft_to_payload_mag_phase(fft)

                # 3) Build framed packet
                tx = self._proto.build_frame(seq=self._seq, payload=payload)

                # 4) Send/receive over SPI (requires MOSI->MISO loopback)
                rx = self._spi_xfer_bytes(tx)

                # 5) Validate framing + CRC (optionally every N)
                ok, why = self._proto.validate_framing(rx)
                st = self._get_stats()

                if not ok:
                    st.bad_parse += 1
                    st.last_err = f"parse:{why}"
                    self._set_stats(st)
                    self._publish(None, False, st)
                    time.sleep(self._dt)
                    continue

                if self._crc_every_n and ((st.frames_ok % self._crc_every_n) == 0):
                    if not self._proto.validate_crc(rx):
                        st.bad_crc += 1
                        st.last_err = "crc"
                        self._set_stats(st)
                        self._publish(None, False, st)
                        time.sleep(self._dt)
                        continue

                # 6) Parse payload back to fft (exercises parser)
                payload_rx = rx[self._proto.header_len : self._proto.header_len + self._proto.payload_len]
                fft_rx = self._proto.payload_to_fft(payload_rx)

                st.frames_ok += 1
                st.last_err = ""
                self._set_stats(st)
                self._publish(fft_rx, True, st)

            except Exception as e:
                st = self._get_stats()
                st.bad_parse += 1
                st.last_err = f"loop_exc:{e}"
                self._set_stats(st)
                self._publish(None, False, st)

            time.sleep(self._dt)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _publish(self, fft: Optional[np.ndarray], ok: bool, st: SourceStats) -> None:
        with self._lock:
            self._latest = LatestFrame(
                fft_data=fft,
                ok=ok,
                stats=st,
                frame_id=self._seq,
            )

    def _get_stats(self) -> SourceStats:
        with self._lock:
            return self._latest.stats

    def _set_stats(self, st: SourceStats) -> None:
        with self._lock:
            self._latest.stats = st

    def _set_err(self, msg: str) -> None:
        st = self._get_stats()
        st.last_err = msg
        self._set_stats(st)

    def _spi_xfer_bytes(self, tx: bytes) -> bytes:
        assert self._spi is not None
        rx = bytearray(len(tx))
        mv = memoryview(tx)
        off = 0
        while off < len(tx):
            end = min(off + self._chunk, len(tx))
            chunk = mv[off:end]
            # some spidev builds prefer list(chunk); if you see issues, switch to list(chunk)
            r = self._spi.xfer3(chunk)
            rx[off:end] = bytes(r)
            off = end
        return bytes(rx)

    def _fft_to_payload_mag_phase(self, fft: np.ndarray) -> bytes:
        """
        Current protocol assumes float32 MAG + float32 PHASE.
        Payload layout: (N_MICS, N_BINS, 2) float32.
        """
        mag = np.abs(fft).astype(np.float32)
        ph = np.angle(fft).astype(np.float32)
        mp = np.stack([mag, ph], axis=-1)  # (mics, bins, 2)
        return mp.tobytes(order="C")

    def _make_fft_frame(self, t_sec: float) -> np.ndarray:
        """
        Synthetic FFT generator (same vibe as your old demo).
        Returns (N_MICS, N_BINS) complex64.
        """
        N_MICS = int(config.N_MICS)
        N_BINS = int(config.N_BINS)

        mp_mag = self._noise_mag * (1.0 + 0.25 * np.random.randn(N_MICS, N_BINS)).astype(np.float32)
        mp_ph = (self._noise_phase * np.random.randn(N_MICS, N_BINS)).astype(np.float32)

        for i, b0 in enumerate(self._bins):
            if not (0 <= b0 < N_BINS):
                continue

            f0 = float(config.f_axis[b0])
            ang = float(self._angles[i]) + float(self._drift[i]) * t_sec
            ang = ((ang + 90.0) % 180.0) - 90.0
            theta = np.deg2rad(ang)

            a0 = np.exp(
                -1j * 2.0 * np.pi * f0 / config.SPEED_SOUND *
                -(config.x_coords_hw * np.cos(theta) + config.y_coords_hw * np.sin(theta))
            ).astype(np.complex64)

            for dbin in range(-self._spread_bins, self._spread_bins + 1):
                b = b0 + dbin
                if b < 0 or b >= N_BINS:
                    continue
                w = float(np.exp(-0.5 * (dbin / self._sigma_bins) ** 2))
                amp = float(self._ampls[i]) * w
                ph0 = 0.15 * dbin
                X = amp * a0 * np.exp(1j * ph0)

                mp_mag[:, b] += np.abs(X).astype(np.float32)
                mp_ph[:, b] += np.angle(X).astype(np.float32)

        fft = (mp_mag * (np.cos(mp_ph) + 1j * np.sin(mp_ph))).astype(np.complex64)
        return fft
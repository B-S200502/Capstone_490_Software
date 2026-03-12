# spi/spi_protocol.py
from __future__ import annotations

import logging
import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple, Any

import numpy as np

from acoustic_imager.custom_types import LatestFrame, SourceStats

# For per-mic firmware packet config (SPI_MAGIC_FW, SPI_MIC_*, etc.)
try:
    from acoustic_imager import config as _config  # type: ignore
except Exception:
    _config = None

log = logging.getLogger(__name__)


# ---------------------------
# Helpers: flexible config read
# ---------------------------
def _get_attr(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _get_cfg():
    """
    Tries to load configuration from config.py in a flexible way.

    Supported patterns:
      1) config.py defines `cfg` object with attributes
      2) config.py defines constants like N_MICS, N_BINS, etc.

    If your project uses different names, adjust this function only.
    """
    try:
        from acoustic_imager import config  # type: ignore
    except Exception as e:
        raise ImportError(f"spi_protocol could not import config.py: {e}")

    cfg_obj = getattr(config, "cfg", None)

    # Prefer cfg object, fallback to module-level constants
    class _CfgWrap:
        pass

    out = _CfgWrap()

    # Required parameters (match your monolith names)
    out.N_MICS = _get_attr(cfg_obj, "N_MICS", getattr(config, "N_MICS", 16))
    out.SAMPLES_PER_CHANNEL = _get_attr(cfg_obj, "SAMPLES_PER_CHANNEL", getattr(config, "SAMPLES_PER_CHANNEL", 512))
    out.SAMPLE_RATE_HZ = _get_attr(cfg_obj, "SAMPLE_RATE_HZ", getattr(config, "SAMPLE_RATE_HZ", 100000))

    # Derived bins (allow override if you store N_BINS)
    default_bins = out.SAMPLES_PER_CHANNEL // 2 + 1
    out.N_BINS = _get_attr(cfg_obj, "N_BINS", getattr(config, "N_BINS", default_bins))

    # SPI framing format (allow override)
    out.MAGIC_START = _get_attr(cfg_obj, "MAGIC_START", getattr(config, "MAGIC_START", 0x46524654))
    out.MAGIC_END = _get_attr(cfg_obj, "MAGIC_END", getattr(config, "MAGIC_END", 0x454E4421))
    out.VERSION = _get_attr(cfg_obj, "VERSION", getattr(config, "VERSION", 1))

    # Header/trailer formats (match your original)
    out.HEADER_FMT = _get_attr(cfg_obj, "HEADER_FMT", getattr(config, "HEADER_FMT", "<IHH I HH I HH I"))
    out.TRAILER_FMT = _get_attr(cfg_obj, "TRAILER_FMT", getattr(config, "TRAILER_FMT", "<II"))

    return out


CFG = _get_cfg()


# ---------------------------
# Protocol: sizes and packing
# ---------------------------
HEADER_LEN = struct.calcsize(CFG.HEADER_FMT)
TRAILER_LEN = struct.calcsize(CFG.TRAILER_FMT)

# payload = N_MICS * N_BINS * (MAG + PHASE) * float32
PAYLOAD_LEN = int(CFG.N_MICS) * int(CFG.N_BINS) * 2 * 4
FRAME_BYTES = HEADER_LEN + PAYLOAD_LEN + TRAILER_LEN


@dataclass(frozen=True)
class FrameHeader:
    magic: int
    version: int
    header_len: int
    seq: int
    mic_count: int
    fft_size: int
    sample_rate: int
    bin_count: int
    reserved: int
    payload_len: int


@dataclass(frozen=True)
class MicPacketHeader:
    """Matches firmware SPI_FrameHeader_t (spi_protocol.h)."""
    magic: int
    version: int
    header_len: int
    frame_counter: int
    batch_id: int
    mic_index: int
    fft_size: int
    sample_rate: int
    flags: int
    payload_len: int
    battery_mv: int
    reserved0: int
    reserved1: int


def unpack_packed_rfft_to_complex(payload: bytes, fft_size: int) -> np.ndarray:
    """
    CMSIS-style packed RFFT: [DC, Nyquist, re1, im1, re2, im2, ...].
    Returns complex array of shape (fft_size // 2 + 1,) for bins 0..N/2.
    """
    n_floats = len(payload) // 4
    arr = np.frombuffer(payload, dtype=np.float32, count=n_floats)
    n_bins = fft_size // 2 + 1
    if n_floats < 2 + (n_bins - 2) * 2:
        return np.zeros(n_bins, dtype=np.complex64)
    out = np.zeros(n_bins, dtype=np.complex64)
    out[0] = arr[0] + 0j
    out[n_bins - 1] = arr[1] + 0j
    for k in range(1, n_bins - 1):
        out[k] = arr[2 + (k - 1) * 2] + 1j * arr[2 + (k - 1) * 2 + 1]
    return out


class SPIProtocol:
    """
    Responsible ONLY for framing/parsing bytes.
    The SPI transport itself (spidev) belongs in spi_manager / spi_source.
    """

    def __init__(self) -> None:
        self.header_fmt = CFG.HEADER_FMT
        self.trailer_fmt = CFG.TRAILER_FMT
        self.header_len = HEADER_LEN
        self.trailer_len = TRAILER_LEN
        self.payload_len = PAYLOAD_LEN
        self.frame_bytes = FRAME_BYTES

        self.magic_start = int(CFG.MAGIC_START)
        self.magic_end = int(CFG.MAGIC_END)
        self.version = int(CFG.VERSION)

        self.n_mics = int(CFG.N_MICS)
        self.fft_size = int(CFG.SAMPLES_PER_CHANNEL)
        self.sample_rate = int(CFG.SAMPLE_RATE_HZ)
        self.n_bins = int(CFG.N_BINS)

        # Per-mic firmware packet config (from config.py)
        self.spi_magic_fw = int(getattr(_config, "SPI_MAGIC_FW", 0xAABBCCDD))
        self.spi_mic_header_fmt = getattr(_config, "SPI_MIC_HEADER_FMT", "<IHHIHBHIHHHHH")
        self.spi_mic_header_bytes = int(getattr(_config, "SPI_MIC_HEADER_BYTES", 31))
        self.spi_mic_payload_bytes = int(getattr(_config, "SPI_MIC_PAYLOAD_BYTES", 2048))
        self.spi_mic_packet_bytes = int(getattr(_config, "SPI_MIC_PACKET_BYTES", 2081))
        self.spi_frame_packet_size_bytes = int(getattr(_config, "SPI_FRAME_PACKET_SIZE_BYTES", 32801))

    # --------
    # CRC helpers
    # --------
    def compute_crc(self, header: bytes, payload: bytes) -> int:
        crc = zlib.crc32(header)
        crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
        return crc

    # --------
    # Header parse/pack
    # --------
    def pack_header(self, seq: int, payload_len: int) -> bytes:
        return struct.pack(
            self.header_fmt,
            self.magic_start,
            self.version,
            self.header_len,
            int(seq),
            self.n_mics,
            self.fft_size,
            self.sample_rate,
            self.n_bins,
            0,
            int(payload_len),
        )

    def unpack_header(self, buf: bytes) -> FrameHeader:
        fields = struct.unpack_from(self.header_fmt, buf, 0)
        return FrameHeader(
            magic=int(fields[0]),
            version=int(fields[1]),
            header_len=int(fields[2]),
            seq=int(fields[3]),
            mic_count=int(fields[4]),
            fft_size=int(fields[5]),
            sample_rate=int(fields[6]),
            bin_count=int(fields[7]),
            reserved=int(fields[8]),
            payload_len=int(fields[9]),
        )

    # --------
    # Framing validation
    # --------
    def validate_framing(self, buf: bytes) -> Tuple[bool, str]:
        if len(buf) != self.frame_bytes:
            return False, "len"

        try:
            (magic,) = struct.unpack_from("<I", buf, 0)
        except struct.error:
            return False, "hdr_unpack0"

        if int(magic) != self.magic_start:
            return False, "magic_start"

        try:
            hdr = self.unpack_header(buf)
        except struct.error:
            return False, "hdr_unpack"

        if hdr.version != self.version or hdr.header_len != self.header_len:
            return False, "hdr_fields"

        # Config mismatch check
        if (
            hdr.mic_count != self.n_mics
            or hdr.fft_size != self.fft_size
            or hdr.sample_rate != self.sample_rate
            or hdr.bin_count != self.n_bins
        ):
            return False, "cfg_mismatch"

        if hdr.payload_len != self.payload_len:
            return False, "payload_len"

        try:
            (_, magic_end) = struct.unpack_from(self.trailer_fmt, buf, self.header_len + self.payload_len)
        except struct.error:
            return False, "trl_unpack"

        if int(magic_end) != self.magic_end:
            return False, "magic_end"

        return True, "ok"

    def validate_crc(self, buf: bytes) -> bool:
        if len(buf) != self.frame_bytes:
            return False

        hdr = buf[: self.header_len]
        payload = buf[self.header_len : self.header_len + self.payload_len]
        (crc_rx, magic_end) = struct.unpack_from(self.trailer_fmt, buf, self.header_len + self.payload_len)

        if int(magic_end) != self.magic_end:
            return False

        crc_calc = self.compute_crc(hdr, payload)
        return (crc_calc & 0xFFFFFFFF) == (int(crc_rx) & 0xFFFFFFFF)

    # --------
    # Per-mic firmware packet parsing
    # --------
    def parse_mic_packet(
        self, buf: bytes
    ) -> Tuple[bool, int, int, Optional[np.ndarray], str]:
        """
        Parse one per-mic packet (SPI_MIC_PACKET_BYTES).
        Returns (ok, batch_id, mic_index, fft_1mic, why).
        fft_1mic is (n_bins,) complex64 or None on failure.
        """
        if len(buf) != self.spi_mic_packet_bytes:
            return False, 0, 0, None, "len"
        try:
            fields = struct.unpack_from(self.spi_mic_header_fmt, buf, 0)
        except struct.error:
            return False, 0, 0, None, "hdr_unpack"
        magic = int(fields[0])
        if magic != self.spi_magic_fw:
            return False, 0, 0, None, "magic"
        version = int(fields[1])
        if version != 1:
            return False, 0, 0, None, "version"
        header_len = int(fields[2])
        if header_len != self.spi_mic_header_bytes:
            return False, 0, 0, None, "header_len"
        frame_counter = int(fields[3])
        batch_id = int(fields[4])
        mic_index = int(fields[5])
        fft_size = int(fields[6])
        sample_rate = int(fields[7])
        flags = int(fields[8])
        payload_len = int(fields[9])
        battery_mv = int(fields[10])
        if sample_rate != self.sample_rate:
            return False, batch_id, mic_index, None, "sample_rate"
        if payload_len != self.spi_mic_payload_bytes:
            return False, batch_id, mic_index, None, "payload_len"
        if mic_index < 0 or mic_index >= self.n_mics:
            return False, batch_id, mic_index, None, "mic_index"
        payload = buf[self.spi_mic_header_bytes : self.spi_mic_header_bytes + self.spi_mic_payload_bytes]
        try:
            fft_1mic = unpack_packed_rfft_to_complex(payload, fft_size)
        except Exception as e:
            return False, batch_id, mic_index, None, f"rfft:{e}"
        return True, batch_id, mic_index, fft_1mic, "ok"

    def parse_full_frame(self, buf: bytes) -> Tuple[bool, int, Optional[np.ndarray], str]:
        """
        Parse one full frame (SPI_FRAME_PACKET_SIZE_BYTES): one header + 16 mic payloads + 2-byte checksum.
        Returns (ok, frame_counter, fft_data, why). fft_data is (N_MICS, N_BINS) complex64 or None on failure.
        """
        if len(buf) != self.spi_frame_packet_size_bytes:
            return False, 0, None, "len"
        try:
            fields = struct.unpack_from(self.spi_mic_header_fmt, buf, 0)
        except struct.error:
            return False, 0, None, "hdr_unpack"
        magic = int(fields[0])
        if magic != self.spi_magic_fw:
            log.warning(
                "parse_full_frame: magic 0x%08X != expected 0x%08X, buf[:8]=%s",
                magic,
                self.spi_magic_fw,
                buf[:8].hex() if len(buf) >= 8 else buf.hex(),
            )
            return False, 0, None, "magic"
        version = int(fields[1])
        if version != 1:
            return False, 0, None, "version"
        header_len = int(fields[2])
        if header_len != self.spi_mic_header_bytes:
            return False, 0, None, "header_len"
        frame_counter = int(fields[3])
        fft_size = int(fields[6])
        sample_rate = int(fields[7])
        payload_len = int(fields[9])
        if sample_rate != self.sample_rate:
            return False, frame_counter, None, "sample_rate"
        expected_payload = self.n_mics * self.spi_mic_payload_bytes
        if payload_len != expected_payload:
            return False, frame_counter, None, "payload_len"
        fft_data = np.zeros((self.n_mics, self.n_bins), dtype=np.complex64)
        for mic in range(self.n_mics):
            start = self.spi_mic_header_bytes + mic * self.spi_mic_payload_bytes
            end = start + self.spi_mic_payload_bytes
            payload = buf[start:end]
            try:
                fft_data[mic, :] = unpack_packed_rfft_to_complex(payload, fft_size)
            except Exception as e:
                return False, frame_counter, None, f"rfft_mic{mic}:{e}"
        # Sanitize so one bad frame (garbage payload or firmware inf/nan) doesn't poison pipeline
        if not np.all(np.isfinite(fft_data)):
            fft_data = fft_data.copy()
            fft_data[~np.isfinite(fft_data)] = 0.0
        return True, frame_counter, fft_data, "ok"

    # --------
    # Payload parsing
    # --------
    def payload_to_fft(self, payload: bytes) -> np.ndarray:
        """
        Payload layout: float32 MAG, float32 PHASE for each mic/bin.
        Returns complex64 array (N_MICS, N_BINS).
        """
        mp = np.frombuffer(payload, dtype=np.float32)
        mp = mp.reshape(self.n_mics, self.n_bins, 2)

        mag = mp[:, :, 0]
        phase = mp[:, :, 1]
        # complex from polar
        fft = (mag * (np.cos(phase) + 1j * np.sin(phase))).astype(np.complex64)
        return fft

    # --------
    # Full parse convenience
    # --------
    def parse_frame(
        self,
        buf: bytes,
        validate_crc: bool = False,
        stats: Optional[SourceStats] = None,
    ) -> LatestFrame:
        """
        Parses raw SPI bytes into LatestFrame.
        Does not do SPI transfer; only interprets the bytes.
        """
        if stats is None:
            stats = SourceStats()

        out = LatestFrame(fft_data=None, frame_id=0, ok=False, stats=stats)

        ok, why = self.validate_framing(buf)
        if not ok:
            out.stats.bad_parse += 1
            out.stats.last_err = f"parse:{why}"
            return out

        if validate_crc and not self.validate_crc(buf):
            out.stats.bad_crc += 1
            out.stats.last_err = "crc"
            return out

        hdr = self.unpack_header(buf)
        payload = buf[self.header_len : self.header_len + self.payload_len]

        try:
            fft = self.payload_to_fft(payload)
        except Exception as e:
            out.stats.bad_parse += 1
            out.stats.last_err = f"payload_exc:{e}"
            return out

        out.fft_data = fft
        out.frame_id = int(hdr.seq)
        out.ok = True
        out.stats.frames_ok += 1
        out.stats.last_err = ""
        return out

    # --------
    # Build frame (for loopback / SIM framing)
    # --------
    def build_frame(self, seq: int, payload: bytes) -> bytes:
        if len(payload) != self.payload_len:
            raise ValueError(f"payload length mismatch: got {len(payload)} expected {self.payload_len}")

        header = self.pack_header(seq=seq, payload_len=len(payload))
        crc = self.compute_crc(header, payload)
        trailer = struct.pack(self.trailer_fmt, int(crc), self.magic_end)
        return header + payload + trailer
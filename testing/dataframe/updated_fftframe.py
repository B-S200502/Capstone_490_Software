import struct
import time
from dataclasses import dataclass, field
from typing import List
import numpy as np

#TODO: Update this with our new data structure

@dataclass
class FFTFrame:
    FRAME_HEADER: int = 0xAA55
    FRAME_FOOTER: int = 0x55AA

    frame_id: int = 0
    timestamp_us: int = field(default_factory=lambda: int(time.time() * 1e6))
    channel_count: int = 16
    sampling_rate: int = 72000
    fft_size: int = 2048
    fft_format: int = 1  # 1 = complex float32
    fft_data: np.ndarray = field(default_factory=lambda: np.zeros((16, 1025), dtype=np.complex64))
    crc: int = 0

    # -----------------------------------------------------------------
    def pack(self) -> bytes:
        """Serialize the frame into bytes for transmission."""
        header = struct.pack('<H', self.FRAME_HEADER)
        footer = struct.pack('<H', self.FRAME_FOOTER)

        meta = struct.pack(
            '<HIBH2H',
            self.frame_id,
            self.timestamp_us,
            self.channel_count,
            self.sampling_rate,
            self.fft_size,
            self.fft_format
        )

        # Flatten FFT data as interleaved (real, imag) float32s
        fft_flat = np.empty((self.channel_count, self.fft_data.shape[1] * 2), dtype=np.float32)
        fft_flat[:, 0::2] = self.fft_data.real
        fft_flat[:, 1::2] = self.fft_data.imag
        fft_bytes = fft_flat.tobytes()

        # Optional CRC: simple sum mod 65535
        crc_val = (sum(fft_flat.flatten().view(np.int32)) & 0xFFFF)
        crc = struct.pack('<H', crc_val)

        frame_bytes = header + meta + fft_bytes + crc + footer
        return frame_bytes

    # -----------------------------------------------------------------
    @classmethod
    def unpack(cls, data: bytes) -> 'FFTFrame':
        """Deserialize bytes into an FFTFrame object."""
        hdr, = struct.unpack_from('<H', data, 0)
        if hdr != cls.FRAME_HEADER:
            raise ValueError("Invalid frame header")

        frame_id, timestamp_us, ch_count, fs, fft_size, fft_fmt = struct.unpack_from('<HIBH2H', data, 2)
        offset = struct.calcsize('<H') + struct.calcsize('<HIBH2H')
        fft_bins = fft_size // 2 + 1
        num_floats = ch_count * fft_bins * 2

        fft_array = np.frombuffer(data, dtype=np.float32, count=num_floats, offset=offset)
        fft_array = fft_array.reshape((ch_count, fft_bins * 2))
        fft_complex = fft_array[:, 0::2] + 1j * fft_array[:, 1::2]

        # Calculate CRC offset and footer
        crc_offset = offset + num_floats * 4
        crc_val, = struct.unpack_from('<H', data, crc_offset)
        footer, = struct.unpack_from('<H', data, crc_offset + 2)
        if footer != cls.FRAME_FOOTER:
            raise ValueError("Invalid frame footer")

        frame = cls(
            frame_id=frame_id,
            timestamp_us=timestamp_us,
            channel_count=ch_count,
            sampling_rate=fs,
            fft_size=fft_size,
            fft_format=fft_fmt,
            fft_data=fft_complex.astype(np.complex64),
            crc=crc_val
        )
        return frame

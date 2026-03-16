# custom_types.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

@dataclass
class SourceStats:
    frames_ok: int = 0
    bad_parse: int = 0
    bad_crc: int = 0
    last_err: str = ""
    sclk_hz_rep: int = 0

@dataclass
class LatestFrame:
    fft_data: Optional[np.ndarray] = None  # (N_MICS, N_BINS) complex64
    frame_id: int = 0
    ok: bool = False
    stats: SourceStats = field(default_factory=SourceStats)

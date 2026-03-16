# sources/sim_source.py
from __future__ import annotations

import time
from typing import List

import numpy as np

from acoustic_imager import config
from acoustic_imager.custom_types import LatestFrame, SourceStats


def _cfg(name: str):
    if not hasattr(config, name):
        raise AttributeError(
            f"config.{name} is missing. sim_source expects it. "
            f"Add `{name} = ...` to config.py"
        )
    return getattr(config, name)


class SimSource:
    """
    Generates synthetic multi-source signals and returns FFT frames as LatestFrame.
    """

    def __init__(self) -> None:
        self.frame_id = 0
        self._t0 = time.time()

        self.N_MICS = int(_cfg("N_MICS"))
        self.SAMPLES_PER_CHANNEL = int(_cfg("SAMPLES_PER_CHANNEL"))
        self.SAMPLE_RATE_HZ = int(_cfg("SAMPLE_RATE_HZ"))
        self.SPEED_SOUND = float(_cfg("SPEED_SOUND"))
        self.NOISE_POWER = float(_cfg("NOISE_POWER"))

        self.x_coords = np.asarray(_cfg("x_coords"), dtype=np.float32)
        self.y_coords = np.asarray(_cfg("y_coords"), dtype=np.float32)

        self.SIM_SOURCE_FREQS: List[float] = list(_cfg("SIM_SOURCE_FREQS"))
        self.SIM_SOURCE_ANGLES: List[float] = list(_cfg("SIM_SOURCE_ANGLES"))
        self.SIM_SOURCE_AMPLS: List[float] = list(_cfg("SIM_SOURCE_AMPLS"))

        if not (len(self.SIM_SOURCE_FREQS) == len(self.SIM_SOURCE_ANGLES) == len(self.SIM_SOURCE_AMPLS)):
            raise ValueError("SIM_SOURCE_FREQS/ANGLES/AMPLS must be same length in config.py")

        # SIM_2 event model config
        self.SIM2_PERSISTENT_BEARINGS: List[float] = list(_cfg("SIM2_PERSISTENT_BEARINGS"))
        self.SIM2_PERSISTENT_FREQS: List[float] = list(_cfg("SIM2_PERSISTENT_FREQS"))
        self.SIM2_PERSISTENT_AMPLS: List[float] = list(_cfg("SIM2_PERSISTENT_AMPLS"))
        self.SIM2_PERSISTENT_JITTER_DEG = float(_cfg("SIM2_PERSISTENT_JITTER_DEG"))
        self.SIM2_TRANSIENT_EVENT_RATE_HZ = float(_cfg("SIM2_TRANSIENT_EVENT_RATE_HZ"))
        self.SIM2_TRANSIENT_MIN_DURATION_S = float(_cfg("SIM2_TRANSIENT_MIN_DURATION_S"))
        self.SIM2_TRANSIENT_MAX_DURATION_S = float(_cfg("SIM2_TRANSIENT_MAX_DURATION_S"))
        self.SIM2_TRANSIENT_FREQ_RANGE = tuple(_cfg("SIM2_TRANSIENT_FREQ_RANGE"))
        self.SIM2_TRANSIENT_AMPL_RANGE = tuple(_cfg("SIM2_TRANSIENT_AMPL_RANGE"))
        self.SIM2_NOISE_POWER_SCALE = float(_cfg("SIM2_NOISE_POWER_SCALE"))
        self._rng = np.random.default_rng(1337)
        self._last_event_t = time.time()
        self._transient_events: List[dict] = []
        self.last_sim2_freqs: List[float] = []
        self.last_sim2_angles: List[float] = []

    def read_frame(self) -> LatestFrame:
        # small deterministic drift so it “moves”
        for k in range(len(self.SIM_SOURCE_ANGLES)):
            self.SIM_SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
            if self.SIM_SOURCE_ANGLES[k] > 90.0:
                self.SIM_SOURCE_ANGLES[k] = -90.0

        self.frame_id += 1
        fft = self._generate_fft(self.SIM_SOURCE_ANGLES)

        stats = SourceStats(frames_ok=self.frame_id, last_err="", sclk_hz_rep=0)
        return LatestFrame(
            fft_data=fft,
            frame_id=self.frame_id,
            ok=True,
            stats=stats,
        )

    def read_frame_sim2(self) -> LatestFrame:
        """
        SIM_2: directional events with persistent bearings + transient bursts.
        No true distance simulation; direction + relative intensity only.
        """
        now = time.time()
        dt = max(0.0, now - self._last_event_t)
        self._last_event_t = now

        # Spawn transient events (Poisson-like)
        p_spawn = max(0.0, min(1.0, self.SIM2_TRANSIENT_EVENT_RATE_HZ * dt))
        if self._rng.random() < p_spawn:
            ev = {
                "bearing": float(self._rng.uniform(-85.0, 85.0)),
                "freq": float(self._rng.uniform(self.SIM2_TRANSIENT_FREQ_RANGE[0], self.SIM2_TRANSIENT_FREQ_RANGE[1])),
                "amp": float(self._rng.uniform(self.SIM2_TRANSIENT_AMPL_RANGE[0], self.SIM2_TRANSIENT_AMPL_RANGE[1])),
                "t_end": now + float(
                    self._rng.uniform(self.SIM2_TRANSIENT_MIN_DURATION_S, self.SIM2_TRANSIENT_MAX_DURATION_S)
                ),
            }
            self._transient_events.append(ev)
        self._transient_events = [e for e in self._transient_events if e["t_end"] > now]

        angles: List[float] = []
        freqs: List[float] = []
        ampls: List[float] = []

        # Persistent directional clusters (small jitter over time)
        for i, bearing in enumerate(self.SIM2_PERSISTENT_BEARINGS):
            jitter = float(self._rng.normal(0.0, self.SIM2_PERSISTENT_JITTER_DEG))
            angles.append(float(np.clip(bearing + jitter, -89.0, 89.0)))
            freqs.append(float(self.SIM2_PERSISTENT_FREQS[min(i, len(self.SIM2_PERSISTENT_FREQS) - 1)]))
            ampls.append(float(self.SIM2_PERSISTENT_AMPLS[min(i, len(self.SIM2_PERSISTENT_AMPLS) - 1)]))

        # Transient events
        for ev in self._transient_events:
            angles.append(float(ev["bearing"]))
            freqs.append(float(ev["freq"]))
            ampls.append(float(ev["amp"]))

        self.last_sim2_freqs = list(freqs)
        self.last_sim2_angles = list(angles)

        self.frame_id += 1
        fft = self._generate_fft_dynamic(angles, freqs, ampls, noise_scale=self.SIM2_NOISE_POWER_SCALE)
        stats = SourceStats(frames_ok=self.frame_id, last_err="", sclk_hz_rep=0)
        return LatestFrame(fft_data=fft, frame_id=self.frame_id, ok=True, stats=stats)

    def _generate_fft(self, angle_degs: List[float]) -> np.ndarray:
        t = np.arange(self.SAMPLES_PER_CHANNEL, dtype=np.float32) / float(self.SAMPLE_RATE_HZ)
        mic_signals = np.zeros((self.N_MICS, t.size), dtype=np.float32)

        for src_idx, angle_deg in enumerate(angle_degs):
            angle_rad = np.deg2rad(float(angle_deg))
            f = float(self.SIM_SOURCE_FREQS[src_idx])
            amp = float(self.SIM_SOURCE_AMPLS[src_idx])

            cth = np.cos(angle_rad)
            sth = np.sin(angle_rad)

            # delays for all mics (vectorized)
            delays = -((self.x_coords * cth + self.y_coords * sth) / float(self.SPEED_SOUND))  # (M,)

            # signal per mic
            phase = 2.0 * np.pi * f * (t[None, :] - delays[:, None])
            mic_signals += amp * np.sin(phase).astype(np.float32)

        # noise
        if self.NOISE_POWER > 0:
            mic_signals += np.random.normal(0.0, np.sqrt(self.NOISE_POWER), mic_signals.shape).astype(np.float32)

        # FFT
        fft_data = np.fft.rfft(mic_signals, axis=1).astype(np.complex64)  # (M, N_BINS)
        return fft_data

    def _generate_fft_dynamic(
        self,
        angle_degs: List[float],
        freqs: List[float],
        ampls: List[float],
        noise_scale: float = 1.0,
    ) -> np.ndarray:
        t = np.arange(self.SAMPLES_PER_CHANNEL, dtype=np.float32) / float(self.SAMPLE_RATE_HZ)
        mic_signals = np.zeros((self.N_MICS, t.size), dtype=np.float32)
        for src_idx, angle_deg in enumerate(angle_degs):
            angle_rad = np.deg2rad(float(angle_deg))
            f = float(freqs[src_idx])
            amp = float(ampls[src_idx])
            cth = np.cos(angle_rad)
            sth = np.sin(angle_rad)
            delays = -((self.x_coords * cth + self.y_coords * sth) / float(self.SPEED_SOUND))
            phase = 2.0 * np.pi * f * (t[None, :] - delays[:, None])
            mic_signals += amp * np.sin(phase).astype(np.float32)
        if self.NOISE_POWER > 0:
            mic_signals += np.random.normal(
                0.0, np.sqrt(self.NOISE_POWER * max(0.0, noise_scale)), mic_signals.shape
            ).astype(np.float32)
        return np.fft.rfft(mic_signals, axis=1).astype(np.complex64)

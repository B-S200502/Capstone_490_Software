"""
Live STM32 FFT Stream Simulator (Multi-Source Version)
------------------------------------------------------
Simulates continuous FFT frames parsed from STM32 output format,
performs Delay-and-Sum, MUSIC, and ESPRIT beamforming in real time
for multiple deterministic plane-wave sources.
"""
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.linalg import eigh, pinv, eig

sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))
from fftframe import FFTFrame


# ===============================================================
# 1. Configuration
# ===============================================================
N_SIDE = 4
MIC_ARRAY_SIZE = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 68001
ARRAY_SIZE = 0.04  # 4 cm × 4 cm
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005
PLOT_INTERVAL = 0.1
COV_AVG_FRAMES = 10

# === Multiple Sources ===
SOURCE_FREQS = [9000, 11000, 30000]  # Hz
SOURCE_ANGLES = [-35.0, 0.0, 40.0]   # degrees initial
N_SOURCES = len(SOURCE_ANGLES)

# ===============================================================
# 2. Geometry setup (Fermat Spiral)
# ===============================================================
N_MICS = 16
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.025  # 5 cm radius (~10 cm diameter)
c = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))

x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

angles = np.linspace(-90, 90, 361)
pitch = np.mean(np.diff(sorted(np.unique(np.sqrt(x_coords**2 + y_coords**2)))))


# ===============================================================
# 3. Multi-source STM32 Frame Generator (deterministic)
# ===============================================================
def generate_fft_frame_from_dataframe(angle_degs):
    """Simulate one STM32 FFT frame with multiple sources."""
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    # Combine all sources
    for src_idx, angle_deg in enumerate(angle_degs):
        angle_rad = np.deg2rad(angle_deg)
        f = SOURCE_FREQS[src_idx]
        for i in range(N_MICS):
            delay = -(x_coords[i] * np.cos(angle_rad) + y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
            delayed_t = t - delay
            mic_signals[i, :] += np.sin(2 * np.pi * f * delayed_t)

    # Add noise once
    mic_signals += np.random.normal(0, np.sqrt(NOISE_POWER), mic_signals.shape)

    # Convert to FFT domain
    fft_data = np.fft.rfft(mic_signals, axis=1)
    frame.fft_data = fft_data.astype(np.complex64)
    return frame


# ===============================================================
# 4. Beamforming algorithms
# ===============================================================
def delay_and_sum(Xf, angles, f_signal):
    power = []
    for ang in angles:
        theta = np.deg2rad(ang)
        phase_shift = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                             (x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        summed = np.sum(Xf.flatten() * phase_shift)
        power.append(np.abs(summed) ** 2)
    return np.array(power) / np.max(power)


def music_v2(R, angles, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]

    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                   -(x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        a = a[:, np.newaxis]
        P = 1 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

    spectrum = np.array(spectrum)
    spectrum /= np.max(spectrum)
    return spectrum


def esprit_v2(R, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    val = -(psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch)
    val = np.clip(np.real(val), -1.0, 1.0)
    theta = np.arcsin(val)
    return np.degrees(theta)


# ===============================================================
# 5. Live plotting setup
# ===============================================================
f_axis = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1 / SAMPLE_RATE_HZ)

plt.ion()
fig, ax = plt.subplots(figsize=(9, 4))
line_das, = ax.plot([], [], label="Delay-and-Sum")
line_music, = ax.plot([], [], label="MUSIC")
ax.set_xlim(-90, 90)
ax.set_ylim(0, 1)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("Normalized Power")
ax.set_title("Live DOA Estimation (Multi-Source FFTFrame-Based)")
ax.legend()
ax.grid(True)
plt.tight_layout()

cov_history = np.zeros((N_MICS, N_MICS), dtype=complex)
true_lines = []
esprit_lines = []

# ===============================================================
# 6. Continuous loop
# ===============================================================
print("Running multi-source FFT simulator (detect all sources)...")


# ===============================================================
# 5b. Persistent plotting setup (multi-source friendly)
# ===============================================================
colors = ["tab:blue", "tab:orange", "tab:green"]
line_music_list, line_das_list = [], []

for k, f_sig in enumerate(SOURCE_FREQS):
    l_das, = ax.plot([], [], lw=2.0, alpha=0.5, color=colors[k],
                     label=f"DAS ({f_sig/1000:.1f} kHz)")
    l_music, = ax.plot([], [], lw=1.5, color=colors[k],
                       label=f"MUSIC ({f_sig/1000:.1f} kHz)")
    line_das_list.append(l_das)
    line_music_list.append(l_music)


try:
    while plt.fignum_exists(fig.number):
        # Animate sources
        for k in range(N_SOURCES):
            SOURCE_ANGLES[k] += (0.15 + 0.05 * k)
            if SOURCE_ANGLES[k] > 90:
                SOURCE_ANGLES[k] = -90

        frame = generate_fft_frame_from_dataframe(SOURCE_ANGLES)

        # Clear dynamic markers
        for ln in true_lines: ln.remove()
        for ln in esprit_lines: ln.remove()
        true_lines.clear()
        esprit_lines.clear()

        # Update each source’s spectrum in place
        for k, f_sig in enumerate(SOURCE_FREQS):
            f_idx = np.argmin(np.abs(f_axis - f_sig))
            Xf = frame.fft_data[:, f_idx][:, np.newaxis]
            R = Xf @ Xf.conj().T

            das_spec = delay_and_sum(Xf, angles, f_sig)
            music_spec = music_v2(R, angles, f_sig, n_sources=N_SOURCES)
            esprit_est = esprit_v2(R, f_sig, n_sources=N_SOURCES)

            # Update both line sets
            line_das_list[k].set_data(angles, das_spec)
            line_music_list[k].set_data(angles, music_spec)

            # Update true-angle and ESPRIT lines
            true_lines.append(ax.axvline(SOURCE_ANGLES[k], color=colors[k],
                                        linestyle=':', lw=1.5))
            # for est in esprit_est:
            #     esprit_lines.append(ax.axvline(est, color=colors[k],
            #                                 linestyle='--', lw=1.0))


        # Refresh plot
        ax.legend(loc="upper right", frameon=False, ncol=1)
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(PLOT_INTERVAL)

except KeyboardInterrupt:
    print("\n⏹ Simulation stopped.")


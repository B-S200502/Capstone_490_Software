"""
Continuous STM32 FFT Stream Simulator (with live ESPRIT plot)
-------------------------------------------------------------
Simulates continuous FFT frames from a 16-mic 4x4 array
and performs real-time beamforming (Delay-and-Sum, MUSIC, ESPRIT).
Press Ctrl-C or close the plot to stop.
"""

import numpy as np
import matplotlib.pyplot as plt
import struct
import time
from numpy.linalg import eigh, pinv, eig

# ===============================================================
# 1. STM32 acquisition parameters
# ===============================================================
N_SIDE = 4
MIC_ARRAY_SIZE = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 72000
ARRAY_SIZE = 0.04
SPEED_SOUND = 343.0
BITS_PER_SAMPLE = 16

# ===============================================================
# 2. Simulation parameters
# ===============================================================
F_SIGNAL = 10000       # 10 kHz tone (below alias limit)
TRUE_ANGLE = 25        # deg (can change dynamically)
NOISE_POWER = 0.002
COV_AVG_FRAMES = 10
PLOT_INTERVAL = 0.1

# ===============================================================
# 3. Array geometry
# ===============================================================
pitch = ARRAY_SIZE / (N_SIDE - 1)
x_coords, y_coords = np.meshgrid(
    np.linspace(-ARRAY_SIZE/2, ARRAY_SIZE/2, N_SIDE),
    np.linspace(-ARRAY_SIZE/2, ARRAY_SIZE/2, N_SIDE)
)
x_coords, y_coords = x_coords.flatten(), y_coords.flatten()
N_MICS = len(x_coords)

# ===============================================================
# 4. Helper: generate synthetic FFT data
# ===============================================================
def generate_fft_frame(angle_deg, f_signal, noise_power):
    """Simulate one FFT frame for a plane wave arriving from angle_deg."""
    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    angle_rad = np.deg2rad(angle_deg)
    mic_signals = np.zeros((N_MICS, len(t)))
    for i in range(N_MICS):
        delay = (x_coords[i]*np.cos(angle_rad) + y_coords[i]*np.sin(angle_rad)) / SPEED_SOUND
        delayed_t = t - delay
        mic_signals[i, :] = np.sin(2*np.pi*f_signal*delayed_t)
        mic_signals[i, :] += np.random.normal(0, np.sqrt(noise_power), len(t))

    mic_signals_int16 = np.int16(mic_signals / np.max(np.abs(mic_signals)) * 32767)
    mic_signals = mic_signals_int16.astype(np.float32) / 32768.0
    fft_data = np.fft.rfft(mic_signals, n=SAMPLES_PER_CHANNEL, axis=1)
    return fft_data

# ===============================================================
# 5. Beamforming algorithms
# ===============================================================
def delay_and_sum(Xf, angles, f_signal):
    power = []
    for ang in angles:
        theta = np.deg2rad(ang)
        phase_shift = np.exp(-1j*2*np.pi*f_signal/SPEED_SOUND *
                             (x_coords*np.cos(theta) + y_coords*np.sin(theta)))
        summed = np.sum(Xf * phase_shift)
        power.append(np.abs(summed)**2)
    return np.array(power) / np.max(power)

def music(R, angles, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    En = eigvecs[:, n_sources:]
    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(-1j*2*np.pi*f_signal/SPEED_SOUND *
                   (x_coords*np.cos(theta) + y_coords*np.sin(theta)))
        a = a[:, np.newaxis]
        P = 1 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])
    return np.array(spectrum) / np.max(spectrum)

def esprit(R, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    theta = np.arcsin((psi * SPEED_SOUND) / (2*np.pi*f_signal*pitch))
    return np.degrees(np.real(theta))

# ===============================================================
# 6. Continuous real-time loop
# ===============================================================
np.random.seed(0)
angles = np.linspace(-90, 90, 361)
f_idx = np.argmin(np.abs(np.fft.rfftfreq(SAMPLES_PER_CHANNEL, 1/SAMPLE_RATE_HZ) - F_SIGNAL))

plt.ion()
fig, ax = plt.subplots(figsize=(9,4))
line_das, = ax.plot([], [], label="Delay-and-Sum")
line_music, = ax.plot([], [], label="MUSIC")
line_esprit = ax.axvline(0, color='m', linestyle='--', label='ESPRIT')
ax.axvline(TRUE_ANGLE, color='k', linestyle='--', label='True Source')
ax.set_xlim(-90, 90)
ax.set_ylim(0, 1)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("Normalized Power")
ax.set_title("Live Beamforming (Delay-and-Sum, MUSIC, ESPRIT)")
ax.legend()
ax.grid(True)
plt.tight_layout()

cov_history = np.zeros((N_MICS, N_MICS), dtype=complex)

print("▶ Running continuous simulator (Ctrl-C or close plot to stop)")
try:
    while plt.fignum_exists(fig.number):
        fft_data = generate_fft_frame(TRUE_ANGLE, F_SIGNAL, NOISE_POWER)
        Xf = fft_data[:, f_idx][:, np.newaxis]
        R_new = Xf @ Xf.conj().T
        cov_history = (cov_history * (COV_AVG_FRAMES - 1) + R_new) / COV_AVG_FRAMES

        das_spectrum = delay_and_sum(Xf.flatten(), angles, F_SIGNAL)
        music_spectrum = music(cov_history, angles, F_SIGNAL)
        esprit_est = esprit(cov_history, F_SIGNAL)

        # Update plots
        line_das.set_data(angles, das_spectrum)
        line_music.set_data(angles, music_spectrum)
        if len(esprit_est) > 0:
            line_esprit.set_xdata([esprit_est[0], esprit_est[0]])
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(PLOT_INTERVAL)
except KeyboardInterrupt:
    print("\n⏹ Simulation stopped.")

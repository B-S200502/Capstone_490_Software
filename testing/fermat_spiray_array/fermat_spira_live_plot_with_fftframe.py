"""
Live STM32 FFT Stream Simulator (with DataFrame Integration)
------------------------------------------------------------
Simulates continuous FFT frames parsed from STM32 output format,
performs Delay-and-Sum, MUSIC, and ESPRIT beamforming in real time.
Deterministic source generation (no random seed variation).
"""
import sys
from pathlib import Path

# Add parent and dataframe folder to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "dataframe"))

from fftframe import FFTFrame

import numpy as np
import matplotlib.pyplot as plt
import time
from numpy.linalg import eigh, pinv, eig


# ===============================================================
# 1. Configuration
# ===============================================================
N_SIDE = 4
MIC_ARRAY_SIZE = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 72000
ARRAY_SIZE = 0.04  # 4 cm × 4 cm
SPEED_SOUND = 343.0
F_SIGNAL = 10000       # Hz
TRUE_ANGLE = 25        # deg
NOISE_POWER = 0.0005
COV_AVG_FRAMES = 10
PLOT_INTERVAL = 0.1

# ===============================================================
# 2. Geometry setup
# ===============================================================
pitch = ARRAY_SIZE / (N_SIDE - 1)
x_coords, y_coords = np.meshgrid(
    np.linspace(-ARRAY_SIZE / 2, ARRAY_SIZE / 2, N_SIDE),
    np.linspace(-ARRAY_SIZE / 2, ARRAY_SIZE / 2, N_SIDE)
)
x_coords, y_coords = x_coords.flatten(), y_coords.flatten()
N_MICS = len(x_coords)
angles = np.linspace(-90, 90, 361)

# ===============================================================
# 3. Fake STM32 Frame Generator (deterministic)
# ===============================================================
def generate_fft_frame_from_dataframe(angle_deg: float) -> FFTFrame:
    """Simulate one STM32 FFT frame as an FFTFrame instance."""
    frame = FFTFrame()
    frame.channel_count = N_MICS
    frame.sampling_rate = SAMPLE_RATE_HZ
    frame.fft_size = SAMPLES_PER_CHANNEL
    frame.frame_id += 1

    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    angle_rad = np.deg2rad(angle_deg)

    # Deterministic (no random noise seed)
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)
    for i in range(N_MICS):
        delay = -(x_coords[i] * np.cos(angle_rad) + y_coords[i] * np.sin(angle_rad)) / SPEED_SOUND
        delayed_t = t - delay
        mic_signals[i, :] = np.sin(2 * np.pi * F_SIGNAL * delayed_t)
        mic_signals[i, :] += np.random.normal(0, np.sqrt(NOISE_POWER), len(t))

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

def music_v1(R, angles, f_signal, n_sources=1):
    # print("DEBUG (R-values):", R)
    eigvals, eigvecs = eigh(R)
    print("DEBUG (eigvecs):", eigvecs)
    idx = eigvals.argsort()[::-1]
    En = eigvecs[:, n_sources:]
    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                   (x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        a = a[:, np.newaxis]
        P = 1 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])
    return np.array(spectrum) / np.max(spectrum)

def music_v2(R, angles, f_signal, n_sources=1):
    # --- Debug: Covariance matrix overview ---
    print("\n=== DEBUG: Covariance Matrix (R) ===")
    np.set_printoptions(precision=3, suppress=True)
    print(R)

    # --- Eigen decomposition ---
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # --- Debug: Eigenvalues summary ---
    print("\n=== DEBUG: Eigenvalues (sorted) ===")
    for i, val in enumerate(eigvals):
        print(f"  λ{i+1:02d}: {val:.5f}")

    # --- Debug: Eigenvectors (magnitude and phase) ---
    print("\n=== DEBUG: Eigenvectors (magnitude & phase per mic) ===")
    for i in range(eigvecs.shape[1]):
        mag = np.abs(eigvecs[:, i])
        phase = np.angle(eigvecs[:, i])
        print(f"\nEigenvector {i+1:02d}:")
        for m in range(len(mag)):
            print(f"  Mic{m+1:02d}: mag={mag[m]:.3f}, phase={phase[m]:+.3f} rad")

    # --- Build noise subspace ---
    En = eigvecs[:, n_sources:]

    # --- MUSIC spectrum calculation ---
    spectrum = []
    for ang in angles:
        theta = np.deg2rad(ang)
        a = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND * -(x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        a = a[:, np.newaxis]
        P = 1 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spectrum.append(P[0, 0])

    spectrum = np.array(spectrum)
    spectrum /= np.max(spectrum)

    # --- Debug: MUSIC peak detection ---
    peak_idx = np.argmax(spectrum)
    peak_angle = angles[peak_idx]
    print(f"\n=== DEBUG: MUSIC Peak ===\n  → Angle = {peak_angle:.2f}°, Power = {spectrum[peak_idx]:.3f}\n")

    return spectrum

def esprit_v1(R, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]
    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)
    theta = np.arcsin((psi * SPEED_SOUND) / (2 * np.pi * f_signal * pitch))
    return np.degrees(np.real(-theta))

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
f_idx = np.argmin(np.abs(f_axis - F_SIGNAL))

plt.ion()
fig, ax = plt.subplots(figsize=(9, 4))
line_das, = ax.plot([], [], label="Delay-and-Sum")
line_music, = ax.plot([], [], label="MUSIC")
line_esprit = ax.axvline(0, color='m', linestyle='--', label='ESPRIT')
line_true = ax.axvline(TRUE_ANGLE, color='k', linestyle='--', label='True Source')
ax.set_xlim(-90, 90)
ax.set_ylim(0, 1)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("Normalized Power")
ax.set_title("Live DOA Estimation (FFTFrame-Based)")
ax.legend()
ax.grid(True)
plt.tight_layout()

cov_history = np.zeros((N_MICS, N_MICS), dtype=complex)

# ===============================================================
# 6. Continuous loop
# ===============================================================
print("▶ Running STM32 FFT frame simulator (Ctrl-C to stop)...")
try:
    while plt.fignum_exists(fig.number):
        TRUE_ANGLE = (TRUE_ANGLE + 0.2) % 90
        frame = generate_fft_frame_from_dataframe(TRUE_ANGLE)

        Xf = frame.fft_data[:, f_idx][:, np.newaxis]

        line_true.set_xdata([TRUE_ANGLE, TRUE_ANGLE])

        # Covariance update
        R_new = Xf @ Xf.conj().T

        # DEBUG : 1
        # cov_history = (cov_history * (COV_AVG_FRAMES - 1) + R_new) / COV_AVG_FRAMES

        # DEBUG : 2
        # alpha = 0.3
        # cov_history = (1 - alpha) * cov_history + alpha * (Xf @ Xf.conj().T)

        # DEBUG : 3
        cov_history = R_new

        das_spectrum = delay_and_sum(Xf, angles, F_SIGNAL)
        music_spectrum = music_v2(cov_history, angles, F_SIGNAL)
        esprit_est = esprit_v2(cov_history, F_SIGNAL)

        # Update plot
        line_das.set_data(angles, das_spectrum)
        line_music.set_data(angles, music_spectrum)
        if len(esprit_est) > 0:
            line_esprit.set_xdata([esprit_est[0], esprit_est[0]])

        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(PLOT_INTERVAL)

except KeyboardInterrupt:
    print("\n⏹ Simulation stopped.")

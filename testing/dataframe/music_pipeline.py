import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigh, pinv, eig
from fftframe import FFTFrame   

# --------------------------------------------------------------------
# 1. Create a fake parsed FFT frame (simulating what would come from STM32)
# --------------------------------------------------------------------
def generate_fake_fft_frame(num_channels=16, fft_bins=1025, f_signal=29500, snr_db=20):
    """
    Generate a fake FFTFrame containing synthetic FFT data for one source.
    """
    frame = FFTFrame()
    frame.channel_count = num_channels
    frame.fft_size = (fft_bins - 1) * 2
    frame.sampling_rate = 72000
    frame.frame_id = 1

    # Simulate phase delays per microphone (for 30° source)
    c = 343.0
    true_angle_deg = 30
    true_angle_rad = np.deg2rad(true_angle_deg)

    # Example Fermat spiral coordinates (reuse from array config)
    golden_angle = np.deg2rad(137.5)
    aperture_radius = 0.025
    r_scale = aperture_radius / np.sqrt(num_channels - 1)
    x_coords = np.array([r_scale * np.sqrt(n) * np.cos(n * golden_angle) for n in range(num_channels)])
    y_coords = np.array([r_scale * np.sqrt(n) * np.sin(n * golden_angle) for n in range(num_channels)])

    # Generate FFT magnitudes and phases for one dominant bin (e.g., near 29.5 kHz)
    fft_data = np.zeros((num_channels, fft_bins), dtype=np.complex64)
    f_bin_index = int((f_signal / frame.sampling_rate) * frame.fft_size)

    for i in range(num_channels):
        phase_shift = -2j * np.pi * f_signal / c * (x_coords[i] * np.cos(true_angle_rad) + y_coords[i] * np.sin(true_angle_rad))
        fft_data[i, f_bin_index] = np.exp(phase_shift)  # coherent phase
        noise = (10 ** (-snr_db / 20)) * (np.random.randn(fft_bins) + 1j * np.random.randn(fft_bins)) / np.sqrt(2)
        fft_data[i, :] += noise

    frame.fft_data = fft_data
    return frame, x_coords, y_coords, f_signal


# --------------------------------------------------------------------
# 2. Perform MUSIC and ESPRIT on parsed FFT data
# --------------------------------------------------------------------
def perform_doa_estimation(frame, x_coords, y_coords, f_signal):
    N = frame.channel_count
    angles = np.linspace(-90, 90, 361)
    c = 343.0

    # Extract FFT snapshot at target bin
    target_bin = np.argmax(np.sum(np.abs(frame.fft_data), axis=0))
    snapshot = frame.fft_data[:, target_bin]

    # Estimate covariance matrix from a single FFT snapshot
    R = np.outer(snapshot, np.conj(snapshot))

    # Eigen decomposition
    eigvals, eigvecs = eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, 1:]  # noise subspace

    # MUSIC pseudo-spectrum
    P_music = []
    for ang in angles:
        theta = np.deg2rad(ang)
        steering = np.exp(-1j * 2 * np.pi * f_signal / c *
                          (x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        steering = steering[:, np.newaxis]
        P = 1 / np.real(np.conj(steering.T) @ En @ En.conj().T @ steering)
        P_music.append(P[0, 0])
    P_music = np.array(P_music)
    P_music /= np.max(P_music)

    # ESPRIT
    Es = eigvecs[:, :1]
    Es1 = Es[:-1]
    Es2 = Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    esprit_angle = np.degrees(np.arcsin((c / (2 * np.pi * f_signal * (0.025 / np.sqrt(N-1)))) * np.angle(eigs_phi)))
    esprit_angle = np.real(esprit_angle[0])

    return angles, P_music, esprit_angle


# --------------------------------------------------------------------
# 3. Visualization
# --------------------------------------------------------------------
def main():
    frame, x_coords, y_coords, f_signal = generate_fake_fft_frame()
    angles, music_spectrum, esprit_angle = perform_doa_estimation(frame, x_coords, y_coords, f_signal)

    true_angle_deg = 30
    plt.figure(figsize=(10,5))
    plt.plot(angles, music_spectrum, label="MUSIC Spectrum")
    plt.axvline(true_angle_deg, color='k', linestyle='--', label='True Source')
    plt.axvline(esprit_angle, color='m', linestyle='--', label=f'ESPRIT ({esprit_angle:.1f}°)')
    plt.title("MUSIC & ESPRIT DOA Estimation (from Parsed Data Frame)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Normalized Power")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

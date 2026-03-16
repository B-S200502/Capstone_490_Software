import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy.linalg import svd, eig

# ----------------------------------
# Fermat Spiral Array Configuration
# -----------------------------------

N = 16                           # number of microphones
golden_angle = np.deg2rad(137.5) # golden angle in radians
aperture_radius = 0.025           # meters (example: 8 cm aperture)
c = aperture_radius / np.sqrt(N-1) # scaling factor to fit aperture

speed_of_sound = 343.0           # m/s
f_max = 36000                    # microphone bandwidth in Hz

x_coords = []
y_coords = []
plt.figure(figsize=(6,6))

for n in range(N):
    r = c * np.sqrt(n)
    theta = n * golden_angle
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    x_coords.append(x)
    y_coords.append(y)
    plt.plot(x, y, 'bo')
    plt.text(x, y, f"({x:.4f}, {y:.4f})", fontsize=7)


# Aperture boundary
circle = plt.Circle((0,0), aperture_radius, color='r',
                    fill=False, linestyle='--', label='Aperture Boundary')
plt.gca().add_artist(circle)

plt.title("16-Mic Fermat Spiral Array")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)
plt.show()

#------------------------------
# --- Scientific Assessment ---
#------------------------------

# 1. Aperture diameter
D = 2 * aperture_radius

# 2. Angular resolution at f_max (radians)
wavelength = speed_of_sound / f_max
ang_res_rad = wavelength / D
ang_res_deg = np.degrees(ang_res_rad)

# 3. Minimum spacing (for aliasing limit)
distances = []
for i in range(N):
    for j in range(i+1, N):
        d = np.sqrt((x_coords[i]-x_coords[j])**2 + (y_coords[i]-y_coords[j])**2)
        distances.append(d)
min_spacing = min(distances)

# spatial aliasing cutoff frequency
f_alias = speed_of_sound / (2 * min_spacing)

print("=== Scientific Assessment ===")
print(f"Aperture diameter: {D:.3f} m")
print(f"Max operating frequency (mic bandwidth): {f_max/1000:.1f} kHz")
print(f"Wavelength at f_max: {wavelength*1000:.2f} mm")
print(f"Angular resolution: {ang_res_deg:.2f} degrees")
print(f"Minimum mic spacing: {min_spacing*1000:.2f} mm")
print(f"Spatial aliasing cutoff: {f_alias/1000:.2f} kHz")
if f_alias < f_max:
    print("Aliasing will occur above cutoff frequency.")
else:
    print("No aliasing expected within mic bandwidth.")

# -------------------------------
# Simulation Parameters
# -------------------------------
fs = 96000            # Sampling rate (Hz)
duration = 0.01       # seconds
f_signal = 29500      # Hz (source frequency)
true_angle_deg = 30   # True azimuth of the source
noise_power = 0.001   # Adjust for SNR

t = np.arange(0, duration, 1/fs)
signal = np.sin(2 * np.pi * f_signal * t)  # base tone

# Convert true angle to radians
true_angle_rad = np.deg2rad(true_angle_deg)

# -------------------------------
# Simulate Received Signals
# -------------------------------
mic_signals = np.zeros((N, len(t)))

for i in range(N):
    x = x_coords[i]
    y = y_coords[i]
    delay = (x * np.cos(true_angle_rad) + y * np.sin(true_angle_rad)) / speed_of_sound
    delayed_t = t - delay
    mic_signal = np.sin(2 * np.pi * f_signal * delayed_t)
    mic_signal += np.random.normal(0, np.sqrt(noise_power), len(t))
    mic_signals[i, :] = mic_signal

# -------------------------------
# Delay-and-Sum Beamforming
# -------------------------------
angles = np.linspace(-90, 90, 361)
power_beam = []

for ang in angles:
    theta = np.deg2rad(ang)
    delays = (np.array(x_coords)*np.cos(theta) + np.array(y_coords)*np.sin(theta)) / speed_of_sound
    aligned = []
    for i in range(N):
        shifted = np.roll(mic_signals[i], int(delays[i]*fs))
        aligned.append(shifted)
    summed = np.sum(aligned, axis=0)
    power_beam.append(np.mean(np.abs(summed)**2))

power_beam = np.array(power_beam)
power_beam /= np.max(power_beam)

# -------------------------------
# MUSIC Algorithm
# -------------------------------
# Covariance matrix
R = np.dot(mic_signals, mic_signals.conj().T) / mic_signals.shape[1]

# Eigen decomposition
eigvals, eigvecs = np.linalg.eigh(R)
idx = eigvals.argsort()[::-1]
eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

# Signal and noise subspaces (assume 1 source)
En = eigvecs[:, 1:]  # Noise subspace

music_spectrum = []
for ang in angles:
    theta = np.deg2rad(ang)
    steering = np.exp(-1j * 2 * np.pi * f_signal / speed_of_sound *
                      (np.array(x_coords)*np.cos(theta) + np.array(y_coords)*np.sin(theta)))
    steering = steering[:, np.newaxis]
    P = 1 / np.real(np.conj(steering.T) @ En @ En.conj().T @ steering)
    music_spectrum.append(P[0, 0])

music_spectrum = np.array(music_spectrum)
music_spectrum /= np.max(music_spectrum)

# -------------------------------
# ESPRIT Algorithm
# -------------------------------
# Subspace
Es = eigvecs[:, :1]
# Form two shifted versions of the subspace
Es1 = Es[:-1]
Es2 = Es[1:]
phi = np.linalg.pinv(Es1) @ Es2
eigs_phi, _ = np.linalg.eig(phi)
angles_esprit = np.angle(eigs_phi)

esprit_angle = np.degrees(np.arcsin((speed_of_sound / (2 * np.pi * f_signal * c)) * angles_esprit))
esprit_angle = np.real(esprit_angle[0])  # single source estimate

# -------------------------------
# Plot Results
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(angles, power_beam, label="Delay-and-Sum Beamformer")
plt.plot(angles, music_spectrum, label="MUSIC Spectrum")
plt.axvline(true_angle_deg, color='k', linestyle='--', label='True Source')
plt.axvline(esprit_angle, color='m', linestyle='--', label=f'ESPRIT ({esprit_angle:.1f}°)')
plt.title("Spatial Spectrum Comparison")
plt.xlabel("Angle (degrees)")
plt.ylabel("Normalized Power")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

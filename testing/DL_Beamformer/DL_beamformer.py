"""
DL-MVDR Beamforming – Multi-Frequency, DOA Classification Version
=================================================================
- Fs = 72 kHz
- Geometry: 16-mic Fermat spiral
- TRAIN_FREQS: one model per frequency (option A)
- Teacher: MVDR beamformer
- Network task: classify the DOA angle bin that MVDR peaks at
- Live demo: single source emitting all TRAIN_FREQS, wideband DAS/MUSIC/DL-MVDR
"""

import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from numpy.linalg import pinv, eigh

# ===============================================================
# 1. GLOBAL CONFIG
# ===============================================================
N_MICS = 16
SAMPLES_PER_CHANNEL = 1024
SAMPLE_RATE_HZ = 72000.0      # 72 kHz
SPEED_SOUND = 343.0
NOISE_POWER = 0.0005

TRAIN_FREQS = [9000.0, 11000.0, 30000.0]   # Hz, edit as desired

ANGLES = np.linspace(-90, 90, 361)         # 0.5° resolution
N_ANGLES = len(ANGLES)

F_AXIS = np.fft.rfftfreq(SAMPLES_PER_CHANNEL, d=1.0 / SAMPLE_RATE_HZ)

# ===============================================================
# 2. GEOMETRY – Fermat spiral
# ===============================================================
golden_angle = np.deg2rad(137.5)
aperture_radius = 0.025
c_geom = aperture_radius / np.sqrt(N_MICS - 1)

x_coords, y_coords = [], []
for n in range(N_MICS):
    r = c_geom * np.sqrt(n)
    theta = n * golden_angle
    x_coords.append(r * np.cos(theta))
    y_coords.append(r * np.sin(theta))
x_coords = np.array(x_coords)
y_coords = np.array(y_coords)

# ===============================================================
# 3. SIGNAL GENERATORS
# ===============================================================
def generate_single_tone_frame(angle_deg: float, f_sig: float):
    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    ang_rad = np.deg2rad(angle_deg)
    for i in range(N_MICS):
        delay = -(x_coords[i] * np.cos(ang_rad) +
                  y_coords[i] * np.sin(ang_rad)) / SPEED_SOUND
        mic_signals[i, :] = np.sin(2 * np.pi * f_sig * (t - delay))

    mic_signals += np.random.normal(0.0, np.sqrt(NOISE_POWER), mic_signals.shape)
    fft_data = np.fft.rfft(mic_signals, axis=1)
    return fft_data.astype(np.complex64)


def generate_multitone_frame(angle_deg: float):
    """Source emits all TRAIN_FREQS at the same DOA."""
    t = np.arange(SAMPLES_PER_CHANNEL) / SAMPLE_RATE_HZ
    mic_signals = np.zeros((N_MICS, len(t)), dtype=np.float32)

    ang_rad = np.deg2rad(angle_deg)
    for i in range(N_MICS):
        delay = -(x_coords[i] * np.cos(ang_rad) +
                  y_coords[i] * np.sin(ang_rad)) / SPEED_SOUND
        delayed_t = t - delay
        for f_sig in TRAIN_FREQS:
            mic_signals[i, :] += np.sin(2 * np.pi * f_sig * delayed_t)

    mic_signals += np.random.normal(0.0, np.sqrt(NOISE_POWER), mic_signals.shape)
    fft_data = np.fft.rfft(mic_signals, axis=1)
    return fft_data.astype(np.complex64)

# ===============================================================
# 4. CLASSICAL BEAMFORMERS
# ===============================================================
def delay_and_sum(Xf, ang_grid, f_signal):
    power = []
    for ang in ang_grid:
        theta = np.deg2rad(ang)
        phase = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                       (x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        s = np.sum(Xf.flatten() * phase)
        power.append(np.abs(s) ** 2)
    power = np.array(power)
    return power / (np.max(power) + 1e-12)


def music_spectrum(R, ang_grid, f_signal, n_sources=1):
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]

    spec = []
    for ang in ang_grid:
        theta = np.deg2rad(ang)
        a = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                   -(x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        a = a[:, None]
        P = 1.0 / np.real(a.conj().T @ En @ En.conj().T @ a)
        spec.append(P[0, 0])
    spec = np.array(spec)
    return spec / (np.max(spec) + 1e-12)


def mvdr_spectrum(R, ang_grid, f_signal):
    M = R.shape[0]
    R_reg = R + 1e-3 * np.eye(M)
    Rinv = pinv(R_reg)

    spec = []
    for ang in ang_grid:
        theta = np.deg2rad(ang)
        a = np.exp(-1j * 2 * np.pi * f_signal / SPEED_SOUND *
                   (x_coords * np.cos(theta) + y_coords * np.sin(theta)))
        a = a[:, None]
        P = 1.0 / np.real(a.conj().T @ Rinv @ a)
        spec.append(P[0, 0])
    spec = np.array(spec)
    return spec / (np.max(spec) + 1e-12)

# ===============================================================
# 5. DATASET – DOA classification per frequency
# ===============================================================
class BeamDataset(Dataset):
    """
    For a given frequency:
      - Random DOA in [-75, 75]
      - Build covariance over `n_cov_frames` frames
      - Compute MVDR spectrum, label = argmax index (DOA bin)
      - Input = real+imag of last frame snapshot
    """

    def __init__(self, n_samples: int, f_signal: float,
                 n_cov_frames: int = 12):
        self.X = []
        self.labels = []

        f_idx = np.argmin(np.abs(F_AXIS - f_signal))

        for _ in range(n_samples):
            src_angle = np.random.uniform(-75.0, 75.0)

            # Temporal covariance averaging
            R = np.zeros((N_MICS, N_MICS), dtype=np.complex128)
            last_snapshot = None
            for _ in range(n_cov_frames):
                frame = generate_single_tone_frame(src_angle, f_signal)
                Xf = frame[:, f_idx][:, None]
                R += Xf @ Xf.conj().T
                last_snapshot = Xf
            R /= n_cov_frames

            mvdr_spec = mvdr_spectrum(R, ANGLES, f_signal)
            doa_idx = int(np.argmax(mvdr_spec))    # class label

            Xc = last_snapshot.squeeze(1)
            x_in = np.stack([np.real(Xc), np.imag(Xc)], axis=0)
            x_in = x_in / (np.linalg.norm(x_in) + 1e-8)

            self.X.append(x_in.astype(np.float32))
            self.labels.append(doa_idx)

        self.X = np.stack(self.X, axis=0)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

# ===============================================================
# 6. MODEL – outputs logits over angles
# ===============================================================
class DOANet(nn.Module):
    def __init__(self, n_mics: int, n_angles: int):
        super().__init__()
        self.conv1 = nn.Conv1d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, n_angles)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)     # (B, n_angles)
        return logits

# ===============================================================
# 7. TRAIN ONE MODEL PER FREQUENCY
# ===============================================================
def train_model_for_frequency(f_sig: float,
                              n_samples: int = 2000,
                              n_epochs: int = 20,
                              batch_size: int = 32,
                              device: str = "cpu"):

    print(f"\n[TRAIN] Frequency = {f_sig/1000:.1f} kHz")
    dataset = BeamDataset(n_samples=n_samples, f_signal=f_sig,
                          n_cov_frames=12)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DOANet(N_MICS, N_ANGLES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()

        avg_loss = total_loss / len(dataset)
        acc = correct / len(dataset) * 100.0
        print(f"  Epoch {epoch:02d} | Loss={avg_loss:.4f} | Acc={acc:.1f}%")

    fname = f"doa_mvdr_{int(f_sig)}Hz.pth"
    torch.save(model.state_dict(), fname)
    print(f"  [✓] Saved {f_sig/1000:.1f} kHz model -> {fname}")
    return model.eval()

# ===============================================================
# 8. MAIN – TRAIN ALL FREQUENCIES, THEN LIVE DEMO
# ===============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    trained_models = {}

    print("\n[1/2] Training DOA models (MVDR teacher, multi-frequency)...")
    for f in TRAIN_FREQS:
        trained_models[f] = train_model_for_frequency(
            f_sig=f,
            n_samples=2000,
            n_epochs=20,
            batch_size=32,
            device=device
        )

    # ---------------- LIVE DEMO ----------------
    print("\n[2/2] Running live wideband demo...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 4))

    line_das, = ax.plot([], [], label="DAS (wideband)")
    line_music, = ax.plot([], [], label="MUSIC (wideband)")
    line_dl, = ax.plot([], [], label="DL-MVDR (wideband)", linestyle="--")

    ax.set_xlim(-90, 90)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Angle (°)")
    ax.set_ylabel("Normalized Power")
    ax.grid(True)
    ax.legend()

    true_line = None
    test_angle = -70.0

    while True:
        test_angle += 1.0
        if test_angle > 70.0:
            test_angle = -70.0

        frame = generate_multitone_frame(test_angle)

        wide_das = np.zeros(N_ANGLES)
        wide_music = np.zeros(N_ANGLES)
        wide_dl = np.zeros(N_ANGLES)

        for f_sig in TRAIN_FREQS:
            f_idx = np.argmin(np.abs(F_AXIS - f_sig))
            Xf = frame[:, f_idx][:, None]
            R = Xf @ Xf.conj().T

            das_spec = delay_and_sum(Xf, ANGLES, f_sig)
            music_spec = music_spectrum(R, ANGLES, f_sig, n_sources=1)

            Xc = Xf.squeeze(1)
            x = np.stack([np.real(Xc), np.imag(Xc)], axis=0)
            x = x / (np.linalg.norm(x) + 1e-8)
            xt = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = trained_models[f_sig](xt)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

            wide_das += das_spec
            wide_music += music_spec
            wide_dl += probs

        wide_das /= (wide_das.max() + 1e-12)
        wide_music /= (wide_music.max() + 1e-12)
        wide_dl /= (wide_dl.max() + 1e-12)

        line_das.set_data(ANGLES, wide_das)
        line_music.set_data(ANGLES, wide_music)
        line_dl.set_data(ANGLES, wide_dl)

        if true_line is not None:
            true_line.remove()
        true_line = ax.axvline(test_angle, color="red",
                               linestyle=":", linewidth=1.5)

        ax.set_title(f"True Angle = {test_angle:.1f}°")
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

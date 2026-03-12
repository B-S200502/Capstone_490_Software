# dsp/beamforming.py
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, pinv, eig


def directivity_ratio(R: np.ndarray) -> float:
    """
    Ratio of largest eigenvalue to sum of all eigenvalues (0 to 1).
    Near 1 = directional (rank-1 like); near 1/M = diffuse. Used to gate diffuse noise.
    """
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1] or R.shape[0] < 2:
        return 0.0
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.maximum(eigvals.real, 0.0)
    total = float(np.sum(eigvals)) + 1e-12
    lam1 = float(np.max(eigvals))
    return lam1 / total


def music_spectrum(
    R: np.ndarray,
    angles: np.ndarray,
    f_signal: float,
    n_sources: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    speed_sound: float,
) -> np.ndarray:
    """
    Vectorized MUSIC spectrum over all angles (no Python loop).
    Output: float32 array (len(angles),), normalized so max=1.

    No globals: x_coords/y_coords/speed_sound are passed in.
    """
    angles = np.asarray(angles)
    A_len = int(angles.size)
    if A_len == 0:
        return np.zeros((0,), dtype=np.float32)

    # Validate R shape
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return np.zeros((A_len,), dtype=np.float32)

    M = int(R.shape[0])
    if M < 2:
        return np.zeros((A_len,), dtype=np.float32)

    # Clamp n_sources
    n_sources = int(np.clip(n_sources, 1, M - 1))

    # Eigendecomposition
    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]

    # Noise subspace projector
    En = eigvecs[:, n_sources:]               # (M, M-nsrc)
    Pn = En @ En.conj().T                     # (M, M)

    # Steering vectors for all angles
    theta = np.deg2rad(angles).astype(np.float32)  # (A,)
    cth = np.cos(theta).astype(np.float32)
    sth = np.sin(theta).astype(np.float32)

    x_coords = np.asarray(x_coords, dtype=np.float32).reshape(M)
    y_coords = np.asarray(y_coords, dtype=np.float32).reshape(M)

    proj = (x_coords[:, None] * cth[None, :] +
            y_coords[:, None] * sth[None, :]).astype(np.float32)  # (M, A)

    k = (2.0 * np.pi * float(f_signal) / float(speed_sound))
    steering = np.exp(1j * k * proj).astype(np.complex64)         # (M, A)

    # MUSIC pseudospectrum: 1 / Re(a^H Pn a)
    PA = Pn @ steering
    denom = np.einsum("ma,ma->a", steering.conj(), PA).real
    denom = np.maximum(denom, 1e-12)

    spec = (1.0 / denom).astype(np.float32)

    # Normalize
    m = float(spec.max()) if spec.size else 1.0
    spec /= (m + 1e-12)
    return spec


def esprit_estimate(
    R: np.ndarray,
    f_signal: float,
    n_sources: int,
    pitch: float,
    speed_sound: float,
) -> np.ndarray:
    """
    ESPRIT estimate (returns degrees).
    No globals: pitch/speed_sound passed in.
    """
    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return np.zeros((max(1, int(n_sources)),), dtype=np.float32)

    M = int(R.shape[0])
    if M < 2:
        return np.zeros((max(1, int(n_sources)),), dtype=np.float32)

    n_sources = int(np.clip(n_sources, 1, M - 1))

    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    Es = eigvecs[:, :n_sources]

    Es1, Es2 = Es[:-1], Es[1:]
    phi = pinv(Es1) @ Es2
    eigs_phi, _ = eig(phi)
    psi = np.angle(eigs_phi)

    if float(pitch) == 0.0:
        return np.zeros((n_sources,), dtype=np.float32)

    val = -(psi * float(speed_sound)) / (2 * np.pi * float(f_signal) * float(pitch))
    val = np.clip(np.real(val), -1.0, 1.0)

    theta = np.arcsin(val)
    return np.degrees(theta).astype(np.float32)

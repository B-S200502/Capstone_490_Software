# dsp/beamforming.py
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh, pinv, eig

# Steering matrix cache for 2D MUSIC: key (f_signal, Nx, Ny) -> steering_flat (M, Nx*Ny)
_STEERING_2D_CACHE: dict[tuple[float, int, int], np.ndarray] = {}
_STEERING_2D_CACHE_ORDER: list[tuple[float, int, int]] = []
_STEERING_2D_CACHE_MAX = 32


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


def music_spectrum_2d(
    R: np.ndarray,
    angles_x: np.ndarray,
    angles_y: np.ndarray,
    f_signal: float,
    n_sources: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    speed_sound: float,
) -> np.ndarray:
    """
    2D MUSIC spectrum over (θ_x, θ_y). Steering phase at mic i:
    k * (x_i * sin(θ_x) + y_i * sin(θ_y)).
    Output: float32 (len(angles_x), len(angles_y)), max=1.
    Steering matrix is cached by (f_signal, Nx, Ny) to avoid recomputing per frame.
    """
    angles_x = np.asarray(angles_x)
    angles_y = np.asarray(angles_y)
    Nx, Ny = int(angles_x.size), int(angles_y.size)
    if Nx == 0 or Ny == 0:
        return np.zeros((Nx, Ny), dtype=np.float32)

    R = np.asarray(R)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        return np.zeros((Nx, Ny), dtype=np.float32)

    M = int(R.shape[0])
    if M < 2:
        return np.zeros((Nx, Ny), dtype=np.float32)

    n_sources = int(np.clip(n_sources, 1, M - 1))
    f_sig = float(f_signal)
    cache_key = (f_sig, Nx, Ny)

    # Reuse cached steering matrix if available (same freq + grid)
    if cache_key in _STEERING_2D_CACHE:
        steering_flat = _STEERING_2D_CACHE[cache_key]
    else:
        x_coords = np.asarray(x_coords, dtype=np.float32).reshape(M)
        y_coords = np.asarray(y_coords, dtype=np.float32).reshape(M)
        k = 2.0 * np.pi * f_sig / float(speed_sound)
        theta_x = np.deg2rad(angles_x).astype(np.float32)
        theta_y = np.deg2rad(angles_y).astype(np.float32)
        sx = np.sin(theta_x)
        sy = np.sin(theta_y)
        proj = (
            x_coords[:, None, None] * sx[None, :, None] +
            y_coords[:, None, None] * sy[None, None, :]
        ).astype(np.float32)
        steering = np.exp(1j * k * proj).astype(np.complex64)
        steering_flat = steering.reshape(M, Nx * Ny).copy()
        # Bounded cache: evict oldest
        _STEERING_2D_CACHE[cache_key] = steering_flat
        _STEERING_2D_CACHE_ORDER.append(cache_key)
        while len(_STEERING_2D_CACHE_ORDER) > _STEERING_2D_CACHE_MAX:
            old_key = _STEERING_2D_CACHE_ORDER.pop(0)
            _STEERING_2D_CACHE.pop(old_key, None)

    eigvals, eigvecs = eigh(R)
    idx = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, idx]
    En = eigvecs[:, n_sources:]
    Pn = En @ En.conj().T

    PA = Pn @ steering_flat
    denom = np.einsum("ma,ma->a", steering_flat.conj(), PA).real
    denom = np.maximum(denom, 1e-12)
    spec_flat = (1.0 / denom).astype(np.float32)
    spec = spec_flat.reshape(Nx, Ny)

    m = float(spec.max()) if spec.size else 1.0
    spec /= (m + 1e-12)
    return spec


def music_2d_peak_angles(
    spec_2d: np.ndarray,
    angles_x: np.ndarray,
    angles_y: np.ndarray,
) -> tuple[float, float]:
    """
    Find peak in 2D MUSIC spectrum and optionally refine with parabolic interpolation.
    Returns (angle_x_deg, angle_y_deg) in degrees.
    """
    spec_2d = np.asarray(spec_2d)
    angles_x = np.asarray(angles_x)
    angles_y = np.asarray(angles_y)
    Nx, Ny = spec_2d.shape[0], spec_2d.shape[1]
    if Nx == 0 or Ny == 0:
        return 0.0, 0.0

    ix = int(np.argmax(spec_2d) // Ny)
    iy = int(np.argmax(spec_2d) % Ny)

    def parabolic_offset(arr: np.ndarray, idx: int, size: int) -> float:
        y0 = float(arr[idx - 1]) if idx > 0 else float(arr[idx])
        y1 = float(arr[idx])
        y2 = float(arr[idx + 1]) if idx < size - 1 else float(arr[idx])
        if idx > 0 and idx < size - 1:
            denom = y0 - 2.0 * y1 + y2
            if abs(denom) >= 1e-12:
                d = 0.5 * (y0 - y2) / denom
                d = np.clip(d, -0.5, 0.5)
                return float(idx) + d
        return float(idx)

    # Refine in x (row at iy)
    row = spec_2d[:, iy]
    ix_frac = parabolic_offset(row, ix, Nx)
    ix_frac = np.clip(ix_frac, 0.0, float(Nx - 1))
    # Refine in y (column at ix)
    col = spec_2d[ix, :]
    iy_frac = parabolic_offset(col, iy, Ny)
    iy_frac = np.clip(iy_frac, 0.0, float(Ny - 1))

    # Map fractional index to angle (same convention as 1D: index 0 -> first angle, N-1 -> last)
    span_x = float(angles_x[-1]) - float(angles_x[0]) if Nx > 1 else 0.0
    span_y = float(angles_y[-1]) - float(angles_y[0]) if Ny > 1 else 0.0
    angle_x_deg = float(angles_x[0]) + span_x * ix_frac / max(1, Nx - 1)
    angle_y_deg = float(angles_y[0]) + span_y * iy_frac / max(1, Ny - 1)
    return angle_x_deg, angle_y_deg


def music_spectrum_2d_refined(
    R: np.ndarray,
    angles_x: np.ndarray,
    angles_y: np.ndarray,
    f_signal: float,
    n_sources: int,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    speed_sound: float,
    coarse_resolution: int = 21,
    refine_half_width: int = 2,
) -> tuple[np.ndarray, float, float]:
    """
    Two-stage 2D MUSIC: coarse grid then small fine patch around peak.
    Returns (spec_coarse, angle_x_deg, angle_y_deg). spec_coarse shape (coarse_resolution, coarse_resolution)
    for display; angles are refined from the fine patch.
    """
    angles_x = np.asarray(angles_x)
    angles_y = np.asarray(angles_y)
    x_min, x_max = float(angles_x[0]), float(angles_x[-1])
    y_min, y_max = float(angles_y[0]), float(angles_y[-1])
    coarse_resolution = max(3, int(coarse_resolution))
    refine_half_width = max(0, int(refine_half_width))
    coarse_x = np.linspace(x_min, x_max, coarse_resolution, dtype=np.float64)
    coarse_y = np.linspace(y_min, y_max, coarse_resolution, dtype=np.float64)
    spec_coarse = music_spectrum_2d(
        R, coarse_x, coarse_y, f_signal, n_sources,
        x_coords, y_coords, speed_sound,
    )
    Ncx, Ncy = spec_coarse.shape[0], spec_coarse.shape[1]
    flat_idx = int(np.argmax(spec_coarse))
    ix = flat_idx // Ncy
    iy = flat_idx % Ncy
    ix = np.clip(ix, 0, Ncx - 1)
    iy = np.clip(iy, 0, Ncy - 1)
    angle_x_center = float(coarse_x[ix])
    angle_y_center = float(coarse_y[iy])
    if refine_half_width <= 0:
        angle_x_deg, angle_y_deg = music_2d_peak_angles(spec_coarse, coarse_x, coarse_y)
        return spec_coarse, angle_x_deg, angle_y_deg
    step_x = (x_max - x_min) / max(1, coarse_resolution - 1)
    step_y = (y_max - y_min) / max(1, coarse_resolution - 1)
    delta_x = step_x * (refine_half_width + 0.5)
    delta_y = step_y * (refine_half_width + 0.5)
    n_fine = 2 * refine_half_width + 1
    fine_x = np.linspace(
        max(x_min, angle_x_center - delta_x),
        min(x_max, angle_x_center + delta_x),
        n_fine,
        dtype=np.float64,
    )
    fine_y = np.linspace(
        max(y_min, angle_y_center - delta_y),
        min(y_max, angle_y_center + delta_y),
        n_fine,
        dtype=np.float64,
    )
    spec_fine = music_spectrum_2d(
        R, fine_x, fine_y, f_signal, n_sources,
        x_coords, y_coords, speed_sound,
    )
    angle_x_deg, angle_y_deg = music_2d_peak_angles(spec_fine, fine_x, fine_y)
    return spec_coarse, angle_x_deg, angle_y_deg


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

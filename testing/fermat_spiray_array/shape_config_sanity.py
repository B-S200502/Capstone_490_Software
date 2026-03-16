import numpy as np
from tabulate import tabulate

# ====== CONFIGURATION ======
N = 16  # microphones
golden_angle = np.deg2rad(137.5)

speed_of_sound = 343.0
f_max = 36000  # mic bandwidth

# Soft aliasing thresholds (Hz)
alias_pass = 26000   # PASS if alias cutoff >= 26 kHz
alias_warn = 20000   # WARN if 20–26 kHz
# <20 kHz = FAIL

# Physical microphone package constraint
mic_body_mm = 3.5  # 3.5 mm body length
min_spacing_required = (mic_body_mm + 0.7) / 1000   # ~4.2 mm realistic

# Aperture radii to test (meters) → diameter = 4 cm to 12 cm
test_radii = np.linspace(0.02, 0.06, 9)


# ====== HELPER FUNCTIONS ======
def fermat_positions(N, radius):
    """Return mic coordinates (x,y) for a Fermat spiral of given max radius."""
    c = radius / np.sqrt(N - 1)
    xs, ys = [], []
    for n in range(N):
        r = c * np.sqrt(n)
        theta = n * golden_angle
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
    return np.array(xs), np.array(ys)


def min_pairwise_distance(xs, ys):
    """Compute minimum inter-microphone spacing (corrected!)."""
    min_d = 1e9
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            # FIXED: correct coordinate subtraction
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < min_d:
                min_d = d
    return min_d


# ====== SWEEP ======
rows = []

for R in test_radii:
    xs, ys = fermat_positions(N, R)
    min_d = min_pairwise_distance(xs, ys)

    wavelength = speed_of_sound / f_max
    D = 2 * R  # diameter
    ang_res_deg = np.degrees(wavelength / D)

    # Aliasing cutoff
    f_alias = speed_of_sound / (2 * min_d)

    # ----- SOFT RATING -----
    if f_alias >= alias_pass:
        rating = "PASS"
    elif f_alias >= alias_warn:
        rating = "WARN"
    else:
        rating = "FAIL"

    rows.append([
        f"{R:.3f}",
        f"{D*100:.1f} cm",
        f"{ang_res_deg:5.2f}°",
        f"{min_d*1000:5.2f} mm",
        f"{f_alias/1000:6.1f} kHz",
        rating
    ])

# Pretty table output
print(tabulate(rows, headers=[
    "Radius (m)", "Diameter", "AngRes", "MinSpacing", "AliasCutoff", "Rating"
]))

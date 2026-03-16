import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ====== CONFIGURATION ======
N = 16  # microphones
golden_angle = np.deg2rad(137.5)

speed_of_sound = 343.0
f_max = 36000  # mic bandwidth

# Soft aliasing thresholds (Hz)
alias_pass = 26000
alias_warn = 20000

# Physical MEMS microphone size
mic_length = 3.7 / 1000   # meters
mic_width  = 2.9 / 1000  # meters
mic_radius_effective = np.sqrt((mic_length/2)**2 + (mic_width/2)**2)

# Minimum spacing requirement (edge-to-edge)
min_spacing_required = mic_length + 0.7/1000  # 3.5mm + 0.7mm buffer

# Radii to sweep
test_radii = np.linspace(0.02, 0.06, 9)


# ==========================================================
# 🌀 FERMAT SPIRAL PLACEMENT
# ==========================================================
def fermat_positions(N, radius):
    c = radius / np.sqrt(N - 1)
    xs, ys = [], []
    for n in range(N):
        r = c * np.sqrt(n)
        theta = n * golden_angle
        xs.append(r * np.cos(theta))
        ys.append(r * np.sin(theta))
    return np.array(xs), np.array(ys)


# ==========================================================
# 🔍 COMPUTE MIN SPACING
# ==========================================================
def min_pairwise_distance(xs, ys):
    min_d = 1e9
    for i in range(len(xs)):
        for j in range(i+1, len(xs)):
            d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
            if d < min_d:
                min_d = d
    return min_d


# ==========================================================
#  SWEEP RESULTS TABLE
# ==========================================================
rows = []

for R in test_radii:
    xs, ys = fermat_positions(N, R)
    min_d = min_pairwise_distance(xs, ys)

    D = 2 * R
    wavelength = speed_of_sound / f_max
    ang_res_deg = np.degrees(wavelength / D)

    f_alias = speed_of_sound / (2 * min_d)

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

print(tabulate(rows, headers=[
    "Radius (m)", "Diameter", "AngRes", "MinSpacing", "AliasCutoff", "Rating"
]))


# =======================================================
# CHOOSE RADIUS TO VISUALIZE & CHECK COLLISIONS ====
# =======================================================
R_plot = 0.025  # 5 cm diameter
xs, ys = fermat_positions(N, R_plot)

# =======================================================
# COLLISION DETECTION
# =======================================================
print("\n=== Collision Detection (for R = %.3f m) ===" % R_plot)

collision_rows = []

for i in range(N):
    for j in range(i+1, N):
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        dist = np.hypot(dx, dy)

        # Edge-to-edge clearance
        clearance = dist - (mic_length)

        if clearance < 0:
            status = "COLLISION"
        elif clearance < 0.001:
            status = "WARNING"
        else:
            status = "OK"

        collision_rows.append([
            f"U{i+1}-U{j+1}",
            f"{dist*1000:6.2f} mm",
            f"{clearance*1000:6.2f} mm",
            status
        ])

print(tabulate(collision_rows, headers=["Pair", "CenterDist", "Clearance", "Status"]))


# =======================================================
# PRINT COORDINATES (mm)
# =======================================================
print("\n=== Microphone Coordinates (mm) for R = %.1f mm (Diameter = %.1f mm) ===" %
      (R_plot * 1000, 2 * R_plot * 1000))

coord_rows = []
for i in range(N):
    coord_rows.append([
        f"U{i+1}",
        f"{xs[i]*1000:.2f}",   # X in mm
        f"{ys[i]*1000:.2f}",   # Y in mm
    ])

print(tabulate(coord_rows, headers=["Mic", "X (mm)", "Y (mm)"]))



# =======================================================
# PLOT ARRAY WITH LABELS
# =======================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.set_title(f"Fermat Spiral Microphone Geometry (Diameter = {2*R_plot*100:.1f} cm)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

# Boundary
circle = plt.Circle((0,0), R_plot, color='red', fill=False, linestyle='--')
ax.add_artist(circle)

# Draw microphones and labels
# Draw microphones and labels
for idx, (x, y) in enumerate(zip(xs, ys)):
    # Draw microphone body
    rect = patches.Rectangle(
        (x - mic_length/2, y - mic_width/2),
        mic_length, mic_width,
        edgecolor="blue",
        facecolor="cyan",
        alpha=0.6
    )
    ax.add_patch(rect)
    ax.plot(x, y, 'ko')

    # === Improved coordinate labeling ===
    label = f"U{idx+1}\n({x*1000:.1f}, {y*1000:.1f}) mm"

    ax.text(
        x + 0.001,      # 3 mm horizontal offset
        y + 0.001,      # 3 mm vertical offset
        label,
        fontsize=7,
        ha='left',
        va='bottom',
        bbox=dict(
            facecolor='white',
            edgecolor='black',
            boxstyle='round,pad=0.2',
            alpha=0.7
        )
    )


ax.set_aspect("equal", "box")
ax.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Parameters
N = 16                             # number of microphones
golden_angle = np.deg2rad(137.5)   # golden angle in radians
aperture_radius = 0.025            # meters (example: 5 cm aperture)
c = aperture_radius / np.sqrt(N-1) # scaling factor to fit aperture

speed_of_sound = 343.0             # m/s
f_max = 36000                      # microphone bandwidth in Hz

#TODO: We need to treat each point as a camera
#that includes dimensions of each mic's packaging
#so we can calculate space between mics more accurately for PCB design
# (keep in mind the points right now represent the center of each microphone).

# Generate mic positions
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

# --- Scientific Assessment ---
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

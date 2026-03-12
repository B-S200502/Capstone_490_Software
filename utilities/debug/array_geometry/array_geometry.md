# Array geometry reference

This document explains the output of `array_geometry.py`: the **table** (and CSV) and the **MUSIC summary**. Use it as a reference when calibrating the heatmap or feeding geometry into the beamformer.

---

## 1. What the table is showing

The table is in the **flipped array-center frame**: origin at the array center, with the 180° flip applied so “camera top-left” matches your physical setup.

### Column meanings

| Column | Meaning |
|--------|--------|
| **ch** | Channel index 0–15. Order of mics in software/SPI (channel 0 = first mic, etc.). |
| **label** | Physical mic name from the board (U1, U2, U6, …, U19). |
| **x_mm, y_mm** | Mic position in the **flipped** array-center frame, in mm. Negative values because after the flip the mics are in the “negative” quadrant relative to the chosen axes. |
| **x_m, y_m** | Same position in **meters**. This is the form used by MUSIC (SI units). |
| **dist_to_camera_mm** | Straight-line distance (mm) from the **camera** to that mic: `sqrt(x_cam² + y_cam² + z_cam²)` from the original camera-frame measurement. |
| **dist_to_center_mm** | 2D distance (mm) from the **array center** to that mic in the array plane. |
| **azimuth_deg** | Azimuth angle of the mic in the flipped frame (see below). |
| **min_dist_mm** | The “Min Dist. (mm)” from your FreeCAD CSV (measured camera–mic distance). Lets you cross-check the script against the raw data. |

---

## 2. What “azimuth_deg” is

In the script it’s computed as:

```python
azimuth_deg = math.degrees(math.atan2(y_f, x_f))
```

So it’s the **angle in the XY plane** from the positive x-axis to the mic, in degrees:

- **0°** = +x  
- **90°** = +y  
- **±180°** = −x  
- **−90°** = −y  

Your values (about −134° to −156°) put all mics in the **third quadrant** (negative x, negative y) in the flipped frame, which matches the geometry.

It’s the same “angle from the array” that shows up when you think of a source at some azimuth: MUSIC and your heatmap work in angles, so having each mic’s azimuth in this table helps you sanity-check that the layout (and any angle convention) is consistent.

---

## 3. Why these columns matter

| What | Why it matters |
|------|----------------|
| **x_mm, y_mm, x_m, y_m** | Define the array geometry in one place. **x_m / y_m** are exactly what you feed into MUSIC. |
| **dist_to_camera_mm / dist_to_center_mm** | For calibration and sanity checks (relative distances, symmetry, alignment with the camera). |
| **azimuth_deg** | Links geometry to angle space (DOA, heatmap direction). |
| **ch + label** | Map channel index (0–15) to physical mic (U1…U19) so firmware, config, and MUSIC all agree on which channel is which mic. |
| **min_dist_mm** | Direct check against your FreeCAD measurements. |

The table is both a **geometry reference** and a **cross-check** for calibration and for feeding the right numbers into the pipeline.

---

## 4. What the MUSIC summary is showing

The MUSIC summary is the **exact input set** your beamformer expects.

### 4.1. x_coords (m) and y_coords (m)

- **What:** Two length-16 arrays of mic positions in **meters**, in **channel order** (index 0 = U1, 1 = U2, …, 15 = U19).
- **Where it’s used:** In `beamforming.music_spectrum()` (and related code) these are the `x_coords` and `y_coords` arguments. For each test angle θ, the code builds the steering vector using  
  `k * (x_coords * cos(θ) + y_coords * sin(θ))`  
  so the phase delay at each mic is determined by these coordinates and the wavenumber *k*.
- **Why it matters:** MUSIC’s direction-of-arrival (DOA) and your heatmap directions depend entirely on this geometry. If you paste these arrays into your config (or calibration module), the pipeline uses your **measured** array layout instead of a generic Fermat spiral, which is what you want for calibrating the heatmap to the real hardware.

### 4.2. pitch (m)

- **What:** Mean **radial spacing** between mics (in meters), computed from the same flipped positions (e.g. from sorted radii).
- **Why it matters:** Used by **ESPRIT** and similar methods that assume or exploit a characteristic spacing (e.g. for a ULA or for wavelength/spacing checks). Having it printed lets you compare with config and with theory.

### 4.3. Channel order line

- **What:** Explicit mapping: `0=U1, 1=U2, … 8=U11, … 15=U19`.
- **Why it matters:** Ensures that whenever you use `x_coords[i]` / `y_coords[i]` (or the i-th FFT channel), you know which physical mic that is. The same mapping must be used in config, SPI parsing, and MUSIC so that the geometry and the data channels stay aligned.

---

## 5. Summary

- **Table:** Documents the full geometry and distances/angles for checking and calibration.
- **MUSIC summary:** Gives you the drop-in arrays and pitch (and channel order) to drive the beamformer and heatmap with your real array layout.

Keep this file next to `array_geometry.py` and the generated CSV so you can quickly look up column meanings and how the numbers feed into the pipeline.

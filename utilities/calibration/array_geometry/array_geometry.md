# Array geometry reference

This document explains the output of `array_geometry.py`: the **table** (and CSV) and the **MUSIC summary**. Use it as a reference when calibrating the heatmap or feeding geometry into the beamformer.

---

## 1. What the table is showing

The table is in the **flipped array-center frame**: origin at the array center, with the 180° flip applied so “camera top-left” matches your physical setup.

### Column meanings

| Column | Meaning |
|--------|--------|
| **pay** | Payload index 0–15. Row index = payload index; matches firmware/SPI order (e.g. payload 0 = U3, payload 1 = U2, …). |
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

Your values (about −138° to −160°) put all mics in the **third quadrant** (negative x, negative y) in the flipped frame, which matches the geometry.

It’s the same “angle from the array” that shows up when you think of a source at some azimuth: MUSIC and your heatmap work in angles, so having each mic’s azimuth in this table helps you sanity-check that the layout (and any angle convention) is consistent.

---

## 3. Why these columns matter

| What | Why it matters |
|------|----------------|
| **x_mm, y_mm, x_m, y_m** | Define the array geometry in one place. **x_m / y_m** are exactly what you feed into MUSIC. |
| **dist_to_camera_mm / dist_to_center_mm** | For calibration and sanity checks (relative distances, symmetry, alignment with the camera). |
| **azimuth_deg** | Links geometry to angle space (DOA, heatmap direction). |
| **pay + label** | Map payload index (0–15) to physical mic so firmware, config, and MUSIC all agree: `x_coords[i]` / `fft_data[i, :]` = mic at `PAYLOAD_TO_MIC[i]`. |
| **min_dist_mm** | Direct check against your FreeCAD measurements. |

The table is both a **geometry reference** and a **cross-check** for calibration and for feeding the right numbers into the pipeline.

---

## 4. What the MUSIC summary is showing

The MUSIC summary is the **exact input set** your beamformer expects.

### 4.1. x_coords (m) and y_coords (m)

- **What:** Two length-16 arrays of mic positions in **meters**, in **payload order** (index i = payload index i: 0=U3, 1=U2, 2=U1, 3=U4, 4=U8, 5=U6, 6=U9, 7=U7, 8=U18, 9=U16, 10=U19, 11=U17, 12=U13, 13=U12, 14=U11, 15=U14).
- **Where it’s used:** In `beamforming.music_spectrum()` (and related code) these are the `x_coords` and `y_coords` arguments. For each test angle θ, the code builds the steering vector using  
  `k * (x_coords * cos(θ) + y_coords * sin(θ))`  
  so the phase delay at each mic is determined by these coordinates and the wavenumber *k*.
- **Why it matters:** MUSIC’s direction-of-arrival (DOA) and your heatmap directions depend entirely on this geometry. If you paste these arrays into your config (or calibration module), the pipeline uses your **measured** array layout instead of a generic Fermat spiral, which is what you want for calibrating the heatmap to the real hardware.

### 4.2. pitch (m)

- **What:** Mean **radial spacing** between mics (in meters), computed from the same flipped positions (e.g. from sorted radii).
- **Why it matters:** Used by **ESPRIT** and similar methods that assume or exploit a characteristic spacing (e.g. for a ULA or for wavelength/spacing checks). Having it printed lets you compare with config and with theory.

### 4.3. Payload order line

- **What:** Explicit mapping of payload index to mic: `0=U3, 1=U2, 2=U1, 3=U4, … 15=U14` (see `PAYLOAD_TO_MIC` in the script).
- **Why it matters:** Ensures that `x_coords[i]` / `y_coords[i]` and `fft_data[i, :]` from SPI refer to the same physical mic. The script is the single reference for this mapping.

### 4.4. Other MUSIC-relevant metrics (sanity checks)

The script also prints these so you can cross-check and tune the pipeline:

| Metric | Meaning | Why it matters |
|--------|--------|----------------|
| **speed_sound (m/s)** | Speed of sound used in steering vector (`k = 2πf/c`). Matches `config.SPEED_SOUND` (343 m/s). | Must match between array_geometry and config so angles and wavelengths are consistent. |
| **aperture_radius (m)** | Max distance from array center to any mic. | Drives angular resolution (~λ/aperture). Compare with config’s design aperture if you had one. |
| **wavelength at 30 kHz (m)** | λ = c/f at a reference frequency (middle of bandpass). Also prints λ/2. | Steering vector uses wavenumber k = 2π/λ. Use λ/2 for spatial Nyquist checks. |
| **pairwise distances (m)** | Min, max, and mean distance between any two mics. | If **max pairwise distance > λ/2** at your operating frequency, you can get grating lobes (ambiguous angles). Use to sanity-check operating band. |

### 4.5. Payload index and FFT layout

- **Payload index:** The script hardcodes `PAYLOAD_TO_MIC` and (optionally) `PAYLOAD_ADC_PIN_MIC` so that payload index 0..15 maps to ADC/PIN and mic label. Table and CSV use `payload_index`; row index = payload index everywhere.
- **FFT packed layout:** Per-mic RFFT from the firmware is 512 floats: `fft_data[0]=Re(DC)`, `fft_data[1]=Re(Nyquist)`; for bins 1..255, `fft_data[2*i]=Re(bin i)`, `fft_data[2*i+1]=Im(bin i)`. The script defines `FFT_FRAME_SIZE`, `FFT_HALF`, `FFT_N_BINS` and documents this layout; `acoustic_imager.spi.spi_protocol.unpack_packed_rfft_to_complex` implements it.

---

## 5. Summary

- **Table:** Documents the full geometry and distances/angles for checking and calibration.
- **MUSIC summary:** Gives you the drop-in arrays (`x_coords`, `y_coords` in payload order), `pitch`, `speed_sound`, payload→mic mapping, plus aperture, wavelength (and λ/2), and pairwise distances for sanity checks and tuning.

Keep this file next to `array_geometry.py` and the generated CSV so you can quickly look up column meanings and how the numbers feed into the pipeline.

---

## 6. Updating config from array_geometry

When you have new measured coordinates (e.g. from FreeCAD), update the app so the heatmap pipeline uses them:

1. **Edit** `MICS_RAW` (and optionally `ARRAY_CENTER_TO_CAMERA_MM`) in [array_geometry.py](array_geometry.py) with the new camera-frame measurements.
2. **Run** the script to regenerate the CSV and print the MUSIC summary:
   ```bash
   python3 utilities/calibration/array_geometry/array_geometry.py
   ```
3. **Copy** the printed `x_coords` and `y_coords` (and `pitch`) from the MUSIC summary into [config.py](../../../src/software/acoustic_imager/config.py) as `x_coords_hw`, `y_coords_hw`, and `pitch_hw` (section 4, HW geometry).

The app uses these values for SRC:HW and LOOP (MUSIC/heatmap); keeping config in sync with array_geometry ensures correct phase relationships in the pipeline.

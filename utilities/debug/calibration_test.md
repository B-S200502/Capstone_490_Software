# Camera–array calibration test setup

Reference for the physical layout used when testing heatmap vs camera alignment. Values are also in [config.py](../../src/software/acoustic_imager/config.py) as `CALIBRATION_*` for use in code.

## Layout

- **Camera on left**, **audio board (mic array) on right**; **same heading** (parallel).
- Camera optical axis and array “forward” (0°) are parallel; offset between them is lateral only.

## Distance

- **5 inches** — distance from camera to mic array center (depth).
- Measured along the common heading (perpendicular to the camera sensor plane).

## Lateral offset

- **7 inches** — from **camera optical center** to **center of audio board** (array center), in the plane at that depth.
- **Sign convention:** Positive = array center is to the **right** of camera center when the camera is on the left. So in this setup the array is +7 in from camera center.

## Units

- All values in **inches** here and in config for the test setup.
- Conversion to meters or to **pixels** (for overlay alignment) will need camera FOV and sensor/resolution; not defined in this doc.

## Use

- For calibration tests and for future overlay code that maps heatmap angle space to camera pixels (so the blob for the array appears over the array in the image).

# Delay / Latency Improvements Reference

Reference for the acoustic imager pipeline timing and where to optimize next. Check this before starting latency work.

## Pipeline stages (ms by revision)

| Stage         | rev0 (ms) | rev1 (ms) | rev2 (ms) | Notes                          |
|---------------|-----------|-----------|-----------|--------------------------------|
| read          | 0.05      | 0.06      | 0.06      |                                |
| heat_music    | 16–19     | ~6.1      | ~4.8      | MUSIC / DSP; rev1 after coarser grid + every-2nd-frame + two-stage |
| heat_stability| 0.01      | 0.01      | 0.01      |                                |
| heat_draw     | 11–12     | ~13.9     | ~12.3     | Heatmap drawing                |
| heat_scale    | 11–20     | ~17.7     | ~15.5     | Scaling; rev2 histogram percentile (stretch kept) |
| heat          | 10–13     | ~11.9     | ~11.5     | Heatmap pipeline (pre-draw)    |
| bg            | 2.7–3     | ~1.9      | ~1.2      | Background                     |
| blend         | 13–15     | ~11.9     | ~9.5      | Heatmap blend (already optimized) |
| bars          | 12–13     | ~15.6     | ~12.0     | DB + freq bar drawing          |
| ui            | 8–9       | ~8.5      | ~12.5     | Rest of UI drawing             |
| imshow        | <1        | ~1.3      | ~1.5      | Display                        |
| waitKey       | 11–13     | ~10.4     | ~9.5      | UI event loop (often fixed)    |

**Total** rev0 ~106–114 ms → rev1 ~99–103 ms → rev2 ~92–93 ms.

- **rev0**: Baseline (before heat_music optimizations).
- **rev1**: After ANGLES_2D_RESOLUTION=35, SPI_MUSIC_EVERY_N_FRAMES=2, two-stage 2D MUSIC (coarse 21 + 5×5 refine).
- **rev2**: After histogram-based percentile for contrast stretch (blobs visible); total ~92–93 ms.

## Priority order for optimization (rev2)

1. **heat_scale** (~15.5 ms) — Subsample percentile or every-N-frames if more gain needed.
2. **heat_draw** (~12.3 ms) — Heatmap drawing; lower res or simpler path.
3. **bars** (~12 ms) — DB + freq bar; reuse or simplify.
4. **ui** (~12.5 ms) — Rest of UI; fewer redraws or cache static elements.
5. **heat** (~11.5 ms) — Upstream of heat_draw.

`waitKey` is often a fixed 10–15 ms and may be hard to reduce.

## Already done

- **Blend**: Fused colormap+weight LUT (no applyColorMap + first multiply); optional half-res blend (`config.BLEND_HALF_RES`). LUT writes use contiguous single-channel buffers to satisfy OpenCV `cv2.LUT` dst layout.
- **heat_music**: (1) `ANGLES_2D_RESOLUTION` reduced to 35 (from 51) for coarser 2D grid; (2) `SPI_MUSIC_EVERY_N_FRAMES = 2` to run 2D MUSIC every other frame and reuse cached angles/spec; (3) two-stage 2D MUSIC when `SPI_MUSIC_2D_COARSE_RESOLUTION > 0` (e.g. 21): coarse grid then small fine patch around peak via `music_spectrum_2d_refined` in `dsp/beamforming.py`, with `SPI_MUSIC_2D_REFINE_HALF_WIDTH` (e.g. 2 → 5×5 patch). Set `SPI_MUSIC_2D_COARSE_RESOLUTION = 0` to disable two-stage and use full grid only.
- **heat_scale**: Contrast stretch uses `percentile_uint8_fast` (256-bin histogram) in `dsp/heatmap.py` instead of `np.percentile`; keeps blobs visible with lower cost.

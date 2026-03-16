# Delay / Latency Improvements Reference

Reference for the acoustic imager pipeline timing and where to optimize next. Check this before starting latency work.

## Pipeline stages (ms by revision)

| Stage         | rev0 (ms) | rev1 (ms) | rev2 (ms) | rev3 (ms) | rev4 (ms) | rev5 (ms) | Notes                          |
|---------------|-----------|-----------|-----------|-----------|-----------|-----------|--------------------------------|
| read          | 0.05      | 0.06      | 0.06      | 0.06      | 0.06      | 0.06      |                                |
| heat_music    | 16–19     | ~6.1      | ~4.8      | ~4.8      | ~5.2      | ~5.5      | MUSIC / DSP; rev1 coarser grid + every-2nd-frame + two-stage |
| heat_stability| 0.01      | 0.01      | 0.01      | 0.01      | 0.01      | 0.01      |                                |
| heat_draw     | 11–12     | ~13.9     | ~12.3     | ~12.3     | ~10.2     | ~9.7      | Heatmap drawing                |
| heat_scale    | 11–20     | ~17.7     | ~15.5     | ~10.5     | ~7.6      | ~7.5      | rev2 histogram; rev3 subsample; rev4 fused pass + level-scaled pct |
| heat          | 10–13     | ~11.9     | ~11.5     | ~11.5     | ~11.5     | ~12.4     | Heatmap pipeline (pre-draw)    |
| bg            | 2.7–3     | ~1.9      | ~1.2      | ~1.2      | ~1.9      | ~2.0      | Background                     |
| blend         | 13–15     | ~11.9     | ~9.5      | ~9.5      | ~9.5      | ~9.5      | Heatmap blend (already optimized) |
| bars          | 12–13     | ~15.6     | ~12.0     | ~12.0     | ~10.0     | ~9.9      | rev5 spectrum bar buffer + full dB bar cache |
| ui            | 8–9       | ~8.5      | ~12.5     | ~12.5     | ~8.0      | ~8.0      | Rest of UI drawing             |
| imshow        | <1        | ~1.3      | ~1.5      | ~1.5      | ~1.6      | ~1.5      | Display                        |
| waitKey       | 11–13     | ~10.4     | ~9.5      | ~9.5      | ~10.6     | ~11.2     | UI event loop (often fixed)    |

**Total** rev0 ~106–114 ms → rev1 ~99–103 ms → rev2 ~92–93 ms → rev3 ~80–82 ms → rev4 ~80 ms → rev5 ~79 ms.

- **rev0**: Baseline (before heat_music optimizations).
- **rev1**: After ANGLES_2D_RESOLUTION=35, SPI_MUSIC_EVERY_N_FRAMES=2, two-stage 2D MUSIC (coarse 21 + 5×5 refine).
- **rev2**: After histogram-based percentile for contrast stretch (blobs visible); total ~92–93 ms.
- **rev3**: After heat_scale subsample (HEATMAP_STRETCH_SUBSAMPLE=4); total ~80–82 ms.
- **rev4**: After heat_scale fused pass (level-scaled percentile), HEATMAP_SMOOTH_ALPHA=0.62, SPI_COV_AVG_FRAMES=6; total ~80 ms.
- **rev5**: After spectrum bar buffer reuse + full dB colorbar cache (labels); total ~79 ms.

## Priority order for optimization (rev5)

1. **heat** (~12.4 ms) — Heatmap pipeline; upstream of heat_draw.
2. **heat_draw** (~9.7 ms) — Heatmap drawing; lower res or simpler path.
3. **waitKey** (~11.2 ms) — Often fixed by Qt; hard to reduce.
4. **blend** (~9.5 ms) — Already optimized.
5. **bars** (~9.9 ms) — Already buffer reuse + full dB cache.

`waitKey` is often a fixed 10–15 ms and may be hard to reduce.

## Already done

- **Blend**: Fused colormap+weight LUT (no applyColorMap + first multiply); optional half-res blend (`config.BLEND_HALF_RES`). LUT writes use contiguous single-channel buffers to satisfy OpenCV `cv2.LUT` dst layout.
- **heat_music**: (1) `ANGLES_2D_RESOLUTION` reduced to 35 (from 51) for coarser 2D grid; (2) `SPI_MUSIC_EVERY_N_FRAMES = 2` to run 2D MUSIC every other frame and reuse cached angles/spec; (3) two-stage 2D MUSIC when `SPI_MUSIC_2D_COARSE_RESOLUTION > 0` (e.g. 21): coarse grid then small fine patch around peak via `music_spectrum_2d_refined` in `dsp/beamforming.py`, with `SPI_MUSIC_2D_REFINE_HALF_WIDTH` (e.g. 2 → 5×5 patch). Set `SPI_MUSIC_2D_COARSE_RESOLUTION = 0` to disable two-stage and use full grid only.
- **heat_scale**: Contrast stretch uses `percentile_uint8_fast` (256-bin histogram); subsample (HEATMAP_STRETCH_SUBSAMPLE=4); fused level+stretch pass; percentile from level-scaled subsample.
- **bars**: Spectrum analyzer reuses `_SPECTRUM_BAR_BUF` (no panel_bg.copy() per frame); dB colorbar uses `_DB_COLORBAR_FULL_CACHE` (gradient + labels) when state is None.

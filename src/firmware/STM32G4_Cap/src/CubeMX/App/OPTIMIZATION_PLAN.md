# App Layer Cycle-Reduction Plan

## Context

- **Target:** This `App` folder (STM32G4 Acoustic Imager firmware).
- **Goal:** Reduce clock cycles per frame. Typical budget is ~1.8M CC/frame with ~1M CC consumed by FFT; the rest is checksum, battery read, clipping scan, ADC/DC steps, and payload assembly.
- **Config:** All options are compile-time in [app_main.h](app_main.h).

---

## Optimizations Implemented

| # | Optimization | Config switch | Location | Effect | Host / protocol impact |
|---|--------------|---------------|----------|--------|-------------------------|
| 1 | **Checksum off** | `SPI_CHECKSUM_ENABLE` (0 = off) | [transport/spi_stream.c](transport/spi_stream.c): `spi_stream_build_frame_header`, `spi_stream_finalize_frame` | ~300k CC/frame saved when off | Host must treat frame length as **header + payload only** (no trailing 2 bytes). |
| 2 | **Battery every N frames** | `APP_BATTERY_READ_EVERY_N_FRAMES` (e.g. 8) | [app_main.c](app_main.c): `app_process_synced_window`; cache in `battery_millivolts_cached`; CLI uses `app_get_cached_battery_mv()` | ~100k CC per frame saved when N > 1 (read every Nth frame) | Battery in header/status updates at most every N frames. |
| 3 | **Clipping scan off** | `APP_CLIP_DETECT_ENABLE` (0 = off) | [app_main.c](app_main.c): `app_process_synced_window` | Saves full clipping scan over all ADCs × FRAME_SIZE × channels | `SPI_FRAME_FLAG_TIME_CLIPPING` never set; clip counters stay 0. |
| 4 | **ADC + DC in one step** | (no switch) | [dsp/dsp_pipeline.c](dsp/dsp_pipeline.c): `process_adc_to_float_and_remove_dc`, used by `process_adc_channel_pipeline` | One fewer full pass over the buffer per channel (merge de-interleave + DC sum + DC remove into two loops) | None. |
| 5 | **Bulk payload copy** | `SPI_FRAME_PAYLOAD_BYTES` (spi_stream.h) | [app_main.c](app_main.c): after the 16× FFT+avg loop, one memcpy of SPI_FRAME_PAYLOAD_BYTES from &fft_avg[0][0]; no runtime size check | Single bulk copy; no per-frame capacity check | None. |
| 6 | **Optional: no FFT repack** | Not implemented | [dsp/dsp_pipeline.c](dsp/dsp_pipeline.c): `apply_fft` → `pack_rfft_complex_bins` | Saves repack step if host accepts CMSIS packed layout | Host must accept **CMSIS packed** FFT output (512 floats) instead of current complex layout. |

---

## Config Summary (app_main.h)

```c
#define SPI_CHECKSUM_ENABLE       0   /* 0 = no checksum; host uses header+payload only */
#define APP_BATTERY_READ_EVERY_N_FRAMES 8u   /* Read battery every N frames */
#define APP_CLIP_DETECT_ENABLE    0   /* 0 = skip clipping scan */
```

---

## Testing Order

1. **CRC off** – Build with `SPI_CHECKSUM_ENABLE 0`, run host with frame length = header + payload (no checksum bytes). Measure CC/frame.
2. **Battery cache** – Set `APP_BATTERY_READ_EVERY_N_FRAMES` to 8 or 16; confirm status/battery CLI still report and that header battery updates every N frames.
3. **Clipping off** – Set `APP_CLIP_DETECT_ENABLE` to 0; confirm no clip flags and that clip counters remain 0.
4. **ADC+DC merge** – No config change; confirm FFT/SPI output unchanged (e.g. compare with previous build).
5. **Bulk copy** – No config change; confirm frame content and length unchanged.

---

## Optional: Remove FFT Repacking (Item 6)

- **Condition:** Host must accept CMSIS real-FFT packed format (same 512 floats, different layout).
- **Change:** In `apply_fft`, call `arm_rfft_fast_f32(..., input, fft_output, 0)` and do **not** call `pack_rfft_complex_bins`. Use `fft_output` (packed) for `update_fft_bin_average` and for SPI payload.
- **Caveat:** Any code that expects the current complex layout (e.g. `calculate_magnitude`) must be updated to use packed format or keep a separate unpacked buffer for that path.

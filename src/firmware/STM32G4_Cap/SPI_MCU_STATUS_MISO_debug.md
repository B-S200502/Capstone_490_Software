# When does MCU_STATUS and MISO toggle?

NSS toggling (Pi side) means the Pi is asserting chip select. If **MCU_STATUS** (frame-ready) and **MISO** (STM32 → Pi data) never toggle, the STM32 is not driving them. This doc summarizes when the firmware does that.

## Firmware flow

1. **ADC DMA** (all 4 ADCs) fills buffers and sets bits in `adc_ready_mask` on half/full complete (see `adc.c`).
2. **app_loop()** (app_main.c) only calls `app_process_synced_window()` when:
   - **All 4 ADCs** have reported (either half or full): `(adc_pending_mask & ADC_READY_HALF_MASK) == ADC_READY_HALF_MASK` or same for `ADC_READY_FULL_MASK`.
   - And `!fft_in_progress && !get_spi_dma_busy()`.
3. **app_process_synced_window()** builds one SPI frame (header + 16 mic FFT payloads), then calls **spi_stream_tx_dma()**.
4. **spi_stream_tx_dma()** (spi_dma.c):
   - Arms `HAL_SPI_Transmit_DMA()` (STM32 will drive MISO when the Pi clocks).
   - Sets **MCU_STATUS (PE0) HIGH** so the Pi sees "frame ready".
5. When the **Pi** asserts NSS and clocks SCK, the STM32 (SPI slave) outputs the frame on **MISO**. When DMA completes, **HAL_SPI_TxCpltCallback** sets **MCU_STATUS LOW** and `spi_dma_busy = 0`.

So: **MCU_STATUS and MISO only toggle if the STM32 has reached step 4**, which requires the ADC path to have produced a synced window (all 4 ADCs ready).

## If MCU_STATUS never goes high

- **All 4 ADCs must be producing DMA half/full events.** If any ADC is not running, not triggered, or not connected, `adc_ready_mask` never has all four bits set, so `app_process_synced_window()` is never called and MCU_STATUS never goes high.
- Check: TIM6 running (triggers ADCs), all four `HAL_ADC_Start_DMA` called in `app_start()`, and ADC DMA callbacks firing (e.g. via USB `status` command or debug).

## If MCU_STATUS goes high once and never toggles (and MISO stays quiet)

- The first time all 4 ADCs are ready, the STM32 arms SPI TX DMA and sets MCU_STATUS high. If the **Pi never clocks** (NSS/SCK not reaching the STM32, or wrong pins), the slave transfer never completes.
- Then `spi_dma_busy` stays 1, so on the next synced window we **skip** `app_process_synced_window()` (because `!get_spi_dma_busy()` is false). So no new frame is sent, MCU_STATUS is never cleared by the callback (it stays high), and MISO never gets clocked out.
- **Fix:** Ensure NSS and SCK from the Pi reach the STM32 (J2 pin 6 = NSS, pin 5 = SCLK; STM32 PE3 = NSS, PE2 = SCK). If the Pi is toggling NSS but the STM32 NSS/SCK pins are not connected or are on the wrong header, the slave will never see the transaction.

## Wiring (STM32 → Pi)

| Signal     | STM32 pin | Pi (SPI0) physical | Pi BCM |
|-----------|-----------|----------------------|--------|
| MCU_STATUS | PE0      | 26                   | 7      |
| MISO      | PE5       | 21                   | 9      |
| NSS       | PE3       | 24                   | 8      |
| SCK       | PE2       | 23                   | 11     |

Confirm PE0 and PE5 are wired to Pi physical 26 and 21 (for SPI0).

## Quick checklist

1. **Firmware:** `main.c` has `#define MODE RELEASE` so the main loop runs `app_loop()` (not a test loop).
2. **ADCs:** All 4 ADCs started with DMA in `app_start()`; TIM6 started so conversions are triggered; ADC DMA callbacks run (e.g. USB `status` shows activity).
3. **Wiring:** NSS (PE3), SCK (PE2) from Pi to STM32 so the slave sees the transaction; MCU_STATUS (PE0) and MISO (PE5) from STM32 to Pi.
4. **One-shot stuck:** If MCU_STATUS went high once and never low, the first SPI transfer did not complete (Pi didn’t clock, or NSS/SCK not connected). Power-cycle or reset the STM32 after fixing wiring so `spi_dma_busy` clears.

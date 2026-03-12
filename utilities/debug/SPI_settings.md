# SPI settings: firmware vs Pi

Reference for SPI alignment between STM32 (slave) and Raspberry Pi (master). Canonical sources: firmware [spi.c](../../src/firmware/STM32G4_Cap/src/CubeMX/Core/Src/spi.c), Pi [config.py](../../src/software/acoustic_imager/config.py).

## Firmware (STM32 SPI4 slave)

| Setting    | Value              | Meaning                    |
|-----------|--------------------|----------------------------|
| Mode      | SPI_MODE_SLAVE     | Slave                      |
| CLKPolarity | SPI_POLARITY_LOW | CPOL = 0                   |
| CLKPhase  | SPI_PHASE_1EDGE    | CPHA = 0                   |
| FirstBit  | SPI_FIRSTBIT_MSB   | MSB first                  |
| DataSize  | SPI_DATASIZE_8BIT  | 8 bits per frame           |

Effective: **SPI mode 0**, **MSB first**.

## Pi (spidev, config)

| Setting   | Value / source           | Meaning        |
|----------|---------------------------|----------------|
| SPI_MODE | config.SPI_MODE (default 0) | CPOL=0, CPHA=0 |
| lsb_first | Not set in code          | Kernel default (MSB first) |

Effective: **SPI mode 0**, **MSB first** (assuming kernel default).

## SPI clock

| Location | Value | Notes |
|----------|--------|-------|
| config.SPI_MAX_SPEED_HZ | 20_000_000 (20 MHz) | Current setting; conservative for 32801-byte full-frame xfer. |

- **21.25 MHz**: Often works if 20 MHz is stable; try `21_250_000` and re-check (see below).
- **30 MHz**: May be possible if STM32 and wiring support it. Overclocking can cause marginal timing, CRC/parse errors, or drops. Try `30_000_000` and run `heatmap_pipeline_debug.py --live`; if bad_parse or timeouts rise, reduce.

**When changing SPI speed, check:**
1. Run `python3 utilities/debug/heatmap_pipeline_debug.py --live` — frames_ok should increase over time; no repeated bad_parse.
2. In the app, watch stats (e.g. overlay): frames_ok, bad_parse, irq_timeout. No sustained errors.
3. Heatmap and spectrum should update smoothly; if frames stall or look frozen, lower speed.

## Alignment

On paper both sides use mode 0 and MSB first. If magic header is still wrong, use [spi_magic_probe.py](spi_magic_probe.py) to try other modes and LSB-first and see if any setting receives 0xAABBCCDD (or its bit-reversed form).

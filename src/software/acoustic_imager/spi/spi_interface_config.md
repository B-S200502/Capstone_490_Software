# Acoustic Imager config reference

See [config.py](config.py) for the live configuration. This doc summarizes the two SPI interface modes and their pinouts.

## SPI interface modes

Set `SPI_INTERFACE` in config.py to `0` or `1` to choose the Pi SPI bus and chip-select. **J2** is the STM32-side header (lines 1–10) per the ribbon cable pinout.


| Item                  | SPI INTERFACE 0     | SPI INTERFACE 1     |
| --------------------- | ------------------- | ------------------- |
| **config.py**         | `SPI_INTERFACE = 0` | `SPI_INTERFACE = 1` |
| **Device**            | `/dev/spidev0.0`    | `/dev/spidev1.2`    |
| **SPI_BUS / SPI_DEV** | 0, 0                | 1, 2                |


---


| Purpose                       | STM-32 PINOUT (J2) | RASPI PINOUTS (SPI0) | RASPI PINOUTS (SPI1) |
| ----------------------------- | ------------------ | -------------------- | -------------------- |
| Gain control (AUTO_GAIN_CNTL) | 1                  | 22                   | 22                   |
| Frame-ready (MCU_STATUS)      | 2                  | 26                   | 26                   |
| RPI status                    | 3                  | 27                   | 27                   |
| GND                           | 4                  | 20                   | 25                   |
| SCLK                          | 5                  | 23                   | 40                   |
| Chip select (NSS)             | 6                  | 24                   | 36                   |
| MISO                          | 7                  | 21                   | 35                   |
| MOSI                          | 8                  | 19                   | 38                   |
| GND                           | 9                  | 25                   | 39                   |
| NRST                          | 10                 | 28                   | 28                   |


## Sanity check: switching `SPI_INTERFACE` (e.g. to 0)

When you set `SPI_INTERFACE = 0` or `1` in config.py, **all** of the following are taken from the same table; there is no separate Pi config file for SPI:

| config.py              | SPI0 (interface 0) | SPI1 (interface 1) |
|------------------------|--------------------|--------------------|
| `_SPI_PIN_SETUPS[n]`   | (0, 0, 7, 25)      | (1, 2, 7, 25)      |
| → Device               | /dev/spidev0.0     | /dev/spidev1.2     |
| → Frame-ready (BCM)    | 7 (physical 26)    | 7 (physical 26)     |
| → Gain ctrl (BCM)      | 25 (physical 22)   | 25 (physical 22)   |

- **SPIManager** (HW source) reads only `config.SPI_BUS`, `config.SPI_DEV`, `config.FRAME_READY_BCM_PIN`, `config.GAIN_CTRL_BCM_PIN` — all derived from `_SPI_PIN_SETUPS[SPI_INTERFACE]`. No environment variables.
- **deploy_imager.sh** does not set `ACOUSTIC_SPI_BUS` / `ACOUSTIC_SPI_DEV`; the app uses config.py only.
- Other deploy scripts (e.g. deploy_imager_acousticgod.sh) may set `ACOUSTIC_SPI_BUS=1 ACOUSTIC_SPI_DEV=0` in the process; those env vars affect only **LOOP** mode (spi_loopback_source), not **HW** mode (spi_manager). If you use the main deploy_imager.sh, ignore them.

So for "SPI open failed": ensure (1) the chosen device exists (`ls /dev/spidev*`), (2) `dtparam=spi=on` or SPI1 overlay is set, (3) user is in group `spi`, and (4) config.py has the intended `SPI_INTERFACE` and no typo in `_SPI_PIN_SETUPS`.

Notes:

- **SPI0** uses the Pi's main 40-pin header (pins 19, 21, 23, 24).
- **SPI1** is the auxiliary SPI; enable with `dtoverlay=spi1-3cs` in `/boot/config.txt`. CE2 gives `/dev/spidev1.2`.
- Frame-ready (MCU_STATUS) is an input to the Pi (STM32 drives it). Gain control (AUTO_GAIN_CNTL) is an output from the Pi (drives STM32).

## Enabling SPI (fix "no such file" / device missing)

If you see **SPI open failed … No such file or directory**:

**Diagnostic first:** From the repo root run:
```bash
bash utilities/calibration/check_spi_devices.sh
```
This prints which boot config is used, whether SPI is enabled there, and whether `/dev/spidev*` exists.

**Boot config location:** On Raspberry Pi OS Bookworm/Trixie and Pi 5, the active config is often `/boot/firmware/config.txt` (not `/boot/config.txt`). Check both; the one that exists and is modified by raspi-config is the one that counts.

1. **SPI0** (`/dev/spidev0.0`): Enable the main SPI interface:
   - `sudo raspi-config` → **Interface Options** → **SPI** → Enable → Finish.
   - **Or manually:** In the active config file (see above), ensure this line is present and **not** commented: `dtparam=spi=on`
   - Reboot. Then check: `ls -l /dev/spidev*` (you should see `/dev/spidev0.0`).

2. **SPI1** (`/dev/spidev1.2`): Enable the auxiliary SPI and reboot:
   - In the active config (e.g. `/boot/firmware/config.txt` or `/boot/config.txt`), add: `dtoverlay=spi1-3cs`
   - Reboot. Then check: `ls -l /dev/spidev*` (you should see `spidev1.0`, `spidev1.1`, `spidev1.2`).

3. Ensure your user can access the device: `sudo usermod -a -G spi $USER` (log out and back in, or reboot).


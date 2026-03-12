"""
Firmware flasher for Acoustic Imager device.
Invoked from System Settings > Flash Firmware button.
"""

from __future__ import annotations


def flash_firmware() -> None:
    """
    Flash firmware to the connected device.
    Implement actual flashing logic (e.g. DFU, stm32flash, openocd) here.
    On success, set button_state.firmware_flash_status = "success".
    On error, set button_state.firmware_flash_status = "error".
    """
    # TODO: Implement firmware flashing (DFU, stm32flash, openocd, etc.)
    # Example when done: from acoustic_imager.state import button_state; button_state.firmware_flash_status = "success"
    pass

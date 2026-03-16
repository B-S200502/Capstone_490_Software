# sources/spi_source.py
from __future__ import annotations

from typing import Optional

from acoustic_imager.custom_types import LatestFrame
from acoustic_imager.io.spi_manager import SPIManager


class SPISource:
    """
    High-level SPI source used by main loop.
    Wraps SPIManager (threaded mailbox).
    """

    def __init__(self, spi_manager: Optional[SPIManager] = None) -> None:
        self.spi_manager = spi_manager or SPIManager(use_frame_ready=False)

    def start(self) -> None:
        self.spi_manager.start()

    def stop(self) -> None:
        self.spi_manager.stop()

    def get_latest(self) -> LatestFrame:
        return self.spi_manager.get_latest()

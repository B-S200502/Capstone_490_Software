from __future__ import annotations

import logging

from acoustic_imager import config

_LOG = logging.getLogger(__name__)


class GainControl:
    """
    Optional Pi GPIO output for analog gain control.

    Keep this disabled (`config.GAIN_CTRL_ENABLED = False`) until firmware path
    is confirmed. When disabled, calls are no-op and UI behavior is unchanged.
    """

    def __init__(self) -> None:
        self.enabled = bool(getattr(config, "GAIN_CTRL_ENABLED", False))
        self.bcm_pin = int(getattr(config, "GAIN_CTRL_BCM_PIN", 25))
        self.active_high = bool(getattr(config, "GAIN_CTRL_ACTIVE_HIGH", True))
        self._device = None

        if not self.enabled:
            return

        try:
            from gpiozero import OutputDevice  # type: ignore

            self._device = OutputDevice(self.bcm_pin, active_high=self.active_high, initial_value=False)
            _LOG.info("Gain GPIO initialized on BCM%s (active_high=%s)", self.bcm_pin, self.active_high)
        except Exception as exc:
            self.enabled = False
            self._device = None
            _LOG.warning("Gain GPIO unavailable; running in no-op mode: %s", exc)

    def set_mode(self, mode: str) -> None:
        if not self.enabled or self._device is None:
            return

        # LOW/HIGH menu values map to pin level; polarity is controlled by active_high.
        drive_high = str(mode).upper() == "HIGH"
        try:
            if drive_high:
                self._device.on()
            else:
                self._device.off()
        except Exception as exc:
            _LOG.warning("Failed to drive gain GPIO: %s", exc)

    def close(self) -> None:
        if self._device is None:
            return
        try:
            self._device.close()
        except Exception:
            pass
        self._device = None


GAIN_CONTROL = GainControl()


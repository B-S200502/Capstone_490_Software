# acoustic_imager/spi/frame_ready.py
from __future__ import annotations

import threading
from typing import Optional

class FrameReady:
    """Interface: something you can wait() on for a new frame-ready pulse."""
    def wait(self, timeout: Optional[float] = None) -> bool:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class FrameReadyGPIO(FrameReady):
    """
    Rising-edge interrupt on a Pi GPIO pin sets an Event.

    Use BCM numbering (e.g. BCM7 == physical pin 26 for MCU_STATUS).
    """
    def __init__(self, bcm_pin: int = 25, pull: str = "down") -> None:
        self._evt = threading.Event()
        self._closed = False

        self.bcm_pin = int(bcm_pin)
        self.pull = pull

        self._backend = None
        self._btn = None  # gpiozero button
        self._gpio = None # RPi.GPIO

        # Try gpiozero first
        try:
            from gpiozero import Button  # type: ignore
            pull_up = (pull == "up")
            # Button triggers on "pressed". For pull-down wiring, pressed == high.
            self._btn = Button(self.bcm_pin, pull_up=pull_up, bounce_time=0.001)
            self._btn.when_pressed = self._on_edge
            self._backend = "gpiozero"
            return
        except Exception:
            pass

        # Fallback: RPi.GPIO
        try:
            import RPi.GPIO as GPIO  # type: ignore
            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)

            pud = GPIO.PUD_DOWN if pull == "down" else GPIO.PUD_UP if pull == "up" else GPIO.PUD_OFF
            GPIO.setup(self.bcm_pin, GPIO.IN, pull_up_down=pud)
            GPIO.add_event_detect(self.bcm_pin, GPIO.RISING, callback=lambda ch: self._on_edge(), bouncetime=1)
            self._backend = "rpigpio"
            return
        except Exception as e:
            raise RuntimeError(
                "No GPIO backend available. Install gpiozero or run on a Raspberry Pi with RPi.GPIO."
            ) from e

    def _on_edge(self) -> None:
        if not self._closed:
            self._evt.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._evt.wait(timeout=timeout)

    def clear(self) -> None:
        self._evt.clear()

    def close(self) -> None:
        self._closed = True
        try:
            if self._backend == "gpiozero" and self._btn is not None:
                self._btn.close()
            elif self._backend == "rpigpio" and self._gpio is not None:
                self._gpio.remove_event_detect(self.bcm_pin)
                self._gpio.cleanup(self.bcm_pin)
        finally:
            self._btn = None
            self._gpio = None
            self._backend = None

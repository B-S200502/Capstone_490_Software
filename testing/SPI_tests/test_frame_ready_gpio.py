#!/usr/bin/env python3
"""
GPIO Frame-Ready Self-Test (no STM32 needed)

Goal:
  Verify that acoustic_imager.spi.frame_ready.FrameReadyGPIO
  correctly detects rising edges on BCM GPIO25 (physical pin 22).

How:
  We generate pulses on BCM GPIO23 (physical pin 16) and jumper-wire:
    GPIO23 (pin 16) -> GPIO25 (pin 22)

Expected:
  For each pulse, FrameReadyGPIO.wait() should return True.

Run:
  python3 testing/SPI_tests/test_frame_ready_gpio.py

Notes:
  - Uses gpiozero if available, otherwise falls back to RPi.GPIO.
  - Assumes FrameReadyGPIO is configured for rising edges (it is).
  - Default pull is "down" in your code; that's perfect for rising edges.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

# Add src/software to PYTHONPATH dynamically
repo_root = Path(__file__).resolve().parents[2]
sys.path.append(str(repo_root / "src" / "software"))


# ---- Adjust import path if needed (usually not needed if you run from repo root) ----
try:
    from acoustic_imager.spi.frame_ready import FrameReadyGPIO
except Exception as e:
    print("ERROR: Could not import FrameReadyGPIO.")
    print("Make sure you're running from repo root and your venv is active.")
    raise

# --- Pins ---
BCM_OUT = 23   # physical pin 16
BCM_IN  = 25   # physical pin 22 (frame-ready input)
PULL    = "down"  # matches your default wiring expectation

PULSE_HIGH_S = 0.010   # 10 ms high
PULSE_LOW_S  = 0.050   # 50 ms low between pulses
N_PULSES     = 10
WAIT_TIMEOUT = 0.250   # seconds


def _make_output():
    """
    Create a GPIO output on BCM_OUT using gpiozero or RPi.GPIO.
    Returns an object with .on()/.off() and a .close()/cleanup call.
    """
    # Try gpiozero first
    try:
        from gpiozero import OutputDevice  # type: ignore
        out = OutputDevice(BCM_OUT, active_high=True, initial_value=False)
        return ("gpiozero", out)
    except Exception:
        pass

    # Fallback: RPi.GPIO
    try:
        import RPi.GPIO as GPIO  # type: ignore
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BCM_OUT, GPIO.OUT, initial=GPIO.LOW)

        class _Out:
            def on(self):
                GPIO.output(BCM_OUT, GPIO.HIGH)

            def off(self):
                GPIO.output(BCM_OUT, GPIO.LOW)

            def close(self):
                GPIO.cleanup(BCM_OUT)

        return ("rpigpio", _Out())
    except Exception as e:
        raise RuntimeError(
            "No GPIO output backend available. Install gpiozero or run on a Raspberry Pi with RPi.GPIO."
        ) from e


def main() -> int:
    print("=" * 72)
    print("FrameReady GPIO Self-Test")
    print("=" * 72)
    print(f"Jumper required: BCM{BCM_OUT} (pin 16) -> BCM{BCM_IN} (pin 22)")
    print(f"Input pull: {PULL} | Pulses: {N_PULSES} | wait timeout: {WAIT_TIMEOUT}s")
    print()

    # Set up input (frame-ready detector)
    fr = FrameReadyGPIO(bcm_pin=BCM_IN, pull=PULL)

    # Set up output (pulse generator)
    backend, out = _make_output()
    print(f"Output backend: {backend}")
    print("Starting pulses...\n")

    passed = 0
    failed = 0

    try:
        for i in range(1, N_PULSES + 1):
            # Clear stale events BEFORE pulsing (important)
            fr.clear()

            # Generate rising edge: LOW -> HIGH -> LOW
            out.off()
            time.sleep(0.005)
            out.on()
            time.sleep(PULSE_HIGH_S)
            out.off()

            # Wait for frame-ready event
            got = fr.wait(timeout=WAIT_TIMEOUT)

            if got:
                passed += 1
                print(f"[{i:02d}] PASS  (edge detected)")
            else:
                failed += 1
                print(f"[{i:02d}] FAIL  (no edge detected within {WAIT_TIMEOUT}s)")

            time.sleep(PULSE_LOW_S)

    finally:
        # Cleanup
        try:
            out.off()
        except Exception:
            pass
        try:
            if hasattr(out, "close"):
                out.close()
        except Exception:
            pass
        try:
            fr.close()
        except Exception:
            pass

    print("\n" + "-" * 72)
    print(f"Result: passed={passed} failed={failed}")
    print("-" * 72)

    if failed == 0:
        print("✅ GPIO frame-ready interrupt path looks GOOD.")
        return 0
    else:
        print("❌ Some pulses were missed. Check wiring / pull / backend.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

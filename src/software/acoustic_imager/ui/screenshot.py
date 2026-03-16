"""
Screenshot capture and flash feedback.
"""

from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np

from ..state import button_state


def save_screenshot(frame: np.ndarray, output_dir: Path) -> Optional[str]:
    from datetime import datetime
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"screenshot_{timestamp}.png"
    if cv2.imwrite(str(filepath), frame):
        button_state.screenshot_flash_time = time.time()
        return str(filepath)
    return None


def draw_screenshot_flash(frame: np.ndarray) -> None:
    """
    Draw a green flash effect when a screenshot is taken.
    Flash lasts for 0.3 seconds with fade out.
    """
    if button_state.screenshot_flash_time is None:
        return

    elapsed = time.time() - button_state.screenshot_flash_time
    flash_duration = 0.3

    if elapsed > flash_duration:
        button_state.screenshot_flash_time = None
        return

    fade = 1.0 - (elapsed / flash_duration)

    overlay = frame.copy()
    overlay[:] = (40, 255, 60)

    alpha = 0.15 * fade
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    border_alpha = 0.8 * fade
    if border_alpha > 0.1:
        border_thickness = int(8 * fade) + 2
        cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1),
                     (255, 255, 255), border_thickness, cv2.LINE_AA)

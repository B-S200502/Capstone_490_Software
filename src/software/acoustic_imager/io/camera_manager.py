#!/usr/bin/env python3
"""
camera_manager.py

Manages camera acquisition for the Acoustic Imager.

Responsibilities:
- Detects and initializes an available camera backend (Picamera2, libcamera GStreamer, or OpenCV/V4L2).
- Runs capture in a background "latest frame" thread so the main loop never blocks on camera IO.
- Provides a thread-safe method to fetch the most recent frame for rendering.
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np
import cv2

from .. import config
from .. import state

# Try picamera2 (optional)
try:
    from picamera2 import Picamera2  # type: ignore
    PICAMERA2_AVAILABLE = True
except Exception:
    Picamera2 = None  # type: ignore
    PICAMERA2_AVAILABLE = False

CameraType = Optional[Literal["picamera2", "opencv"]]


@dataclass
class LatestFrame:
    lock: threading.Lock
    frame: Optional[np.ndarray] = None
    ok: bool = False


class CameraManager:
    """
    Mirrors monolith camera behavior:
      - Detects camera backend (picamera2, then gstreamer libcamerasrc, then V4L2)
      - Starts a worker thread that keeps ONLY the latest frame (main loop never blocks)
      - Exposes get_latest_frame() for non-blocking retrieval
    """

    def __init__(
        self,
        width: int = config.WIDTH,
        height: int = config.HEIGHT,
        camera_width: int = config.CAMERA_WIDTH,
        camera_height: int = config.CAMERA_HEIGHT,
        use_camera: bool = config.USE_CAMERA,
        use_libcamera: bool = config.USE_LIBCAMERA,
        camera_index: int = config.CAMERA_INDEX,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.camera_width = int(camera_width)
        self.camera_height = int(camera_height)
        self.use_camera = bool(use_camera)
        self.use_libcamera = bool(use_libcamera)
        self.camera_index = int(camera_index)

        self.camera_type: CameraType = None
        self._picam2 = None
        self._cam = None

        self._latest = LatestFrame(lock=threading.Lock())
        self._stop = False
        self._thread: Optional[threading.Thread] = None

    # --------------------------
    # Public API
    # --------------------------
    def detect_and_open(self) -> bool:
        """
        Detect a usable camera backend and open it.
        Sets state.CAMERA_AVAILABLE accordingly.
        """
        if not self.use_camera:
            self.camera_type = None
            state.CAMERA_AVAILABLE = False
            return False

        self.close()

        # 1) picamera2
        if self.use_libcamera and PICAMERA2_AVAILABLE and Picamera2 is not None:
            if self._try_open_picamera2():
                self.camera_type = "picamera2"
                state.CAMERA_AVAILABLE = True
                return True

        # 2) libcamerasrc via GStreamer (OpenCV)
        if self.use_libcamera:
            if self._try_open_gstreamer():
                self.camera_type = "opencv"
                state.CAMERA_AVAILABLE = True
                return True

        # 3) V4L2 / default OpenCV
        if self._try_open_v4l2():
            self.camera_type = "opencv"
            state.CAMERA_AVAILABLE = True
            return True

        self.camera_type = None
        state.CAMERA_AVAILABLE = False
        return False

    def start(self) -> bool:
        """
        Start the latest-frame worker thread.
        Respects state.button_state.camera_enabled.
        """
        # Need a backend first
        if self.camera_type is None:
            if not self.detect_and_open():
                return False

        if not state.button_state.camera_enabled:
            return False

        if self._thread is not None and self._thread.is_alive():
            return True

        self._stop = False
        with self._latest.lock:
            self._latest.frame = None
            self._latest.ok = False

        if self.camera_type == "picamera2":
            self._thread = threading.Thread(target=self._worker_picam2, daemon=True)
        else:
            self._thread = threading.Thread(target=self._worker_opencv, daemon=True)

        self._thread.start()
        return True

    def stop(self) -> None:
        self._stop = True
        time.sleep(0.02)
        self._thread = None

    def close(self) -> None:
        """
        Stop worker + release camera resources.
        """
        self.stop()

        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        self._cam = None

        if self._picam2 is not None:
            try:
                self._picam2.stop()
            except Exception:
                pass
            try:
                self._picam2.close()
            except Exception:
                pass
        self._picam2 = None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        Non-blocking. Returns the most recent frame if available.
        Note: returns the internal stored numpy array reference (like monolith).
        """
        with self._latest.lock:
            if self._latest.ok and self._latest.frame is not None:
                return self._latest.frame
        return None

    # --------------------------
    # Internal: backend open
    # --------------------------
    def _try_open_picamera2(self) -> bool:
        try:
            picam2 = Picamera2()  # type: ignore
            cfg = picam2.create_preview_configuration(
                 main={"size": (self.camera_width, self.camera_height), "format": "BGR888"}
            )
            picam2.configure(cfg)
            try:
                picam2.set_controls({"AeEnable": True, "AwbEnable": True})
                picam2.set_controls({"ExposureValue": 0.7})  # try 0.3 to 0.7
            except Exception:
                pass

            picam2.start()
            time.sleep(1.5)

            test = picam2.capture_array()
            if test is not None and getattr(test, "size", 0) > 0:
                self._picam2 = picam2
                return True

            try:
                picam2.stop()
                picam2.close()
            except Exception:
                pass
            return False
        except Exception:
            return False

    def _try_open_gstreamer(self) -> bool:
        try:
            gst = (
                f"libcamerasrc ! video/x-raw,width={self.camera_width},height={self.camera_height},framerate=30/1 ! "
                f"videoconvert ! appsink"
            )
            cam = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if not cam.isOpened():
                cam.release()
                return False

            ret, tf = cam.read()
            if ret and tf is not None and getattr(tf, "size", 0) > 0:
                self._cam = cam
                return True

            cam.release()
            return False
        except Exception:
            return False

    def _try_open_v4l2(self) -> bool:
        # Try CAP_V4L2 first, then default
        for api in (cv2.CAP_V4L2, 0):
            try:
                cam = cv2.VideoCapture(self.camera_index, api)
                if not cam.isOpened():
                    cam.release()
                    continue

                cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                for _ in range(3):
                    cam.read()

                ret, tf = cam.read()
                if ret and tf is not None and getattr(tf, "size", 0) > 0:
                    self._cam = cam
                    return True

                cam.release()
            except Exception:
                pass
        return False

    # --------------------------
    # Internal: worker threads
    # --------------------------
    def _worker_picam2(self) -> None:
        while not self._stop:
            try:
                if self._picam2 is None:
                    time.sleep(0.01)
                    continue

                fr = self._picam2.capture_array()

                if fr is not None and getattr(fr, "size", 0) > 0:
                    # Picamera2 often returns RGB even if you request BGR888
                    fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)

                    # ---- brightness control ----
                    fr = cv2.convertScaleAbs(
                        fr,
                        alpha=config.CAMERA_BRIGHTNESS_ALPHA,
                        beta=config.CAMERA_BRIGHTNESS_BETA
                    )
                    with self._latest.lock:
                        self._latest.frame = fr
                        self._latest.ok = True
            except Exception:
                time.sleep(0.005)

    def _worker_opencv(self) -> None:
        while not self._stop:
            try:
                if self._cam is None:
                    time.sleep(0.01)
                    continue

                ok = self._cam.grab()
                if not ok:
                    time.sleep(0.005)
                    continue

                ok, fr = self._cam.retrieve()
                if ok and fr is not None and getattr(fr, "size", 0) > 0:
                    with self._latest.lock:
                        self._latest.frame = fr
                        self._latest.ok = True
            except Exception:
                time.sleep(0.005)

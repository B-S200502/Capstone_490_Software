#!/usr/bin/env python3
"""
video_recorder.py

Provides video recording functionality for the Acoustic Imager.

This module implements the VideoRecorder class, which:
- Starts and stops video capture sessions
- Streams raw BGR frames to an ffmpeg subprocess
- Encodes the frames into an H.264 MP4 file
- Supports pausing and resuming recording
- Tracks the output file and number of recorded frames

It is used by the UI to record the rendered visualization frames to disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import time
import threading
import queue

import os
import subprocess
import numpy as np


class VideoRecorder:
    def __init__(self, output_dir: Path, width: int, height: int, fps: int = 30):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_recording = False
        self.is_paused = False
        self.current_file = None
        self.frame_count = 0

        # Timestamp tracking
        self.start_time = None
        self.pause_time = None
        self.total_paused_time = 0.0
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=8)  # small keeps latency low
        self.worker: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    def start_recording(self) -> bool:
        if self.is_recording:
            return False
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_file = self.output_dir / f"recording_{timestamp}.mp4"
            command = [
                "ffmpeg", "-y",
                "-loglevel", "error", "-nostats",
                "-thread_queue_size", "512",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-vcodec", "libx264",
                "-preset", "ultrafast",
                "-crf", "30",
                "-pix_fmt", "yuv420p",
                "-tune", "zerolatency",
                "-g", str(self.fps),        # keyframe every 1s
                "-keyint_min", str(self.fps),
                str(self.current_file),
            ]
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._stop_evt.clear()
            self.q = queue.Queue(maxsize=8)
            self.worker = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker.start()
            self.is_recording = True
            self.is_paused = False
            self.frame_count = 0
            self.start_time = time.time()
            self.pause_time = None
            self.total_paused_time = 0.0
            print(f"Recording started: {self.current_file}")
            return True
        except Exception as e:
            print(f"Failed to start recording: {e}")
            self.is_recording = False
            return False

    def _worker_loop(self) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            while not self._stop_evt.is_set():
                item = self.q.get()
                if item is None:
                    break
                try:
                    self.process.stdin.write(item)
                except Exception:
                    break
        finally:
            # best-effort flush/close
            try:
                if self.process and self.process.stdin:
                    self.process.stdin.flush()
            except Exception:
                pass

    def write_frame(self, frame: np.ndarray) -> bool:
        if not self.is_recording or self.process is None:
            return False
        if self.is_paused:
            return True

        try:
            b = frame.tobytes()
            try:
                self.q.put_nowait(b)
            except queue.Full:
                # drop one frame to make room (keep newest)
                try:
                    _ = self.q.get_nowait()
                except Exception:
                    pass
                try:
                    self.q.put_nowait(b)
                except Exception:
                    pass
            self.frame_count += 1
            return True
        except Exception:
            return False

    def pause_recording(self) -> bool:
        if not self.is_recording or self.is_paused:
            return False
        self.is_paused = True
        self.pause_time = time.time()
        return True

    def resume_recording(self) -> bool:
        if not self.is_recording or not self.is_paused:
            return False
        if self.pause_time is not None:
            self.total_paused_time += time.time() - self.pause_time
            self.pause_time = None
        self.is_paused = False
        return True

    def stop_recording(self) -> Optional[str]:
        if not self.is_recording or self.process is None:
            return None
        try:
            # stop worker first
            self._stop_evt.set()
            try:
                self.q.put_nowait(None)
            except Exception:
                pass
            if self.worker is not None:
                self.worker.join(timeout=2)

            # then close ffmpeg
            if self.process.stdin:
                self.process.stdin.close()
            self.process.wait(timeout=10)
            out = str(self.current_file)
            print(f"Recording stopped: {out} ({self.frame_count} frames)")
            return out
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return None
        finally:
            self.is_recording = False
            self.is_paused = False
            self.current_file = None
            self.frame_count = 0
            self.process = None
            self.worker = None
            self.start_time = None
            self.pause_time = None
            self.total_paused_time = 0.0

    def cleanup(self) -> None:
        if self.is_recording:
            self.stop_recording()

    def get_elapsed_time(self) -> float:
        """Get the elapsed recording time (excluding paused time)."""
        if not self.is_recording or self.start_time is None:
            return 0.0

        current_time = time.time()
        elapsed = current_time - self.start_time - self.total_paused_time

        # If currently paused, subtract the current pause duration
        if self.is_paused and self.pause_time is not None:
            elapsed -= (current_time - self.pause_time)

        return max(0.0, elapsed)

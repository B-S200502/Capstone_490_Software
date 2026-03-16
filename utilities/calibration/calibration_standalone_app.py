#!/usr/bin/env python3
"""
Standalone calibration suite app: single window, runs full suite (main app must be stopped).
Shows silence countdown during step 0, then log and PASSED/FAILED. At end, restarts the main app.

Usage (from repo root; usually launched by main app):
  python3 utilities/calibration/calibration_standalone_app.py [--restart-main] [--gain high|low]
When --restart-main is set, restarts the acoustic imager main app when user dismisses the result.
"""
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

_CAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CAL_DIR.parents[1]
_MAIN_PY = _REPO_ROOT / "src" / "software" / "acoustic_imager" / "main.py"

# Fullscreen (no desktop visible) — title and log use entire screen
FS_W = 1920
FS_H = 1080
PAD = 28
HEADER_H = 72
LOG_LINE_H = 34
LOG_TOP = HEADER_H + PAD
LOG_H = FS_H - LOG_TOP - 40
LOG_W = FS_W - 2 * PAD - 20
SCROLLBAR_W = 24
SB_X = FS_W - PAD - SCROLLBAR_W  # scrollbar left edge (for hit-test)

STEP_LABELS = {
    "silence": "Silence check",
    "pins": "Pin monitor",
    "geometry": "Geometry",
    "spi_devices": "SPI devices",
    "parsing_ok": "Parsing & rates",
    "metrics": "Metrics",
    "post_gain": "Post-gain",
}


def _read_failed_steps() -> list[str]:
    path = _CAL_DIR / "calibration_suite_last_result.txt"
    failed = []
    try:
        if not path.exists():
            return failed
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            if key in STEP_LABELS and val.strip().lower() in ("false", "0"):
                failed.append(STEP_LABELS[key])
    except Exception:
        pass
    return failed


def _run_suite(gain: str, log_lines: list[str], silence_start_callback: callable, silence_triggered: list) -> None:
    run_suite_py = _CAL_DIR / "run_suite.py"
    cmd = [sys.executable, str(run_suite_py), "--gain", gain]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        cwd=str(_REPO_ROOT),
    )
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            line = line.rstrip()
            log_lines.append(line[:400])  # keep enough for full display and dump
            m = re.match(r"SILENCE_START\s+([\d.]+)", line.strip())
            if m:
                silence_start_callback(float(m.group(1)))
                silence_triggered[0] = True
            elif "[0] Silence check" in line and not silence_triggered[0]:
                silence_start_callback(2.0)  # fallback if SILENCE_START was buffered
                silence_triggered[0] = True
    try:
        err = proc.stderr.read()
    except Exception:
        err = ""
    for line in (err or "").splitlines():
        log_lines.append("[stderr] " + line[:380].rstrip())
    if proc.returncode is not None and proc.returncode != 0:
        log_lines.append("Suite exited with code %d" % proc.returncode)
    # Always dump log when suite finishes (same path as main-thread write)
    try:
        (_CAL_DIR / "calibration_suite_cal_dump.txt").write_text("\n".join(log_lines), encoding="utf-8")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Standalone calibration suite (stops main app; restarts when done)")
    ap.add_argument("--restart-main", action="store_true", help="Restart acoustic imager main app when done")
    ap.add_argument("--gain", choices=("high", "low"), default="high", help="Gain for metrics step")
    args = ap.parse_args()

    log_lines: list[str] = []
    scroll_offset = 0
    max_scroll = 0
    suite_done = False
    result_passed = False
    result_count = 0
    result_total = 0
    show_result_overlay = False
    ok_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
    silence_remaining_sec: float | None = None  # set by thread when SILENCE_START seen; main loop counts down
    silence_triggered: list[bool] = [False]  # so we only start countdown once (SILENCE_START or "[0] Silence check")
    user_dismissed_result: list[bool] = [False]  # list so closure can set from mouse callback
    scroll_ref: list[int] = [0]  # mutable so mouse callback can update scroll
    drag_start_y: list[int | None] = [None]  # for touch/drag scroll: last y when dragging, or None
    scrollbar_drag: list[bool] = [False]  # True while user is dragging in scrollbar (tap/drag to scroll when modal up)

    def on_silence_start(sec: float) -> None:
        nonlocal silence_remaining_sec
        silence_remaining_sec = sec

    def worker() -> None:
        nonlocal suite_done
        _run_suite(args.gain, log_lines, on_silence_start, silence_triggered)
        suite_done = True

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    cv2.namedWindow("Calibration Suite", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Calibration Suite", FS_W, FS_H)
    cv2.setWindowProperty("Calibration Suite", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    SCROLL_STEP = LOG_LINE_H * 3  # 3 lines per wheel tick

    def _on_mouse(event: int, x: int, y: int, _a: int, flags: int) -> None:
        total_log_h = len(log_lines) * LOG_LINE_H
        max_s = max(0, total_log_h - LOG_H)
        in_scrollbar = SB_X <= x < SB_X + SCROLLBAR_W and LOG_TOP <= y < LOG_TOP + LOG_H

        # Click on "Return to app" button (only when overlay is shown)
        if event == cv2.EVENT_LBUTTONDOWN and show_result_overlay and ok_rect[2] > 0:
            ox, oy, ow, oh = ok_rect
            if ox <= x < ox + ow and oy <= y < oy + oh:
                user_dismissed_result[0] = True
                drag_start_y[0] = None
                scrollbar_drag[0] = False
                return
        if event == cv2.EVENT_LBUTTONUP:
            drag_start_y[0] = None
            scrollbar_drag[0] = False
            return
        # Scrollbar tap/drag: jump or drag to position (works when modal is up; touch screens often don't send MOUSEMOVE for log drag)
        if event == cv2.EVENT_LBUTTONDOWN and in_scrollbar and len(log_lines) > 0 and max_s > 0:
            scrollbar_drag[0] = True
            # Map y to scroll offset (thumb center logic)
            t = (y - LOG_TOP) / LOG_H
            scroll_ref[0] = max(0, min(max_s, int(t * max_s)))
            return
        if event == cv2.EVENT_MOUSEMOVE and scrollbar_drag[0] and max_s > 0:
            t = (y - LOG_TOP) / LOG_H
            scroll_ref[0] = max(0, min(max_s, int(t * max_s)))
            return
        # Start log drag on any other touch/click (so user can scroll when modal is up)
        if event == cv2.EVENT_LBUTTONDOWN and len(log_lines) > 0:
            drag_start_y[0] = y
            return
        # Log drag: once drag started, any move updates scroll
        if event == cv2.EVENT_MOUSEMOVE and drag_start_y[0] is not None and len(log_lines) > 0:
            delta = drag_start_y[0] - y  # drag up -> positive delta -> scroll up (decrease offset)
            scroll_ref[0] = max(0, min(max_s, scroll_ref[0] + delta))
            drag_start_y[0] = y
            return
        # Mouse wheel: scroll the log (works whether or not result overlay is shown)
        if event == cv2.EVENT_MOUSEWHEEL and len(log_lines) > 0:
            total_log_h = len(log_lines) * LOG_LINE_H
            max_s = max(0, total_log_h - LOG_H)
            # flags: positive = scroll up (content up = decrease offset), negative = scroll down
            delta = -SCROLL_STEP if flags > 0 else SCROLL_STEP
            scroll_ref[0] = max(0, min(max_s, scroll_ref[0] + delta))

    cv2.setMouseCallback("Calibration Suite", _on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_color = (255, 255, 255)
    blue = (255, 180, 80)
    green = (100, 255, 0)
    red = (0, 100, 255)

    LOG_FONT_SCALE = 0.72
    LOG_FONT_THICKNESS = 2
    # Compute max chars that fit in LOG_W so lines are not cut off before the right edge
    (char_w, _), _ = cv2.getTextSize("W", font, LOG_FONT_SCALE, LOG_FONT_THICKNESS)
    LOG_MAX_CHARS = max(60, (LOG_W - 24) // char_w) if char_w > 0 else 120

    while True:
        frame = np.zeros((FS_H, FS_W, 3), dtype=np.uint8)
        frame[:] = (28, 28, 28)
        cv2.rectangle(frame, (0, 0), (FS_W, FS_H), (70, 70, 70), 2, cv2.LINE_AA)

        cv2.putText(frame, "Calibration Suite", (PAD, 52), font, 1.25, text_color, 2, cv2.LINE_AA)

        # Progress indicator (top right): "N/7 tests" from explicit STEP_PROGRESS lines
        NUM_STEPS = 7
        current_step = 0
        for line in log_lines:
            m = re.search(r"STEP_PROGRESS\s+(\d+)\s+(\d+)", line)
            if m:
                current_step = min(int(m.group(1)) + 1, int(m.group(2)))
        if not suite_done or current_step > 0:
            progress_text = "%d/%d tests" % (current_step, NUM_STEPS)
            PROG_FONT = 1.05
            (pw, ph), _ = cv2.getTextSize(progress_text, font, PROG_FONT, 2)
            prog_x = FS_W - PAD - pw - 20
            prog_y = 52
            cv2.rectangle(frame, (prog_x - 10, prog_y - ph - 10), (FS_W - PAD + 10, prog_y + 10), (40, 40, 40), -1, cv2.LINE_AA)
            cv2.putText(frame, progress_text, (prog_x, prog_y), font, PROG_FONT, (200, 200, 200), 2, cv2.LINE_AA)

        total_log_h = len(log_lines) * LOG_LINE_H
        max_scroll = max(0, total_log_h - LOG_H)
        scroll_offset = scroll_ref[0]
        scroll_offset = min(scroll_offset, max_scroll)
        scroll_ref[0] = scroll_offset
        start_idx = scroll_offset // LOG_LINE_H
        y_off = LOG_TOP + 8 - (scroll_offset % LOG_LINE_H)
        cv2.rectangle(frame, (PAD, LOG_TOP), (PAD + LOG_W, LOG_TOP + LOG_H), (18, 18, 18), -1, cv2.LINE_AA)
        cv2.rectangle(frame, (PAD, LOG_TOP), (PAD + LOG_W, LOG_TOP + LOG_H), (60, 60, 60), 1, cv2.LINE_AA)
        for i in range(start_idx, len(log_lines)):
            ly = y_off + (i - start_idx) * LOG_LINE_H
            if ly < LOG_TOP or ly + LOG_LINE_H > LOG_TOP + LOG_H:
                continue
            line = log_lines[i]
            if len(line) > LOG_MAX_CHARS:
                line = line[: LOG_MAX_CHARS - 3] + "..."
            cv2.putText(frame, line, (PAD + 10, ly + LOG_LINE_H - 8), font, LOG_FONT_SCALE, (245, 245, 245), LOG_FONT_THICKNESS, cv2.LINE_AA)

        if total_log_h > LOG_H:
            sb_x = FS_W - PAD - SCROLLBAR_W
            cv2.rectangle(frame, (sb_x, LOG_TOP), (sb_x + SCROLLBAR_W, LOG_TOP + LOG_H), (45, 45, 45), -1, cv2.LINE_AA)
            thumb_ratio = LOG_H / total_log_h
            thumb_h = max(40, int(LOG_H * thumb_ratio))
            thumb_y = LOG_TOP + int((LOG_H - thumb_h) * scroll_offset / max_scroll) if max_scroll > 0 else LOG_TOP
            cv2.rectangle(frame, (sb_x + 3, thumb_y), (sb_x + SCROLLBAR_W - 3, thumb_y + thumb_h), (120, 120, 120), -1, cv2.LINE_AA)

        # Silence countdown overlay (centered on full screen, large and visible)
        if silence_remaining_sec is not None and silence_remaining_sec > 0:
            sec_int = int(math.ceil(silence_remaining_sec))
            msg = "Stay quiet: %d sec" % sec_int
            (tw, th), _ = cv2.getTextSize(msg, font, 1.2, 2)
            cx, cy = FS_W // 2, FS_H // 2
            bw, bh = tw + 80, th + 50
            bx, by = cx - bw // 2, cy - bh // 2
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (30, 60, 30), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 200, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, msg, (cx - tw // 2, cy + th // 2 - 5), font, 1.2, (0, 255, 100), 2, cv2.LINE_AA)

        # When suite just finished, parse and show result
        if suite_done and not show_result_overlay:
            done_match = None
            for line in reversed(log_lines):
                m = re.search(r"=== Done: (\d+)/(\d+) steps passed ===", line)
                if m:
                    done_match = m
                    break
            if done_match:
                result_count = int(done_match.group(1))
                result_total = int(done_match.group(2))
                result_passed = result_count == result_total
            show_result_overlay = True
            dump_path = _CAL_DIR / "calibration_suite_cal_dump.txt"
            try:
                dump_path.write_text("\n".join(log_lines), encoding="utf-8")
            except Exception:
                pass

        # Result overlay (bottom right corner so user can scroll log; dismiss via button)
        if show_result_overlay:
            res_w, res_h = 420, 200
            res_x = FS_W - res_w - PAD * 2
            res_y = FS_H - res_h - PAD * 2
            cv2.rectangle(frame, (res_x, res_y), (res_x + res_w, res_y + res_h), (35, 35, 35), -1, cv2.LINE_AA)
            cv2.rectangle(frame, (res_x, res_y), (res_x + res_w, res_y + res_h), (140, 140, 140), 2, cv2.LINE_AA)
            if result_passed:
                msg = "PASSED (%d/%d steps)" % (result_count, result_total)
                msg_color = green
            else:
                msg = "FAILED (%d/%d steps)" % (result_count, result_total)
                msg_color = red
            (tw, _), _ = cv2.getTextSize(msg, font, 1.0, 2)
            cv2.putText(frame, msg, (res_x + (res_w - tw) // 2, res_y + 52), font, 1.0, msg_color, 2, cv2.LINE_AA)
            failed_steps = _read_failed_steps()
            if failed_steps:
                failed_str = ", ".join(failed_steps)[:45]
                cv2.putText(frame, "Failed: " + failed_str, (res_x + 16, res_y + 92), font, 0.52, (220, 160, 160), 2, cv2.LINE_AA)
            btn_label = "Return to app"
            BTN_FONT = 0.82
            (tw_btn, _), _ = cv2.getTextSize(btn_label, font, BTN_FONT, 2)
            ok_w, ok_h = max(280, tw_btn + 56), 58
            ok_x = res_x + (res_w - ok_w) // 2
            ok_y = res_y + res_h - ok_h - 20
            ok_rect = (ok_x, ok_y, ok_w, ok_h)
            cv2.rectangle(frame, (ok_x, ok_y), (ok_x + ok_w, ok_y + ok_h), blue, -1, cv2.LINE_AA)
            cv2.rectangle(frame, (ok_x, ok_y), (ok_x + ok_w, ok_y + ok_h), (200, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, btn_label, (ok_x + (ok_w - tw_btn) // 2, ok_y + ok_h // 2 + 10), font, BTN_FONT, text_color, 2, cv2.LINE_AA)

        cv2.imshow("Calibration Suite", frame)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            break
        if user_dismissed_result[0]:
            break
        if show_result_overlay and key in (ord("\r"), ord("\n"), ord(" "), ord("o"), ord("O")):
            break
        # Key-based scroll (Up/Down/Page Up/Page Down) so user can scroll log with keyboard
        if key == 82:  # Up
            scroll_ref[0] = max(0, scroll_ref[0] - SCROLL_STEP)
        elif key == 84:  # Down
            scroll_ref[0] = min(max_scroll, scroll_ref[0] + SCROLL_STEP)
        elif key == 83:  # Page Up (often 83 on Linux)
            scroll_ref[0] = max(0, scroll_ref[0] - LOG_H)
        elif key == 85:  # Page Down
            scroll_ref[0] = min(max_scroll, scroll_ref[0] + LOG_H)

        # Count down silence timer (main thread drives time)
        if silence_remaining_sec is not None:
            silence_remaining_sec -= 0.05
            if silence_remaining_sec <= 0:
                silence_remaining_sec = None

    cv2.destroyAllWindows()

    if args.restart_main and _MAIN_PY.exists():
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT / "src" / "software")
        try:
            subprocess.Popen(
                [sys.executable, str(_MAIN_PY)],
                cwd=str(_REPO_ROOT),
                env=env,
                start_new_session=True,
            )
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

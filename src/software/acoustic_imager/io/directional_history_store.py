"""
Directional detection persistence with daily JSONL rotation.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class DirectionalHistoryStore:
    def __init__(
        self,
        base_dir: str,
        retention_days: int = 7,
        flush_interval_s: float = 1.0,
        enabled: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.base_dir = Path(base_dir)
        self.retention_days = max(1, int(retention_days))
        self.flush_interval_s = max(0.0, float(flush_interval_s))
        self._buffer: List[str] = []
        self._last_flush_s = 0.0
        self._current_date = ""
        self._current_file: Optional[Path] = None
        if self.enabled:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(force=True)

    def add_event(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self._rotate_if_needed()
            line = json.dumps(event, separators=(",", ":"), ensure_ascii=True)
            self._buffer.append(line)
            now = time.time()
            if self.flush_interval_s <= 0.0 or (now - self._last_flush_s) >= self.flush_interval_s:
                self.flush()
        except Exception:
            # Persistence must not block UI loop.
            return

    def flush(self) -> None:
        if not self.enabled or not self._buffer:
            return
        if self._current_file is None:
            self._rotate_if_needed(force=True)
        if self._current_file is None:
            return
        payload = "\n".join(self._buffer) + "\n"
        with self._current_file.open("a", encoding="utf-8") as f:
            f.write(payload)
        self._buffer.clear()
        self._last_flush_s = time.time()

    def close(self) -> None:
        try:
            self.flush()
        except Exception:
            pass

    def _rotate_if_needed(self, force: bool = False) -> None:
        if not self.enabled:
            return
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if force or date_str != self._current_date or self._current_file is None:
            self._current_date = date_str
            self._current_file = self.base_dir / f"directional-history-{date_str}.jsonl"
            self._prune_old_files()

    def _prune_old_files(self) -> None:
        cutoff = datetime.now(timezone.utc).date().toordinal() - self.retention_days
        for p in self.base_dir.glob("directional-history-*.jsonl"):
            stem = p.stem  # directional-history-YYYY-MM-DD
            parts = stem.split("-")
            if len(parts) < 5:
                continue
            try:
                y = int(parts[-3])
                m = int(parts[-2])
                d = int(parts[-1])
                ord_day = datetime(y, m, d, tzinfo=timezone.utc).date().toordinal()
            except Exception:
                continue
            if ord_day < cutoff:
                try:
                    p.unlink()
                except Exception:
                    pass


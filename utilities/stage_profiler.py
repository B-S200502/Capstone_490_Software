# utilities/stage_profiler.py

import time
from collections import defaultdict, deque


class StageProfiler:
    def __init__(self, keep=120):
        self.t0 = None
        self.cur = {}
        self.hist = defaultdict(lambda: deque(maxlen=keep))

    def start_frame(self):
        self.t0 = time.perf_counter()
        self.cur = {}

    def mark(self, name):
        t = time.perf_counter()
        self.cur[name] = t

    def end_frame(self):
        names = list(self.cur.keys())
        prev = self.t0
        for n in names:
            dt = self.cur[n] - prev
            self.hist[n].append(dt)
            prev = self.cur[n]
        self.hist["frame_total"].append(time.perf_counter() - self.t0)

    def ms(self, name):
        h = self.hist.get(name)
        if not h:
            return 0.0
        return 1000.0 * (sum(h) / len(h))

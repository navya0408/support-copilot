"""Tiny in-memory protections for a public demo (they reset when the server restarts).

RateLimiter  - at most N requests per minute per client address.
DailyBudget  - at most N paid/free-tier LLM calls per day in total, so strangers cannot use up your quota.
A limit of 0 (or less) switches the protection off.
"""
import threading
import time
from collections import deque
from datetime import date


class RateLimiter:
    def __init__(self, limit: int, window_seconds: float = 60.0, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self._hits = {}
        self._lock = threading.Lock()
        self._calls = 0

    def allow(self, key: str) -> bool:
        if self.limit <= 0:
            return True
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            self._calls += 1
            if self._calls % 500 == 0:            # occasionally forget clients we have not seen for a while
                for k in [k for k, q in self._hits.items() if not q or now - q[-1] >= self.window]:
                    del self._hits[k]
            return True


class DailyBudget:
    def __init__(self, limit: int, today=date.today):
        self.limit = limit
        self.today = today
        self._day = today()
        self._used = 0
        self._lock = threading.Lock()

    def try_consume(self) -> bool:
        if self.limit <= 0:
            return True
        with self._lock:
            d = self.today()
            if d != self._day:
                self._day, self._used = d, 0
            if self._used >= self.limit:
                return False
            self._used += 1
            return True

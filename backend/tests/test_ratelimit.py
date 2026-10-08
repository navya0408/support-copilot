from datetime import date

from app.ratelimit import DailyBudget, RateLimiter


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_blocks_after_limit_then_recovers():
    clock = FakeClock()
    rl = RateLimiter(2, window_seconds=60, clock=clock)
    assert rl.allow("a") and rl.allow("a")
    assert not rl.allow("a")                 # third request inside the window
    clock.t = 61
    assert rl.allow("a")                     # window passed


def test_clients_are_counted_separately():
    rl = RateLimiter(1, clock=FakeClock())
    assert rl.allow("a")
    assert rl.allow("b")
    assert not rl.allow("a")


def test_zero_limit_disables():
    rl = RateLimiter(0, clock=FakeClock())
    assert all(rl.allow("a") for _ in range(100))


def test_daily_budget_resets_next_day():
    day = {"d": date(2026, 10, 8)}
    b = DailyBudget(2, today=lambda: day["d"])
    assert b.try_consume() and b.try_consume()
    assert not b.try_consume()
    day["d"] = date(2026, 10, 9)
    assert b.try_consume()


def test_daily_budget_zero_means_unlimited():
    b = DailyBudget(0)
    assert all(b.try_consume() for _ in range(1000))

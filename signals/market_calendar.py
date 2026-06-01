"""NSE trading-calendar helpers for QuantPilot.

Thin wrapper over pandas_market_calendars so the rest of the codebase can ask
"what is the next trading day after X" and "is X a trading day" without touching
calendar internals.

Import-safe: building the calendar object at import time is cheap and performs
no network I/O.
"""
from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal

_NSE = mcal.get_calendar("NSE")


def next_market_open(after) -> str:
    """Return the next NSE trading day (ISO date) strictly AFTER ``after``."""
    ts = pd.Timestamp(after)
    sched = _NSE.schedule(start_date=ts, end_date=ts + pd.Timedelta(days=45))
    days = [d.date().isoformat() for d in sched.index if d.date() > ts.date()]
    return days[0]


def first_market_open(on_or_after) -> str:
    """Return the first NSE trading day (ISO date) on or after ``on_or_after``."""
    ts = pd.Timestamp(on_or_after)
    sched = _NSE.schedule(start_date=ts, end_date=ts + pd.Timedelta(days=20))
    days = [d.date().isoformat() for d in sched.index if d.date() >= ts.date()]
    return days[0]


def is_trading_day(d) -> bool:
    """Return True if ``d`` is an NSE trading day."""
    ts = pd.Timestamp(d)
    s = _NSE.schedule(start_date=ts, end_date=ts)
    return len(s) > 0

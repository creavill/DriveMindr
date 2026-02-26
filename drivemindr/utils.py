"""
DriveMindr utilities — formatting helpers, size conversions, timing.

No network calls. No external dependencies beyond the standard library.
"""

from __future__ import annotations

import time
from typing import Callable


def format_bytes(n: int) -> str:
    """Human-readable byte string: ``1234567`` → ``1.18 MB``."""
    if n < 0:
        return f"-{format_bytes(-n)}"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0  # type: ignore[assignment]
    return f"{n:.2f} PB"


def format_count(n: int) -> str:
    """Comma-separated integer string: ``1234567`` → ``1,234,567``."""
    return f"{n:,}"


def timed(label: str | None = None) -> Callable:
    """Decorator that logs execution time at DEBUG level.

    Usage::

        @timed("scan phase")
        def scan_drive(...): ...
    """
    import functools
    import logging

    log = logging.getLogger("drivemindr.timing")

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - start
            log.debug("%s completed in %.2fs", label or fn.__name__, elapsed)
            return result
        return wrapper
    return decorator


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp *value* between *low* and *high*."""
    return max(low, min(high, value))

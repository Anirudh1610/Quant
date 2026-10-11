"""
core/kite_client.py
--------------------
Thin wrapper around KiteConnect that adds:
  - client-side rate limiting (token bucket, sleeps instead of erroring)
  - retry with exponential backoff on transient/network errors
  - a clear, non-retried failure on auth errors (stale token needs a human
    to re-run scripts/generate_access_token.py, retrying won't help)

Every data-layer module should call the Kite API through a KiteRateLimited
instance rather than importing kiteconnect directly, so rate limiting and
retry behavior stay in one place.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable, TypeVar

from kiteconnect.exceptions import (
    DataException,
    InputException,
    KiteException,
    NetworkException,
    TokenException,
)

from config import RATE_LIMIT, RateLimitConfig
from core.session import get_kite_client

logger = logging.getLogger("nifty_options.kite_client")

T = TypeVar("T")

# Errors worth retrying (transient). TokenException/InputException are NOT
# retried — retrying a stale-token or malformed-request error just burns the
# rate-limit budget for the same guaranteed failure.
_RETRYABLE = (NetworkException, DataException)


class RateLimiter:
    """Sliding-window token bucket, sleeps the caller instead of raising."""

    def __init__(self, requests_per_second: float) -> None:
        self._interval = 1.0 / requests_per_second
        self._last_call: float = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


class KiteRateLimited:
    """
    Wraps a KiteConnect instance. Call Kite API methods via `.call(fn_name,
    *args, **kwargs)` (or the convenience passthroughs below) so every
    request goes through the rate limiter and retry logic.
    """

    def __init__(self, cfg: RateLimitConfig = RATE_LIMIT, kite=None) -> None:
        self.kite = kite or get_kite_client()
        self.cfg = cfg
        self._limiter = RateLimiter(cfg.requests_per_second)

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        method: Callable[..., T] = getattr(self.kite, method_name)
        last_exc: Exception | None = None

        for attempt in range(1, self.cfg.max_retries + 1):
            self._limiter.wait()
            try:
                return method(*args, **kwargs)
            except TokenException:
                raise  # not retryable — surface immediately, needs re-login
            except InputException:
                raise  # not retryable — caller passed bad params
            except _RETRYABLE as e:
                last_exc = e
                backoff = self.cfg.backoff_base_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Kite call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    method_name, attempt, self.cfg.max_retries, e, backoff,
                )
                time.sleep(backoff)
            except KiteException:
                raise  # any other Kite error: surface, don't mask it

        raise RuntimeError(
            f"Kite call {method_name} failed after {self.cfg.max_retries} attempts"
        ) from last_exc

    def call_in_chunks(self, method_name: str, items: list[str],
                        chunk_size: int | None = None) -> dict:
        """
        For endpoints like quote()/ltp()/ohlc() that accept a list of
        instrument identifiers and return a dict keyed by identifier. Splits
        `items` into chunks (Kite historically caps these calls around
        several hundred symbols per request) and merges the results.
        """
        chunk_size = chunk_size or self.cfg.quote_chunk_size
        merged: dict = {}
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            merged.update(self.call(method_name, chunk))
        return merged

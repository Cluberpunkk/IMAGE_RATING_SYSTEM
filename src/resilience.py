"""
resilience.py — production hardening for the paid VLM calls
===========================================================
Two layers that turn "works in the demo" into "survives a bad night":

  1) retry_call()   — bounded exponential backoff + jitter on TRANSIENT errors
                      (timeout, 429 rate-limit, 5xx, connection reset). Non-
                      transient errors (bad request, auth, schema) fail FAST —
                      retrying them just burns latency and quota.

  2) CircuitBreaker — after N consecutive failures the breaker OPENS and calls
                      fast-fail for a cooldown window instead of hammering a
                      sick API on every image. It then HALF-OPENs to probe
                      recovery and CLOSES again on the first success.

Why this matters here: each image costs 2 paid Gemini calls. Without these,
one Gemini incident means every image pays full timeout latency and then falls
back — slow AND expensive. With them, a sick dependency is detected once, the
pipeline fast-fails to its safe FALLBACK (score=0.25, not cached, flagged for
retry), and recovers automatically the moment Gemini is healthy again.

Dependency-free (stdlib only) and fully deterministic under test: inject
`sleep`, `now`, and `rng` so CI and the simulation path stay hermetic.
"""

from __future__ import annotations

import socket
import threading
import time
from typing import Callable, Iterable, Optional


class CircuitOpenError(RuntimeError):
    """Raised when the breaker is OPEN and fast-failing (the API is presumed
    sick). The pipeline treats this like any other failure -> safe fallback."""


# Substrings that mark a TRANSIENT, worth-retrying failure. Matched against the
# exception's type name and message, case-insensitively. Anything not matched
# is treated as permanent (auth, malformed request, bad schema) and fails fast.
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "deadline", "temporarily", "unavailable",
    "connection", "reset", "broken pipe", "rate", "quota", "overloaded",
    "resource exhausted", "429", "500", "502", "503", "504",
)
_TRANSIENT_TYPES: tuple = (
    TimeoutError, ConnectionError, socket.timeout,
)


def is_transient(exc: BaseException) -> bool:
    """Best-effort classifier: is this failure worth a retry?

    Conservative on purpose — we'd rather retry a permanent error a couple of
    times than fail fast on a recoverable one. Auth/validation errors carry
    none of the markers, so they correctly fail fast.
    """
    if isinstance(exc, CircuitOpenError):
        return False  # the breaker already decided; do not retry behind it
    if isinstance(exc, _TRANSIENT_TYPES):
        return True
    blob = (type(exc).__name__ + " " + str(exc)).lower()
    # A status code attribute, if the SDK exposes one, is the strongest signal.
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(code, int) and (code == 429 or 500 <= code <= 599):
        return True
    return any(m in blob for m in _TRANSIENT_MARKERS)


def retry_call(
    fn: Callable,
    *args,
    max_attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: float = 0.25,
    retryable: Callable[[BaseException], bool] = is_transient,
    breaker: "Optional[CircuitBreaker]" = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[], float] = None,
    **kwargs,
):
    """Call `fn(*args, **kwargs)` with bounded exponential backoff + jitter.

    - Retries only when `retryable(exc)` is True, up to `max_attempts` tries.
    - Backoff: base_delay * 2**(attempt-1), capped at max_delay, then a random
      jitter of ±`jitter` fraction is applied to avoid thundering-herd sync.
    - If a `breaker` is supplied, it gates entry (fast-fail when OPEN) and is
      told of every success/failure so it can trip and recover.
    - `sleep`/`rng` are injectable so tests run with zero real delay.
    """
    if rng is None:
        # Imported lazily and tolerant of the no-randomness sandbox: a fixed
        # mid-point jitter is still correct, just not de-synchronised.
        try:
            import random
            rng = random.random
        except Exception:  # pragma: no cover - defensive
            rng = lambda: 0.5

    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        if breaker is not None:
            breaker.before_call()  # raises CircuitOpenError if fast-failing
        try:
            result = fn(*args, **kwargs)
            if breaker is not None:
                breaker.on_success()
            return result
        except BaseException as exc:  # noqa: BLE001 - we re-raise below
            if breaker is not None:
                breaker.on_failure()
            last_exc = exc
            if attempt >= max_attempts or not retryable(exc):
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1.0 + jitter * (2.0 * rng() - 1.0)  # ±jitter fraction
            sleep(max(0.0, delay))
    # Unreachable (loop either returns or raises), but keeps type-checkers calm.
    raise last_exc  # type: ignore[misc]  # pragma: no cover


class CircuitBreaker:
    """A minimal, thread-safe three-state breaker.

    CLOSED      -> calls flow; consecutive failures are counted.
    OPEN        -> calls fast-fail with CircuitOpenError until `reset_timeout`
                   elapses since the trip.
    HALF_OPEN   -> a single probe call is allowed; success CLOSES the breaker,
                   failure RE-OPENS it for another cooldown.

    Thread-safe because the Gemini path issues its two calls concurrently.
    """

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        fail_max: int = 5,
        reset_timeout: float = 30.0,
        now: Callable[[], float] = time.monotonic,
        name: str = "vlm",
    ):
        self.fail_max = fail_max
        self.reset_timeout = reset_timeout
        self._now = now
        self.name = name
        self._lock = threading.Lock()
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self.trip_count = 0  # observability: how many times we've opened

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def before_call(self) -> None:
        """Gate entry. Raises CircuitOpenError when fast-failing."""
        with self._lock:
            if self._state == self.OPEN:
                if (self._now() - self._opened_at) >= self.reset_timeout:
                    self._state = self.HALF_OPEN  # allow one probe
                else:
                    raise CircuitOpenError(
                        f"circuit '{self.name}' OPEN — fast-failing for "
                        f"{self.reset_timeout - (self._now() - self._opened_at):.1f}s"
                    )

    def on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = self.CLOSED

    def on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._state == self.HALF_OPEN:
                self._open_locked()  # probe failed -> back to OPEN
            elif self._consecutive_failures >= self.fail_max:
                self._open_locked()

    def _open_locked(self) -> None:
        self._state = self.OPEN
        self._opened_at = self._now()
        self.trip_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "trip_count": self.trip_count,
            }

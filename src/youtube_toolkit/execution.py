"""Execution policies for retried operations and partial-result worker pools."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import urllib.error

_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})


def retry_transient[Result](
    operation: Callable[[], Result],
    *,
    attempts: int,
    backoff_seconds: float,
    retryable_errors: tuple[type[Exception], ...],
    sleep: Callable[[float], None] = time.sleep,
) -> Result:
    """Retry one idempotent operation while preserving its final underlying exception."""
    if attempts <= 0:
        raise ValueError("retry attempts must be greater than 0")
    if backoff_seconds < 0:
        raise ValueError("retry backoff must be non-negative")

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == attempts or not _is_retryable_error(error, retryable_errors):
                raise
            sleep(_retry_delay_seconds(error, attempt, backoff_seconds))
    raise AssertionError("retry loop exhausted without returning or raising")


def _is_retryable_error(
    error: Exception,
    retryable_errors: tuple[type[Exception], ...],
) -> bool:
    """Treat configured transport errors and selected HTTP statuses as transient."""
    if isinstance(error, urllib.error.HTTPError):
        return error.code in _RETRYABLE_HTTP_STATUS_CODES
    return isinstance(error, retryable_errors)


def _retry_delay_seconds(
    error: Exception,
    attempt: int,
    backoff_seconds: float,
) -> float:
    """Honor numeric Retry-After responses before bounded exponential backoff."""
    if isinstance(error, urllib.error.HTTPError):
        retry_after = error.headers.get("Retry-After") if error.headers is not None else None
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return backoff_seconds * (2 ** (attempt - 1))


def collect_concurrently[JobKey, Job, Result](
    jobs: Mapping[JobKey, Job],
    worker: Callable[[Job], Result],
    *,
    max_workers: int,
) -> tuple[dict[JobKey, Result], dict[JobKey, Exception]]:
    """Return keyed successes and failures without letting one worker stop the batch."""
    if max_workers <= 0:
        raise ValueError("concurrent worker count must be greater than 0")
    if not jobs:
        return {}, {}

    completed: dict[JobKey, Result] = {}
    failures: dict[JobKey, Exception] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        futures = {executor.submit(worker, job): key for key, job in jobs.items()}
        for future in as_completed(futures):
            key = futures[future]
            try:
                completed[key] = future.result()
            except Exception as error:
                failures[key] = error
    return completed, failures

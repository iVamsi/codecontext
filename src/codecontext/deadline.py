"""One sync budget shared by nested Git, parser, and database operations."""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_deadline: ContextVar[float | None] = ContextVar("sync_deadline", default=None)


class SyncDeadlineExceeded(Exception):
    pass


def expired() -> bool:
    deadline = _deadline.get()
    return deadline is not None and time.monotonic() >= deadline


def check_deadline() -> None:
    if expired():
        raise SyncDeadlineExceeded


def remaining_timeout(limit: float) -> float:
    check_deadline()
    deadline = _deadline.get()
    return limit if deadline is None else min(limit, max(0.001, deadline - time.monotonic()))


@contextmanager
def sync_deadline(seconds: float) -> Iterator[None]:
    token = _deadline.set(time.monotonic() + seconds)
    try:
        yield
    finally:
        _deadline.reset(token)

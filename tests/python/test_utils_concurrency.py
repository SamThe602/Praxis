"""Tests for the concurrency helpers."""

from __future__ import annotations

import time
from typing import Callable

import pytest

from packages.utils.concurrency import ParallelExecutionError, ParallelTimeoutError, run_parallel


def test_run_parallel_preserves_order() -> None:
    """Results should mirror the order of the input tasks."""

    def make_task(value: int) -> Callable[[], int]:
        def _task() -> int:
            time.sleep(0.01)
            return value * 2

        return _task

    tasks = [make_task(index) for index in range(5)]
    outcome = run_parallel(tasks, mode="thread", task_kind="io")
    assert outcome == [index * 2 for index in range(5)]


def test_run_parallel_propagates_exceptions() -> None:
    """The first failing task should surface as a ParallelExecutionError."""

    def ok() -> int:
        return 1

    def fail() -> int:
        raise ValueError("boom")

    with pytest.raises(ParallelExecutionError) as excinfo:
        run_parallel([ok, fail, ok], mode="thread")

    assert isinstance(excinfo.value.cause, ValueError)
    assert excinfo.value.index == 1


def test_run_parallel_times_out() -> None:
    """The helper should raise when execution exceeds the deadline."""

    def slow() -> int:
        time.sleep(0.2)
        return 42

    with pytest.raises(ParallelTimeoutError):
        run_parallel([slow, slow], mode="thread", timeout=0.05)

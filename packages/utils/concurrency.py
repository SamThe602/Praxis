"""Parallel execution helpers with deterministic ordering and robust error handling."""

from __future__ import annotations

import concurrent.futures
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence, TypeVar

from packages.utils.config import load_config

T = TypeVar("T")

__all__ = ["ParallelExecutionError", "ParallelTimeoutError", "run_parallel"]


@dataclass(frozen=True)
class _ConcurrencyDefaults:
    mode: Literal["auto", "thread", "process"]
    kind: Literal["io", "cpu"]
    thread_max_workers: int | None
    thread_timeout: float | None
    process_max_workers: int | None
    process_timeout: float | None


class ParallelExecutionError(RuntimeError):
    """Raised when a parallel task fails.

    The exception records the index of the failed task, the execution mode, and a
    lightweight representation of the callable to aid debugging.
    """

    def __init__(self, *, index: int, mode: str, task: str, cause: BaseException) -> None:
        message = f"parallel task {index} failed in {mode} mode: {cause}"
        super().__init__(message)
        self.index = index
        self.mode = mode
        self.task = task
        self.cause = cause


class ParallelTimeoutError(TimeoutError):
    """Raised when ``run_parallel`` exceeds the configured deadline."""

    def __init__(self, *, timeout: float, completed: int, total: int) -> None:
        super().__init__(f"parallel execution timed out after {timeout:.3f}s")
        self.timeout = timeout
        self.completed = completed
        self.total = total


def run_parallel(
    tasks: Sequence[Callable[[], T]] | Iterable[Callable[[], T]],
    *,
    mode: Literal["auto", "thread", "process"] | None = None,
    task_kind: Literal["auto", "io", "cpu"] = "auto",
    max_workers: int | None = None,
    timeout: float | None = None,
    cancel_on_error: bool = True,
) -> list[T]:
    """Execute ``tasks`` in parallel while preserving input order.

    Parameters
    ----------
    tasks:
        Iterable of parameterless callables to execute. Results are returned in
        the same order as the input sequence.
    mode:
        ``"thread"`` to force a :class:`ThreadPoolExecutor`, ``"process"`` for a
        :class:`ProcessPoolExecutor`, or ``"auto"`` (default) to derive the mode
        from ``task_kind`` and configuration defaults.
    task_kind:
        Hint indicating whether tasks are IO-bound (``"io"``) or CPU-bound
        (``"cpu"``). ``"auto"`` falls back to the configured default and
        ultimately treats tasks as IO-bound.
    max_workers:
        Maximum parallel workers. When omitted the value is derived from the
        configuration file or executor heuristics.
    timeout:
        Optional overall timeout in seconds. A :class:`ParallelTimeoutError` is
        raised if the deadline elapses before all tasks finish.
    cancel_on_error:
        Whether to cancel remaining futures as soon as one task fails.

    Raises
    ------
    ParallelExecutionError
        When a task raises an exception; the original exception is attached as
        ``.__cause__``.
    ParallelTimeoutError
        When the aggregate execution exceeds the timeout.
    """

    task_list = list(tasks)
    if not task_list:
        return []
    for index, task in enumerate(task_list):
        if not callable(task):
            raise TypeError(f"task at position {index} is not callable: {task!r}")

    defaults = _load_defaults()
    selected_mode = _select_mode(mode, task_kind, defaults)
    selected_timeout = timeout if timeout is not None else _default_timeout(selected_mode, defaults)
    selected_workers = _resolve_max_workers(
        selected_mode,
        max_workers if max_workers is not None else _default_max_workers(selected_mode, defaults),
        len(task_list),
    )

    executor: concurrent.futures.Executor
    if selected_mode == "process":
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=selected_workers)
    else:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=selected_workers)

    futures: list[concurrent.futures.Future[tuple[int, T]]] = []
    start_time = time.perf_counter()
    try:
        for index, task in enumerate(task_list):
            future = executor.submit(_call_wrapper, index, task)
            setattr(future, "_task_index", index)
            futures.append(future)

        results: list[T | None] = [None] * len(task_list)
        pending = set(futures)
        completed = 0
        while pending:
            wait_timeout = None
            if selected_timeout is not None:
                remaining = selected_timeout - (time.perf_counter() - start_time)
                if remaining <= 0.0:
                    _cancel_pending(pending)
                    raise ParallelTimeoutError(
                        timeout=selected_timeout,
                        completed=completed,
                        total=len(task_list),
                    )
                wait_timeout = max(0.0, min(0.5, remaining))
            done, pending = concurrent.futures.wait(
                pending, timeout=wait_timeout, return_when=concurrent.futures.FIRST_COMPLETED
            )
            if not done:
                continue
            for future in done:
                try:
                    index, value = future.result()
                except BaseException as exc:  # inclusive of KeyboardInterrupt to ensure cleanup
                    task_index = _future_index(future)
                    if cancel_on_error:
                        _cancel_pending(pending)
                    task_repr = (
                        _safe_callable_repr(task_list[task_index])
                        if 0 <= task_index < len(task_list)
                        else "<unknown>"
                    )
                    raise ParallelExecutionError(
                        index=task_index if task_index >= 0 else -1,
                        mode=selected_mode,
                        task=task_repr,
                        cause=exc,
                    ) from exc
                results[index] = value
                completed += 1
        assert all(item is not None for item in results)
        return [item for item in results if item is not None]
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def _load_defaults() -> _ConcurrencyDefaults:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "utils.yaml"
    data: dict[str, object] = {}
    if config_path.exists():
        try:
            raw = load_config(config_path)
            if isinstance(raw, dict):
                data = raw
        except Exception:  # pragma: no cover - config parsing should not break runtime
            data = {}

    section = data.get("concurrency") if isinstance(data, dict) else {}
    if not isinstance(section, dict):
        section = {}
    mode = str(section.get("default_mode", "auto")).lower()
    kind = str(section.get("default_kind", "io")).lower()
    thread_cfg = section.get("thread", {}) if isinstance(section.get("thread"), dict) else {}
    process_cfg = section.get("process", {}) if isinstance(section.get("process"), dict) else {}

    return _ConcurrencyDefaults(
        mode=_normalise_mode(mode),
        kind=_normalise_kind(kind),
        thread_max_workers=(
            _coerce_int(thread_cfg.get("max_workers")) if isinstance(thread_cfg, dict) else None
        ),
        thread_timeout=(
            _coerce_float(thread_cfg.get("timeout")) if isinstance(thread_cfg, dict) else None
        ),
        process_max_workers=(
            _coerce_int(process_cfg.get("max_workers")) if isinstance(process_cfg, dict) else None
        ),
        process_timeout=(
            _coerce_float(process_cfg.get("timeout")) if isinstance(process_cfg, dict) else None
        ),
    )


_DEFAULT_CPU_KIND: Literal["io", "cpu"] = "cpu"
_DEFAULT_IO_KIND: Literal["io", "cpu"] = "io"


def _select_mode(
    mode: Literal["auto", "thread", "process"] | None,
    task_kind: Literal["auto", "io", "cpu"],
    defaults: _ConcurrencyDefaults,
) -> Literal["thread", "process"]:
    if mode and mode != "auto":
        return "process" if mode == "process" else "thread"
    selected_kind = task_kind
    if selected_kind == "auto":
        selected_kind = defaults.kind
    if selected_kind == "cpu":
        cpu_count = os.cpu_count()
        return "process" if cpu_count is not None and cpu_count > 1 else "thread"
    return "thread"


def _default_timeout(
    mode: Literal["thread", "process"], defaults: _ConcurrencyDefaults
) -> float | None:
    if mode == "process":
        return defaults.process_timeout
    return defaults.thread_timeout


def _default_max_workers(
    mode: Literal["thread", "process"], defaults: _ConcurrencyDefaults
) -> int | None:
    if mode == "process":
        return defaults.process_max_workers
    return defaults.thread_max_workers


def _resolve_max_workers(
    mode: Literal["thread", "process"], configured: int | None, task_count: int
) -> int:
    if configured is not None and configured > 0:
        return min(configured, max(1, task_count))

    if mode == "process":
        cpu_count = os.cpu_count() or 1
        return min(cpu_count, max(1, task_count))

    # Thread pools default to 5 * CPU count, but we tame it to avoid oversubscription in tests.
    cpu_count = os.cpu_count() or 1
    heuristic = max(1, min(task_count, cpu_count * 4))
    return heuristic


def _cancel_pending(pending: Iterable[concurrent.futures.Future[Any]]) -> None:
    for future in pending:
        future.cancel()


def _future_index(future: concurrent.futures.Future[tuple[int, T]]) -> int:
    index = getattr(future, "_task_index", None)
    return index if isinstance(index, int) else -1


def _call_wrapper(index: int, task: Callable[[], T]) -> tuple[int, T]:
    if hasattr(task, "__wrapped__"):
        task = getattr(task, "__wrapped__")  # type: ignore[assignment]
    return index, task()


def _safe_callable_repr(task: Callable[[], T]) -> str:
    name = getattr(task, "__qualname__", None) or getattr(task, "__name__", None)
    module = getattr(task, "__module__", None)
    if name and module:
        return f"{module}.{name}"
    return repr(task)


def _normalise_mode(value: str) -> Literal["auto", "thread", "process"]:
    if value not in {"auto", "thread", "process"}:
        return "auto"
    return value  # type: ignore[return-value]


def _normalise_kind(value: str) -> Literal["io", "cpu"]:
    if value not in {"io", "cpu"}:
        return "io"
    return value  # type: ignore[return-value]


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None

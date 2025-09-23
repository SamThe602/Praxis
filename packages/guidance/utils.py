"""Utility helpers shared across the lightweight training flows.

The production project uses richer infrastructure for data pipelines and
training orchestration.  The unit-test focused environment in this repository
needs a compact, deterministic subset of those helpers so tests can exercise
behaviour without pulling in heavy runtime dependencies.  The primitives below
cover the most common needs: seeding, shuffling, batching, and early stopping
checks.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Sequence, Tuple, TypeVar

T = TypeVar("T")

__all__ = [
    "seed_everything",
    "deterministic_shuffle",
    "batched",
    "EarlyStoppingMonitor",
    "should_stop_early",
]


def seed_everything(seed: int | None, *, deterministic: bool = False) -> object | None:
    """Seed Python, NumPy, and PyTorch RNGs when available.

    Parameters
    ----------
    seed:
        The seed value to apply.  ``None`` leaves the global state untouched
        which is useful during ad-hoc experiments.  Negative seeds are rejected
        to catch accidental casts from signed 32-bit types.
    deterministic:
        When ``True`` the helper also toggles PyTorch deterministic algorithms
        (if PyTorch is installed).  This mirrors the behaviour of the training
        scripts which prefer reproducibility over absolute throughput during
        tests.

    Returns
    -------
    Optional ``torch.Generator`` instance seeded with the provided value.  When
    PyTorch is unavailable the function returns ``None`` while still seeding the
    standard library RNG.
    """

    if seed is None:
        return None
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    generator: Any = None

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - NumPy is available in CI
        np = None  # type: ignore

    if np is not None:
        np.random.seed(seed)

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - PyTorch is available in CI
        torch = None  # type: ignore

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - CUDA disabled in CI
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        generator = torch.Generator()
        generator.manual_seed(seed)

    return generator


def deterministic_shuffle(items: Sequence[T], seed: int) -> List[T]:
    """Return a deterministically shuffled copy of ``items``.

    The helper mirrors :func:`random.shuffle` but isolates the RNG via
    ``random.Random(seed)`` so that other code relying on the global RNG state is
    unaffected.  A new ``list`` is returned to avoid modifying the caller's
    container in-place.
    """

    if seed < 0:
        raise ValueError("seed must be non-negative")
    values = list(items)
    rnd = random.Random(seed)
    rnd.shuffle(values)
    return values


def batched(
    iterable: Iterable[T], batch_size: int, *, drop_last: bool = False
) -> Iterator[List[T]]:
    """Yield successive batches from ``iterable``.

    Parameters
    ----------
    iterable:
        Source items to split into batches.
    batch_size:
        Maximum number of elements per batch.  Must be strictly positive.
    drop_last:
        When ``True`` the final partially filled batch is discarded.  This is
        handy for gradient-accumulation steps that expect uniform batch sizes.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    buffer: List[T] = []
    for item in iterable:
        buffer.append(item)
        if len(buffer) == batch_size:
            yield buffer
            buffer = []

    if buffer and not drop_last:
        yield buffer


@dataclass(slots=True)
class EarlyStoppingMonitor:
    """Track metric improvements and decide when to stop training.

    The monitor implements the classic patience-based early stopping heuristic.
    Call :meth:`update` with the latest metric value; the method returns
    ``True`` when the patience budget has been exhausted (i.e. the caller should
    interrupt training).
    """

    patience: int = 5
    min_delta: float = 0.0
    mode: str = "max"
    best: float | None = None
    best_step: int | None = None
    _bad_steps: int = 0

    def __post_init__(self) -> None:
        mode_normalised = self.mode.lower()
        if mode_normalised not in {"min", "max"}:
            raise ValueError("mode must be either 'min' or 'max'")
        if self.patience < 1:
            raise ValueError("patience must be at least 1")
        self.mode = mode_normalised

    def update(self, value: float, *, step: int | None = None) -> bool:
        """Register ``value`` and return ``True`` if training should stop."""

        if self.best is None:
            self.best = value
            self.best_step = step
            self._bad_steps = 0
            return False

        improved = self._is_improvement(value)
        if improved:
            self.best = value
            self.best_step = step
            self._bad_steps = 0
        else:
            self._bad_steps += 1

        return self._bad_steps >= self.patience

    def state(self) -> Tuple[float | None, int | None, int]:
        """Expose internal counters for diagnostics and tests."""

        return self.best, self.best_step, self._bad_steps

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value >= self.best + self.min_delta
        return value <= self.best - self.min_delta


def should_stop_early(
    values: Sequence[float],
    *,
    patience: int = 5,
    min_delta: float = 0.0,
    mode: str = "max",
) -> bool:
    """Convenience wrapper for evaluating an entire metric history at once."""

    monitor = EarlyStoppingMonitor(patience=patience, min_delta=min_delta, mode=mode)
    for index, value in enumerate(values):
        if monitor.update(value, step=index):
            return True
    return False

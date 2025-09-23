"""Tests for deterministic guidance utilities."""

from __future__ import annotations

import random
from typing import Optional, cast

import numpy as np
import pytest
import torch

from packages.guidance import utils


def _draw_random_triplet(seed: int) -> tuple[float, float, float]:
    generator_obj = utils.seed_everything(seed)
    python_value = random.random()
    numpy_value = float(np.random.rand())
    torch_generator = cast(Optional[torch.Generator], generator_obj)
    torch_value = float(torch.rand(1, generator=torch_generator).item()) if torch_generator else 0.0
    return python_value, numpy_value, torch_value


def test_seed_everything_is_reproducible() -> None:
    first = _draw_random_triplet(13_579)
    # Advance the RNGs to confirm reseeding resets state.
    random.random()
    np.random.rand()
    torch.rand(1)
    second = _draw_random_triplet(13_579)
    assert first == pytest.approx(second)


def test_deterministic_shuffle_does_not_touch_global_rng() -> None:
    random.seed(2024)
    _ = random.random()
    expected_next = random.random()

    random.seed(2024)
    _ = random.random()
    shuffled = utils.deterministic_shuffle([1, 2, 3, 4], seed=17)
    assert shuffled != [1, 2, 3, 4]
    assert shuffled == utils.deterministic_shuffle([1, 2, 3, 4], seed=17)
    assert random.random() == expected_next


def test_batched_handles_partial_and_drop_last() -> None:
    items = list(range(7))
    batches = list(utils.batched(items, batch_size=3))
    assert batches == [[0, 1, 2], [3, 4, 5], [6]]

    drop_last = list(utils.batched(items, batch_size=3, drop_last=True))
    assert drop_last == [[0, 1, 2], [3, 4, 5]]

    with pytest.raises(ValueError):
        list(utils.batched(items, batch_size=0))


def test_early_stopping_monitor_and_helper() -> None:
    monitor = utils.EarlyStoppingMonitor(patience=2, min_delta=0.05, mode="max")
    assert monitor.update(0.2) is False
    assert monitor.update(0.22) is False  # improvement within delta -> no reset
    assert monitor.update(0.35) is False  # significant improvement resets counter
    assert monitor.update(0.34) is False
    assert monitor.update(0.33) is True

    should_stop = utils.should_stop_early(
        [0.5, 0.51, 0.49, 0.48], patience=2, min_delta=0.02, mode="max"
    )
    assert should_stop is True

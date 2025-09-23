"""Pruning scaffold."""

from __future__ import annotations

from typing import Any


def should_prune(state: Any) -> bool:
    raise NotImplementedError("Pruning scaffold.")

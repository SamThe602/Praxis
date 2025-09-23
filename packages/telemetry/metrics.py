"""Metrics scaffold."""

from __future__ import annotations

from typing import Any


def emit(metric: str, value: Any) -> None:
    raise NotImplementedError("Telemetry metrics scaffold.")

"""Hook scaffolding for telemetry."""

from __future__ import annotations

from typing import Any, Callable


def register_hook(name: str, fn: Callable[[Any], None]) -> None:
    raise NotImplementedError("Telemetry hooks scaffold.")

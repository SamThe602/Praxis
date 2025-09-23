"""Interfaces between synthesizer and other components (scaffold)."""

from __future__ import annotations

from typing import Protocol, Any


class Verifier(Protocol):
    """Verifier protocol scaffold."""

    def verify(self, program: Any) -> Any:  # pragma: no cover - scaffold
        ...

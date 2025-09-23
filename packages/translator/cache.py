"""Translation cache scaffold."""

from __future__ import annotations

from typing import Dict, Tuple

_CACHE: Dict[Tuple[str, str], object] = {}


def get(prompt: str, locale: str = "en") -> object | None:
    return _CACHE.get((prompt, locale))


def set(prompt: str, locale: str, spec: object) -> None:
    _CACHE[(prompt, locale)] = spec

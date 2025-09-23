"""Proof cache scaffold."""

from __future__ import annotations

from typing import Dict, Any

_CACHE: Dict[str, Any] = {}


def get(key: str) -> Any:
    return _CACHE.get(key)


def set(key: str, value: Any) -> None:
    _CACHE[key] = value

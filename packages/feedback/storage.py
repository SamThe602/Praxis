"""Feedback storage scaffold."""

from __future__ import annotations

from typing import Any, List

_STORAGE: List[Any] = []


def add(record: Any) -> None:
    _STORAGE.append(record)


def all_feedback() -> List[Any]:
    return list(_STORAGE)

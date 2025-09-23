"""Persistent translation cache.

This module provides a lightweight JSON-backed cache so repeated natural
language translations do not recompute deterministic rule-based logic. The
cache is intentionally simple (single-process writes guarded by a lock) and
keeps payloads human-readable for debugging.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from threading import RLock
from typing import Any, Dict

_CACHE_LOCK = RLock()
_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache" / "translator"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class CacheEntry:
    """Payload stored on disk for a translation request."""

    prompt: str
    locale: str
    payload: Dict[str, Any]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, sort_keys=True, indent=2)


def _path_for(prompt: str, locale: str) -> Path:
    digest = sha256(f"{locale}::{prompt}".encode("utf-8")).hexdigest()
    return _CACHE_ROOT / f"{digest}.json"


def get(prompt: str, locale: str = "en") -> Dict[str, Any] | None:
    """Return cached payload for prompt/locale pair if it exists."""
    path = _path_for(prompt, locale)
    if not path.exists():
        return None
    with _CACHE_LOCK:
        try:
            loaded: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
    if loaded.get("version") != 1 or "payload" not in loaded:
        return None
    return loaded["payload"]  # type: ignore[return-value]


def set(prompt: str, locale: str, payload: Dict[str, Any]) -> None:
    """Persist payload for the given prompt/locale pair."""
    entry = CacheEntry(prompt=prompt, locale=locale, payload=payload)
    path = _path_for(prompt, locale)
    with _CACHE_LOCK:
        path.write_text(entry.to_json(), encoding="utf-8")


def clear() -> None:
    """Delete every cached translation. Mostly useful for tests."""
    with _CACHE_LOCK:
        for candidate in _CACHE_ROOT.glob("*.json"):
            try:
                candidate.unlink()
            except FileNotFoundError:
                continue

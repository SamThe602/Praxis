"""File-backed cache for proof obligations."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from . import obligations

__all__ = [
    "DEFAULT_CACHE_DIR",
    "cache_path",
    "get",
    "set",
    "store_result",
    "load_result",
]

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = _ROOT / "data" / "processed" / "proofs"

_LOCK = threading.RLock()
_MEMORY: dict[str, Any] = {}


def _ensure_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cache_path(key: str, *, cache_dir: Path | str | None = None) -> Path:
    base = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    return _ensure_dir(base) / f"{key}.json"


def get(key: str, *, cache_dir: Path | str | None = None) -> Any | None:
    with _LOCK:
        if key in _MEMORY:
            return _MEMORY[key]
    path = cache_path(key, cache_dir=cache_dir)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    with _LOCK:
        _MEMORY[key] = data
    return data


def set(key: str, value: Any, *, cache_dir: Path | str | None = None) -> None:
    path = cache_path(key, cache_dir=cache_dir)
    payload = json.dumps(value, sort_keys=True, indent=2)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)
    with _LOCK:
        _MEMORY[key] = value


def store_result(
    obligation: obligations.ProofObligation,
    result: obligations.ProofResult,
    *,
    cache_dir: Path | str | None = None,
) -> None:
    key = obligation.cache_key()
    payload = result.to_dict()
    payload.setdefault("obligation_id", obligation.identifier)
    payload.setdefault("backend", obligation.backend)
    set(key, payload, cache_dir=cache_dir)


def load_result(
    obligation: obligations.ProofObligation, *, cache_dir: Path | str | None = None
) -> obligations.ProofResult | None:
    key = obligation.cache_key()
    data = get(key, cache_dir=cache_dir)
    if data is None:
        return None
    record = dict(data)
    record.setdefault("obligation_id", obligation.identifier)
    record.setdefault("backend", obligation.backend)
    result = obligations.ProofResult.from_mapping(record).mark_cached()
    return result

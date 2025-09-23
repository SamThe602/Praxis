"""Proof obligation utilities for the SMT/Coq/Lean bridge."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

__all__ = [
    "ProofObligation",
    "ProofResult",
    "ProofStatus",
    "ObligationQueue",
    "collect_from_program",
    "queue_obligation",
]


class ProofStatus(str, Enum):
    """Normalised status labels returned by prover backends."""

    PROVED = "proved"
    REFUTED = "refuted"
    UNKNOWN = "unknown"
    ERROR = "error"


def _json_friendly(value: Any) -> Any:
    """Best-effort conversion so payloads hash and serialise predictably."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_friendly(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_friendly(val) for key, val in sorted(value.items())}
    if isinstance(value, set):
        return sorted(_json_friendly(item) for item in value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return repr(value)


def _coerce_assumptions(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, Mapping):
        return tuple(f"{key}={val}" for key, val in sorted(value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value)
    return (str(value),)


@dataclass(slots=True, frozen=True)
class ProofObligation:
    """Structured representation of a single proof goal."""

    identifier: str
    backend: str
    payload: Any
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend = self.backend.lower()
        if backend not in {"smt", "coq", "lean"}:
            raise ValueError(f"Unsupported prover backend '{self.backend}'")
        object.__setattr__(self, "backend", backend)
        if not isinstance(self.identifier, str) or not self.identifier:
            raise ValueError("Obligation identifier must be a non-empty string")
        if not isinstance(self.assumptions, tuple):
            object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if not isinstance(self.metadata, Mapping):
            object.__setattr__(self, "metadata", dict(self.metadata))

    def cache_key(self) -> str:
        """Return a stable cache key based on backend + logical payload."""

        canonical = {
            "backend": self.backend,
            "payload": _json_friendly(self.payload),
            "assumptions": list(self.assumptions),
        }
        raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "backend": self.backend,
            "payload": _json_friendly(self.payload),
            "assumptions": list(self.assumptions),
            "metadata": _json_friendly(dict(self.metadata)),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProofObligation":
        identifier = (
            data.get("id") or data.get("identifier") or data.get("name") or data.get("label")
        )
        payload = data.get("payload") or data.get("goal") or data.get("statement")
        if not identifier:
            payload_stub = _json_friendly(payload)
            seed = json.dumps(payload_stub, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
            identifier = f"obligation_{digest}"
        backend = str(data.get("backend", "smt"))
        assumptions = data.get("assumptions") or ()
        metadata = data.get("metadata") or {}
        return cls(
            identifier=str(identifier),
            backend=backend,
            payload=payload,
            assumptions=_coerce_assumptions(assumptions),
            metadata=metadata if isinstance(metadata, Mapping) else dict(metadata),
        )


@dataclass(slots=True, frozen=True)
class ProofResult:
    """Outcome returned by a prover backend."""

    obligation_id: str
    backend: str
    status: ProofStatus
    counterexample: Mapping[str, Any] | None = None
    certificate: Any | None = None
    diagnostics: str | None = None
    duration_ms: float | None = None
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "obligation_id": self.obligation_id,
            "backend": self.backend,
            "status": self.status.value,
            "counterexample": _json_friendly(
                dict(self.counterexample) if self.counterexample else None
            ),
            "certificate": _json_friendly(self.certificate),
            "diagnostics": self.diagnostics,
            "duration_ms": self.duration_ms,
            "cached": self.cached,
        }

    def with_duration(self, duration_ms: float) -> "ProofResult":
        if self.duration_ms is not None:
            return self
        return replace(self, duration_ms=float(duration_ms))

    def mark_cached(self) -> "ProofResult":
        if self.cached:
            return self
        return replace(self, cached=True)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProofResult":
        status = data.get("status")
        if status is None:
            raise ValueError("Proof result mapping requires a 'status' field")
        obligation_id = data.get("obligation_id") or data.get("id")
        if not obligation_id:
            raise ValueError("Proof result mapping requires an 'obligation_id'")
        return cls(
            obligation_id=str(obligation_id),
            backend=str(data.get("backend", "smt")),
            status=ProofStatus(str(status)),
            counterexample=data.get("counterexample"),
            certificate=data.get("certificate"),
            diagnostics=data.get("diagnostics"),
            duration_ms=data.get("duration_ms"),
            cached=bool(data.get("cached", False)),
        )


class ObligationQueue:
    """FIFO queue that deduplicates obligations by cache key."""

    def __init__(self) -> None:
        self._items: list[ProofObligation] = []
        self._seen: dict[str, ProofObligation] = {}

    def add(self, obligation: ProofObligation | Mapping[str, Any]) -> ProofObligation:
        obj = (
            obligation
            if isinstance(obligation, ProofObligation)
            else ProofObligation.from_mapping(obligation)
        )
        key = obj.cache_key()
        existing = self._seen.get(key)
        if existing is not None:
            return existing
        self._seen[key] = obj
        self._items.append(obj)
        return obj

    def __iter__(self) -> Iterator[ProofObligation]:
        return iter(self._items)

    def __len__(self) -> int:  # pragma: no cover - convenience helper
        return len(self._items)

    def drain(self) -> tuple[ProofObligation, ...]:
        payload = tuple(self._items)
        self._items.clear()
        self._seen.clear()
        return payload


_DEFAULT_QUEUE = ObligationQueue()


def queue_obligation(obligation: Mapping[str, Any] | ProofObligation) -> ProofObligation:
    """Insert ``obligation`` into the process-wide queue."""

    return _DEFAULT_QUEUE.add(obligation)


def collect_from_program(program: Mapping[str, Any]) -> tuple[ProofObligation, ...]:
    """Extract proof obligations from a program dictionary."""

    raw_items: Iterable[MutableMapping[str, Any]] | None = None
    if isinstance(program, Mapping):
        candidate = program.get("proof_obligations")
        if isinstance(candidate, Iterable):
            raw_items = candidate
    if not raw_items:
        return ()
    queue = ObligationQueue()
    for entry in raw_items:
        queue.add(entry)
    return tuple(queue)

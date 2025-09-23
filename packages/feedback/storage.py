"""In-memory storage backend for human feedback records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Mapping

__all__ = ["FeedbackRecord", "add", "all_feedback", "clear", "extend"]


@dataclass(slots=True)
class FeedbackRecord:
    """Normalised representation of a human feedback datapoint."""

    solution_id: str
    spec_id: str
    rating: str
    notes: str | None = None
    reviewer: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FeedbackRecord":
        if "solution_id" not in data or "spec_id" not in data:
            raise KeyError("feedback records require solution_id and spec_id")
        rating = str(data.get("rating", "unrated"))
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"solution_id", "spec_id", "rating", "notes", "reviewer", "timestamp"}
        }
        timestamp = data.get("timestamp")
        if timestamp is None:
            timestamp_str = datetime.now(timezone.utc).isoformat()
        else:
            timestamp_str = str(timestamp)
        return cls(
            solution_id=str(data["solution_id"]),
            spec_id=str(data["spec_id"]),
            rating=rating,
            notes=data.get("notes"),
            reviewer=data.get("reviewer"),
            timestamp=timestamp_str,
            metadata=dict(metadata),
        )


_LOCK = RLock()
_STORAGE: list[FeedbackRecord] = []


def add(record: FeedbackRecord | Mapping[str, Any]) -> FeedbackRecord:
    if not isinstance(record, FeedbackRecord):
        record = FeedbackRecord.from_mapping(record)
    with _LOCK:
        _STORAGE.append(record)
    return record


def extend(records: Iterable[FeedbackRecord | Mapping[str, Any]]) -> list[FeedbackRecord]:
    added: list[FeedbackRecord] = []
    for record in records:
        added.append(add(record))
    return added


def all_feedback() -> list[FeedbackRecord]:
    with _LOCK:
        return list(_STORAGE)


def clear() -> None:
    with _LOCK:
        _STORAGE.clear()

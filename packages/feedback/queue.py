"""Priority queue for human feedback review tasks."""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from itertools import count
from typing import Any, Iterable, List, Mapping

__all__ = [
    "FeedbackItem",
    "clear",
    "dequeue",
    "enqueue",
    "enqueue_many",
    "peek",
    "pending",
    "size",
]


@dataclass(slots=True)
class FeedbackItem:
    """Item representing a solution awaiting human feedback."""

    solution_id: str
    spec_id: str
    prompt: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "FeedbackItem":
        if "solution_id" not in data or "spec_id" not in data:
            raise KeyError("feedback items require solution_id and spec_id")
        prompt = data.get("prompt") or data.get("natural_prompt")
        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"solution_id", "spec_id", "prompt", "natural_prompt"}
        }
        return cls(
            solution_id=str(data["solution_id"]),
            spec_id=str(data["spec_id"]),
            prompt=str(prompt) if prompt is not None else None,
            metadata=dict(metadata),
        )


_QUEUE: List[tuple[int, int, FeedbackItem]] = []
_LOCK = threading.RLock()
_COUNTER = count()


def enqueue(item: FeedbackItem, *, priority: int = 0) -> None:
    if not isinstance(item, FeedbackItem):
        raise TypeError("item must be a FeedbackItem instance")
    with _LOCK:
        heapq.heappush(_QUEUE, (-int(priority), next(_COUNTER), item))


def enqueue_many(items: Iterable[FeedbackItem], *, priority: int = 0) -> None:
    for item in items:
        enqueue(item, priority=priority)


def dequeue() -> FeedbackItem | None:
    with _LOCK:
        if not _QUEUE:
            return None
        _, _, item = heapq.heappop(_QUEUE)
        return item


def peek() -> FeedbackItem | None:
    with _LOCK:
        if not _QUEUE:
            return None
        return _QUEUE[0][2]


def pending() -> list[FeedbackItem]:
    with _LOCK:
        return [entry[2] for entry in sorted(_QUEUE)]


def size() -> int:
    with _LOCK:
        return len(_QUEUE)


def clear() -> None:
    with _LOCK:
        _QUEUE.clear()

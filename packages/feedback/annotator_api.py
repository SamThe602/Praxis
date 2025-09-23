"""Convenience API used by the human feedback tooling."""

from __future__ import annotations

from typing import Mapping

from packages.feedback import queue, storage

__all__ = [
    "queue_for_review",
    "request_next_item",
    "submit_feedback",
]


def queue_for_review(record: Mapping[str, object], *, priority: int = 0) -> queue.FeedbackItem:
    """Normalise ``record`` and push it onto the queue."""

    item = queue.FeedbackItem.from_mapping(record)
    queue.enqueue(item, priority=priority)
    return item


def request_next_item() -> queue.FeedbackItem | None:
    """Return the highest-priority item awaiting review."""

    return queue.dequeue()


def submit_feedback(
    record: Mapping[str, object],
    *,
    requeue: bool = False,
    priority: int = 0,
) -> storage.FeedbackRecord:
    """Persist a human rating and optionally push the item back to the queue."""

    stored = storage.add(record)
    if requeue:
        queue.enqueue(queue.FeedbackItem.from_mapping(record), priority=priority)
    return stored

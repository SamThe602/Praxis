"""Sampling helpers used to surface items for human review or training."""

from __future__ import annotations

from packages.feedback import queue

__all__ = ["sample_for_review"]


def sample_for_review(max_items: int = 1, *, consume: bool = False) -> list[queue.FeedbackItem]:
    """Return up to ``max_items`` highest-priority queue entries.

    When ``consume`` is ``True`` the items are removed from the underlying
    queue, otherwise the function simply peeks at the pending work without
    mutating the queue.  The helper keeps ordering deterministic by relying on
    the queue's internal priority ordering.
    """

    if max_items <= 0:
        return []

    if consume:
        collected: list[queue.FeedbackItem] = []
        for _ in range(max_items):
            item = queue.dequeue()
            if item is None:
                break
            collected.append(item)
        return collected

    pending = queue.pending()
    return pending[:max_items]

"""Frontier management with beam filtering and deduplication."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, Hashable, Optional


@dataclass(slots=True)
class FrontierItem:
    """Container returned by :class:`Frontier.pop`."""

    state: Any
    score: float
    depth: int
    metadata: dict[str, Any] = field(default_factory=dict)
    key: Hashable | None = None
    active: bool = True


class Frontier:
    """Queue-like container used by the BFS synthesiser.

    The frontier behaves like a FIFO queue but supports optional beam limiting
    and state deduplication.  When ``beam_width`` is provided the frontier keeps
    at most ``beam_width`` active items per depth level, replacing the worst
    scoring entry when a better candidate arrives.
    """

    def __init__(
        self,
        *,
        beam_width: int | None = None,
        dedupe: bool = True,
        key_fn: Callable[[Any], Hashable] | None = None,
    ) -> None:
        self._queue: Deque[FrontierItem] = deque()
        self._beam_width = beam_width
        self._key_fn = key_fn or (lambda state: state)
        self._dedupe = dedupe
        self._seen: set[Hashable] = set()
        self._per_depth: Dict[int, list[FrontierItem]] = {}
        self._active_count = 0

    def __len__(self) -> int:
        return self._active_count

    def __bool__(self) -> bool:
        return self._active_count > 0

    def push(
        self,
        state: Any,
        *,
        score: float,
        depth: int,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Insert ``state`` into the frontier.

        Returns ``True`` when the state is accepted.  Items rejected because of
        deduplication or beam filtering yield ``False`` so callers can avoid
        unnecessary bookkeeping.
        """

        key: Hashable | None = None
        if self._dedupe:
            key = self._key_fn(state)
            if key in self._seen:
                return False

        item = FrontierItem(
            state=state, score=score, depth=depth, metadata=dict(metadata or {}), key=key
        )

        if self._beam_width is not None:
            bucket = self._per_depth.setdefault(depth, [])
            if len(bucket) >= self._beam_width:
                worst_index = min(range(len(bucket)), key=lambda idx: bucket[idx].score)
                worst_item = bucket[worst_index]
                if worst_item.score >= item.score:
                    return False
                bucket.pop(worst_index)
                if worst_item.active:
                    worst_item.active = False
                    if self._dedupe and worst_item.key is not None:
                        self._seen.discard(worst_item.key)
                    self._active_count -= 1
            bucket.append(item)
        else:
            bucket = self._per_depth.setdefault(depth, [])
            bucket.append(item)

        self._queue.append(item)
        if self._dedupe and key is not None:
            self._seen.add(key)
        self._active_count += 1
        return True

    def pop(self) -> FrontierItem:
        """Remove and return the next active item."""

        while self._queue:
            item = self._queue.popleft()
            if not item.active:
                continue
            if self._dedupe and item.key is not None:
                self._seen.discard(item.key)
            bucket = self._per_depth.get(item.depth)
            if bucket:
                try:
                    bucket.remove(item)
                except ValueError:
                    pass
                if not bucket:
                    self._per_depth.pop(item.depth, None)
            self._active_count -= 1
            return item
        raise IndexError("pop from an empty frontier")

    def clear(self) -> None:
        """Remove all pending items."""

        self._queue.clear()
        self._seen.clear()
        self._per_depth.clear()
        self._active_count = 0


__all__ = ["Frontier", "FrontierItem"]

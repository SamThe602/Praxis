"""Canonical synthesiser state representation used by the BFS baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

HashableItem = Any


def _to_hashable(value: Any) -> HashableItem:
    """Recursively convert ``value`` into a hashable structure."""

    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((key, _to_hashable(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_to_hashable(item) for item in value)
    if isinstance(value, bytes):
        return value
    return repr(value)


@dataclass(slots=True)
class SynthState:
    """Immutable-ish container describing a partial synthesis candidate.

    ``steps`` records the high-level actions applied so far (statements, guards,
    snippet insertions, ...). ``covered_requirements`` mirrors the set of spec
    obligations that this candidate already satisfied, while
    ``pending_requirements`` keeps the to-do list in insertion order.  ``metadata``
    stores lightweight scalar metrics (latency estimates, reuse counters) that
    guide heuristic scoring, and ``analysis`` caches heavier results such as
    static analysis outcomes so downstream passes can avoid recomputation.
    """

    steps: tuple[str, ...] = ()
    covered_requirements: frozenset[str] = field(default_factory=frozenset)
    pending_requirements: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.steps, tuple):
            self.steps = tuple(self.steps)
        if not isinstance(self.pending_requirements, tuple):
            self.pending_requirements = tuple(self.pending_requirements)
        if not isinstance(self.covered_requirements, frozenset):
            self.covered_requirements = frozenset(self.covered_requirements)
        if not isinstance(self.metadata, dict):
            self.metadata = dict(self.metadata)
        if not isinstance(self.analysis, dict):
            self.analysis = dict(self.analysis)

    def is_goal(self) -> bool:
        return not self.pending_requirements

    def next_requirement(self) -> str | None:
        return self.pending_requirements[0] if self.pending_requirements else None

    def coverage_ratio(self, total_requirements: int) -> float:
        if total_requirements <= 0:
            return 1.0
        return min(1.0, len(self.covered_requirements) / total_requirements)

    def fingerprint(self) -> tuple[Any, ...]:
        """Stable fingerprint used for frontier deduplication."""

        metadata_fingerprint = tuple(sorted((k, _to_hashable(v)) for k, v in self.metadata.items()))
        return (
            self.steps,
            tuple(sorted(self.covered_requirements)),
            self.pending_requirements,
            metadata_fingerprint,
        )

    def with_step(
        self,
        step: str,
        *,
        covers: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SynthState":
        """Return a new state with ``step`` appended and bookkeeping updated."""

        new_steps = self.steps + (step,)
        covered = set(self.covered_requirements)
        if covers:
            covered.update(covers)
        new_metadata = dict(self.metadata)
        if metadata:
            for key, value in metadata.items():
                if value is None:
                    continue
                new_metadata[key] = value
        new_covered = frozenset(covered)
        new_pending = tuple(req for req in self.pending_requirements if req not in new_covered)
        return SynthState(
            steps=new_steps,
            covered_requirements=new_covered,
            pending_requirements=new_pending,
            metadata=new_metadata,
            analysis=dict(self.analysis),
        )

    def annotate(self, **metadata: Any) -> "SynthState":
        """Return a new state with ``metadata`` merged into ``self.metadata``."""

        if not metadata:
            return self
        merged = dict(self.metadata)
        merged.update({k: v for k, v in metadata.items() if v is not None})
        return SynthState(
            steps=self.steps,
            covered_requirements=self.covered_requirements,
            pending_requirements=self.pending_requirements,
            metadata=merged,
            analysis=dict(self.analysis),
        )


__all__ = ["SynthState"]

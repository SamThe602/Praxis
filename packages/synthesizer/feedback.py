"""Helpers for folding verifier feedback into synthesiser state metadata."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .state import SynthState

__all__ = ["incorporate"]


def incorporate(state: SynthState | Any, feedback: Any) -> SynthState | Any:
    """Return ``state`` annotated with verifier feedback.

    ``feedback`` accepts a single mapping or an iterable of mappings.  Each
    entry is appended to the cumulative ``metadata['feedback']`` list and any
    counterexamples are mirrored under ``analysis['counterexamples']``.  Entries
    whose ``severity`` is ``error`` or whose ``kind`` ends with ``_failure``
    mark the state as invalid so search can prioritise alternative branches.
    """

    if feedback is None or not isinstance(state, SynthState):
        return state

    items = _normalise_feedback(feedback)
    if not items:
        return state

    metadata = dict(state.metadata)
    analysis = dict(state.analysis)

    stored_feedback = list(metadata.get("feedback", ()))
    counterexamples = list(analysis.get("counterexamples", ()))
    invalid = metadata.get("invalid", False)

    for entry in items:
        stored_feedback.append(entry)
        counterexample = entry.get("counterexample") if isinstance(entry, Mapping) else None
        if counterexample is not None:
            counterexamples.append(counterexample)
        kind = entry.get("kind") if isinstance(entry, Mapping) else None
        severity = entry.get("severity") if isinstance(entry, Mapping) else None
        if severity == "error" or (isinstance(kind, str) and kind.endswith("_failure")):
            invalid = True

    metadata["feedback"] = tuple(stored_feedback)
    if invalid:
        metadata["invalid"] = True
    if counterexamples:
        analysis["counterexamples"] = tuple(counterexamples)

    return SynthState(
        steps=state.steps,
        covered_requirements=state.covered_requirements,
        pending_requirements=state.pending_requirements,
        metadata=metadata,
        analysis=analysis,
    )


def _normalise_feedback(feedback: Any) -> list[Mapping[str, Any]]:
    if isinstance(feedback, Mapping):
        return [feedback]
    if isinstance(feedback, Iterable) and not isinstance(feedback, (str, bytes)):
        return [item for item in feedback if isinstance(item, Mapping)]
    return []

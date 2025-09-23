"""Confidence scoring for translator outputs.

We expose a small heuristic estimator so the orchestrator can decide whether to
accept the translation automatically. The default acceptance threshold lives in
``DEFAULT_THRESHOLD``; callers may override it but the comment here documents
why we default to a conservative value.
"""

from __future__ import annotations

from typing import Mapping

DEFAULT_THRESHOLD = 0.72  # Conservative default; see text_interface for gating rationale.


def score(candidate: Mapping[str, object], prompt: str | None = None) -> float:
    """Return a heuristic confidence estimate in ``[0.0, 1.0]``.

    The estimator rewards:
    - Known task identifiers recognised by the glossary-backed decoder.
    - Presence of concrete examples and metadata fields.
    - Agreement between prompt tokens and the inferred task id.
    """

    if not isinstance(candidate, Mapping):
        return 0.0

    prompt_text = (prompt or str(candidate.get("natural_prompt", ""))).lower()
    task_id = str(candidate.get("id", "")).lower()
    examples = candidate.get("examples", [])
    metadata = candidate.get("metadata", {})
    inputs = candidate.get("inputs", [])
    outputs = candidate.get("outputs", {})

    score_value = 0.2  # Base prior.

    if task_id in {"array_sort", "array_reverse", "graph_path", "histogram", "matrix_mult"}:
        score_value += 0.35
    elif task_id:
        score_value += 0.25

    if prompt_text and task_id and task_id.split("_")[0] in prompt_text:
        score_value += 0.15

    if isinstance(examples, list) and examples:
        score_value += 0.1

    if isinstance(metadata, Mapping) and metadata.get("difficulty"):
        score_value += 0.05

    if isinstance(inputs, list) and inputs:
        score_value += 0.05

    if isinstance(outputs, Mapping) and outputs.get("type"):
        score_value += 0.05

    return max(0.0, min(1.0, round(score_value, 4)))

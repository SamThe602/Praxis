"""Expansion operators for the synthesiser BFS baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, List, Mapping, Sequence

from .state import SynthState

if TYPE_CHECKING:  # pragma: no cover - type checking helper
    from packages.orchestrator.spec_loader import StructuredSpec


def expand(state: SynthState, spec: "StructuredSpec" | None = None) -> Iterable[SynthState]:
    """Yield child states obtained by applying baseline expansion operators."""

    if not isinstance(state, SynthState):
        raise TypeError("expand expects SynthState inputs")

    if not state.pending_requirements:
        return ()

    requirement = state.pending_requirements[0]
    candidates: List[SynthState] = []

    candidates.append(_add_statement(state, requirement))
    candidates.append(_refine_expression(state, requirement))

    if requirement.startswith(("constraint:", "contract:")):
        candidates.append(_insert_contract_guard(state, requirement))
    if spec is not None:
        candidates.extend(_inline_snippets(state, requirement, spec))
        candidates.extend(_specialise_input(state, requirement, spec))
    candidates.append(_apply_proof_hint(state, requirement))

    return _unique_candidates(candidates)


def _add_statement(state: SynthState, requirement: str) -> SynthState:
    metadata = _next_metadata(state, operator="AddStatement", latency_increment=1.0)
    step = f"AddStatement::{requirement}"
    return state.with_step(step, covers=[requirement], metadata=metadata)


def _refine_expression(state: SynthState, requirement: str) -> SynthState:
    metadata = _next_metadata(state, operator="RefineExpression", latency_increment=0.8)
    step = f"RefineExpression::{requirement}"
    return state.with_step(step, covers=[requirement], metadata=metadata)


def _insert_contract_guard(state: SynthState, requirement: str) -> SynthState:
    metadata = _next_metadata(state, operator="InsertContractGuard", latency_increment=0.5)
    metadata["guards_added"] = int(state.metadata.get("guards_added", 0)) + 1
    step = f"InsertContractGuard::{requirement}"
    return state.with_step(step, covers=[requirement], metadata=metadata)


def _inline_snippets(
    state: SynthState, requirement: str, spec: "StructuredSpec"
) -> Iterable[SynthState]:
    results: List[SynthState] = []
    reuse_pool: Sequence[str] = getattr(spec, "reuse_pool", ())
    if not reuse_pool:
        return ()

    lookup = _retrieval_lookup(spec)
    base_hits = float(state.metadata.get("reuse_hits", 0.0))

    for rank, snippet in enumerate(reuse_pool):
        snippet_text = str(snippet)
        norm = _normalise_snippet(snippet_text)
        info = lookup.get(norm, {})
        score = max(0.0, float(info.get("score", 0.0)))
        source = info.get("spec_id")
        metadata = _next_metadata(state, operator="InlineSnippet", latency_increment=0.3)
        metadata["reuse_hits"] = base_hits + 1.0 + score
        metadata["snippet_reuse"] = score
        metadata["reuse_rank"] = rank
        if source:
            metadata["reuse_source"] = source
        step = f"InlineSnippet::{snippet_text}->{requirement}"
        results.append(state.with_step(step, covers=[requirement], metadata=metadata))
    return results


def _specialise_input(
    state: SynthState, requirement: str, spec: "StructuredSpec"
) -> Iterable[SynthState]:
    if not requirement.startswith("use_input:"):
        return ()
    input_name = requirement.split(":", 1)[1]
    input_type = getattr(spec, "input_types", {}).get(input_name)
    if not input_type:
        return ()
    operator = (
        "SpecializeLoop"
        if "vector" in input_type or "matrix" in input_type
        else "ReparameterizeLambda"
    )
    metadata = _next_metadata(state, operator=operator, latency_increment=1.2)
    step = f"{operator}::{input_name}:{input_type}"
    return (state.with_step(step, covers=[requirement], metadata=metadata),)


def _apply_proof_hint(state: SynthState, requirement: str) -> SynthState:
    metadata = _next_metadata(state, operator="ApplyProofHint", latency_increment=0.2)
    metadata["proof_hints"] = int(state.metadata.get("proof_hints", 0)) + 1
    step = f"ApplyProofHint::{requirement}"
    return state.with_step(step, covers=[requirement], metadata=metadata)


def _next_metadata(
    state: SynthState,
    *,
    operator: str,
    latency_increment: float,
) -> dict[str, Any]:
    estimated_cost = float(state.metadata.get("estimated_cost", 0.0)) + float(latency_increment)
    return {
        "last_operator": operator,
        "estimated_cost": estimated_cost,
        "latency_estimate": estimated_cost,
    }


def _unique_candidates(candidates: Sequence[SynthState]) -> Iterable[SynthState]:
    seen = set()
    unique: List[SynthState] = []
    for candidate in candidates:
        fingerprint = candidate.fingerprint()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(candidate)
    return tuple(unique)


def _retrieval_lookup(spec: "StructuredSpec") -> dict[str, dict[str, Any]]:
    metadata = getattr(spec, "metadata", {})
    if not isinstance(metadata, Mapping):
        return {}
    retrieval_info = metadata.get("retrieval")
    if not isinstance(retrieval_info, Mapping):
        return {}
    results = retrieval_info.get("results", ())
    lookup: dict[str, dict[str, Any]] = {}
    if not isinstance(results, Sequence):
        return lookup
    for item in results:
        if not isinstance(item, Mapping):
            continue
        snippet = item.get("snippet")
        if not isinstance(snippet, str):
            continue
        key = _normalise_snippet(snippet)
        lookup[key] = {
            "score": float(item.get("score", 0.0)),
            "spec_id": str(item.get("spec_id", "")) if item.get("spec_id") else "",
        }
    return lookup


def _normalise_snippet(snippet: str) -> str:
    return " ".join(str(snippet).strip().split())


__all__ = ["expand"]

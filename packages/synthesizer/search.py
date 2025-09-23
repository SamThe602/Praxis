"""Breadth-first synthesiser search with optional beam pruning."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import ModuleType
from typing import Any, List

from packages.telemetry import hooks as telemetry_hooks
from packages.telemetry import metrics as telemetry_metrics
from packages.utils.concurrency import run_parallel

from . import expansions, heuristics, scoring
from .frontier import Frontier
from .state import SynthState

pruning: ModuleType | None
try:
    from . import pruning as pruning_module
except ImportError:  # pragma: no cover - defensive fallback
    pruning = None
else:
    pruning = pruning_module


def _frontier_key(state: Any) -> Any:
    return state.fingerprint() if isinstance(state, SynthState) else state


@dataclass(slots=True)
class SearchConfig:
    beam_width: int = 32
    max_depth: int = 12
    max_nodes: int = 1024
    dedupe: bool = True
    partial_evaluation: bool = True
    record_trace: bool = False
    parallel_evaluation: bool = False


@dataclass(slots=True)
class SearchResult:
    solution: SynthState | None
    best: SynthState | None
    visited: int
    expanded: int
    reason: str
    trace: tuple[SynthState, ...] = ()
    best_score: float = 0.0


def search(
    initial_state: SynthState,
    *,
    spec: Any | None = None,
    config: SearchConfig | None = None,
    heuristic_config: heuristics.HeuristicConfig | None = None,
    score_weights: scoring.ScoreWeights | None = None,
) -> SearchResult:
    """Run BFS/beam search starting from ``initial_state``.

    Parameters
    ----------
    initial_state:
        Root synthesis state for the exploration.
    spec:
        Optional structured specification passed down to expansions and
        heuristics.
    config:
        Search control knobs (beam width, depth/visit limits, ...).
    heuristic_config:
        Weights controlling the heuristic scoring function.
    score_weights:
        Weights used by the final composite scoring function.
    """

    config = config or SearchConfig()
    heuristic_config = heuristic_config or heuristics.HeuristicConfig()
    score_weights = score_weights or scoring.ScoreWeights()

    total_requirements = len(getattr(spec, "goal_tokens", ())) or len(
        initial_state.covered_requirements
    ) + len(initial_state.pending_requirements)
    frontier = Frontier(
        beam_width=config.beam_width,
        dedupe=config.dedupe,
        key_fn=_frontier_key,
    )

    root_score = scoring.composite_score(
        initial_state,
        spec=spec,
        heuristic_config=heuristic_config,
        weights=score_weights,
        total_requirements=total_requirements,
    )
    frontier.push(initial_state, score=root_score.total, depth=0, metadata={"score": root_score})

    best_state = initial_state
    best_score = root_score.total
    trace: List[SynthState] = []

    visited = 0
    expanded = 0

    while frontier and visited < config.max_nodes:
        item = frontier.pop()
        state = item.state
        score_info = item.metadata.get("score") if isinstance(item.metadata, dict) else None
        if score_info is None:
            score_info = scoring.composite_score(
                state,
                spec=spec,
                heuristic_config=heuristic_config,
                weights=score_weights,
                total_requirements=total_requirements,
            )
        visited += 1
        if config.record_trace:
            trace.append(state)

        if item.depth > config.max_depth:
            continue

        if state.is_goal():
            # Return immediately on first goal to preserve BFS semantics.
            best_state = state
            best_score = score_info.total
            reason = "goal_found"
            break

        if config.partial_evaluation and _fails_fast(state):
            continue
        if pruning is not None and pruning.should_prune(state):
            continue

        children_iter = tuple(expansions.expand(state, spec=spec))
        expanded += 1
        child_depth = item.depth + 1
        valid_children: list[SynthState] = []
        for child in children_iter:
            if not isinstance(child, SynthState):
                raise TypeError("expansions.expand must yield SynthState instances")
            if child_depth > config.max_depth:
                continue
            valid_children.append(child)
        if not valid_children:
            continue

        if config.parallel_evaluation and len(valid_children) > 1:

            def _score_child(child: SynthState) -> tuple[SynthState, scoring.ScoreBreakdown]:
                return (
                    child,
                    scoring.composite_score(
                        child,
                        spec=spec,
                        heuristic_config=heuristic_config,
                        weights=score_weights,
                        total_requirements=total_requirements,
                    ),
                )

            tasks = [functools.partial(_score_child, child) for child in valid_children]
            scored_children = run_parallel(
                tasks,
                mode="thread",
                task_kind="io",
            )
        else:
            scored_children = [
                (
                    child,
                    scoring.composite_score(
                        child,
                        spec=spec,
                        heuristic_config=heuristic_config,
                        weights=score_weights,
                        total_requirements=total_requirements,
                    ),
                )
                for child in valid_children
            ]

        for child, child_score in scored_children:
            accepted = frontier.push(
                child,
                score=child_score.total,
                depth=child_depth,
                metadata={"score": child_score},
            )
            if not accepted:
                continue
            if child_score.total > best_score:
                best_state = child
                best_score = child_score.total
    else:
        reason = "frontier_exhausted" if not frontier else "max_nodes"

    if frontier:
        # Loop broke via ``break`` (goal) or via max_nodes.
        reason = reason if "reason" in locals() else "max_nodes"
    result = SearchResult(
        solution=best_state if best_state.is_goal() else None,
        best=best_state,
        visited=visited,
        expanded=expanded,
        reason=reason,
        trace=tuple(trace) if config.record_trace else (),
        best_score=best_score,
    )
    _record_search_telemetry(result, spec)
    return result


def _fails_fast(state: SynthState) -> bool:
    """Cheap rejection hooks before expensive analysis."""

    marker = state.metadata.get("invalid")
    if marker:
        return True
    marker = state.metadata.get("partial_invalid")
    if marker:
        return True
    return False


__all__ = ["SearchConfig", "SearchResult", "search"]


def _record_search_telemetry(result: SearchResult, spec: Any | None) -> None:
    try:
        telemetry_metrics.emit(
            "praxis.synth.search_nodes",
            result.visited,
            tags={
                "reason": result.reason,
                "solution": "found" if result.solution is not None else "pending",
            },
            extra={"expanded": result.expanded},
        )
        telemetry_hooks.dispatch(
            telemetry_hooks.SYNTH_SEARCH_COMPLETED,
            {
                "visited": result.visited,
                "expanded": result.expanded,
                "best_score": result.best_score,
                "solution_found": result.solution is not None,
                "reason": result.reason,
                "spec_id": getattr(spec, "identifier", None),
            },
        )
    except Exception:  # pragma: no cover - metrics should not break search
        telemetry_hooks.dispatch(
            "synthesizer.telemetry_error",
            {"reason": result.reason, "visited": result.visited},
        )

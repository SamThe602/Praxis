"""Heuristic scoring utilities for the synthesiser BFS baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import SynthState


@dataclass(slots=True)
class HeuristicConfig:
    coverage_weight: float = 1.0
    depth_penalty: float = 0.05
    guard_penalty: float = 0.15
    latency_weight: float = 0.02
    pending_penalty: float = 0.01


@dataclass(slots=True)
class HeuristicFeatures:
    coverage: float
    depth: int
    guard_debt: int
    latency_estimate: float
    pending: int


def compute_features(
    state: SynthState,
    *,
    spec: Any | None = None,
    total_requirements: int | None = None,
) -> HeuristicFeatures:
    """Extract heuristic features from ``state``.

    The baseline uses lightweight signals directly available from the search
    state: coverage vs pending requirements, a cheap latency proxy derived from
    the current step count, and whether contract/guard obligations remain.
    """

    if not isinstance(state, SynthState):
        raise TypeError("heuristic scoring expects SynthState inputs")

    total_goals = total_requirements
    if total_goals is None:
        total_goals = len(state.covered_requirements) + len(state.pending_requirements)
    coverage = state.coverage_ratio(total_goals)
    depth = len(state.steps)
    guard_debt = sum(
        1
        for requirement in state.pending_requirements
        if requirement.startswith(("contract:", "constraint:"))
    )
    latency_estimate = state.metadata.get("latency_estimate")
    if latency_estimate is None:
        base_cost = state.metadata.get("estimated_cost", depth)
        latency_estimate = float(base_cost)
    latency_estimate = float(latency_estimate)
    pending = len(state.pending_requirements)
    return HeuristicFeatures(
        coverage=coverage,
        depth=depth,
        guard_debt=guard_debt,
        latency_estimate=latency_estimate,
        pending=pending,
    )


def score(
    state: SynthState,
    *,
    spec: Any | None = None,
    config: HeuristicConfig | None = None,
    total_requirements: int | None = None,
) -> tuple[float, HeuristicFeatures]:
    """Return the heuristic score and contributing features."""

    config = config or HeuristicConfig()
    features = compute_features(state, spec=spec, total_requirements=total_requirements)
    heuristic = (
        config.coverage_weight * features.coverage
        - config.depth_penalty * features.depth
        - config.guard_penalty * features.guard_debt
        - config.latency_weight * features.latency_estimate
        - config.pending_penalty * features.pending
    )
    return heuristic, features


__all__ = ["HeuristicConfig", "HeuristicFeatures", "compute_features", "score"]

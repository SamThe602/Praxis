"""Composite scoring that blends heuristic, policy, and latency signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import heuristics
from .state import SynthState


@dataclass(slots=True)
class ScoreWeights:
    policy: float = 0.0
    heuristic: float = 1.0
    latency_penalty: float = 0.05
    reuse_bonus: float = 0.1


@dataclass(slots=True)
class ScoreBreakdown:
    total: float
    policy: float
    heuristic: float
    latency_penalty: float
    reuse_bonus: float
    features: heuristics.HeuristicFeatures


def composite_score(
    state: SynthState,
    *,
    spec: Any | None = None,
    heuristic_config: heuristics.HeuristicConfig | None = None,
    weights: ScoreWeights | None = None,
    total_requirements: int | None = None,
) -> ScoreBreakdown:
    """Return the weighted score for ``state`` and its breakdown."""

    if not isinstance(state, SynthState):
        raise TypeError("composite_score expects SynthState inputs")

    heuristic_config = heuristic_config or heuristics.HeuristicConfig()
    weights = weights or ScoreWeights()

    heuristic_score, features = heuristics.score(
        state,
        spec=spec,
        config=heuristic_config,
        total_requirements=total_requirements,
    )
    policy_score = _policy_score(state)
    latency_penalty = _latency_penalty(state, spec, features)
    reuse_bonus = _reuse_bonus(state)

    total = (
        weights.policy * policy_score
        + weights.heuristic * heuristic_score
        - weights.latency_penalty * latency_penalty
        + weights.reuse_bonus * reuse_bonus
    )
    return ScoreBreakdown(
        total=total,
        policy=policy_score,
        heuristic=heuristic_score,
        latency_penalty=latency_penalty,
        reuse_bonus=reuse_bonus,
        features=features,
    )


def _policy_score(state: SynthState) -> float:
    if "policy_score" in state.metadata:
        return float(state.metadata["policy_score"])
    if "policy_bias" in state.metadata:
        return float(state.metadata["policy_bias"])
    return 0.0


def _latency_penalty(
    state: SynthState,
    spec: Any | None,
    features: heuristics.HeuristicFeatures,
) -> float:
    estimate = state.metadata.get("latency_estimate", features.latency_estimate)
    target = None
    if spec is not None:
        target = getattr(spec, "latency_target_ms", None)
    if target is None:
        target = state.metadata.get("latency_target")
    if target is None:
        return float(estimate)
    return max(0.0, float(estimate) - float(target))


def _reuse_bonus(state: SynthState) -> float:
    reuse_hits = state.metadata.get("reuse_hits")
    if reuse_hits is None:
        reuse_hits = state.metadata.get("snippet_reuse", 0)
    return float(reuse_hits or 0)


__all__ = ["ScoreBreakdown", "ScoreWeights", "composite_score"]

"""High-level synthesiser interface used by the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from packages.orchestrator.spec_loader import StructuredSpec, load_spec

from . import heuristics, retrieval, scoring, search
from .state import SynthState


@dataclass(slots=True)
class SynthesizerSettings:
    search: search.SearchConfig
    heuristics: heuristics.HeuristicConfig
    scoring: scoring.ScoreWeights


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "synth" / "default.yaml"


class Synthesizer:
    """Convenience wrapper combining config loading and BFS search."""

    def __init__(
        self,
        *,
        settings: SynthesizerSettings | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        if settings is None:
            config_data = _load_config(config_path or DEFAULT_CONFIG_PATH)
            settings = build_settings(config_data)
        self.settings = settings

    def initialise_state(self, spec: StructuredSpec) -> SynthState:
        metadata = {
            "spec_id": spec.identifier,
            "latency_target": spec.latency_target_ms,
            "estimated_cost": 0.0,
            "latency_estimate": 0.0,
        }
        return SynthState(
            steps=(),
            covered_requirements=frozenset(),
            pending_requirements=spec.goal_tokens,
            metadata=metadata,
        )

    def run(self, spec: StructuredSpec) -> search.SearchResult:
        spec_with_reuse = retrieval.apply_retrieval(spec)
        initial_state = self.initialise_state(spec_with_reuse)
        return search.search(
            initial_state,
            spec=spec_with_reuse,
            config=self.settings.search,
            heuristic_config=self.settings.heuristics,
            score_weights=self.settings.scoring,
        )


def synthesise_from_path(
    path: str | Path, *, config_path: str | Path | None = None
) -> search.SearchResult:
    """Load a structured spec from ``path`` and run the synthesiser."""

    spec = load_spec(path)
    engine = Synthesizer(config_path=config_path)
    return engine.run(spec)


def _load_config(path: str | Path) -> Mapping[str, Any]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"synthesiser config not found: {path_obj}")
    data = yaml.safe_load(path_obj.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("synth config must be a mapping")
    return data


def build_settings(data: Mapping[str, Any]) -> SynthesizerSettings:
    search_cfg = data.get("search", {})
    heuristic_cfg = data.get("heuristics", {})
    scoring_cfg = data.get("scoring", {})

    search_settings = search.SearchConfig(
        beam_width=int(search_cfg.get("beam_width", 32)),
        max_depth=int(search_cfg.get("max_depth", 12)),
        max_nodes=int(search_cfg.get("max_nodes", 1024)),
        dedupe=bool(search_cfg.get("dedupe", True)),
        partial_evaluation=bool(search_cfg.get("partial_evaluation", True)),
        record_trace=bool(search_cfg.get("record_trace", False)),
        parallel_evaluation=bool(search_cfg.get("parallel_evaluation", False)),
    )

    heuristic_settings = heuristics.HeuristicConfig(
        coverage_weight=float(heuristic_cfg.get("coverage_weight", 1.0)),
        depth_penalty=float(heuristic_cfg.get("depth_penalty", 0.05)),
        guard_penalty=float(heuristic_cfg.get("guard_penalty", 0.15)),
        latency_weight=float(heuristic_cfg.get("latency_weight", 0.02)),
        pending_penalty=float(heuristic_cfg.get("pending_penalty", 0.01)),
    )

    scoring_settings = scoring.ScoreWeights(
        policy=float(scoring_cfg.get("policy", 0.0)),
        heuristic=float(scoring_cfg.get("heuristic", 1.0)),
        latency_penalty=float(scoring_cfg.get("latency_penalty", 0.05)),
        reuse_bonus=float(scoring_cfg.get("reuse_bonus", 0.1)),
    )

    return SynthesizerSettings(
        search=search_settings,
        heuristics=heuristic_settings,
        scoring=scoring_settings,
    )


__all__ = [
    "Synthesizer",
    "SynthesizerSettings",
    "build_settings",
    "DEFAULT_CONFIG_PATH",
    "synthesise_from_path",
]

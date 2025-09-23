"""Curriculum weighting heuristics used in the lightweight tests.

The production system uses a considerably more sophisticated controller that
interfaces with telemetry storage, taxonomy databases, and multi-stage task
generators.  For unit tests we emulate a trimmed version that captures the
shape of the interface: ingest metrics, adjust sampling weights, and persist the
result so scripts can inspect the new curriculum state.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

CURRICULUM_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "curriculum_v1"
STATE_PATH = CURRICULUM_DIR / "state.json"
HISTORY_LIMIT = 32
MIN_WEIGHT = 1e-3


@dataclass(slots=True)
class _TaskSignal:
    """Container describing the signals that influence sampling weights."""

    task_id: str
    success_rate: float
    novelty: float
    failure_taxonomy: Mapping[str, int]
    attempts: int
    previous_weight: float

    def compute_weight(self) -> float:
        """Mix success, novelty, and failures into a sampling weight."""

        success_penalty = 1.0 - max(0.0, min(1.0, self.success_rate))
        novelty_boost = max(0.0, self.novelty)
        failure_count = sum(int(v) for v in self.failure_taxonomy.values())
        failure_term = failure_count**0.5  # diminishing returns for large counts

        candidate = 1.0 + 1.5 * success_penalty + 0.5 * novelty_boost + 0.75 * failure_term

        # Combine with previous weight so the curriculum evolves smoothly.
        return 0.6 * self.previous_weight + 0.4 * candidate


def _load_state() -> dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Corrupted state should not brick the tests; fall back to defaults.
            pass
    return {"version": 1, "tasks": {}, "history": []}


def _normalise(weights: Mapping[str, float]) -> dict[str, float]:
    total = sum(max(MIN_WEIGHT, value) for value in weights.values())
    if total <= 0.0:
        equal_weight = 1.0 / max(1, len(weights))
        return {task_id: equal_weight for task_id in weights}
    return {task_id: max(MIN_WEIGHT, value) / total for task_id, value in weights.items()}


def update_curriculum(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Update sampling weights based on the latest metrics snapshot.

    Parameters
    ----------
    metrics:
        Mapping containing a ``tasks`` dictionary.  Each entry should provide
        ``success_rate`` (0-1), ``novelty`` (>=0), ``failures`` (taxonomy →
        counts), and ``attempts``.

    Returns
    -------
    The updated curriculum state dictionary.  The caller usually ignores the
    return value, but it is convenient for tests.
    """

    if not isinstance(metrics, Mapping):  # pragma: no cover - defensive clause
        raise TypeError("metrics must be a mapping")

    task_metrics = metrics.get("tasks")
    if not isinstance(task_metrics, Mapping) or not task_metrics:
        raise ValueError("metrics must include a non-empty 'tasks' mapping")

    CURRICULUM_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = _load_state()
    existing_tasks: MutableMapping[str, Any] = state.setdefault("tasks", {})

    timestamp = metrics.get("timestamp")
    if not isinstance(timestamp, str):
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    weights: dict[str, float] = {}
    taxonomy_counter: Counter[str] = Counter()

    for task_id, data in task_metrics.items():
        if not isinstance(task_id, str):
            raise TypeError("task identifiers must be strings")
        if not isinstance(data, Mapping):
            raise TypeError("task metrics must be mappings")

        success_rate = float(data.get("success_rate", 0.0))
        novelty = float(data.get("novelty", 0.0))
        attempts = int(data.get("attempts", 0))
        failures = data.get("failures") or {}
        if not isinstance(failures, Mapping):
            raise TypeError("failures must be a mapping of taxonomy labels")

        taxonomy_counter.update({label: int(count) for label, count in failures.items()})

        previous_weight = float(existing_tasks.get(task_id, {}).get("weight", 1.0))
        signal = _TaskSignal(
            task_id=task_id,
            success_rate=success_rate,
            novelty=novelty,
            failure_taxonomy=failures,
            attempts=attempts,
            previous_weight=previous_weight,
        )
        weights[task_id] = signal.compute_weight()

        task_state = {
            "weight": previous_weight,  # placeholder until normalisation occurs
            "success_rate": success_rate,
            "novelty": max(0.0, novelty),
            "failures": {label: int(count) for label, count in failures.items()},
            "attempts": max(0, attempts),
            "last_updated": timestamp,
        }
        existing_tasks[task_id] = task_state

    # Apply a mild decay to tasks that did not appear in the latest metrics.
    missing_tasks = set(existing_tasks.keys()) - set(task_metrics.keys())
    for task_id in missing_tasks:
        existing_tasks[task_id]["weight"] = max(
            MIN_WEIGHT, float(existing_tasks[task_id].get("weight", 1.0)) * 0.85
        )

    normalised = _normalise(
        {**{task_id: existing_tasks[task_id]["weight"] for task_id in missing_tasks}, **weights}
    )
    for task_id, weight in normalised.items():
        existing_tasks[task_id]["weight"] = weight

    history = state.setdefault("history", [])
    summary = {
        "timestamp": timestamp,
        "total_tasks": len(existing_tasks),
        "avg_success": sum(t["success_rate"] for t in existing_tasks.values())
        / max(1, len(existing_tasks)),
        "top_failures": [label for label, _ in taxonomy_counter.most_common(5)],
    }
    history.append(summary)
    if len(history) > HISTORY_LIMIT:
        del history[:-HISTORY_LIMIT]

    state["version"] = 1
    state["updated_at"] = timestamp

    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    return state

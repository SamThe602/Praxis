"""Tests for the lightweight curriculum update logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from packages.guidance import curriculum
from scripts import export_checkpoint, seed_curriculum


@pytest.fixture
def isolated_curriculum(tmp_path, monkeypatch):
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(curriculum, "CURRICULUM_DIR", tmp_path)
    monkeypatch.setattr(curriculum, "STATE_PATH", state_path)
    return state_path


def test_update_curriculum_prioritises_struggling_tasks(isolated_curriculum):
    metrics = {
        "timestamp": "2024-09-05T12:00:00Z",
        "tasks": {
            "easy": {
                "success_rate": 0.9,
                "novelty": 0.1,
                "failures": {"syntax": 1},
                "attempts": 12,
            },
            "hard": {
                "success_rate": 0.25,
                "novelty": 0.6,
                "failures": {"runtime": 3, "logic": 2},
                "attempts": 15,
            },
        },
    }

    state = curriculum.update_curriculum(metrics)
    weights = {name: data["weight"] for name, data in state["tasks"].items()}

    assert weights["hard"] > weights["easy"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert "runtime" in state["history"][-1]["top_failures"]
    assert isolated_curriculum.exists()

    persisted = json.loads(isolated_curriculum.read_text(encoding="utf-8"))
    assert persisted["tasks"]["hard"]["success_rate"] == 0.25


def test_missing_tasks_decay_weight(isolated_curriculum):
    initial_metrics = {
        "tasks": {
            "alpha": {
                "success_rate": 0.1,
                "novelty": 0.4,
                "failures": {"logic": 4},
                "attempts": 9,
            },
            "beta": {
                "success_rate": 0.6,
                "novelty": 0.2,
                "failures": {},
                "attempts": 7,
            },
        }
    }
    state_first = curriculum.update_curriculum(initial_metrics)
    alpha_weight = state_first["tasks"]["alpha"]["weight"]

    follow_up_metrics = {
        "tasks": {
            "beta": {
                "success_rate": 0.7,
                "novelty": 0.1,
                "failures": {},
                "attempts": 5,
            }
        }
    }

    state_second = curriculum.update_curriculum(follow_up_metrics)
    assert state_second["tasks"]["alpha"]["weight"] < alpha_weight
    assert len(state_second["history"]) == 2


def test_curriculum_scripts_round_trip(tmp_path):
    curriculum_dir = tmp_path / "curriculum"
    result = seed_curriculum.build_curriculum(seed=99, limit=2, source=None, output=curriculum_dir)
    manifest_path = Path(result["manifest"])
    assert manifest_path.exists()
    metadata_path = curriculum_dir / "metadata.json"
    assert metadata_path.exists()

    checkpoint_root = tmp_path / "checkpoints"
    (checkpoint_root / "supervised").mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_root / "supervised" / "demo.pt"
    ckpt_path.write_bytes(b"demo-checkpoint")

    exports_dir = tmp_path / "exports"
    export_result = export_checkpoint.package_checkpoints(
        root=checkpoint_root, output=exports_dir, seed=0
    )
    export_manifest = Path(export_result["manifest"])
    assert export_result["entries"] == 1
    assert export_manifest.exists()

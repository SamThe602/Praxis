"""Integration tests for the Python↔native verifier bridge."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from packages.verifier import metamorphic as metamorphic_mod
from packages.verifier import native_bridge, property_tests, static_analysis


@contextlib.contextmanager
def configure_native_bridge(**overrides: Any):
    """Temporarily override bridge settings within a test."""

    settings = {
        "enabled": True,
        "static_analysis": True,
        "metamorphic": True,
        "checkers": True,
    }
    settings.update(overrides)
    original_cache = native_bridge._CONFIG_CACHE  # type: ignore[attr-defined]
    try:
        native_bridge._CONFIG_CACHE = {"native_bridge": settings}  # type: ignore[attr-defined]
        yield
    finally:
        native_bridge._CONFIG_CACHE = original_cache  # type: ignore[attr-defined]


@pytest.fixture(scope="module", autouse=True)
def ensure_library_built() -> None:
    """Ensure the native library is available before running bridge tests."""

    if not native_bridge.is_available():
        pytest.skip("native verifier library is not available")


def test_static_analysis_native_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    with configure_native_bridge(static_analysis=True, metamorphic=False, checkers=False):
        # Guard against falling back to the Python analyzer.
        def _fail_run(self: Any) -> Any:  # pragma: no cover - defensive guard
            raise AssertionError("Python analyzer should not run when native payload is provided")

        monkeypatch.setattr(static_analysis._Analyzer, "run", _fail_run)
        monkeypatch.setattr(
            native_bridge,
            "record_fallback",
            lambda surface: pytest.fail(f"unexpected fallback for {surface}"),
        )

        program = {
            "native_static": {
                "arrays": [
                    {"name": "values", "length": {"min": 0, "max": 3}},
                ],
                "array_accesses": [
                    {"array": 0, "index": {"min": 0, "max": 5}},
                ],
                "labels": {"0": {"function": "solve", "instruction_index": 4}},
            }
        }

        result = static_analysis.analyze(program)
        assert not result.ok
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure.kind == "array_out_of_bounds"
        assert failure.function == "solve"
        assert failure.detail.get("subject") == 0


def test_static_analysis_fallback_when_payload_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = static_analysis.StaticAnalysisResult((), {})

    def _run(self: Any) -> static_analysis.StaticAnalysisResult:
        return captured

    monkeypatch.setattr(static_analysis._Analyzer, "run", _run)
    fallbacks: list[str] = []
    monkeypatch.setattr(native_bridge, "record_fallback", lambda surface: fallbacks.append(surface))
    with configure_native_bridge(static_analysis=True, metamorphic=False, checkers=False):
        result = static_analysis.analyze({})
        assert result is captured
    assert fallbacks.count("static_analysis") == 1


def test_metamorphic_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}
    original = native_bridge.evaluate_relation

    def _wrapped(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        called["count"] += 1
        return original(payload)

    monkeypatch.setattr(native_bridge, "evaluate_relation", _wrapped)
    monkeypatch.setattr(
        native_bridge,
        "record_fallback",
        lambda surface: pytest.fail(f"unexpected fallback for {surface}"),
    )

    with configure_native_bridge(metamorphic=True, static_analysis=False, checkers=False):
        program = {
            "metamorphic_tests": [
                {
                    "name": "sum_permutation",
                    "function": sum,
                    "cases": [
                        {
                            "name": "preserves_sum",
                            "base": [1, 2, 3],
                            "variants": [
                                {"inputs": [3, 2, 1]},
                                {"inputs": [2, 1, 3]},
                            ],
                            "relation": "permutation_invariance",
                        },
                        {
                            "name": "detects_difference",
                            "base": [1, 2, 3],
                            "variants": [
                                {"inputs": [1, 1, 1]},
                            ],
                            "relation": "permutation_invariance",
                        },
                    ],
                }
            ]
        }

        run = metamorphic_mod.run_metamorphic_tests(program)
        assert len(run.results) == 3  # two cases -> three variants evaluated
        statuses = [item.status for item in run.results]
        assert statuses.count("passed") == 2
        assert statuses.count("failed") == 1
        assert called["count"] == 3


def test_property_tests_native_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"count": 0}
    original = native_bridge.run_checker

    def _wrapped(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        called["count"] += 1
        return original(payload)

    monkeypatch.setattr(native_bridge, "run_checker", _wrapped)
    monkeypatch.setattr(
        native_bridge,
        "record_fallback",
        lambda surface: pytest.fail(f"unexpected fallback for {surface}"),
    )

    with configure_native_bridge(checkers=True, static_analysis=False, metamorphic=False):
        program = {
            "property_tests": [
                {
                    "name": "sorted_checker",
                    "cases": [
                        {
                            "name": "already_sorted",
                            "inputs": {"values": [1, 2, 3]},
                            "function": lambda values: values,
                            "native_checker": {"name": "sorted", "values": "actual"},
                        },
                        {
                            "name": "detects_unsorted",
                            "inputs": {"values": [3, 2, 1]},
                            "function": lambda values: values,
                            "native_checker": {"name": "sorted", "values": "actual"},
                        },
                    ],
                }
            ]
        }

        run = property_tests.run_property_tests(program)
        assert len(run.results) == 2
        statuses = [result.status for result in run.results]
        assert statuses == ["passed", "failed"]
        assert called["count"] == 2

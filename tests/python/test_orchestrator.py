"""Integration coverage for the orchestrator pipeline."""

from __future__ import annotations

from pathlib import Path

from packages.orchestrator import orchestrator

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "specs"


def test_run_structured_generates_artifacts(tmp_path: Path) -> None:
    config = orchestrator.load_configuration()
    spec_path = FIXTURES / "histogram.yaml"

    report = orchestrator.run_structured(spec_path, config=config, generate_explanation=True)

    assert report.spec_id == "histogram"
    assert report.synthesizer.status == "solved"
    assert report.verifier.status == "ok"
    assert report.explanation is not None

    export = orchestrator.export_report(
        report, tmp_path, formats=("json", "markdown"), basename="histogram"
    )
    exported_files = {path.suffix for path in export.paths}
    assert {".json", ".md"} <= exported_files


def test_run_prompt_requires_confirmation_by_default(tmp_path: Path) -> None:
    config = orchestrator.load_configuration()
    prompt = "Explain an ambiguous idea without concrete details."

    result = orchestrator.run_prompt(prompt, config=config)

    assert result.status == "pending_confirmation"
    assert result.solution is None
    assert result.translation.status == "needs_confirmation"
    assert "fallback_available" in result.notes


def test_run_prompt_accepts_low_confidence_override(tmp_path: Path) -> None:
    overrides = {
        "translator": {"accept_low_confidence": True},
        "output": {"directory": str(tmp_path)},
    }
    config = orchestrator.load_configuration(overrides=overrides)

    prompt = "Provide a generic discussion without a clear spec."
    result = orchestrator.run_prompt(prompt, config=config, generate_explanation=True)
    assert result.status == "completed"
    assert result.solution is not None
    assert result.solution.verifier.status == "ok"
    assert result.solution.explanation is not None

    export = orchestrator.export_report(
        result.solution, tmp_path, formats=("json",), basename="reverse_prompt"
    )
    assert any(path.name.startswith("reverse_prompt") for path in export.paths)

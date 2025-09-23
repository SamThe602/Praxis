"""Smoke tests for the CLI structured workflow."""

from __future__ import annotations

from pathlib import Path

from packages.orchestrator import cli

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "specs"


def test_cli_structured_export(tmp_path):
    spec_path = FIXTURES / "array_reverse.yaml"
    args = [
        "--config",
        str(Path("configs/orchestrator/default.yaml")),
        "--set",
        f"output.directory={tmp_path}",
        "structured",
        str(spec_path),
        "--export",
        str(tmp_path),
        "--format",
        "json",
        "--format",
        "markdown",
    ]
    exit_code = cli.main(args)
    assert exit_code == 0
    exported = list(tmp_path.glob("*.json"))
    assert exported, "expected JSON export"
    assert any(path.with_suffix(".md").exists() for path in exported)

"""Smoke tests for the CLI prompt workflow."""

from __future__ import annotations

from pathlib import Path

from packages.orchestrator import cli


def test_cli_prompt_run(tmp_path) -> None:
    args = [
        "--config",
        str(Path("configs/orchestrator/default.yaml")),
        "--set",
        f"output.directory={tmp_path}",
        "prompt",
        "--prompt",
        "Sort the array ascending.",
        "--export",
        str(tmp_path),
        "--format",
        "json",
    ]
    exit_code = cli.main(args)
    assert exit_code == 0
    exported = list(tmp_path.glob("*.json"))
    assert exported, "expected JSON export"

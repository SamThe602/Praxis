#!/usr/bin/env python3
"""Evaluation harness running orchestrator suites defined under configs/eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from packages.orchestrator import orchestrator
from packages.orchestrator.cli import DEFAULT_CONFIG_PATH as CLI_DEFAULT_CONFIG
from packages.orchestrator.types import OrchestratorConfig

_FORMAT_CHOICES = ("json", "yaml", "markdown")
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SUITES = _ROOT / "configs" / "eval"
_DEFAULT_OUTPUT = _ROOT / "data" / "cache" / "eval_reports"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Praxis evaluation suites")
    parser.add_argument(
        "--suite",
        dest="suites",
        action="append",
        help="Suite name or path (repeat to run multiple suites)",
    )
    parser.add_argument("--config", type=Path, help="Optional orchestrator configuration file")
    parser.add_argument(
        "--set", dest="overrides", action="append", help="Configuration overrides (key=value)"
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUTPUT, help="Directory where reports will be stored"
    )
    parser.add_argument(
        "--formats",
        action="append",
        choices=_FORMAT_CHOICES,
        help="Preferred export formats overriding suite defaults",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Abort on first task failure")

    args = parser.parse_args(argv)

    overrides = _parse_overrides(args.overrides)
    config_path = args.config or (CLI_DEFAULT_CONFIG if CLI_DEFAULT_CONFIG.exists() else None)
    config = orchestrator.load_configuration(config_path, overrides=overrides)

    suite_paths = list(_discover_suite_paths(args.suites))
    if not suite_paths:
        print("[eval] no evaluation suites found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    overall = []
    all_success = True

    for suite_path in suite_paths:
        summary = _run_suite(
            suite_path,
            config=config,
            output_root=args.out,
            override_formats=args.formats,
            fail_fast=args.fail_fast,
        )
        overall.append(summary)
        all_success = all_success and summary.get("success", False)

    index_path = args.out / "index.json"
    index_payload = {
        "root": str(args.out),
        "suites": overall,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[eval] summary written to {index_path}")
    return 0 if all_success else 2


# ---------------------------------------------------------------------------
# Suite execution helpers


def _discover_suite_paths(selected: Sequence[str] | None) -> Sequence[Path]:
    """Return concrete suite file paths to execute."""

    if selected:
        paths: list[Path] = []
        for item in selected:
            candidate = Path(item)
            if candidate.exists():
                paths.append(candidate)
                continue
            yaml_path = _DEFAULT_SUITES / f"{item}.yaml"
            yml_path = _DEFAULT_SUITES / f"{item}.yml"
            if yaml_path.exists():
                paths.append(yaml_path)
            elif yml_path.exists():
                paths.append(yml_path)
            else:
                raise FileNotFoundError(f"evaluation suite '{item}' not found")
        return paths
    return sorted(_DEFAULT_SUITES.glob("*.y*ml"))


def _run_suite(
    path: Path,
    *,
    config: OrchestratorConfig,
    output_root: Path,
    override_formats: Sequence[str] | None,
    fail_fast: bool,
) -> Mapping[str, Any]:
    """Execute a single evaluation suite and return a serialisable summary."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"evaluation suite at {path} must be a mapping")

    suite_meta = data.get("suite") if isinstance(data.get("suite"), Mapping) else {}
    name = str(suite_meta.get("name", path.stem))
    description = str(suite_meta.get("description", "")).strip()
    defaults = data.get("defaults") if isinstance(data.get("defaults"), Mapping) else {}
    tasks = data.get("tasks")
    if not isinstance(tasks, Sequence):
        raise ValueError(f"evaluation suite '{name}' must declare a list of tasks")

    suite_dir = output_root / name
    suite_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    completed = 0
    for index, entry in enumerate(tasks, start=1):
        task_id = f"task_{index}"
        if isinstance(entry, Mapping) and entry.get("id"):
            task_id = str(entry["id"])
        try:
            record = _run_task_entry(
                entry,
                task_id=task_id,
                config=config,
                suite_dir=suite_dir,
                defaults=defaults,
                override_formats=override_formats,
            )
            if record.get("status") == "completed":
                completed += 1
        except Exception as exc:
            record = {
                "task_id": task_id,
                "status": "error",
                "error": str(exc),
            }
            print(f"[eval] {name}/{task_id} failed: {exc}", file=sys.stderr)
            if fail_fast:
                records.append(record)
                break
        records.append(record)
    success = completed == len(tasks)
    index_payload = {
        "suite": name,
        "description": description,
        "path": str(path),
        "total": len(tasks),
        "completed": completed,
        "success": success,
        "tasks": records,
    }
    (suite_dir / "index.json").write_text(
        json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[eval] suite {name}: {completed}/{len(tasks)} tasks completed -> {suite_dir}")
    return index_payload


def _run_task_entry(
    entry: Any,
    *,
    task_id: str,
    config: OrchestratorConfig,
    suite_dir: Path,
    defaults: Mapping[str, Any],
    override_formats: Sequence[str] | None,
) -> dict[str, Any]:
    """Execute an individual suite task and normalise the resulting metadata."""
    if not isinstance(entry, Mapping):
        raise TypeError(f"task '{task_id}' must be a mapping")

    mode = str(entry.get("mode") or defaults.get("mode", "spec")).lower()
    explain = bool(entry.get("explain", defaults.get("explain", False)))

    formats = _resolve_formats(entry, defaults, override_formats)
    basename = str(entry.get("output_name") or defaults.get("output_name", task_id))

    if mode in {"spec", "structured"}:
        spec_value = entry.get("spec") or defaults.get("spec")
        if not spec_value:
            raise ValueError(f"task '{task_id}' requires a 'spec'")
        spec_path = _resolve_path(spec_value)
        if not spec_path.exists():
            raise FileNotFoundError(f"structured spec not found: {spec_path}")
        report = orchestrator.run_structured(spec_path, config=config, generate_explanation=explain)
        export = orchestrator.export_report(report, suite_dir, formats=formats, basename=basename)
        record = {
            "task_id": task_id,
            "mode": "spec",
            "status": "completed",
            "spec": str(spec_path),
            "outputs": [str(path) for path in export.paths],
            "synth_status": report.synthesizer.status,
            "verifier_status": report.verifier.status,
        }
        return record

    if mode in {"prompt", "text"}:
        prompt = entry.get("prompt") or defaults.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task '{task_id}' requires a non-empty 'prompt'")
        locale = entry.get("locale") or defaults.get("locale")
        result = orchestrator.run_prompt(
            prompt,
            config=config,
            locale=locale,
            generate_explanation=explain,
        )
        record = {
            "task_id": task_id,
            "mode": "prompt",
            "status": result.status,
            "prompt": prompt,
            "confidence": result.translation.confidence,
            "translation_status": result.translation.status,
            "notes": list(result.notes),
        }
        if result.solution is not None:
            export = orchestrator.export_report(
                result.solution, suite_dir, formats=formats, basename=basename
            )
            record.update(
                {
                    "status": "completed",
                    "outputs": [str(path) for path in export.paths],
                    "synth_status": result.solution.synthesizer.status,
                    "verifier_status": result.solution.verifier.status,
                }
            )
        else:
            report_path = suite_dir / f"{basename}.json"
            report_path.write_text(
                json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
            )
            record.setdefault("outputs", [str(report_path)])
        return record

    raise ValueError(f"unknown task mode '{mode}' for task '{task_id}'")


# ---------------------------------------------------------------------------
# Utility helpers


def _resolve_formats(
    entry: Mapping[str, Any],
    defaults: Mapping[str, Any],
    override_formats: Sequence[str] | None,
) -> tuple[str, ...]:
    """Return the ordered list of export formats respecting overrides."""
    merged: list[str] = []
    default_formats = defaults.get("export_formats")
    if isinstance(default_formats, Sequence):
        merged.extend(str(item) for item in default_formats if item)
    entry_formats = entry.get("export_formats")
    if isinstance(entry_formats, Sequence):
        merged.extend(str(item) for item in entry_formats if item)
    if override_formats:
        merged.extend(str(item) for item in override_formats if item)
    merged.append("json")
    seen: dict[str, None] = {}
    for item in merged:
        if item not in _FORMAT_CHOICES:
            raise ValueError(f"unsupported export format '{item}'")
        seen.setdefault(item, None)
    return tuple(seen.keys())


def _resolve_path(value: Any) -> Path:
    """Interpret ``value`` as a path relative to the repository root."""
    path = Path(str(value))
    if not path.is_absolute():
        return (_ROOT / path).resolve()
    return path


def _parse_overrides(raw: Sequence[str] | None) -> Mapping[str, Any]:
    """Convert ``KEY=VALUE`` strings into a nested mapping of overrides."""
    overrides: dict[str, Any] = {}
    if not raw:
        return overrides
    for item in raw:
        key, sep, value_text = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' is missing '='")
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError("override key must not be empty")
        value = _coerce_literal(value_text)
        cursor: dict[str, Any] = overrides
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
            if not isinstance(cursor, dict):
                raise ValueError(f"override '{key}' conflicts with an existing value")
        cursor[parts[-1]] = value
    return overrides


def _coerce_literal(value: str) -> Any:
    """Parse ``value`` into a Python literal using JSON/YAML fallbacks."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return yaml.safe_load(value)
        except yaml.YAMLError:
            return value


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

"""Praxis orchestrator command-line interface."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import orchestrator
from .types import OrchestratorConfig

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "orchestrator" / "default.yaml"
)

_FORMAT_CHOICES = ("json", "yaml", "markdown")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="praxis", description="Praxis orchestration CLI")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None,
        help="Optional path to an orchestrator configuration YAML file.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        help="Override configuration values using dot notation (e.g. synthesizer.overrides.search.max_nodes=256).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_structured_parser(subparsers)
    _add_prompt_parser(subparsers)
    _add_explain_parser(subparsers)
    _add_verify_parser(subparsers)
    _add_export_parser(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        overrides = _parse_overrides(args.overrides)
        config = orchestrator.load_configuration(args.config, overrides=overrides)
        if args.command == "structured":
            return _cmd_structured(args, config)
        if args.command == "prompt":
            return _cmd_prompt(args, config)
        if args.command == "explain":
            return _cmd_explain(args)
        if args.command == "verify":
            return _cmd_verify(args, config)
        if args.command == "export":
            return _cmd_export(args)
    except Exception as exc:  # pragma: no cover - defensive CLI guard
        print(f"[praxis] error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1


# ---------------------------------------------------------------------------
# Sub-command wiring


def _add_structured_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "structured", help="Run a structured specification through the pipeline"
    )
    parser.add_argument("spec", type=Path, help="Path to the structured YAML specification")
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate a Markdown explanation inline.",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Directory where artifacts should be exported",
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=_FORMAT_CHOICES,
        dest="formats",
        help="Formats to emit when exporting (default derived from config)",
    )
    parser.add_argument(
        "--output-name",
        dest="basename",
        help="Override the basename used for exported artifacts",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the full report as JSON to stdout",
    )


def _add_prompt_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "prompt", help="Run a natural-language prompt through the pipeline"
    )
    parser.add_argument("--prompt", help="Prompt text (omit to read from stdin)")
    parser.add_argument("--prompt-file", type=Path, help="File containing the prompt")
    parser.add_argument("--locale", help="Override translator locale")
    parser.add_argument(
        "--explain", action="store_true", help="Generate explanation when synthesis succeeds"
    )
    parser.add_argument("--export", type=Path, help="Directory for exported artifacts")
    parser.add_argument(
        "--format",
        action="append",
        choices=_FORMAT_CHOICES,
        dest="formats",
        help="Formats to emit when exporting",
    )
    parser.add_argument("--output-name", dest="basename", help="Basename for exported artifacts")
    parser.add_argument("--json", action="store_true", help="Emit the result bundle as JSON")


def _add_explain_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "explain", help="Render a Markdown explanation for a stored report"
    )
    parser.add_argument(
        "report", type=Path, help="Path to a JSON report produced by the orchestrator"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional path where the explanation should be written"
    )


def _add_verify_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("verify", help="Run the verifier on a bytecode module")
    parser.add_argument(
        "module",
        type=Path,
        help="JSON file containing either the full program request or the module itself",
    )
    parser.add_argument("--entry", help="Override the entrypoint (defaults to solve)")
    parser.add_argument(
        "--arg",
        dest="args",
        action="append",
        help="Entry-point argument literal (repeat for multiple values)",
    )
    parser.add_argument(
        "--limits",
        help="JSON literal describing execution limits (e.g. '{\"fuel\": 1024}')",
    )
    parser.add_argument("--json", action="store_true", help="Emit the verifier response as JSON")


def _add_export_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export artifacts from a stored report")
    parser.add_argument("report", type=Path, help="Report JSON produced by the orchestrator")
    parser.add_argument(
        "--dir", type=Path, required=True, help="Target directory for exported artifacts"
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=_FORMAT_CHOICES,
        dest="formats",
        help="Formats to emit (default derived from report config)",
    )
    parser.add_argument("--output-name", dest="basename", help="Basename for exported artifacts")


# ---------------------------------------------------------------------------
# Command implementations


def _cmd_structured(args: argparse.Namespace, config: OrchestratorConfig) -> int:
    report = orchestrator.run_structured(
        args.spec, config=config, generate_explanation=args.explain
    )
    print(f"[praxis] structured run completed: {report.summary()}")

    if args.export:
        formats = tuple(args.formats) if args.formats else _preferred_formats(config)
        export = orchestrator.export_report(
            report, args.export, formats=formats, basename=args.basename
        )
        print(_format_export(export))
    elif not args.export and _should_auto_export(config):
        export = orchestrator.export_report(
            report,
            config.output.directory,
            formats=_preferred_formats(config),
        )
        print(_format_export(export))

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_prompt(args: argparse.Namespace, config: OrchestratorConfig) -> int:
    prompt_text = _load_prompt(args)
    result = orchestrator.run_prompt(
        prompt_text,
        config=config,
        locale=args.locale,
        generate_explanation=args.explain,
    )
    translation = result.translation
    print(
        f"[praxis] prompt status={result.status} confidence={translation.confidence:.3f} "
        f"mode={translation.status}"
    )
    if translation.fallback:
        print(
            f"[praxis] fallback metadata available (fields: {sorted(translation.fallback.keys())})"
        )

    if result.solution is not None:
        print(f"[praxis] synthesis summary: {result.solution.summary()}")
        target_dir = args.export or (
            config.output.directory if _should_auto_export(config) else None
        )
        if target_dir:
            formats = tuple(args.formats) if args.formats else _preferred_formats(config)
            export = orchestrator.export_report(
                result.solution, target_dir, formats=formats, basename=args.basename
            )
            print(_format_export(export))
        if args.json:
            print(json.dumps(result.solution.to_dict(), indent=2, sort_keys=True))
    elif args.json:
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_explain(args: argparse.Namespace) -> int:
    payload = _load_json(args.report)
    explanation = orchestrator.explain_report(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(explanation, encoding="utf-8")
        print(f"[praxis] explanation written to {args.output}")
    else:
        print(explanation)
    return 0


def _cmd_verify(args: argparse.Namespace, config: OrchestratorConfig) -> int:
    payload = _load_json(args.module)
    limits = _parse_limits(args.limits)
    parsed_args = tuple(_coerce_literal(value) for value in (args.args or ()))
    report = orchestrator.verify_module(
        payload,
        config=config,
        entry=args.entry,
        args=parsed_args,
        limits=limits,
    )
    print(f"[praxis] verifier status={report.status} entry={report.entrypoint}")
    if report.error:
        print(f"[praxis] error={report.error} detail={report.detail}")
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    payload = _load_json(args.report)
    formats = tuple(args.formats) if args.formats else ("json",)
    export = orchestrator.export_report(payload, args.dir, formats=formats, basename=args.basename)
    print(_format_export(export))
    return 0


# ---------------------------------------------------------------------------
# Helpers


def _parse_overrides(raw: Sequence[str] | None) -> Mapping[str, Any]:
    overrides: dict[str, Any] = {}
    if not raw:
        return overrides
    for item in raw:
        key, sep, value_text = item.partition("=")
        if not sep:
            raise ValueError(f"override '{item}' is missing '='")
        key_parts = [part for part in key.split(".") if part]
        if not key_parts:
            raise ValueError("override key must not be empty")
        value = _coerce_literal(value_text)
        cursor = overrides
        for part in key_parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
            if not isinstance(cursor, dict):
                raise ValueError(f"override '{key}' conflicts with an existing value")
        cursor[key_parts[-1]] = value
    return overrides


def _coerce_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _parse_limits(raw: str | None) -> Mapping[str, Any] | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        coerced = _coerce_literal(raw)
        if isinstance(coerced, Mapping):
            return coerced
        raise ValueError("limits must be a JSON or literal mapping")


def _preferred_formats(config: OrchestratorConfig) -> tuple[str, ...]:
    formats: list[str] = []
    if config.output.save_json:
        formats.append("json")
    if config.output.save_yaml:
        formats.append("yaml")
    if config.output.save_markdown:
        formats.append("markdown")
    return tuple(formats) if formats else ("json",)


def _should_auto_export(config: OrchestratorConfig) -> bool:
    return config.output.directory is not None and (
        config.output.save_json or config.output.save_yaml or config.output.save_markdown
    )


def _format_export(export: Any) -> str:
    payload = export.to_dict() if hasattr(export, "to_dict") else export
    paths = ", ".join(payload.get("paths", []))
    return f"[praxis] exported artifacts ({', '.join(payload.get('formats', []))}) -> {paths}"


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

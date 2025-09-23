#!/usr/bin/env python3
"""Run a structured Praxis specification end-to-end."""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.orchestrator import cli

_FORMAT_CHOICES = ("json", "yaml", "markdown")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a structured Praxis task")
    parser.add_argument("spec", type=Path, help="Path to the structured YAML specification")
    parser.add_argument("--config", type=Path, help="Optional orchestrator configuration file")
    parser.add_argument(
        "--set", dest="overrides", action="append", help="Configuration overrides (key=value)"
    )
    parser.add_argument("--explain", action="store_true", help="Generate Markdown explanation")
    parser.add_argument("--export", type=Path, help="Directory where artifacts should be exported")
    parser.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=_FORMAT_CHOICES,
        help="Export formats to emit",
    )
    parser.add_argument(
        "--output-name", dest="basename", help="Override the basename for exported artifacts"
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit the full report as JSON to stdout"
    )

    args = parser.parse_args(argv)

    cli_args: list[str] = []
    if args.config:
        cli_args.extend(["--config", str(args.config)])
    for override in args.overrides or ():
        cli_args.extend(["--set", override])

    cli_args.extend(["structured", str(args.spec)])
    if args.explain:
        cli_args.append("--explain")
    if args.export:
        cli_args.extend(["--export", str(args.export)])
    for fmt in args.formats or ():
        cli_args.extend(["--format", fmt])
    if args.basename:
        cli_args.extend(["--output-name", args.basename])
    if args.json:
        cli_args.append("--json")

    return cli.main(cli_args)


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())

"""Utility script to import/export Praxis knowledge base shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from packages.synthesizer import retrieval


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synchronise the local knowledge base store")
    sub = parser.add_subparsers(dest="command", required=True)

    export_cmd = sub.add_parser("export", help="Write the current knowledge base to a JSON file")
    export_cmd.add_argument("output", type=Path, help="Destination JSON path")
    export_cmd.add_argument("--pretty", action="store_true", help="Pretty-print the exported JSON")
    export_cmd.set_defaults(func=_handle_export)

    import_cmd = sub.add_parser("import", help="Merge entries from a JSON file into the KB")
    import_cmd.add_argument("input", type=Path, help="Source JSON path")
    import_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing entries that share the same spec/snippet key",
    )
    import_cmd.set_defaults(func=_handle_import)

    list_cmd = sub.add_parser("list", help="Display a quick summary of stored snippets")
    list_cmd.add_argument(
        "--limit", type=int, default=20, help="Maximum number of records to display"
    )
    list_cmd.set_defaults(func=_handle_list)

    return parser.parse_args()


def _handle_export(args: argparse.Namespace) -> None:
    kb = retrieval.KnowledgeBase.load()
    payload = {
        "dimension": kb.dimension,
        "metadata": kb.metadata,
        "entries": [entry.as_payload() for entry in kb.entries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.pretty:
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = json.dumps(payload, separators=(",", ":"))
    args.output.write_text(text + "\n", encoding="utf-8")
    print(f"exported {len(kb.entries)} entries to {args.output}")


def _handle_import(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise FileNotFoundError(f"input file not found: {args.input}")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("knowledge base payload must be a JSON object")
    kb = retrieval.KnowledgeBase.load()
    merged = kb.merge_payload(payload, overwrite=bool(args.overwrite))
    kb.save()
    print(f"merged {merged} entries into {kb.path}")


def _handle_list(args: argparse.Namespace) -> None:
    kb = retrieval.KnowledgeBase.load()
    total = len(kb.entries)
    print(f"knowledge base root: {kb.root}")
    print(f"dimension: {kb.dimension}")
    print(f"entries: {total}")
    limit = max(0, int(args.limit))
    for index, entry in enumerate(kb.entries[:limit]):
        summary = entry.snippet.replace("\n", " ")
        if len(summary) > 60:
            summary = summary[:57] + "..."
        print(
            f"[{index:03d}] spec={entry.spec_id!r} score_hint={entry.metadata.get('score', 'n/a')} snippet={summary}"
        )
    if total > limit:
        print(f"... and {total - limit} more")


def main() -> None:
    args = _parse_args()
    handler = getattr(args, "func", None)
    if handler is None:
        raise RuntimeError("no command handler registered")
    handler(args)


if __name__ == "__main__":
    main()

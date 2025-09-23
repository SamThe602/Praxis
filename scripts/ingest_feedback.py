"""CLI helper that ingests JSONL feedback files into the local store."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator

from packages.feedback import annotator_api, queue, storage


def _load_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    if str(path) == "-":
        import sys

        stream = sys.stdin
        for line in stream:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
        return

    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Path to a JSONL file or '-' for stdin")
    parser.add_argument(
        "--enqueue",
        action="store_true",
        help="Also push the ingested records onto the feedback queue",
    )
    parser.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Queue priority to use when --enqueue is provided",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    records = list(_load_jsonl(input_path))
    storage.extend(records)

    if args.enqueue:
        for record in records:
            record_priority = args.priority
            if isinstance(record, Mapping):
                priority_value = record.get("priority", record_priority)
                if priority_value is not None:
                    try:
                        record_priority = int(priority_value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        record_priority = args.priority
            annotator_api.queue_for_review(record, priority=record_priority)

    print(f"ingested {len(records)} feedback records")
    if args.enqueue:
        print(f"queue size: {queue.size()}")


if __name__ == "__main__":
    main()

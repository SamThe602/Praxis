"""Create deterministic curriculum artifacts from bundled fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from packages.dsl.generators import GeneratedProgram, generate_curriculum_tasks
from packages.guidance.utils import deterministic_shuffle, seed_everything

DEFAULT_OUTPUT = Path("data/processed/curriculum_v1")


def _hash_text(text: str) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _serialise_task(
    task: GeneratedProgram, *, index: int, base_dir: Path, output_dir: Path
) -> dict:
    program_dir = output_dir / "programs"
    program_dir.mkdir(parents=True, exist_ok=True)

    target_name = f"{task.name or f'task_{index:04d}'}.dsl"
    target_path = program_dir / target_name
    target_path.write_text(task.source, encoding="utf-8")

    rel_path = target_path.relative_to(base_dir)
    num_lines = task.source.count("\n") + 1 if task.source else 0
    text_hash = _hash_text(task.source)

    return {
        "id": task.name or f"task_{index:04d}",
        "index": index,
        "program_path": str(rel_path),
        "sha256": text_hash,
        "line_count": num_lines,
        "byte_length": len(task.source.encode("utf-8")),
    }


def _ensure_unique(tasks: Sequence[GeneratedProgram]) -> Iterable[GeneratedProgram]:
    """Yield tasks while ensuring stable unique identifiers."""

    seen: dict[str, int] = {}
    for task in tasks:
        if task.name not in seen:
            seen[task.name] = 0
            yield task
            continue
        seen[task.name] += 1
        suffix = seen[task.name]
        yield GeneratedProgram(
            name=f"{task.name}_{suffix}",
            path=task.path,
            source=task.source,
            module=task.module,
        )


def build_curriculum(
    *, seed: int, limit: int | None, source: Path | None, output: Path | None
) -> dict:
    output_dir = output or DEFAULT_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    program_dir = output_dir / "programs"
    if program_dir.exists():
        for path in program_dir.glob("*.dsl"):
            path.unlink()

    seed_everything(seed)
    raw_tasks = list(generate_curriculum_tasks(source))
    if limit is not None:
        raw_tasks = raw_tasks[:limit]
    shuffled = deterministic_shuffle(raw_tasks, seed)
    deduped = list(_ensure_unique(shuffled))

    manifest = []
    base_dir = output_dir
    for idx, task in enumerate(deduped):
        manifest.append(_serialise_task(task, index=idx, base_dir=base_dir, output_dir=output_dir))

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "task_count": len(manifest),
        "source": str(source) if source is not None else "default-fixtures",
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return {"manifest": str(manifest_path), "metadata": str(metadata_path)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "Seed curriculum artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed to apply")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional upper bound on the number of tasks to export",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional directory containing .dsl fixtures",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory (defaults to data/processed/curriculum_v1)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    return build_curriculum(
        seed=args.seed,
        limit=args.limit,
        source=args.source,
        output=args.output,
    )


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))

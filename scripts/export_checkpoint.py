"""Collect the latest checkpoints and bundle them with provenance metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from packages.guidance.utils import seed_everything

CHECKPOINT_ROOT = Path("data/processed/checkpoints")
EXPORT_ROOT = CHECKPOINT_ROOT / "exports"
CHECKPOINT_EXTENSIONS = {".pt", ".pth", ".ckpt", ".bin"}
CONFIG_MAP = {
    "supervised": Path("configs/training/supervised.yaml"),
    "rl": Path("configs/training/rl.yaml"),
    "preference": Path("configs/training/preference.yaml"),
}


def _hash_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hash_config(name: str) -> dict | None:
    config_path = CONFIG_MAP.get(name)
    if config_path is None or not config_path.exists():
        return None
    return {"path": str(config_path), "sha256": _hash_bytes(config_path)}


def _iter_latest(root: Path) -> Iterable[tuple[str, Path]]:
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir() or subdir.name == "exports":
            continue
        candidates = [
            path
            for path in subdir.rglob("*")
            if path.is_file() and path.suffix.lower() in CHECKPOINT_EXTENSIONS
        ]
        if not candidates:
            continue
        latest = max(candidates, key=lambda path: path.stat().st_mtime)
        yield subdir.name, latest


def package_checkpoints(
    *, root: Path | None = None, output: Path | None = None, seed: int | None = None
) -> dict:
    root_dir = root or CHECKPOINT_ROOT
    export_base = output or EXPORT_ROOT
    export_base.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        seed_everything(seed)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    export_dir = export_base / f"export_{timestamp}"
    export_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    iterator: Iterable[tuple[str, Path]] = _iter_latest(root_dir) if root_dir.exists() else ()

    for name, path in iterator:
        target_dir = export_dir / name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / path.name
        shutil.copy2(path, target_path)

        entry = {
            "name": name,
            "source_path": str(path),
            "export_path": str(target_path),
            "size": path.stat().st_size,
            "mtime": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            "sha256": _hash_bytes(path),
            "config": _hash_config(name),
        }
        entries.append(entry)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_count": len(entries),
        "root": str(root_dir),
        "entries": entries,
    }

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    latest_path = export_base / "latest.json"
    latest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "manifest": str(manifest_path),
        "entries": len(entries),
        "export_dir": str(export_dir),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__ or "Export checkpoints")
    parser.add_argument("--root", type=Path, default=None, help="Checkpoint root directory")
    parser.add_argument("--output", type=Path, default=None, help="Export base directory")
    parser.add_argument(
        "--seed", type=int, default=None, help="Optional RNG seed for reproducibility"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    return package_checkpoints(root=args.root, output=args.output, seed=args.seed)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))

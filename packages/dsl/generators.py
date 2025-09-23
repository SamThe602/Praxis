"""Helper utilities for producing synthetic DSL programs used in tests.

The project includes several small `.dsl` fixtures that capture the flavour of
programs we expect parsers and serializers to handle.  The helpers here load and
parse those fixtures on demand so tests can focus on semantic checks instead of
I/O boilerplate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from . import grammar
from .ast import Module

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests" / "python" / "fixtures" / "programs"


@dataclass(frozen=True)
class GeneratedProgram:
    """Container bundling a parsed module with context metadata."""

    name: str
    path: Path
    source: str
    module: Module


def generate_curriculum_tasks(root: Path | None = None) -> Iterable[GeneratedProgram]:
    """Yield parsed representations of the bundled curriculum fixtures.

    Parameters
    ----------
    root:
        Optional directory override.  When omitted the default fixture
        collection under ``tests/python/fixtures/programs`` is used.
    """

    yield from _iter_programs(root or FIXTURE_ROOT)


def _iter_programs(root: Path) -> Iterator[GeneratedProgram]:
    if not root.exists():
        return
    for path in sorted(root.glob("*.dsl")):
        source = path.read_text(encoding="utf-8")
        module = grammar.parse_module(source, filename=str(path))
        yield GeneratedProgram(name=path.stem, path=path, source=source, module=module)


__all__ = ["GeneratedProgram", "generate_curriculum_tasks"]

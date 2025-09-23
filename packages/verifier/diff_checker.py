"""Robust structural and behavioural equivalence checker for Praxis modules."""

from __future__ import annotations

import ast
import difflib
import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from packages.verifier.runner import VmExecution, VmExecutionError, execute_vm

TModule = Mapping[str, Any] | Any
TRunner = Callable[..., VmExecution]

__all__ = [
    "BehaviouralDiff",
    "DiffConfig",
    "DiffReport",
    "StructuralDiff",
    "compare",
]


@dataclass(slots=True, frozen=True)
class StructuralDiff:
    """Outcome of the structural comparison stage."""

    equivalent: bool
    candidate_repr: str
    baseline_repr: str
    differences: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BehaviouralDiff:
    """Outcome of the behavioural comparison stage."""

    equivalent: bool
    mismatches: tuple[Mapping[str, Any], ...]
    checks_run: int


@dataclass(slots=True, frozen=True)
class DiffReport:
    """Aggregate diff report returned by :func:`compare`."""

    equivalent: bool
    structural: StructuralDiff
    behavioural: BehaviouralDiff | None


@dataclass(slots=True, frozen=True)
class DiffConfig:
    """Configuration knobs for :func:`compare`."""

    whitespace_insensitive: bool = True
    fold_constants: bool = True
    behavioural_checks: int = 3
    behavioural_inputs: Sequence[Any] | None = None
    behavioural_runner: TRunner | None = None
    entrypoint: str | None = None
    limits: Mapping[str, Any] | None = None


def compare(
    candidate: TModule, baseline: TModule, *, config: DiffConfig | None = None
) -> DiffReport:
    """Compare ``candidate`` and ``baseline`` modules.

    The checker runs two complementary passes:

    * **Structural** – normalises the payloads (AST for strings, JSON for
      mappings) and emits a unified diff when they diverge.  Whitespace and
      trivial constant folding differences are ignored by default.
    * **Behavioural** – executes both modules inside the Praxis VM on a small
      battery of generated inputs to confirm observable equivalence.  This stage
      relies on :func:`packages.verifier.runner.execute_vm` unless an alternate
      runner is supplied via :class:`DiffConfig` (useful in unit tests).
    """

    cfg = config or DiffConfig()
    structural = _compare_structural(candidate, baseline, cfg)
    behavioural: BehaviouralDiff | None = None
    equivalent = structural.equivalent

    if cfg.behavioural_checks > 0:
        behavioural = _compare_behaviour(candidate, baseline, cfg)
        if behavioural is not None:
            equivalent = equivalent and behavioural.equivalent

    return DiffReport(equivalent=equivalent, structural=structural, behavioural=behavioural)


def _compare_structural(candidate: Any, baseline: Any, cfg: DiffConfig) -> StructuralDiff:
    c_repr = _normalise_structure(candidate, cfg)
    b_repr = _normalise_structure(baseline, cfg)
    if c_repr == b_repr:
        return StructuralDiff(
            equivalent=True, candidate_repr=c_repr, baseline_repr=b_repr, differences=()
        )

    diff = tuple(
        difflib.unified_diff(
            b_repr.splitlines(),
            c_repr.splitlines(),
            fromfile="baseline",
            tofile="candidate",
            lineterm="",
        )
    )
    return StructuralDiff(
        equivalent=False, candidate_repr=c_repr, baseline_repr=b_repr, differences=diff
    )


def _compare_behaviour(candidate: Any, baseline: Any, cfg: DiffConfig) -> BehaviouralDiff | None:
    candidate_program = _coerce_program(candidate, cfg)
    baseline_program = _coerce_program(baseline, cfg)
    if candidate_program is None or baseline_program is None:
        return None

    runner = cfg.behavioural_runner or execute_vm
    inputs = list(cfg.behavioural_inputs or _default_inputs(cfg.behavioural_checks))
    mismatches: list[Mapping[str, Any]] = []
    checks_run = 0

    for index, args in enumerate(inputs[: cfg.behavioural_checks]):
        args_list = _normalise_args(args)
        payload = {
            "module": candidate_program["module"],
            "entry": candidate_program["entry"],
            "args": args_list,
            "limits": candidate_program.get("limits"),
        }
        reference_payload = {
            "module": baseline_program["module"],
            "entry": baseline_program["entry"],
            "args": args_list,
            "limits": baseline_program.get("limits"),
        }
        checks_run += 1
        try:
            candidate_result = runner(**payload)
        except VmExecutionError as exc:
            mismatches.append(
                {
                    "input_index": index,
                    "stage": "candidate",
                    "error": exc.kind,
                    "message": str(exc),
                }
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            mismatches.append(
                {
                    "input_index": index,
                    "stage": "candidate",
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            continue

        try:
            baseline_result = runner(**reference_payload)
        except VmExecutionError as exc:
            mismatches.append(
                {
                    "input_index": index,
                    "stage": "baseline",
                    "error": exc.kind,
                    "message": str(exc),
                }
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            mismatches.append(
                {
                    "input_index": index,
                    "stage": "baseline",
                    "error": exc.__class__.__name__,
                    "message": str(exc),
                }
            )
            continue

        if candidate_result.return_value != baseline_result.return_value:
            mismatches.append(
                {
                    "input_index": index,
                    "stage": "compare",
                    "candidate": candidate_result.return_value,
                    "baseline": baseline_result.return_value,
                    "args": args_list,
                }
            )

    return BehaviouralDiff(
        equivalent=not mismatches, mismatches=tuple(mismatches), checks_run=checks_run
    )


def _normalise_structure(payload: Any, cfg: DiffConfig) -> str:
    if isinstance(payload, str):
        return _normalise_source(payload, cfg)
    if isinstance(payload, Path):
        return _normalise_source(Path(payload).read_text(encoding="utf-8"), cfg)
    if isinstance(payload, Mapping):
        try:
            return json.dumps(payload, sort_keys=True, indent=2)
        except TypeError:
            # Fallback to deterministic repr for non-JSON serialisable content.
            return _repr_structure(payload)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return _repr_structure(payload)
    return repr(payload)


def _normalise_source(source: str, cfg: DiffConfig) -> str:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return " ".join(source.split()) if cfg.whitespace_insensitive else source
    if cfg.fold_constants:
        tree = _ConstantFolder().visit(tree)
        ast.fix_missing_locations(tree)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


class _ConstantFolder(ast.NodeTransformer):
    """Simplistic constant folder used to smooth out equivalent arithmetic."""

    _BIN_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.MatMult: operator.matmul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
    }
    _UNARY_OPS: dict[type[ast.AST], Callable[[Any], Any]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.Invert: operator.invert,
    }

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # type: ignore[override]
        visited = self.generic_visit(node)
        bin_op = cast(ast.BinOp, visited)
        if isinstance(bin_op.left, ast.Constant) and isinstance(bin_op.right, ast.Constant):
            operator_fn = self._BIN_OPS.get(type(bin_op.op))
            if operator_fn is not None:
                try:
                    value = operator_fn(bin_op.left.value, bin_op.right.value)
                except Exception:
                    return bin_op
                return ast.copy_location(ast.Constant(value=value), bin_op)
        return bin_op

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:  # type: ignore[override]
        visited = self.generic_visit(node)
        unary_op = cast(ast.UnaryOp, visited)
        operand = getattr(unary_op, "operand", None)
        if isinstance(operand, ast.Constant):
            operator_fn = self._UNARY_OPS.get(type(unary_op.op))
            if operator_fn is not None:
                try:
                    value = operator_fn(operand.value)
                except Exception:
                    return unary_op
                return ast.copy_location(ast.Constant(value=value), unary_op)
        return unary_op

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:  # type: ignore[override]
        visited = self.generic_visit(node)
        bool_op = cast(ast.BoolOp, visited)
        constants = [value for value in bool_op.values if isinstance(value, ast.Constant)]
        if len(constants) != len(bool_op.values):
            return bool_op
        if isinstance(bool_op.op, ast.And):
            result = all(const.value for const in constants)
        elif isinstance(bool_op.op, ast.Or):
            result = any(const.value for const in constants)
        else:
            return bool_op
        return ast.copy_location(ast.Constant(value=result), bool_op)


def _repr_structure(payload: Any) -> str:
    if isinstance(payload, Mapping):
        items = ", ".join(
            f"{repr(key)}: {_repr_structure(value)}"
            for key, value in sorted(payload.items(), key=lambda item: repr(item[0]))
        )
        return "{" + items + "}"
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return "[" + ", ".join(_repr_structure(item) for item in payload) + "]"
    return repr(payload)


def _coerce_program(payload: Any, cfg: DiffConfig) -> Mapping[str, Any] | None:
    if isinstance(payload, Mapping):
        module = payload.get("module", payload)
        entry = payload.get("entry") or cfg.entrypoint or "solve"
        limits = payload.get("limits") or cfg.limits
        if not isinstance(module, Mapping):
            return None
        return {
            "module": module,
            "entry": entry,
            "limits": limits,
        }
    return None


def _default_inputs(count: int) -> Sequence[Sequence[Any]]:
    if count <= 0:
        return [()]
    seeds = [(), (0,), (1,), (-1,), (2,)]
    return seeds[: max(1, count)]


def _normalise_args(args: Any) -> list[Any]:
    if args is None:
        return []
    if isinstance(args, Mapping):
        return [args]
    if isinstance(args, Sequence) and not isinstance(args, (str, bytes)):
        return list(args)
    return [args]

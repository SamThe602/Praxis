"""Abstract interpretation based static checks for Praxis bytecode.

A large portion of the synthesis search space can be rejected without running
candidate programs.  This module implements a forward abstract interpreter
covering the container instructions used by the Praxis VM (``List*``, ``Map*``,
``Length``) as well as simple arithmetic ranges.  When we can prove an
operation would trap at runtime – popping from an empty list, reading a missing
map key, etc. – we emit a structured failure that the synthesiser can use for
pruning and the telemetry reporter can surface.

The domain intentionally stays lightweight: registers hold an
:class:`AbstractValue` describing numeric intervals, container cardinalities, or
known map keys.  The lattice is a cartesian product with a distinguished ``Top``
for unknown values.  Transfer functions refine these abstracts but immediately
fall back to ``Top`` once operations mix incompatible types so we avoid noisy
false positives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import contracts as contract_utils
from . import native_bridge

__all__ = [
    "AbstractValue",
    "AnalysisFailure",
    "StaticAnalysisResult",
    "analyze",
]


# ---------------------------------------------------------------------------
# Abstract domain primitives


@dataclass(slots=True, frozen=True)
class Interval:
    """Closed numeric interval. ``None`` represents an unknown bound."""

    lower: Optional[float]
    upper: Optional[float]

    def join(self, other: Interval) -> Interval:
        return Interval(
            lower=_min_optional(self.lower, other.lower),
            upper=_max_optional(self.upper, other.upper),
        )

    def widen_with(self, other: Interval) -> Interval:
        return Interval(
            lower=self.lower if other.lower is not None else None,
            upper=self.upper if other.upper is not None else None,
        )

    def add(self, other: Interval) -> Interval:
        return Interval(_add_bounds(self.lower, other.lower), _add_bounds(self.upper, other.upper))

    def sub(self, other: Interval) -> Interval:
        return Interval(_sub_bounds(self.lower, other.upper), _sub_bounds(self.upper, other.lower))

    def decrement(self) -> Interval:
        return self.sub(Interval(1, 1))

    def increment(self) -> Interval:
        return self.add(Interval(1, 1))

    def floor_at_zero(self) -> Interval:
        lower = self.lower if self.lower is None else max(0, self.lower)
        upper = self.upper if self.upper is None else max(0, self.upper)
        return Interval(lower, upper)

    def is_point(self) -> bool:
        return self.lower is not None and self.lower == self.upper


@dataclass(slots=True, frozen=True)
class ListAbstract:
    """Abstract list characterised by a length interval."""

    length: Interval

    def push(self) -> ListAbstract:
        return ListAbstract(self.length.increment())

    def pop(self) -> ListAbstract:
        return ListAbstract(self.length.decrement().floor_at_zero())

    def min_length(self) -> Optional[float]:
        return self.length.lower


@dataclass(slots=True, frozen=True)
class MapAbstract:
    """Map abstraction tracking known keys and whether unknown keys may exist."""

    known_keys: frozenset[Any] = field(default_factory=frozenset)
    may_have_unknown: bool = False

    def insert_key(self, key: Any | None) -> MapAbstract:
        if key is None:
            return MapAbstract(self.known_keys, True)
        return MapAbstract(self.known_keys | {key}, self.may_have_unknown)

    def has_key(self, key: Any | None) -> bool:
        if key is None or self.may_have_unknown:
            return True
        return key in self.known_keys


@dataclass(slots=True, frozen=True)
class AbstractValue:
    """Value lattice element used for register states."""

    kind: str
    interval: Optional[Interval] = None
    bool_value: Optional[bool] = None
    string_value: Optional[str] = None
    list_value: Optional[ListAbstract] = None
    map_value: Optional[MapAbstract] = None

    # -- constructors -----------------------------------------------------
    @staticmethod
    def top() -> AbstractValue:
        return AbstractValue("unknown")

    @staticmethod
    def bottom() -> AbstractValue:
        return AbstractValue("bottom")

    @staticmethod
    def integer(lower: Optional[float], upper: Optional[float]) -> AbstractValue:
        return AbstractValue("int", interval=Interval(lower, upper))

    @staticmethod
    def integer_point(value: float) -> AbstractValue:
        return AbstractValue.integer(value, value)

    @staticmethod
    def boolean(value: Optional[bool]) -> AbstractValue:
        return AbstractValue("bool", bool_value=value)

    @staticmethod
    def string(value: Optional[str]) -> AbstractValue:
        return AbstractValue("string", string_value=value)

    @staticmethod
    def list_(length: Interval) -> AbstractValue:
        return AbstractValue("list", list_value=ListAbstract(length))

    @staticmethod
    def map_(known: Optional[frozenset[Any]] = None, unknown: bool = False) -> AbstractValue:
        return AbstractValue("map", map_value=MapAbstract(known or frozenset(), unknown))

    # -- helpers ----------------------------------------------------------
    def join(self, other: AbstractValue) -> AbstractValue:
        if self.kind == "bottom":
            return other
        if other.kind == "bottom":
            return self
        if self.kind != other.kind:
            return AbstractValue.top()
        if self.kind == "int" and self.interval and other.interval:
            return AbstractValue("int", interval=self.interval.join(other.interval))
        if self.kind == "bool":
            if self.bool_value == other.bool_value:
                return AbstractValue.boolean(self.bool_value)
            return AbstractValue.boolean(None)
        if self.kind == "string":
            if self.string_value == other.string_value:
                return AbstractValue.string(self.string_value)
            return AbstractValue.string(None)
        if self.kind == "list" and self.list_value and other.list_value:
            return AbstractValue.list_(self.list_value.length.join(other.list_value.length))
        if self.kind == "map" and self.map_value and other.map_value:
            combined = self.map_value.known_keys | other.map_value.known_keys
            may_unknown = self.map_value.may_have_unknown or other.map_value.may_have_unknown
            return AbstractValue.map_(frozenset(combined), may_unknown)
        return AbstractValue.top()

    def as_list(self) -> Optional[ListAbstract]:
        return self.list_value if self.kind == "list" else None

    def as_map(self) -> Optional[MapAbstract]:
        return self.map_value if self.kind == "map" else None

    def constant(self) -> Any | None:
        if self.kind == "int" and self.interval and self.interval.is_point():
            return self.interval.lower
        if self.kind == "bool":
            return self.bool_value if self.bool_value is not None else None
        if self.kind == "string":
            return self.string_value
        return None

    def widen(self, other: AbstractValue) -> AbstractValue:
        if self.kind != other.kind:
            return AbstractValue.top()
        if self.kind == "int" and self.interval and other.interval:
            return AbstractValue("int", interval=self.interval.widen_with(other.interval))
        return self.join(other)


# ---------------------------------------------------------------------------
# Analysis results


@dataclass(slots=True, frozen=True)
class AnalysisFailure:
    """Structured description of a static analysis violation."""

    kind: str
    message: str
    function: str
    instruction_index: Optional[int] = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kind": self.kind,
            "message": self.message,
            "function": self.function,
        }
        if self.instruction_index is not None:
            payload["instruction_index"] = self.instruction_index
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(slots=True)
class FunctionSummary:
    """Lightweight per-function summary consumed by telemetry."""

    final_registers: dict[int, AbstractValue]


@dataclass(slots=True)
class StaticAnalysisResult:
    """Aggregate outcome of the static analysis pass."""

    failures: tuple[AnalysisFailure, ...]
    summaries: Mapping[str, FunctionSummary]

    @property
    def ok(self) -> bool:
        return not self.failures


# ---------------------------------------------------------------------------
# Public entry point


def analyze(program: Any) -> StaticAnalysisResult:
    """Run the static analyzer over ``program`` leveraging the native bridge when available."""

    if native_bridge.should_use("static_analysis"):
        candidate = _prepare_native_static_payload(program)
        if candidate is not None:
            request, labels = candidate
            try:
                response = native_bridge.static_analyze(request)
            except native_bridge.NativeBridgeError:
                native_bridge.record_fallback("static_analysis")
            else:
                return _build_native_result(response, labels)
        else:
            native_bridge.record_fallback("static_analysis")

    return _analyze_python(program)


def _analyze_python(program: Any) -> StaticAnalysisResult:
    """Pure-Python analysis fallback used when the native fast path is disabled or unavailable."""

    analyzer = _Analyzer(program)
    return analyzer.run()


# ---------------------------------------------------------------------------
# Internal implementation


@dataclass(slots=True)
class _Instruction:
    opcode: str
    operands: Mapping[str, Any]


@dataclass(slots=True)
class _Function:
    name: str
    instructions: Sequence[_Instruction]
    contracts: Optional[contract_utils.FunctionContracts] = None


class _Analyzer:
    def __init__(self, program: Any) -> None:
        self.program = program
        self.failures: list[AnalysisFailure] = []
        self.summaries: dict[str, FunctionSummary] = {}

    def run(self) -> StaticAnalysisResult:
        for function in self._iter_functions():
            summary = self._analyze_function(function)
            self.summaries[function.name] = summary
        return StaticAnalysisResult(tuple(self.failures), self.summaries)

    # ------------------------------------------------------------------
    # Normalisation helpers

    def _iter_functions(self) -> Iterable[_Function]:
        functions: Iterable[Any]
        if isinstance(self.program, Mapping):
            functions = self.program.get("functions", [])
        else:
            functions = getattr(self.program, "functions", [])

        contracts_index = self._extract_contract_index(self.program)

        for entry in functions:
            if isinstance(entry, Mapping):
                name = entry.get("name", "<anonymous>")
                instructions = [
                    _Instruction(
                        str(instr.get("opcode")),
                        self._normalise_operands(instr.get("operands", {})),
                    )
                    for instr in entry.get("instructions", [])
                ]
                contract_meta = contracts_index.get(name)
            else:
                name = getattr(entry, "name", "<anonymous>")
                raw_instructions = getattr(entry, "instructions", [])
                instructions = [
                    _Instruction(str(instr.opcode), getattr(instr, "operands", {}))
                    for instr in raw_instructions
                ]
                contract_meta = contracts_index.get(name)
            yield _Function(name=name, instructions=instructions, contracts=contract_meta)

    def _extract_contract_index(self, program: Any) -> dict[str, contract_utils.FunctionContracts]:
        index: dict[str, contract_utils.FunctionContracts] = {}
        if isinstance(program, Mapping):
            raw = program.get("contracts")
            if isinstance(raw, Mapping):
                for name, payload in raw.items():
                    converted = _coerce_contracts(name, payload)
                    if converted is not None:
                        index[name] = converted
        return index

    def _normalise_operands(self, operands: Mapping[str, Any]) -> Mapping[str, Any]:
        if not operands:
            return {"kind": "None", "value": None}
        kind = operands.get("kind")
        value = operands.get("value")
        return {"kind": kind, "value": value}

    # ------------------------------------------------------------------
    # Core analysis

    def _analyze_function(self, function: _Function) -> FunctionSummary:
        registers: dict[int, AbstractValue] = {}
        for index, instruction in enumerate(function.instructions):
            self._transfer(function.name, index, instruction, registers)
        if function.contracts is not None:
            self._check_contracts(function.name, function.contracts)
        return FunctionSummary(final_registers=registers)

    def _transfer(
        self,
        function: str,
        index: int,
        instruction: _Instruction,
        registers: dict[int, AbstractValue],
    ) -> None:
        opcode = instruction.opcode
        operands = instruction.operands
        kind = operands.get("kind")
        value = operands.get("value")

        if opcode == "LoadImmediate" and kind == "RegImmediate":
            target, immediate = value
            registers[int(target)] = _abstract_from_immediate(immediate)
            return
        if opcode == "LoadConst" and kind == "RegImmediate":
            target, _ = value
            registers[int(target)] = AbstractValue.top()
            return
        if opcode == "Move" and kind == "RegPair":
            dst, src = value
            registers[int(dst)] = registers.get(int(src), AbstractValue.top())
            return
        if (
            opcode
            in {"Add", "Sub", "Mul", "Div", "Mod", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "And", "Or"}
            and kind == "RegTriple"
        ):
            dst, lhs, rhs = value
            self._handle_binary(opcode, int(dst), int(lhs), int(rhs), registers)
            return
        if opcode == "ListNew" and kind == "RegImmediate":
            target, immediate = value
            length = _immediate_to_int(immediate)
            initial = max(0, length) if length is not None else 0
            registers[int(target)] = AbstractValue.list_(Interval(initial, initial))
            return
        if opcode == "ListPush" and kind == "RegPair":
            list_reg, _ = value
            reg_id = int(list_reg)
            current = registers.get(reg_id, AbstractValue.top())
            list_value = current.as_list()
            if list_value is None:
                if current.kind != "unknown":
                    self._emit_failure(
                        "list_type_error",
                        function,
                        index,
                        f"ListPush expects a list in r{reg_id}",
                    )
                registers[reg_id] = AbstractValue.list_(Interval(0, None))
            else:
                # Natural-number interval lattice where push corresponds to
                # shifting the interval by one.  We deliberately preserve ``None``
                # to keep the transfer monotone when the upper bound is already
                # unbounded.
                registers[reg_id] = AbstractValue.list_(list_value.push().length)
            return
        if opcode == "ListPop" and kind == "RegPair":
            list_reg, target = value
            reg_id = int(list_reg)
            current = registers.get(reg_id, AbstractValue.top())
            list_value = current.as_list()
            if list_value is None:
                self._emit_failure(
                    "list_type_error",
                    function,
                    index,
                    f"ListPop expects a list in r{reg_id}",
                )
                registers[int(target)] = AbstractValue.top()
                registers[reg_id] = AbstractValue.list_(Interval(0, None))
                return
            minimum = list_value.min_length()
            if minimum is not None and minimum <= 0:
                self._emit_failure(
                    "list_underflow",
                    function,
                    index,
                    "ListPop may observe an empty list",
                    detail={"register": reg_id, "min_length": minimum},
                )
            registers[int(target)] = AbstractValue.top()
            registers[reg_id] = AbstractValue.list_(list_value.pop().length)
            return
        if opcode == "MapInsert" and kind == "RegTriple":
            map_reg, key_reg, _ = value
            reg_id = int(map_reg)
            current = registers.get(reg_id, AbstractValue.map_())
            map_value = current.as_map() or MapAbstract()
            key_value = registers.get(int(key_reg), AbstractValue.top()).constant()
            updated = map_value.insert_key(key_value)
            registers[reg_id] = AbstractValue.map_(updated.known_keys, updated.may_have_unknown)
            return
        if opcode == "MapGet" and kind == "RegTriple":
            map_reg, key_reg, target_reg = value
            reg_id = int(map_reg)
            map_value = registers.get(reg_id, AbstractValue.top()).as_map()
            key_value = registers.get(int(key_reg), AbstractValue.top()).constant()
            if map_value is None:
                self._emit_failure(
                    "map_type_error",
                    function,
                    index,
                    f"MapGet expects a map in r{reg_id}",
                )
            else:
                # The powerset lattice over known keys lets us reject lookups
                # that provably miss.  Unknown keys collapse to ``Top`` to keep
                # false positives at bay.
                if not map_value.has_key(key_value):
                    self._emit_failure(
                        "map_missing_key",
                        function,
                        index,
                        "MapGet may fail: key not proven to exist",
                        detail={"register": reg_id, "key": key_value},
                    )
            registers[int(target_reg)] = AbstractValue.top()
            return
        if opcode == "Length" and kind == "RegPair":
            dst_reg, src_reg = value
            container = registers.get(int(src_reg), AbstractValue.top())
            if container.kind == "list" and container.list_value:
                registers[int(dst_reg)] = AbstractValue.integer(
                    container.list_value.length.lower,
                    container.list_value.length.upper,
                )
            elif container.kind == "map" and container.map_value:
                upper = (
                    None
                    if container.map_value.may_have_unknown
                    else len(container.map_value.known_keys)
                )
                registers[int(dst_reg)] = AbstractValue.integer(0, upper)
            else:
                registers[int(dst_reg)] = AbstractValue.top()
            return
        if opcode == "Return":
            return

    def _handle_binary(
        self,
        opcode: str,
        dst: int,
        lhs: int,
        rhs: int,
        registers: dict[int, AbstractValue],
    ) -> None:
        left = registers.get(lhs, AbstractValue.top())
        right = registers.get(rhs, AbstractValue.top())
        if (
            opcode in {"Add", "Sub"}
            and left.kind == right.kind == "int"
            and left.interval
            and right.interval
        ):
            result_interval = (
                left.interval.add(right.interval)
                if opcode == "Add"
                else left.interval.sub(right.interval)
            )
            registers[dst] = AbstractValue("int", interval=result_interval)
            return
        if opcode in {"Eq", "Ne", "Lt", "Le", "Gt", "Ge", "And", "Or"}:
            registers[dst] = AbstractValue.boolean(None)
            return
        registers[dst] = AbstractValue.top()

    # ------------------------------------------------------------------
    # Contract checks

    def _check_contracts(self, function: str, contracts: contract_utils.FunctionContracts) -> None:
        for condition in contracts.preconditions + contracts.postconditions:
            if _condition_is_unsatisfiable(condition.expression):
                self._emit_failure(
                    "contract_unsatisfiable",
                    function,
                    None,
                    f"{condition.kind} condition {condition.description} is unsatisfiable",
                )
        for clause in contracts.requires + contracts.ensures:
            if not clause.resources:
                self._emit_failure(
                    "contract_empty_resource",
                    function,
                    None,
                    f"@{clause.kind} must reference at least one resource",
                )

    # ------------------------------------------------------------------
    # Failure recording

    def _emit_failure(
        self,
        kind: str,
        function: str,
        instruction_index: Optional[int],
        message: str,
        detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.failures.append(
            AnalysisFailure(
                kind=kind,
                message=message,
                function=function,
                instruction_index=instruction_index,
                detail=detail or {},
            )
        )


# ---------------------------------------------------------------------------
# Helpers


def _min_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    values = [v for v in (a, b) if v is not None]
    return min(values) if values else None


def _max_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
    values = [v for v in (a, b) if v is not None]
    return max(values) if values else None


def _prepare_native_static_payload(
    program: Any,
) -> tuple[Mapping[str, Any], Mapping[int, Mapping[str, Any]]] | None:
    if not isinstance(program, Mapping):
        return None
    candidate = program.get("native_static")
    if not isinstance(candidate, Mapping):
        return None

    payload = dict(candidate)
    raw_labels = payload.pop("labels", {})
    labels: dict[int, Mapping[str, Any]] = {}
    if isinstance(raw_labels, Mapping):
        for key, meta in raw_labels.items():
            subject = _coerce_int(key)
            if subject is None or not isinstance(meta, Mapping):
                continue
            labels[subject] = dict(meta)
    return payload, labels


def _build_native_result(
    response: Mapping[str, Any], labels: Mapping[int, Mapping[str, Any]]
) -> StaticAnalysisResult:
    violations: list[AnalysisFailure] = []
    entries = response.get("violations") if isinstance(response, Mapping) else None
    if isinstance(entries, Iterable):
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            subject = _coerce_int(entry.get("subject"))
            metadata = labels.get(subject, {}) if subject is not None else {}
            function = str(metadata.get("function", "<native>"))
            instruction_index_raw = metadata.get("instruction_index")
            instruction_index = (
                int(instruction_index_raw) if isinstance(instruction_index_raw, int) else None
            )
            detail: dict[str, Any] = {
                "subject": subject,
                "observed": entry.get("observed"),
                "expected": entry.get("expected"),
            }
            if metadata:
                detail["metadata"] = dict(metadata)
            if subject is None:
                detail["raw_subject"] = entry.get("subject")
            violations.append(
                AnalysisFailure(
                    kind=str(entry.get("kind", "native_violation")),
                    message=str(entry.get("message", "")),
                    function=function,
                    instruction_index=instruction_index,
                    detail=detail,
                )
            )
    return StaticAnalysisResult(tuple(violations), {})


def _coerce_int(value: Any) -> Optional[int]:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _add_bounds(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a + b


def _sub_bounds(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def _abstract_from_immediate(payload: Any) -> AbstractValue:
    literal = _decode_immediate(payload)
    if isinstance(literal, bool):
        return AbstractValue.boolean(literal)
    if isinstance(literal, int):
        return AbstractValue.integer_point(literal)
    if isinstance(literal, float):
        return AbstractValue.integer(literal, literal)
    if isinstance(literal, str):
        return AbstractValue.string(literal)
    return AbstractValue.top()


def _decode_immediate(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        if "Scalar" in payload:
            literal = payload["Scalar"]
            if isinstance(literal, Mapping):
                if "Int" in literal:
                    return int(literal["Int"])
                if "Float" in literal:
                    return float(literal["Float"])
                if "Bool" in literal:
                    return bool(literal["Bool"])
                if "String" in literal:
                    return str(literal["String"])
                if "Unit" in literal:
                    return None
        if "Constant" in payload:
            return None
    return payload


def _immediate_to_int(payload: Any) -> Optional[int]:
    value = _decode_immediate(payload)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    return None


def _coerce_contracts(name: str, payload: Any) -> Optional[contract_utils.FunctionContracts]:
    if isinstance(payload, contract_utils.FunctionContracts):
        return payload
    if isinstance(payload, Mapping):
        pre = payload.get("preconditions") or payload.get("pre") or []
        post = payload.get("postconditions") or payload.get("post") or []
        requires = payload.get("requires") or []
        ensures = payload.get("ensures") or []
        pre_conditions = tuple(
            contract_utils.ContractCondition("pre", cond, str(cond), None) for cond in pre
        )
        post_conditions = tuple(
            contract_utils.ContractCondition("post", cond, str(cond), None) for cond in post
        )
        req_clauses = tuple(
            contract_utils.ResourceClause(
                "requires",
                _normalise_resource_tuple(clause),
                None,
            )
            for clause in requires
        )
        ens_clauses = tuple(
            contract_utils.ResourceClause(
                "ensures",
                _normalise_resource_tuple(clause),
                None,
            )
            for clause in ensures
        )
        return contract_utils.FunctionContracts(
            function=name,
            preconditions=pre_conditions,
            postconditions=post_conditions,
            requires=req_clauses,
            ensures=ens_clauses,
        )
    return None


def _condition_is_unsatisfiable(expression: object) -> bool:
    if isinstance(expression, bool):
        return not expression
    try:
        from packages.dsl import ast as dsl_ast
    except Exception:  # pragma: no cover - conservative fallback
        return False
    if isinstance(expression, dsl_ast.Literal) and expression.literal_type == "bool":
        return expression.value is False
    return False


def _normalise_resource_tuple(clause: Any) -> tuple[str, ...]:
    if isinstance(clause, (list, tuple, set)):  # common metadata shape
        return tuple(str(item) for item in clause)
    return (str(clause),)

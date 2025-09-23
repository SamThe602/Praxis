"""Hindley–Milner type inference and contract harvesting for the Praxis DSL.

This module performs a dedicated semantic pass over the parser AST and annotates
nodes with static type information.  The implementation intentionally embraces a
classic Hindley–Milner discipline so that DSL authors obtain predictable,
principled inference behaviour (uniform substitution, principal types, and
rank-1 polymorphism via ``let`` generalisation).  The checker is written in pure
Python to keep the pass debuggable and easy to extend, while remaining fast
enough for the relatively small DSL programs handled in unit tests.

Key responsibilities handled here:

* Construct and maintain a polymorphic type environment with lexical scoping.
* Infer types for expressions/statements, including literals, control-flow,
  lambdas, comprehensions, and user-defined/builtin calls.
* Respect explicit annotations, raising informative errors when the inferred
  types disagree with declarations.
* Collect contract annotations (``@pre``, ``@post``, ``@requires``,
  ``@ensures``) as structured metadata so later passes (verifier/runtime) can
  enforce them without re-parsing decorator arguments.

The module is intentionally self-contained.  It avoids any dependency on the
parser beyond the AST definitions so the pass can be reused from tests or other
pipelines without needing to invoke the full grammar again.
"""

from __future__ import annotations

import string
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Iterator, Mapping, Optional, Sequence, cast

from . import ast

__all__ = [
    "InferenceResult",
    "Type",
    "TypeBinding",
    "TypeEnv",
    "TypeOperator",
    "TypeScheme",
    "TypeSystemError",
    "TypeVariable",
    "infer",
    "format_type",
]


# ---------------------------------------------------------------------------
# Error types


class TypeSystemError(RuntimeError):
    """Raised when semantic typing fails."""

    def __init__(self, message: str, *, node: Optional[ast.Node] = None) -> None:
        if node and node.span:
            span = node.span
            prefix = f"{span.start_line}:{span.start_column}: "
            super().__init__(prefix + message)
        else:
            super().__init__(message)


# ---------------------------------------------------------------------------
# Type representation (classic HM operators/variables)


class Type:
    """Structural base class used for type annotations."""


class TypeVariable(Type):
    """Unification variable used by the Hindley–Milner solver.

    ``instance`` implements the union-find representative for the current
    substitution, while ``level`` tracks the lexical scope depth where the
    variable was created.  The latter allows us to generalise only variables
    that were introduced in strictly nested scopes (rank-1 polymorphism).
    """

    __slots__ = ("id", "instance", "level", "hint")

    _counter = 0

    def __init__(self, *, level: int, hint: str | None = None) -> None:
        self.id = TypeVariable._counter
        TypeVariable._counter += 1
        self.instance: Optional[Type] = None
        self.level = level
        self.hint = hint

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        suffix = f":{self.hint}" if self.hint else ""
        if self.instance is not None:
            return f"TVar({self.id}{suffix}={self.instance!r})"
        return f"TVar({self.id}{suffix})"


class TypeOperator(Type):
    """General type constructor (e.g. ``list`` or function types)."""

    __slots__ = ("name", "types")

    def __init__(self, name: str, types: Sequence[Type] = ()) -> None:
        self.name = name
        self.types: tuple[Type, ...] = tuple(types)

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        if not self.types:
            return f"TypeOperator({self.name})"
        joined = ", ".join(repr(t) for t in self.types)
        return f"TypeOperator({self.name}, [{joined}])"


class TypeLiteral(Type):
    """Opaque literal used for things like sized arrays or resource keys."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"TypeLiteral({self.value!r})"


TypeLike = Type | TypeVariable | TypeOperator | TypeLiteral


@dataclass(slots=True)
class TypeScheme:
    """Polymorphic type with a list of generalised variables."""

    variables: tuple[TypeVariable, ...]
    type: Type


@dataclass(slots=True)
class TypeBinding:
    """Binding stored in the environment (type + mutability metadata)."""

    scheme: TypeScheme
    mutable: bool
    origin: Optional[object] = None


@dataclass(slots=True)
class TypeEnv:
    """Lexically-scoped environment supporting HM generalisation."""

    bindings: Dict[str, TypeBinding] = field(default_factory=dict)
    parent: Optional[TypeEnv] = None
    level: int = 0

    def child(self) -> "TypeEnv":
        return TypeEnv(parent=self, level=self.level + 1)

    def define(self, name: str, binding: TypeBinding) -> None:
        self.bindings[name] = binding

    def lookup(self, name: str) -> TypeBinding:
        env: Optional[TypeEnv] = self
        while env is not None:
            if name in env.bindings:
                return env.bindings[name]
            env = env.parent
        raise KeyError(name)


@dataclass(slots=True)
class InferenceResult:
    """Aggregate outcome of a successful type inference run."""

    root_type: Type
    node_types: dict[int, Type]
    environment: TypeEnv
    contract_metadata: dict[str, dict[str, list[list[str]]]]

    def type_of(self, node: ast.Node) -> Type:
        """Convenience accessor returning the (pruned) type of ``node``."""

        return self.node_types[id(node)]


# ---------------------------------------------------------------------------
# Helper constructors for frequently used types


UNIT = TypeOperator("unit")
BOOL = TypeOperator("bool")
INT = TypeOperator("int")
FLOAT = TypeOperator("float")
STRING = TypeOperator("string")


def function_type(parameters: Sequence[Type], result: Type) -> TypeOperator:
    return TypeOperator("fn", tuple(parameters) + (result,))


def list_type(element: Type) -> TypeOperator:
    return TypeOperator("list", (element,))


def set_type(element: Type) -> TypeOperator:
    return TypeOperator("set", (element,))


def map_type(key: Type, value: Type) -> TypeOperator:
    return TypeOperator("map", (key, value))


def tuple_type(elements: Sequence[Type]) -> TypeOperator:
    return TypeOperator("tuple", tuple(elements))


# ---------------------------------------------------------------------------
# Pretty-printing


def format_type(typ: TypeLike) -> str:
    """Return a human-readable representation used in tests and errors."""

    mapping: Dict[TypeVariable, str] = {}
    counter = 0

    def pretty(t: TypeLike) -> str:
        nonlocal counter
        t = prune(t)
        if isinstance(t, TypeVariable):
            if t.instance is not None:
                return pretty(t.instance)
            if t not in mapping:
                if counter < len(string.ascii_lowercase):
                    mapping[t] = f"'{string.ascii_lowercase[counter]}"
                else:
                    mapping[t] = f"t{counter}"
                counter += 1
            return mapping[t]
        if isinstance(t, TypeOperator):
            if t.name == "fn":
                if len(t.types) == 1:
                    return f"() -> {pretty(t.types[0])}"
                params = ", ".join(pretty(arg) for arg in t.types[:-1])
                return f"({params}) -> {pretty(t.types[-1])}"
            if t.name in {"list", "set"}:
                return f"{t.name}<{pretty(t.types[0])}>"
            if t.name == "map":
                key, value = t.types
                return f"map<{pretty(key)}, {pretty(value)}>"
            if t.name == "tuple":
                inside = ", ".join(pretty(arg) for arg in t.types)
                return f"({inside})"
            if not t.types:
                return t.name
            inside = ", ".join(pretty(arg) for arg in t.types)
            return f"{t.name}<{inside}>"
        if isinstance(t, TypeLiteral):
            return t.value
        raise AssertionError(f"Unknown type node: {t!r}")

    return pretty(typ)


# ---------------------------------------------------------------------------
# Core inference engine


def infer(node: ast.Node) -> InferenceResult:
    """Infer types for ``node`` and return the accumulated metadata."""

    checker = _TypeChecker()
    result = checker.analyse(node)
    return result


def prune(typ: TypeLike) -> Type:
    """Chase union-find representatives to yield the most specific type."""

    if isinstance(typ, TypeVariable) and typ.instance is not None:
        typ.instance = prune(typ.instance)
        return typ.instance
    return typ  # type: ignore[return-value]


def occurs_in_type(variable: TypeVariable, typ: TypeLike) -> bool:
    """Return True when ``variable`` occurs inside ``typ`` (occurs check)."""

    typ = prune(typ)
    if typ is variable:
        return True
    if isinstance(typ, TypeOperator):
        return any(occurs_in_type(variable, arg) for arg in typ.types)
    return False


def iter_type_variables(typ: TypeLike) -> Iterator[TypeVariable]:
    typ = prune(typ)
    if isinstance(typ, TypeVariable):
        if typ.instance is None:
            yield typ
        else:
            yield from iter_type_variables(typ.instance)
    elif isinstance(typ, TypeOperator):
        for arg in typ.types:
            yield from iter_type_variables(arg)


@dataclass(slots=True)
class _FunctionContext:
    name: str
    return_type: Type


class _TypeChecker:
    """Implements a single HM inference pass over the Praxis DSL AST."""

    def __init__(self) -> None:
        self.current_level = 0
        self.node_types: dict[int, Type] = {}
        self.contracts: dict[str, dict[str, list[list[str]]]] = {}
        self.function_signatures: dict[str, TypeScheme] = {}
        self.function_stack: list[_FunctionContext] = []

    # ------------------------------------------------------------------
    # Environment/variable helpers

    def analyse(self, node: ast.Node) -> InferenceResult:
        env = self._initial_environment()
        root_type = self._infer(node, env)
        return InferenceResult(
            root_type=root_type,
            node_types=self.node_types,
            environment=env,
            contract_metadata=self.contracts,
        )

    def _initial_environment(self) -> TypeEnv:
        env = TypeEnv(level=0)
        for name, scheme in self._builtin_schemes().items():
            env.define(name, TypeBinding(scheme=scheme, mutable=False))
        return env

    def _builtin_schemes(self) -> Mapping[str, TypeScheme]:
        def fresh() -> TypeVariable:
            return self._fresh_variable("t", level=0)

        # len: forall a. list<a> -> int
        a = fresh()
        len_scheme = TypeScheme((a,), function_type((list_type(a),), INT))
        # sorted: forall a. list<a> -> list<a>
        b = fresh()
        sorted_scheme = TypeScheme((b,), function_type((list_type(b),), list_type(b)))
        # enumerate: forall a. list<a> -> list<tuple<int, a>>
        c = fresh()
        enumerate_scheme = TypeScheme(
            (c,),
            function_type((list_type(c),), list_type(tuple_type((INT, c)))),
        )
        # sum/min/max: list<int>/list<float> simplified to numeric lists.
        num = fresh()
        numeric_list = list_type(num)
        numeric_scheme = TypeScheme((num,), function_type((numeric_list,), num))
        # range: (int, int?) -> list<int>.  Simplify to int -> int -> list<int>
        range_scheme = TypeScheme(
            tuple(),
            function_type((INT, INT), list_type(INT)),
        )
        return {
            "len": len_scheme,
            "sorted": sorted_scheme,
            "enumerate": enumerate_scheme,
            "sum": numeric_scheme,
            "min": numeric_scheme,
            "max": numeric_scheme,
            "range": range_scheme,
        }

    def _fresh_variable(
        self, hint: str | None = None, *, level: Optional[int] = None
    ) -> TypeVariable:
        return TypeVariable(level=self.current_level if level is None else level, hint=hint)

    @contextmanager
    def _new_level(self) -> Iterator[None]:
        self.current_level += 1
        try:
            yield
        finally:
            self.current_level -= 1

    def _generalise(self, typ: Type, env: TypeEnv) -> TypeScheme:
        pruned = prune(typ)

        quantifiable: dict[int, TypeVariable] = {}

        for var in iter_type_variables(pruned):
            if var.instance is not None:
                continue
            if var.level > env.level:
                quantifiable[var.id] = var

        return TypeScheme(tuple(quantifiable.values()), pruned)

    def _instantiate(self, scheme: TypeScheme, env: TypeEnv) -> Type:
        substitutions: dict[int, TypeVariable] = {}

        def replace(t: TypeLike) -> Type:
            t = prune(t)
            if isinstance(t, TypeVariable):
                if t.instance is not None:
                    return replace(t.instance)
                if t in scheme.variables:
                    if t.id not in substitutions:
                        substitutions[t.id] = self._fresh_variable(level=env.level + 1)
                    return substitutions[t.id]
                return t
            if isinstance(t, TypeOperator):
                return TypeOperator(t.name, tuple(replace(arg) for arg in t.types))
            return t  # TypeLiteral

        return replace(scheme.type)

    # ------------------------------------------------------------------
    # Unification

    def _unify(self, left: TypeLike, right: TypeLike, node: Optional[ast.Node] = None) -> None:
        a = prune(left)
        b = prune(right)
        if isinstance(a, TypeVariable):
            if a is b:
                return
            if occurs_in_type(a, b):
                raise TypeSystemError("Recursive type detected", node=node)
            self._update_level(b, a.level)
            a.instance = b
            return
        if isinstance(b, TypeVariable):
            self._unify(b, a, node=node)
            return
        if isinstance(a, TypeOperator) and isinstance(b, TypeOperator):
            if a.name != b.name or len(a.types) != len(b.types):
                raise TypeSystemError(
                    f"Type mismatch: expected {format_type(a)} but found {format_type(b)}",
                    node=node,
                )
            for sub_a, sub_b in zip(a.types, b.types):
                self._unify(sub_a, sub_b, node=node)
            return
        if isinstance(a, TypeLiteral) and isinstance(b, TypeLiteral):
            if a.value != b.value:
                raise TypeSystemError(
                    f"Literal type mismatch: {a.value!r} != {b.value!r}", node=node
                )
            return
        raise TypeSystemError(
            f"Type mismatch: expected {format_type(a)} but found {format_type(b)}", node=node
        )

    def _update_level(self, typ: TypeLike, level: int) -> None:
        typ = prune(typ)
        if isinstance(typ, TypeVariable):
            if typ.instance is not None:
                self._update_level(typ.instance, level)
            else:
                typ.level = min(typ.level, level)
            return
        if isinstance(typ, TypeOperator):
            for arg in typ.types:
                self._update_level(arg, level)

    # ------------------------------------------------------------------
    # Inference dispatch

    def _infer(self, node: ast.Node, env: TypeEnv) -> Type:
        self._sync_level_with_env(env)
        if isinstance(node, ast.Module):
            return self._infer_module(node, env)
        if isinstance(node, ast.FunctionDecl):
            return self._infer_function(node, env)
        if isinstance(node, ast.Block):
            return self._infer_block(node, env)
        if isinstance(node, ast.Let):
            return self._infer_let(node, env)
        if isinstance(node, ast.Assign):
            return self._infer_assign(node, env)
        if isinstance(node, ast.Loop):
            return self._infer_loop(node, env)
        if isinstance(node, ast.Conditional):
            return self._infer_conditional(node, env)
        if isinstance(node, ast.MatchArm):
            return self._infer_match_arm(node, env)
        if isinstance(node, ast.BuiltinCall):
            return self._infer_builtin_call(node, env)
        if isinstance(node, ast.Call):
            return self._infer_call(node, env)
        if isinstance(node, ast.Lambda):
            return self._infer_lambda(node, env)
        if isinstance(node, ast.Literal):
            return self._infer_literal(node, env)
        if isinstance(node, ast.BinaryOp):
            return self._infer_binary(node, env)
        if isinstance(node, ast.UnaryOp):
            return self._infer_unary(node, env)
        if isinstance(node, ast.Comprehension):
            return self._infer_comprehension(node, env)
        # Fallback for statements that do not affect typing (e.g. expression statements)
        result = UNIT
        self._annotate(node, result)
        return result

    def _sync_level_with_env(self, env: TypeEnv) -> None:
        if self.current_level < env.level:
            self.current_level = env.level

    # ------------------------------------------------------------------
    # Module/function handling

    def _infer_module(self, module: ast.Module, env: TypeEnv) -> Type:
        # Pre-register function signatures for recursive references.
        for fn in module.functions:
            with self._new_level():
                signature = self._prepare_function_signature(fn, env)
            env.define(fn.name, TypeBinding(signature, mutable=False, origin=fn))
        for fn in module.functions:
            self._infer(fn, env)
        self._annotate(module, UNIT)
        return UNIT

    def _prepare_function_signature(self, fn: ast.FunctionDecl, env: TypeEnv) -> TypeScheme:
        parameter_types: list[Type] = []
        for param in fn.parameters:
            param_type = self._fresh_variable(param.name, level=env.level + 1)
            if param.type_annotation:
                annotated = self._parse_annotation(param.type_annotation)
                self._unify(param_type, annotated, node=fn)
            parameter_types.append(param_type)
        return_type = self._fresh_variable(f"{fn.name}_ret", level=env.level + 1)
        if fn.return_type:
            annotated_ret = self._parse_annotation(fn.return_type)
            self._unify(return_type, annotated_ret, node=fn)
        fn_type = function_type(parameter_types, return_type)
        unique_vars = tuple(dict.fromkeys(iter_type_variables(fn_type)))
        scheme = TypeScheme(unique_vars, fn_type)
        self.function_signatures[fn.name] = scheme
        return scheme

    def _infer_function(self, fn: ast.FunctionDecl, env: TypeEnv) -> Type:
        binding = env.lookup(fn.name)
        fn_type = self._instantiate(binding.scheme, env)
        assert isinstance(fn_type, TypeOperator)
        params = list(fn_type.types[:-1])
        return_type = fn_type.types[-1]

        local_env = env.child()

        parameter_metadata: dict[str, Type] = {}

        with self._function_context(fn.name, return_type):
            for param, param_type in zip(fn.parameters, params):
                local_env.define(
                    param.name,
                    TypeBinding(
                        scheme=TypeScheme(tuple(), param_type),
                        mutable=True,
                        origin=param,
                    ),
                )
                parameter_metadata[param.name] = prune(param_type)
                if param.default is not None:
                    default_type = self._infer(param.default, local_env)
                    self._unify(default_type, param_type, node=fn)

            self._harvest_contracts(fn, local_env)
            self._infer(fn.body, local_env)

        generalised = self._generalise(fn_type, env)
        env.define(fn.name, TypeBinding(generalised, mutable=False, origin=fn))
        if parameter_metadata:
            fn.metadata["parameter_types"] = parameter_metadata
        fn.metadata["return_type_inferred"] = prune(return_type)
        self._annotate(fn, fn_type)
        return fn_type

    @contextmanager
    def _function_context(self, name: str, return_type: Type) -> Iterator[None]:
        self.function_stack.append(_FunctionContext(name=name, return_type=return_type))
        try:
            yield
        finally:
            self.function_stack.pop()

    def _harvest_contracts(self, fn: ast.FunctionDecl, env: TypeEnv) -> None:
        allowed = {"pre", "post", "requires", "ensures"}
        collected: dict[str, list[list[str]]] = {key: [] for key in allowed}
        for contract in fn.contracts:
            if contract.name not in allowed:
                raise TypeSystemError(f"Unknown contract @{contract.name}", node=contract)
            resources: list[str] = []
            for arg in contract.arguments:
                literal = self._expect_literal_string(arg)
                resources.append(literal)
            collected[contract.name].append(resources)
        if any(collected[name] for name in allowed):
            self.contracts[fn.name] = collected
            fn.metadata["contracts"] = collected

    def _expect_literal_string(self, expr: ast.Expression) -> str:
        if isinstance(expr, ast.Literal) and expr.literal_type == "string":
            return str(expr.value)
        if isinstance(expr, ast.Literal) and expr.literal_type == "identifier":
            return str(expr.value)
        raise TypeSystemError(
            "Contract arguments must be string literals or identifiers",
            node=expr,
        )

    # ------------------------------------------------------------------
    # Statements / blocks

    def _infer_block(self, block: ast.Block, env: TypeEnv) -> Type:
        scope = env.child()
        with self._new_level():
            for stmt in block.statements:
                self._infer(stmt, scope)
        self._annotate(block, UNIT)
        return UNIT

    def _infer_let(self, stmt: ast.Let, env: TypeEnv) -> Type:
        with self._new_level():
            value_type = self._infer(stmt.value, env)
        if stmt.type_annotation:
            annotated = self._parse_annotation(stmt.type_annotation)
            self._unify(value_type, annotated, node=stmt)
        pruned_value = prune(value_type)
        scheme = (
            TypeScheme(tuple(), pruned_value)
            if stmt.mutable
            else self._generalise(pruned_value, env)
        )
        env.define(stmt.name, TypeBinding(scheme=scheme, mutable=stmt.mutable, origin=stmt))
        stmt.metadata["binding_type"] = pruned_value
        self._annotate(stmt, UNIT)
        return UNIT

    def _infer_assign(self, stmt: ast.Assign, env: TypeEnv) -> Type:
        try:
            binding = env.lookup(stmt.target)
        except KeyError as exc:  # pragma: no cover - defensive
            raise TypeSystemError(f"Unknown identifier {stmt.target!r}", node=stmt) from exc
        if not binding.mutable:
            raise TypeSystemError(f"Cannot assign to immutable binding '{stmt.target}'", node=stmt)
        target_type = self._instantiate(binding.scheme, env)
        value_type = self._infer(stmt.value, env)
        self._unify(target_type, value_type, node=stmt)
        binding.scheme = TypeScheme(tuple(), prune(target_type))
        self._annotate(stmt, UNIT)
        return UNIT

    def _infer_loop(self, loop: ast.Loop, env: TypeEnv) -> Type:
        if loop.kind == "for":
            if loop.iterable is None or loop.target is None:
                raise TypeSystemError("Malformed for loop", node=loop)
            iterable_expr = loop.iterable
            target_name = cast(str, loop.target)
            iterable_type = self._infer(iterable_expr, env)
            element_type = self._fresh_variable("iter_element")
            self._unify(iterable_type, list_type(element_type), node=loop)
            scope = env.child()
            scope.define(
                target_name,
                TypeBinding(TypeScheme(tuple(), element_type), mutable=True, origin=loop),
            )
            self._infer(loop.body, scope)
        elif loop.kind == "while":
            if loop.condition is None:
                raise TypeSystemError("Malformed while loop", node=loop)
            condition_expr = loop.condition
            condition_type = self._infer(condition_expr, env)
            self._unify(condition_type, BOOL, node=loop)
            self._infer(loop.body, env.child())
        else:  # pragma: no cover - parser restricts kinds
            raise TypeSystemError(f"Unknown loop kind {loop.kind}", node=loop)
        self._annotate(loop, UNIT)
        return UNIT

    def _infer_conditional(self, node: ast.Conditional, env: TypeEnv) -> Type:
        if node.kind == "if":
            for test, block in node.branches:
                if test is not None:
                    condition_type = self._infer(test, env)
                    self._unify(condition_type, BOOL, node=test)
                self._infer(block, env.child())
            self._annotate(node, UNIT)
            return UNIT
        if node.kind == "match":
            target_type = self._infer(node.test, env)
            for arm in node.arms:
                self._infer_match_arm(arm, env.child(), target_type)
            self._annotate(node, UNIT)
            return UNIT
        raise TypeSystemError(f"Unknown conditional kind {node.kind}", node=node)

    def _infer_match_arm(
        self, arm: ast.MatchArm, env: TypeEnv, expected: Type | None = None
    ) -> Type:
        if expected is None:
            expected = self._fresh_variable("match_target")
        pattern_type = self._pattern_type(arm.pattern.text)
        self._unify(pattern_type, expected, node=arm)
        if arm.guard is not None:
            guard_type = self._infer(arm.guard, env)
            self._unify(guard_type, BOOL, node=arm.guard)
        self._infer(arm.body, env)
        self._annotate(arm, UNIT)
        return UNIT

    def _pattern_type(self, text: str) -> Type:
        if text == "_":
            return self._fresh_variable("wildcard")
        if text in {"true", "false"}:
            return BOOL
        normalized = text.replace("_", "")
        if normalized.isdigit() or (normalized.startswith("-") and normalized[1:].isdigit()):
            return INT
        return self._fresh_variable(text)

    # ------------------------------------------------------------------
    # Calls

    def _infer_call(self, node: ast.Call, env: TypeEnv) -> Type:
        binding = env.lookup(node.function)
        fn_type = self._instantiate(binding.scheme, env)
        return self._apply_callable(fn_type, node.arguments, env, node)

    def _infer_builtin_call(self, node: ast.BuiltinCall, env: TypeEnv) -> Type:
        if node.name == "return":
            return self._infer_return(node, env)
        if node.name == "range" and len(node.arguments) == 1:
            # Support Python-style single-argument range by desugaring to (0, arg)
            zero = ast.Literal(literal_type="int", value=0)
            node.arguments.insert(0, zero)
        try:
            binding = env.lookup(node.name)
        except KeyError as exc:
            raise TypeSystemError(f"Unknown builtin {node.name}", node=node) from exc
        fn_type = self._instantiate(binding.scheme, env)
        return self._apply_callable(fn_type, node.arguments, env, node)

    def _infer_return(self, node: ast.BuiltinCall, env: TypeEnv) -> Type:
        if not self.function_stack:
            raise TypeSystemError("'return' outside of function", node=node)
        current = self.function_stack[-1]
        if not node.arguments:
            self._unify(current.return_type, UNIT, node=node)
        elif len(node.arguments) == 1:
            value_type = self._infer(node.arguments[0], env)
            self._unify(current.return_type, value_type, node=node)
        else:  # pragma: no cover - parser disallows
            raise TypeSystemError("Return takes zero or one argument", node=node)
        self._annotate(node, UNIT)
        return UNIT

    def _apply_callable(
        self,
        fn_type: Type,
        arguments: Sequence[ast.Expression],
        env: TypeEnv,
        node: ast.Node,
    ) -> Type:
        fn_type = prune(fn_type)
        if not isinstance(fn_type, TypeOperator) or fn_type.name != "fn":
            raise TypeSystemError(f"{format_type(fn_type)} is not callable", node=node)
        param_types = list(fn_type.types[:-1])
        if len(param_types) != len(arguments):
            raise TypeSystemError(
                f"Expected {len(param_types)} arguments but received {len(arguments)}",
                node=node,
            )
        for arg_expr, param_type in zip(arguments, param_types):
            arg_type = self._infer(arg_expr, env)
            self._unify(param_type, arg_type, node=arg_expr)
        result = fn_type.types[-1]
        self._annotate(node, result)
        return result

    # ------------------------------------------------------------------
    # Expressions

    def _infer_lambda(self, node: ast.Lambda, env: TypeEnv) -> Type:
        param_types: list[Type] = []
        scope = env.child()
        for parameter in node.parameters:
            param_type = self._fresh_variable(parameter.name)
            scope.define(
                parameter.name,
                TypeBinding(TypeScheme(tuple(), param_type), mutable=True, origin=node),
            )
            param_types.append(param_type)
        result_type = self._infer(node.body, scope)
        fn_type = function_type(param_types, result_type)
        self._annotate(node, fn_type)
        return fn_type

    def _infer_literal(self, node: ast.Literal, env: TypeEnv) -> Type:
        literal_type = node.literal_type
        result: Type
        if literal_type == "int":
            result = INT
        elif literal_type == "float":
            result = FLOAT
        elif literal_type == "bool":
            result = BOOL
        elif literal_type == "string":
            result = STRING
        elif literal_type == "identifier":
            binding = env.lookup(str(node.value))
            result = self._instantiate(binding.scheme, env)
        elif literal_type == "list":
            values = cast(list[ast.Expression], node.value)
            if not values:
                element_type: Type = self._fresh_variable("list_element")
            else:
                element_type = self._infer(values[0], env)
                for expr in values[1:]:
                    current = self._infer(expr, env)
                    self._unify(element_type, current, node=expr)
            result = list_type(element_type)
        elif literal_type == "map":
            entries = cast(list[tuple[ast.Expression, ast.Expression]], node.value)
            if not entries:
                key_type: Type = self._fresh_variable("map_key")
                value_type: Type = self._fresh_variable("map_value")
            else:
                first_key, first_value = entries[0]
                key_type = self._infer(first_key, env)
                value_type = self._infer(first_value, env)
                for key_expr, value_expr in entries[1:]:
                    k = self._infer(key_expr, env)
                    v = self._infer(value_expr, env)
                    self._unify(key_type, k, node=key_expr)
                    self._unify(value_type, v, node=value_expr)
            result = map_type(key_type, value_type)
        elif literal_type == "tuple":
            items = cast(list[ast.Expression], node.value)
            element_types = [self._infer(expr, env) for expr in items]
            result = tuple_type(element_types)
        else:  # pragma: no cover - grammar restricts literal types
            raise TypeSystemError(f"Unknown literal type {literal_type}", node=node)
        self._annotate(node, result)
        return result

    def _infer_binary(self, node: ast.BinaryOp, env: TypeEnv) -> Type:
        operator = node.operator
        left_type = self._infer(node.left, env)
        right_type = self._infer(node.right, env)
        result: Type
        if operator in {"+", "-", "*", "/", "%"}:
            self._unify(left_type, right_type, node=node)
            if operator == "/":
                self._unify(left_type, FLOAT, node=node)
                result = FLOAT
            else:
                self._ensure_numeric(left_type, node=node)
                result = prune(left_type)
        elif operator in {"<", "<=", ">", ">=", "==", "!="}:
            self._unify(left_type, right_type, node=node)
            if operator in {"<", "<=", ">", ">="}:
                self._ensure_numeric(left_type, node=node)
            result = BOOL
        elif operator in {"and", "or"}:
            self._unify(left_type, BOOL, node=node)
            self._unify(right_type, BOOL, node=node)
            result = BOOL
        else:  # pragma: no cover
            raise TypeSystemError(f"Unknown operator {operator}", node=node)
        self._annotate(node, result)
        return result

    def _infer_unary(self, node: ast.UnaryOp, env: TypeEnv) -> Type:
        operand_type = self._infer(node.operand, env)
        result: Type
        if node.operator == "not":
            self._unify(operand_type, BOOL, node=node)
            result = BOOL
        elif node.operator == "-":
            self._ensure_numeric(operand_type, node=node)
            result = operand_type
        else:  # pragma: no cover
            raise TypeSystemError(f"Unknown unary operator {node.operator}", node=node)
        self._annotate(node, result)
        return result

    def _infer_comprehension(self, node: ast.Comprehension, env: TypeEnv) -> Type:
        iterable_type = self._infer(node.iterable, env)
        element_type = self._fresh_variable("comp_element")
        self._unify(iterable_type, list_type(element_type), node=node.iterable)
        scope = env.child()
        scope.define(
            node.target,
            TypeBinding(TypeScheme(tuple(), element_type), mutable=True, origin=node),
        )
        expr_type = self._infer(node.expression, scope)
        if node.condition is not None:
            cond_type = self._infer(node.condition, scope)
            self._unify(cond_type, BOOL, node=node.condition)
        if node.kind == "list":
            result = list_type(expr_type)
        elif node.kind == "set":
            result = set_type(expr_type)
        elif node.kind == "map":
            if (
                not isinstance(expr_type, TypeOperator)
                or expr_type.name != "tuple"
                or len(expr_type.types) != 2
            ):
                raise TypeSystemError(
                    "Map comprehension expressions must yield (key, value)", node=node
                )
            key_type, value_type = expr_type.types
            result = map_type(key_type, value_type)
        else:  # pragma: no cover
            raise TypeSystemError(f"Unknown comprehension kind {node.kind}", node=node)
        self._annotate(node, result)
        return result

    # ------------------------------------------------------------------
    # Misc helpers

    def _ensure_numeric(self, typ: TypeLike, *, node: ast.Node) -> None:
        concrete = prune(typ)
        if concrete in {INT, FLOAT}:
            return
        if isinstance(concrete, TypeVariable) and concrete.instance is None:
            concrete.instance = INT
            return
        raise TypeSystemError("Expected numeric type", node=node)

    def _parse_annotation(self, annotation: str) -> Type:
        tokens = _TypeAnnotationTokenizer(annotation).tokenize()
        parser = _TypeAnnotationParser(tokens)
        parsed = parser.parse()
        return parsed

    def _annotate(self, node: ast.Node, typ: Type) -> None:
        pruned = prune(typ)
        node.metadata["type"] = pruned
        self.node_types[id(node)] = pruned


# ---------------------------------------------------------------------------
# Simple recursive-descent parser for type annotations


@dataclass(slots=True)
class _Token:
    kind: str
    text: str


class _TypeAnnotationTokenizer:
    def __init__(self, source: str) -> None:
        self.source = source

    def tokenize(self) -> list[_Token]:
        tokens: list[_Token] = []
        index = 0
        length = len(self.source)
        while index < length:
            ch = self.source[index]
            if ch.isspace():
                index += 1
                continue
            if ch.isalpha() or ch == "_":
                start = index
                index += 1
                while index < length and (
                    self.source[index].isalnum() or self.source[index] in {"_"}
                ):
                    index += 1
                tokens.append(_Token("IDENT", self.source[start:index]))
                continue
            if ch.isdigit():
                start = index
                index += 1
                while index < length and self.source[index].isdigit():
                    index += 1
                tokens.append(_Token("INT", self.source[start:index]))
                continue
            if ch in "<>,[]":
                tokens.append(_Token(ch, ch))
                index += 1
                continue
            raise TypeSystemError(f"Unexpected character {ch!r} in type annotation")
        tokens.append(_Token("EOF", ""))
        return tokens


class _TypeAnnotationParser:
    def __init__(self, tokens: Sequence[_Token]) -> None:
        self.tokens = list(tokens)
        self.index = 0

    def parse(self) -> Type:
        result = self._parse_type()
        self._expect("EOF")
        return result

    def _peek(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _expect(self, kind: str) -> _Token:
        token = self._peek()
        if token.kind != kind:
            raise TypeSystemError(f"Expected {kind} in type annotation but found {token.kind}")
        return self._advance()

    def _parse_type(self) -> Type:
        token = self._expect("IDENT")
        name = token.text
        arguments: list[Type] = []
        if self._match("<"):
            arguments.append(self._parse_type())
            while self._match(","):
                arguments.append(self._parse_type())
            self._expect(">")
        if self._match("["):
            size_token = self._expect_any({"IDENT", "INT"})
            size_literal = TypeLiteral(size_token.text)
            self._expect("]")
            arguments.append(size_literal)
        if name == "list" and arguments:
            return list_type(arguments[0])
        if name == "set" and arguments:
            return set_type(arguments[0])
        if name == "map" and len(arguments) >= 2:
            return map_type(arguments[0], arguments[1])
        if name == "tuple" and arguments:
            return tuple_type(arguments)
        if not arguments:
            if name == "int":
                return INT
            if name == "float":
                return FLOAT
            if name == "bool":
                return BOOL
            if name == "string":
                return STRING
            if name == "unit":
                return UNIT
        if arguments and isinstance(arguments[-1], TypeLiteral):
            base_args = [arg for arg in arguments if not isinstance(arg, TypeLiteral)]
            result = TypeOperator(name, tuple(base_args) + (arguments[-1],))
            return result
        return TypeOperator(name, tuple(arguments))

    def _match(self, kind: str) -> bool:
        if self._peek().kind == kind:
            self._advance()
            return True
        return False

    def _expect_any(self, kinds: set[str]) -> _Token:
        token = self._peek()
        if token.kind not in kinds:
            expected = ", ".join(sorted(kinds))
            raise TypeSystemError(f"Expected one of {expected} but found {token.kind}")
        return self._advance()

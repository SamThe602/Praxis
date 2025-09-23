"""Lightweight transpiler from the Praxis DSL AST to Python code.

The transpiler is intentionally conservative: it only covers the subset of
constructs currently present in fixtures and round-trip tests.  The generated
Python is not intended for execution in production, but it provides a convenient
debugging aid while developing the grammar and serializer.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, cast

from . import ast


def transpile(node: ast.Node) -> str:
    """Render ``node`` (typically a :class:`ast.Module`) into Python source."""

    if isinstance(node, ast.Module):
        return "\n\n".join(_emit_function(fn) for fn in node.functions)
    if isinstance(node, ast.FunctionDecl):
        return _emit_function(node)
    raise TypeError(f"Unsupported root node {node.node_type}")


def _emit_function(fn: ast.FunctionDecl) -> str:
    header = f"def {fn.name}({_emit_parameter_list(fn.parameters)}):"
    body_lines = _emit_block(fn.body, indent=1)
    return "\n".join([header, *body_lines])


def _emit_parameter_list(params: Iterable[ast.Parameter]) -> str:
    rendered: List[str] = []
    for param in params:
        fragment = param.name
        if param.default is not None:
            fragment += f"={_emit_expression(param.default)}"
        rendered.append(fragment)
    return ", ".join(rendered)


def _emit_block(block: ast.Block, indent: int) -> List[str]:
    lines: List[str] = []
    if not block.statements:
        lines.append(_indent("pass", indent))
        return lines
    for statement in block.statements:
        lines.extend(_emit_statement(statement, indent))
    return lines


def _emit_statement(node: ast.Node, indent: int) -> List[str]:
    if isinstance(node, ast.Let):
        return [_indent(f"{node.name} = {_emit_expression(node.value)}", indent)]
    if isinstance(node, ast.Assign):
        return [_indent(f"{node.target} = {_emit_expression(node.value)}", indent)]
    if isinstance(node, ast.BuiltinCall) and node.name == "return":
        if node.arguments:
            return [_indent(f"return {_emit_expression(node.arguments[0])}", indent)]
        return [_indent("return", indent)]
    if isinstance(node, ast.BuiltinCall):
        return [_indent(f"{node.name}({_emit_arguments(node.arguments)})", indent)]
    if isinstance(node, ast.Call):
        return [_indent(f"{node.function}({_emit_arguments(node.arguments)})", indent)]
    if isinstance(node, ast.Conditional) and node.kind == "if":
        return _emit_if(node, indent)
    if isinstance(node, ast.Conditional) and node.kind == "match":
        return _emit_match(node, indent)
    if isinstance(node, ast.Loop) and node.kind == "for":
        if node.iterable is None or node.target is None:
            return [_indent("# Unsupported statement: ill-formed loop", indent)]
        header = f"for {node.target} in {_emit_expression(node.iterable)}:"
        body = _emit_block(node.body, indent + 1)
        return [_indent(header, indent), *body]
    if isinstance(node, ast.Loop) and node.kind == "while":
        if node.condition is None:
            return [_indent("# Unsupported statement: ill-formed loop", indent)]
        header = f"while {_emit_expression(node.condition)}:"
        body = _emit_block(node.body, indent + 1)
        return [_indent(header, indent), *body]
    if isinstance(node, ast.Block):
        return _emit_block(node, indent)
    return [_indent(f"# Unsupported statement: {node.node_type}", indent)]


def _emit_if(node: ast.Conditional, indent: int) -> List[str]:
    lines: List[str] = []
    for idx, (condition, block) in enumerate(node.branches):
        if idx == 0:
            prefix = "if"
        elif condition is None:
            lines.append(_indent("else:", indent))
            lines.extend(_emit_block(block, indent + 1))
            continue
        else:
            prefix = "elif"
        if condition is None:
            raise ValueError("Missing condition for non-final branch in conditional")
        header = f"{prefix} {_emit_expression(condition)}:"
        lines.append(_indent(header, indent))
        lines.extend(_emit_block(block, indent + 1))
    return lines


def _emit_match(node: ast.Conditional, indent: int) -> List[str]:
    lines: List[str] = [_indent(f"match {_emit_expression(node.test)}:", indent)]
    for arm in node.arms:
        guard = f" if {_emit_expression(arm.guard)}" if arm.guard else ""
        pattern = _render_pattern(arm.pattern)
        lines.append(_indent(f"case {pattern}{guard}:", indent + 1))
        lines.extend(_emit_block(arm.body, indent + 2))
    return lines


def _emit_expression(node: ast.Node) -> str:
    if isinstance(node, ast.Literal):
        return _emit_literal(node)
    if isinstance(node, ast.BinaryOp):
        return f"{_emit_expression(node.left)} {node.operator} {_emit_expression(node.right)}"
    if isinstance(node, ast.UnaryOp):
        return f"{node.operator} {_emit_expression(node.operand)}"
    if isinstance(node, ast.Call):
        return f"{node.function}({_emit_arguments(node.arguments)})"
    if isinstance(node, ast.BuiltinCall):
        return f"{node.name}({_emit_arguments(node.arguments)})"
    if isinstance(node, ast.Comprehension):
        expr = _emit_expression(node.expression)
        iterable = _emit_expression(node.iterable)
        clause = f" for {node.target} in {iterable}"
        if node.condition is not None:
            clause += f" if {_emit_expression(node.condition)}"
        return f"[{expr}{clause}]"
    if isinstance(node, ast.Lambda):
        params = ", ".join(param.name for param in node.parameters)
        return f"lambda {params}: {_emit_expression(node.body)}"
    if isinstance(node, ast.Block):
        lines = _emit_block(node, indent=0)
        return "\n".join(lines)
    if isinstance(node, ast.Conditional) and node.kind == "if":
        # Inline conditional expression using a Python ternary if possible.
        branches = [branch for branch in node.branches if branch[0] is not None]
        if len(branches) == 2 and node.branches[-1][0] is None:
            true_branch, false_branch = node.branches[0], node.branches[-1]
            true_expr = _extract_expression(true_branch[1])
            false_expr = _extract_expression(false_branch[1])
            if true_expr is not None and false_expr is not None:
                condition_expr = node.branches[0][0]
                if condition_expr is None:
                    return f"/*{node.node_type}*/"
                return f"{true_expr} if {_emit_expression(condition_expr)} else {false_expr}"
    return f"/*{node.node_type}*/"


def _emit_literal(node: ast.Literal) -> str:
    if node.literal_type == "string":
        return repr(node.value)
    if node.literal_type == "identifier":
        return str(node.value)
    if node.literal_type == "list":
        items = cast(List[ast.Node], node.value)
        return f"[{', '.join(_emit_expression(item) for item in items)}]"
    if node.literal_type == "tuple":
        items = cast(List[ast.Node], node.value)
        inner = ", ".join(_emit_expression(item) for item in items)
        if len(items) == 1:
            inner += ","
        return f"({inner})"
    if node.literal_type == "map":
        entries_data = cast(List[Tuple[ast.Node, ast.Node]], node.value)
        entries = [f"{_emit_expression(k)}: {_emit_expression(v)}" for k, v in entries_data]
        return f"{{{', '.join(entries)}}}"
    return repr(node.value)


def _emit_arguments(arguments: Iterable[ast.Node]) -> str:
    return ", ".join(_emit_expression(arg) for arg in arguments)


def _extract_expression(block: ast.Block) -> str | None:
    if len(block.statements) != 1:
        return None
    statement = block.statements[0]
    if (
        isinstance(statement, ast.BuiltinCall)
        and statement.name == "return"
        and statement.arguments
    ):
        return _emit_expression(statement.arguments[0])
    return None


def _indent(text: str, level: int) -> str:
    return "    " * level + text


def _render_pattern(pattern: ast.Pattern) -> str:
    text = pattern.text
    if text == "_":
        return "_"
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered.capitalize()
    try:
        int(text)
        return text
    except ValueError:
        try:
            float(text)
            return text
        except ValueError:
            pass
    if text.isidentifier():
        return text
    return repr(text)


__all__ = ["transpile"]

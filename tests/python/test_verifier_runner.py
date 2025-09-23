"""Tests for the verifier runner's FFI integration."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from packages.verifier.runner import VmExecutionError, execute_vm


@pytest.fixture(scope="module")
def constant_module() -> dict[str, object]:
    """Minimal bytecode module that returns the integer 42."""

    return {
        "constants": [],
        "functions": [
            {
                "name": "solve",
                "registers": 1,
                "stack_slots": 0,
                "contracts": [],
                "instructions": [
                    {
                        "opcode": "LoadImmediate",
                        "operands": {
                            "kind": "RegImmediate",
                            "value": [0, {"Scalar": {"Int": 42}}],
                        },
                    },
                    {
                        "opcode": "Return",
                        "operands": {"kind": "Reg", "value": 0},
                    },
                ],
            }
        ],
    }


def _ffi_built() -> bool:
    if os.environ.get("PRAXIS_FFI_LIB"):
        return True
    root = Path(__file__).resolve().parents[2]
    candidates = ["libpraxis_ffi.so"]
    if sys.platform == "darwin":
        candidates = ["libpraxis_ffi.dylib"]
    elif os.name == "nt":
        candidates = ["praxis_ffi.dll"]
    for build in ("release", "debug"):
        for name in candidates:
            if (root / "target" / build / name).exists():
                return True
    return False


@pytest.mark.skipif(not _ffi_built(), reason="praxis_ffi shared library not built")
def test_execute_vm_returns_value(constant_module: dict[str, object]) -> None:
    """Ensure the VM returns the final register value and trace metadata."""

    outcome = execute_vm(module=constant_module)
    assert outcome.return_value == {"type": "Scalar", "value": {"Int": 42}}
    trace = cast(Mapping[str, Any], outcome.trace)
    trace_body = cast(Mapping[str, Any], trace.get("trace", {}))
    metrics = cast(Mapping[str, Any], trace_body.get("metrics", {}))
    assert metrics.get("instructions") == 2


@pytest.mark.skipif(not _ffi_built(), reason="praxis_ffi shared library not built")
def test_execute_vm_raises_on_invalid_register(constant_module: dict[str, object]) -> None:
    """Returning an out-of-range register should surface a typed error."""

    broken = copy.deepcopy(constant_module)
    functions = cast(list[dict[str, Any]], broken["functions"])
    broken_function = copy.deepcopy(functions[0])
    broken_function["instructions"] = [
        {
            "opcode": "Return",
            "operands": {"kind": "Reg", "value": 3},
        }
    ]
    broken["functions"] = [broken_function]

    with pytest.raises(VmExecutionError) as exc:
        execute_vm(module=broken)

    assert exc.value.kind == "invalid_register"

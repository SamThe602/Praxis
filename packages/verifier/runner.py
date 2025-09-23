"""Runtime runner for executing Praxis bytecode via the native VM."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from packages.telemetry import hooks as telemetry_hooks
from packages.telemetry import metrics as telemetry_metrics

_LIB_CACHE: Optional[ctypes.CDLL] = None


@dataclass
class VmExecution:
    """Successful execution record returned by the VM."""

    return_value: Any
    trace: Dict[str, Any]


class VmExecutionError(RuntimeError):
    """Raised when the native runtime reports an error."""

    def __init__(self, kind: str, message: str, detail: Optional[Any] = None) -> None:
        super().__init__(message)
        self.kind = kind
        self.detail = detail


def verify(program: Dict[str, Any]) -> VmExecution:
    """Execute the provided bytecode module and return the VM trace.

    The *program* dictionary must include the serialised bytecode module under the
    ``module`` key. Optional keys ``entry`` (default: ``"solve"``), ``args`` and
    ``limits`` mirror the JSON contract expected by the Rust FFI layer.
    """

    module = program.get("module")
    if module is None:
        raise ValueError("program['module'] is required for verification")

    entry = program.get("entry", "solve")
    args = program.get("args", [])
    limits = program.get("limits")

    return execute_vm(module=module, entry=entry, args=args, limits=limits)


def execute_vm(
    *,
    module: Dict[str, Any],
    entry: str = "solve",
    args: Optional[Iterable[Any]] = None,
    limits: Optional[Dict[str, Any]] = None,
) -> VmExecution:
    """Run the VM through the FFI bridge and return the execution outcome."""

    start_time = time.perf_counter()
    payload = {
        "module": module,
        "entry": entry,
        "args": list(args or []),
    }
    if limits is not None:
        payload["limits"] = limits

    blob = json.dumps(payload).encode("utf-8")
    lib = _load_library()

    ptr = lib.praxis_vm_execute(blob)
    if not ptr:
        raise RuntimeError("FFI call returned a null pointer")

    try:
        raw = ctypes.c_char_p(ptr).value
    finally:
        lib.praxis_vm_free(ptr)

    if raw is None:
        raise RuntimeError("FFI returned null data")

    response = json.loads(raw.decode("utf-8"))
    if not response.get("ok"):
        error = response.get("error") or {}
        raise VmExecutionError(
            kind=error.get("kind", "runtime_error"),
            message=error.get("message", "runtime execution failed"),
            detail=error.get("detail"),
        )

    trace = response.get("trace") or {}
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    _record_vm_telemetry(entry=entry, trace=trace, latency_ms=latency_ms, payload=payload)
    return VmExecution(return_value=response.get("value"), trace=trace)


def _load_library() -> ctypes.CDLL:
    """Load and memoise the Praxis FFI shared library."""

    global _LIB_CACHE
    if _LIB_CACHE is not None:
        return _LIB_CACHE

    path = _resolve_library_path()
    library = ctypes.CDLL(str(path))
    library.praxis_vm_execute.argtypes = [ctypes.c_char_p]
    library.praxis_vm_execute.restype = ctypes.c_void_p
    library.praxis_vm_free.argtypes = [ctypes.c_void_p]
    library.praxis_vm_free.restype = None
    _LIB_CACHE = library
    return library


def _resolve_library_path() -> Path:
    """Determine the FFI shared library location.

    The lookup honours the ``PRAXIS_FFI_LIB`` environment variable before
    falling back to the default cargo build output in ``target/{debug,release}``.
    """

    override = os.environ.get("PRAXIS_FFI_LIB")
    if override:
        return Path(override)

    candidates = [Path("target") / build / _ffi_filename() for build in ("release", "debug")]

    root = Path(__file__).resolve().parents[2]
    for candidate in candidates:
        location = root / candidate
        if location.exists():
            return location

    raise FileNotFoundError(
        "Unable to locate praxis_ffi shared library. Set PRAXIS_FFI_LIB or build the crate."
    )


def _ffi_filename() -> str:
    if sys.platform == "darwin":
        return "libpraxis_ffi.dylib"
    if os.name == "nt":
        return "praxis_ffi.dll"
    return "libpraxis_ffi.so"


__all__ = [
    "VmExecution",
    "VmExecutionError",
    "execute_vm",
    "verify",
]


def _record_vm_telemetry(
    *, entry: str, trace: Any, latency_ms: float, payload: Mapping[str, Any]
) -> None:
    try:
        telemetry_metrics.emit(
            "praxis.vm.latency_ms",
            latency_ms,
            tags={"entry": entry},
            extra={"args": len(payload.get("args", []))},
        )
        if isinstance(trace, Mapping):
            smt_calls = trace.get("smt_calls")
            if isinstance(smt_calls, (int, float)):
                telemetry_metrics.emit(
                    "praxis.verifier.smt_calls",
                    smt_calls,
                    tags={"entry": entry},
                )
        telemetry_hooks.dispatch(
            telemetry_hooks.VM_EXECUTION_COMPLETED,
            {
                "entry": entry,
                "latency_ms": round(latency_ms, 3),
                "smt_calls": trace.get("smt_calls") if isinstance(trace, Mapping) else None,
                "trace_keys": sorted(trace.keys()) if isinstance(trace, Mapping) else (),
            },
        )
    except Exception:  # pragma: no cover - telemetry should not break execution
        telemetry_hooks.dispatch(
            "vm.telemetry_error",
            {"entry": entry, "latency_ms": latency_ms},
        )

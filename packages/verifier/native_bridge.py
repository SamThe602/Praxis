"""Shared helpers for invoking Praxis native verifier routines via FFI."""

from __future__ import annotations

import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from packages.telemetry import metrics as telemetry_metrics
from packages.utils.config import load_config

__all__ = [
    "NativeBridgeError",
    "is_available",
    "record_fallback",
    "should_use",
    "static_analyze",
    "evaluate_relation",
    "run_checker",
]


class NativeBridgeError(RuntimeError):
    """Raised when the native bridge fails to respond successfully."""

    def __init__(self, surface: str, message: str) -> None:
        super().__init__(message)
        self.surface = surface


_LIB_LOCK = threading.RLock()
_LIB_HANDLE: ctypes.CDLL | None = None

_CONFIG_LOCK = threading.RLock()
_CONFIG_CACHE: dict[str, Any] | None = None

_SURFACE_TO_SYMBOL = {
    "static_analysis": "praxis_verifier_static_analyze",
    "metamorphic": "praxis_verifier_metamorphic",
    "checkers": "praxis_verifier_checker",
}


def should_use(surface: str) -> bool:
    """Return ``True`` when the native bridge is enabled for ``surface``."""

    settings = _load_settings()
    if not settings.get("enabled", False):
        return False
    return bool(settings.get(surface, True))


def record_fallback(surface: str) -> None:
    """Emit a telemetry datapoint indicating a fallback occurred."""

    telemetry_metrics.emit(
        "praxis.verifier.native.fallbacks",
        1,
        tags={"surface": surface},
    )


def is_available() -> bool:
    """Return whether the native shared library could be loaded."""

    try:
        _load_library()
    except NativeBridgeError:
        return False
    return True


def static_analyze(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Invoke the native static analyzer with ``payload``."""

    return _invoke("static_analysis", payload)


def evaluate_relation(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Invoke the native metamorphic relation evaluator."""

    return _invoke("metamorphic", payload)


def run_checker(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Invoke the native checker routines."""

    return _invoke("checkers", payload)


def _invoke(surface: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    symbol = _SURFACE_TO_SYMBOL.get(surface)
    if not symbol:
        raise NativeBridgeError(surface, f"unknown surface '{surface}'")

    telemetry_metrics.emit(
        "praxis.verifier.native.calls",
        1,
        tags={"surface": surface},
    )

    library = _load_library()
    function = getattr(library, symbol)

    blob = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    start = time.perf_counter()
    ptr = function(blob)
    if not ptr:
        telemetry_metrics.emit(
            "praxis.verifier.native.errors",
            1,
            tags={"surface": surface, "kind": "null_pointer"},
        )
        raise NativeBridgeError(surface, "FFI call returned null pointer")

    try:
        raw = ctypes.c_char_p(ptr).value
    finally:
        library.praxis_vm_free(ptr)

    if raw is None:
        telemetry_metrics.emit(
            "praxis.verifier.native.errors",
            1,
            tags={"surface": surface, "kind": "null_payload"},
        )
        raise NativeBridgeError(surface, "FFI returned empty payload")

    latency_ms = (time.perf_counter() - start) * 1000.0
    telemetry_metrics.emit(
        "praxis.verifier.native.latency_ms",
        latency_ms,
        tags={"surface": surface},
    )

    try:
        response = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        telemetry_metrics.emit(
            "praxis.verifier.native.errors",
            1,
            tags={"surface": surface, "kind": "decode"},
        )
        raise NativeBridgeError(surface, f"failed to decode native response: {exc}") from exc

    if _is_bridge_error(response):
        message = _extract_error_message(response)
        telemetry_metrics.emit(
            "praxis.verifier.native.errors",
            1,
            tags={"surface": surface, "kind": "bridge"},
        )
        raise NativeBridgeError(surface, message)
    if not isinstance(response, Mapping):
        telemetry_metrics.emit(
            "praxis.verifier.native.errors",
            1,
            tags={"surface": surface, "kind": "type"},
        )
        raise NativeBridgeError(surface, "native bridge returned non-mapping payload")

    return dict(response)


def _is_bridge_error(response: Any) -> bool:
    if not isinstance(response, Mapping):
        return True
    error = response.get("error")
    if not isinstance(error, Mapping):
        return False
    kind = error.get("kind")
    return isinstance(kind, str) and kind.startswith("native_bridge")


def _extract_error_message(response: Mapping[str, Any]) -> str:
    error = response.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return "native bridge call failed"


def _load_library() -> ctypes.CDLL:
    global _LIB_HANDLE
    with _LIB_LOCK:
        if _LIB_HANDLE is not None:
            return _LIB_HANDLE

        path = _resolve_library_path()
        library = ctypes.CDLL(str(path))
        library.praxis_verifier_static_analyze.argtypes = [ctypes.c_char_p]
        library.praxis_verifier_static_analyze.restype = ctypes.c_void_p
        library.praxis_verifier_metamorphic.argtypes = [ctypes.c_char_p]
        library.praxis_verifier_metamorphic.restype = ctypes.c_void_p
        library.praxis_verifier_checker.argtypes = [ctypes.c_char_p]
        library.praxis_verifier_checker.restype = ctypes.c_void_p
        library.praxis_vm_free.argtypes = [ctypes.c_void_p]
        library.praxis_vm_free.restype = None
        _LIB_HANDLE = library
        return library


def _resolve_library_path() -> Path:
    override = os.environ.get("PRAXIS_FFI_LIB")
    if override:
        candidate = Path(override)
        if candidate.exists():
            return candidate
        raise NativeBridgeError("loader", f"PRAXIS_FFI_LIB points to missing path: {override}")

    root = Path(__file__).resolve().parents[2]
    for build in ("release", "debug"):
        location = root / "target" / build / _ffi_filename()
        if location.exists():
            return location

    raise NativeBridgeError(
        "loader",
        "Unable to locate praxis_ffi shared library. Build the crate or set PRAXIS_FFI_LIB.",
    )


def _ffi_filename() -> str:
    if os.name == "nt":
        return "praxis_ffi.dll"
    if sys.platform == "darwin":
        return "libpraxis_ffi.dylib"
    return "libpraxis_ffi.so"


def _load_settings() -> MutableMapping[str, Any]:
    global _CONFIG_CACHE
    with _CONFIG_LOCK:
        if _CONFIG_CACHE is not None:
            config = _CONFIG_CACHE
        else:
            config = _read_config()
            _CONFIG_CACHE = config
    bridge = config.get("native_bridge", {})
    if isinstance(bridge, Mapping):
        return dict(bridge)
    return {}


def _read_config() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    path = root / "configs" / "verifier" / "static.yaml"
    try:
        data = load_config(path)
    except FileNotFoundError:
        return {}
    except Exception:  # pragma: no cover - defensive config guard
        return {}
    if isinstance(data, dict):
        return data
    return {}

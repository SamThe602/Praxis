"""Helpers for executing untrusted callables inside an isolated sandbox."""

from __future__ import annotations

import builtins
import contextlib
import functools
import io
import multiprocessing
import os
import pickle
import socket
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Mapping, Sequence, TypeVar

from packages.utils.config import load_config

try:  # ``resource`` is POSIX-only; gracefully degrade elsewhere.
    import resource  # type: ignore
except ImportError:  # pragma: no cover - exercised on non-POSIX platforms
    resource = None  # type: ignore

try:  # ``cloudpickle`` allows serialising lambdas/closures if available.
    import cloudpickle  # type: ignore
except Exception:  # pragma: no cover - dependency optional
    cloudpickle = None  # type: ignore

T = TypeVar("T")

_PAYLOAD_REGISTRY: dict[str, Callable[[], Any]] = {}
_REGISTRY_LOCK = threading.Lock()

__all__ = [
    "SandboxError",
    "SandboxTimeoutError",
    "SandboxExecutionError",
    "SandboxResult",
    "SandboxTelemetry",
    "run_in_sandbox",
]


@dataclass(slots=True, frozen=True)
class SandboxTelemetry:
    """Execution metrics collected from the sandboxed process."""

    duration: float
    memory_peak_kb: int | None


@dataclass(slots=True, frozen=True)
class SandboxResult(Generic[T]):
    """Return payload for :func:`run_in_sandbox`."""

    value: T
    stdout: str
    stderr: str
    telemetry: SandboxTelemetry


class SandboxError(RuntimeError):
    """Base error emitted by the sandbox helpers."""


class SandboxTimeoutError(SandboxError):
    """Raised when the payload exceeds the configured wall-clock timeout."""

    def __init__(self, *, timeout: float, stdout: str = "", stderr: str = "") -> None:
        super().__init__(f"sandbox execution timed out after {timeout:.3f}s")
        self.timeout = timeout
        self.stdout = stdout
        self.stderr = stderr


class SandboxExecutionError(SandboxError):
    """Raised when the payload fails inside the sandbox."""

    def __init__(
        self,
        *,
        message: str,
        stdout: str,
        stderr: str,
        traceback_frames: Sequence[str],
        telemetry: SandboxTelemetry,
        kind: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr
        self.traceback = tuple(traceback_frames)
        self.telemetry = telemetry
        self.kind = kind or "sandbox_error"


@dataclass(slots=True, frozen=True)
class _SandboxDefaults:
    timeout: float | None
    memory_limit_mb: int | None
    env_allowlist: tuple[str, ...]
    temp_root: Path | None


def run_in_sandbox(
    payload: Callable[[], T],
    *,
    timeout: float | None = None,
    memory_limit_mb: int | None = None,
    env_allowlist: Sequence[str] | None = None,
) -> SandboxResult[T]:
    """Execute ``payload`` in an isolated subprocess with resource limits.

    The sandbox applies conservative safeguards intended for untrusted code:

    * Wall-clock timeout via the parent process watchdog
    * Memory limit enforced with ``resource.setrlimit`` when available
    * Network access disabled by monkey-patching :mod:`socket`
    * CWD switched to a throw-away directory inside ``/tmp`` (customisable
      through ``utils.yaml``)
    * Environment variables restricted to a small allowlist

    Stdout and stderr emitted by the payload are captured and returned alongside
    the callable's value and timing data.  Any exception raised inside the
    sandbox triggers :class:`SandboxExecutionError` which exposes the captured
    output and traceback for diagnostics.
    """

    if not callable(payload):
        raise TypeError(f"payload must be callable, received {payload!r}")

    defaults = _load_defaults()
    deadline = timeout if timeout is not None else defaults.timeout
    memory_limit = memory_limit_mb if memory_limit_mb is not None else defaults.memory_limit_mb
    allowlist = tuple(env_allowlist) if env_allowlist is not None else defaults.env_allowlist

    sandbox_root = defaults.temp_root
    original_env = dict(os.environ)
    env_snapshot = {name: original_env[name] for name in allowlist if name in original_env}

    ctx = _select_multiprocessing_context()
    payload_spec = _prepare_payload_spec(payload, ctx)
    config: dict[str, Any] = {
        "memory_limit_bytes": None if memory_limit is None else int(memory_limit) * 1024 * 1024,
        "env": env_snapshot,
        "disable_network": True,
    }

    with tempfile.TemporaryDirectory(dir=sandbox_root) as tmpdir:
        parent_conn, child_conn = ctx.Pipe(duplex=False)  # type: ignore[attr-defined]
        process = ctx.Process(  # type: ignore[attr-defined]
            target=_sandbox_entry,
            args=(payload_spec, child_conn, config, tmpdir),
            daemon=True,
        )
        process.start()
        child_conn.close()

        try:
            snapshot = _await_result(process, parent_conn, timeout=deadline)
        finally:
            parent_conn.close()
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)

    try:
        if payload_spec[0] == "registry":
            _unregister_payload(payload_spec[1])
    except Exception:  # pragma: no cover - best-effort cleanup
        pass

    if snapshot is None:
        raise SandboxError("sandbox process terminated without reporting a result")

    status = snapshot.get("status")
    stdout = snapshot.get("stdout", "")
    stderr = snapshot.get("stderr", "")
    telemetry = SandboxTelemetry(
        duration=float(snapshot.get("duration", 0.0)),
        memory_peak_kb=_coerce_int(snapshot.get("memory_peak")),
    )

    if status == "ok":
        value = _deserialize_object(snapshot.get("value"))
        return SandboxResult(value=value, stdout=stdout, stderr=stderr, telemetry=telemetry)

    if status == "timeout":
        timeout_value = float(snapshot.get("timeout", deadline or 0.0))
        raise SandboxTimeoutError(timeout=timeout_value, stdout=stdout, stderr=stderr)

    if status == "error":
        exception = snapshot.get("exception") or {}
        message = str(exception.get("message") or "sandbox execution failed")
        frames = tuple(exception.get("traceback") or ())
        kind = exception.get("kind")
        raise SandboxExecutionError(
            message=message,
            stdout=stdout,
            stderr=stderr,
            traceback_frames=frames,
            telemetry=telemetry,
            kind=str(kind) if kind else None,
        )

    raise SandboxError(f"unrecognised sandbox status: {status!r}")


def _await_result(
    process: multiprocessing.Process,
    conn: multiprocessing.connection.Connection,
    timeout: float | None,
) -> Mapping[str, Any] | None:
    poll_interval = 0.1
    if timeout is not None and timeout <= 0:
        timeout = 0.0
    deadline = (time.perf_counter() + timeout) if timeout is not None else None
    captured: Mapping[str, Any] | None = None
    timed_out = False

    while True:
        wait_time = poll_interval
        if deadline is not None:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                timed_out = True
                break
            wait_time = min(wait_time, max(0.0, remaining))
        if conn.poll(wait_time):
            try:
                captured = conn.recv()
            except EOFError:
                captured = None
            break
        if not process.is_alive():
            process.join(timeout=0.1)
            break

    if captured is None and timed_out:
        return {"status": "timeout", "timeout": timeout or 0.0, "stdout": "", "stderr": ""}
    return captured


def _sandbox_entry(
    payload_spec: tuple[str, Any],
    conn: multiprocessing.connection.Connection,
    config: Mapping[str, Any],
    temp_dir: str,
) -> None:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    start = time.perf_counter()

    result: dict[str, Any]
    try:
        _initialise_process_environment(config, temp_dir)
        payload = _deserialize_callable(payload_spec)
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            value: Any = payload()
        result = {
            "status": "ok",
            "value": _serialise_object(value),
        }
    except BaseException as exc:  # pragma: no cover - exercised via tests raising payload errors
        result = {
            "status": "error",
            "exception": {
                "type": exc.__class__.__name__,
                "kind": getattr(exc, "kind", exc.__class__.__name__),
                "message": str(exc),
                "traceback": traceback.format_exception(exc.__class__, exc, exc.__traceback__),
            },
        }
    finally:
        duration = time.perf_counter() - start
        memory_peak = _read_memory_peak()
        limit_bytes = config.get("memory_limit_bytes")
        if (
            result.get("status") == "ok"
            and isinstance(limit_bytes, int)
            and limit_bytes > 0
            and memory_peak is not None
            and memory_peak * 1024 > limit_bytes
        ):
            overuse = memory_peak * 1024
            result = {
                "status": "error",
                "exception": {
                    "type": "MemoryError",
                    "kind": "MemoryError",
                    "message": f"memory usage {overuse} bytes exceeds sandbox limit {limit_bytes} bytes",
                    "traceback": [],
                },
            }
        result["duration"] = duration
        result["memory_peak"] = memory_peak
        result["stdout"] = stdout_buffer.getvalue()
        result["stderr"] = stderr_buffer.getvalue()
        conn.send(result)
        conn.close()


def _initialise_process_environment(config: Mapping[str, Any], temp_dir: str) -> None:
    os.chdir(temp_dir)
    env = config.get("env")
    if isinstance(env, Mapping):
        os.environ.clear()
        for key, value in env.items():
            os.environ[str(key)] = str(value)
    if config.get("disable_network"):
        _disable_network()
    _install_filesystem_guard(Path(temp_dir))
    memory_limit = config.get("memory_limit_bytes")
    if memory_limit and resource is not None:
        limit = int(memory_limit)
        # Enforce limits on both address space and RSS when available to catch runaway allocations.
        for rlimit in (
            getattr(resource, "RLIMIT_AS", None),
            getattr(resource, "RLIMIT_DATA", None),
        ):
            if rlimit is not None:
                resource.setrlimit(rlimit, (limit, limit))


def _disable_network() -> None:
    def _blocked(*_: Any, **__: Any) -> None:
        raise RuntimeError("network access is disabled inside the sandbox")

    class _BlockedSocket(socket.socket):  # type: ignore[misc]
        def __new__(cls, *args: Any, **kwargs: Any):  # type: ignore[override]
            _blocked()

    for name in (
        "socket",
        "create_connection",
        "create_server",
        "socketpair",
        "fromfd",
        "fromshare",
        "getaddrinfo",
        "gethostbyname",
        "gethostbyname_ex",
    ):
        if hasattr(socket, name):
            setattr(socket, name, _blocked)
    setattr(socket, "socket", _BlockedSocket)


def _install_filesystem_guard(root: Path) -> None:
    base = root.resolve()

    def _normalise(path: Any) -> Path:
        candidate = Path(path)
        try:
            return candidate.resolve(strict=False)
        except Exception:
            return (base / candidate).resolve(strict=False)

    def _ensure_within_root(path: Any) -> None:
        resolved = _normalise(path)
        try:
            resolved.relative_to(base)
        except ValueError:
            raise PermissionError(f"access to {resolved} is outside sandbox root {base}")

    def _is_write_mode(mode: str) -> bool:
        flags = set(mode)
        return bool({"w", "a", "x", "+"} & flags)

    original_open = builtins.open

    @functools.wraps(original_open)
    def guarded_open(file, mode="r", *args, **kwargs):  # type: ignore[override]
        if _is_write_mode(str(mode)):
            _ensure_within_root(file)
        return original_open(file, mode, *args, **kwargs)

    original_path_open = Path.open

    @functools.wraps(original_path_open)
    def guarded_path_open(self, mode="r", *args, **kwargs):  # type: ignore[override]
        if _is_write_mode(str(mode)):
            _ensure_within_root(self)
        return original_path_open(self, mode, *args, **kwargs)

    original_os_open = os.open

    @functools.wraps(original_os_open)
    def guarded_os_open(path, flags, mode=0o777):
        write_flags = getattr(os, "O_WRONLY", 1) | getattr(os, "O_RDWR", 2)
        if flags & write_flags:
            _ensure_within_root(path)
        return original_os_open(path, flags, mode)

    def _wrap_simple(fn, *, allow_destination=False):
        original = fn

        @functools.wraps(original)
        def wrapper(path, *args, **kwargs):
            _ensure_within_root(path)
            return original(path, *args, **kwargs)

        def wrapper_with_dest(src, dst, *args, **kwargs):
            _ensure_within_root(src)
            _ensure_within_root(dst)
            return original(src, dst, *args, **kwargs)

        return wrapper_with_dest if allow_destination else wrapper

    builtins.open = guarded_open  # type: ignore[assignment]
    Path.open = guarded_path_open  # type: ignore[assignment]
    os.open = guarded_os_open  # type: ignore[assignment]
    os.mkdir = _wrap_simple(os.mkdir)
    os.makedirs = _wrap_simple(os.makedirs)
    os.remove = _wrap_simple(os.remove)
    os.unlink = _wrap_simple(os.unlink)
    os.rmdir = _wrap_simple(os.rmdir)
    os.replace = _wrap_simple(os.replace, allow_destination=True)
    os.rename = _wrap_simple(os.rename, allow_destination=True)


def _read_memory_peak() -> int | None:
    if resource is None:  # pragma: no cover - non-POSIX path
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = getattr(usage, "ru_maxrss", None)
    if peak is None:
        return None
    # ``ru_maxrss`` is expressed in kilobytes on Linux and bytes on macOS; normalise to KB.
    if sys.platform == "darwin":
        peak = int(peak / 1024)
    return int(peak)


def _prepare_payload_spec(
    payload: Callable[[], T], ctx: multiprocessing.context.BaseContext
) -> tuple[str, Any]:
    try:
        return "pickle", pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    if cloudpickle is not None:
        try:
            return "cloudpickle", cloudpickle.dumps(payload)
        except Exception:
            pass
    if ctx.get_start_method() == "fork":
        return "registry", _register_payload(payload)
    raise TypeError(
        "sandbox payload must be picklable; install cloudpickle or use a top-level callable"
    )


def _register_payload(payload: Callable[[], Any]) -> str:
    key = f"payload-{uuid.uuid4().hex}"
    with _REGISTRY_LOCK:
        _PAYLOAD_REGISTRY[key] = payload
    return key


def _unregister_payload(key: Any) -> None:
    with _REGISTRY_LOCK:
        _PAYLOAD_REGISTRY.pop(str(key), None)


def _deserialize_callable(spec: tuple[str, Any]) -> Callable[[], T]:
    kind, data = spec
    if kind == "pickle":
        return pickle.loads(data)
    if kind == "cloudpickle":
        if cloudpickle is None:
            raise RuntimeError("cloudpickle is required to deserialize sandbox payloads")
        return cloudpickle.loads(data)
    if kind == "registry":
        with _REGISTRY_LOCK:
            payload = _PAYLOAD_REGISTRY.get(str(data))
        if payload is None:
            raise RuntimeError("sandbox payload registry entry missing")
        return payload  # type: ignore[return-value]
    raise ValueError(f"unknown payload spec kind: {kind}")


def _serialise_object(value: Any) -> bytes:
    try:
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        if cloudpickle is None:
            raise
        return cloudpickle.dumps(value)


def _deserialize_object(data: Any) -> Any:
    if isinstance(data, bytes):
        try:
            return pickle.loads(data)
        except Exception:
            if cloudpickle is None:
                raise
            return cloudpickle.loads(data)
    return data


def _select_multiprocessing_context() -> multiprocessing.context.BaseContext:
    if sys.platform == "win32":  # pragma: no cover - windows not in CI
        return multiprocessing.get_context("spawn")
    try:
        return multiprocessing.get_context("fork")
    except ValueError:  # pragma: no cover - platforms without fork fallback to spawn
        return multiprocessing.get_context("spawn")


def _load_defaults() -> _SandboxDefaults:
    config_path = Path(__file__).resolve().parents[2] / "configs" / "utils.yaml"
    data: Mapping[str, Any] | None = None
    if config_path.exists():
        try:
            raw = load_config(config_path)
            if isinstance(raw, Mapping):
                data = raw
        except Exception:  # pragma: no cover - config misconfiguration should not break runtime
            data = None

    section = data.get("sandbox") if isinstance(data, Mapping) else None
    if not isinstance(section, Mapping):
        section = {}

    timeout = _coerce_float(section.get("timeout"))
    memory_limit = _coerce_int(section.get("memory_limit_mb"))
    env_allowlist = section.get("env_allowlist")
    if isinstance(env_allowlist, Sequence) and not isinstance(env_allowlist, (str, bytes)):
        allowlist = tuple(str(item) for item in env_allowlist)
    else:
        allowlist = (
            "PATH",
            "PYTHONPATH",
            "PYTHONHASHSEED",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "HOME",
        )
    temp_root_raw = section.get("temp_root")
    temp_root = Path(str(temp_root_raw)) if temp_root_raw else None
    return _SandboxDefaults(
        timeout=timeout,
        memory_limit_mb=memory_limit,
        env_allowlist=allowlist,
        temp_root=temp_root,
    )


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None

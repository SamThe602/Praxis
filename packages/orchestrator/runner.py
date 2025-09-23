"""High-level orchestrator entry point invoking the native VM."""

from __future__ import annotations

import uuid
from typing import Any, Mapping

from packages.orchestrator.types import SandboxOptions
from packages.telemetry import hooks as telemetry_hooks
from packages.telemetry import metrics as telemetry_metrics
from packages.telemetry.logger import get_logger
from packages.utils.sandbox import (
    SandboxError,
    SandboxExecutionError,
    SandboxResult,
    SandboxTimeoutError,
    run_in_sandbox,
)
from packages.verifier.runner import VmExecution, execute_vm

_LOGGER = get_logger("praxis.orchestrator.runner")


def run(task: Mapping[str, Any]) -> VmExecution:
    """Run a synthesis task using the native VM interpreter within a sandbox."""

    if "module" not in task:
        raise ValueError("task must include a 'module' entry")

    entry = task.get("entry", "solve")
    args = task.get("args")
    limits = task.get("limits")
    sandbox_options = _resolve_sandbox_options(task)
    sandbox_kwargs = sandbox_options.to_kwargs()
    correlation_id = str(task.get("correlation_id") or uuid.uuid4().hex)
    task_id = task.get("id") if isinstance(task, Mapping) else None

    _LOGGER.info(
        "sandbox run start | cid=%s entry=%s task=%s", correlation_id, entry, task_id or "anonymous"
    )

    try:
        sandbox_result: SandboxResult[VmExecution] = run_in_sandbox(
            lambda: execute_vm(module=task["module"], entry=entry, args=args, limits=limits),
            **sandbox_kwargs,
        )
    except SandboxTimeoutError as exc:
        _LOGGER.warning(
            "sandbox timeout | cid=%s entry=%s timeout=%.3f", correlation_id, entry, exc.timeout
        )
        _record_sandbox_security_event(
            status="timeout",
            correlation_id=correlation_id,
            entry=entry,
            sandbox=sandbox_options,
            task_id=task_id,
            error_kind="timeout",
        )
        raise
    except SandboxExecutionError as exc:
        _LOGGER.warning(
            "sandbox violation | cid=%s entry=%s kind=%s", correlation_id, entry, exc.kind
        )
        _record_sandbox_security_event(
            status="blocked",
            correlation_id=correlation_id,
            entry=entry,
            sandbox=sandbox_options,
            task_id=task_id,
            error_kind=str(exc.kind or exc.__class__.__name__),
        )
        raise
    except SandboxError as exc:
        _LOGGER.warning(
            "sandbox error | cid=%s entry=%s error=%s",
            correlation_id,
            entry,
            exc.__class__.__name__,
        )
        _record_sandbox_security_event(
            status="error",
            correlation_id=correlation_id,
            entry=entry,
            sandbox=sandbox_options,
            task_id=task_id,
            error_kind=exc.__class__.__name__,
        )
        raise

    execution = sandbox_result.value
    if not isinstance(execution, VmExecution):  # pragma: no cover - defensive guard
        raise TypeError("sandbox payload must return a VmExecution instance")

    trace = dict(execution.trace) if isinstance(execution.trace, Mapping) else {}
    sandbox_meta = {
        "duration_ms": round(sandbox_result.telemetry.duration * 1000.0, 3),
        "memory_peak_kb": sandbox_result.telemetry.memory_peak_kb,
        "correlation_id": correlation_id,
    }
    if sandbox_result.stdout:
        sandbox_meta["stdout"] = sandbox_result.stdout
    if sandbox_result.stderr:
        sandbox_meta["stderr"] = sandbox_result.stderr
    trace["sandbox"] = sandbox_meta
    _record_sandbox_security_event(
        status="ok",
        correlation_id=correlation_id,
        entry=entry,
        sandbox=sandbox_options,
        task_id=task_id,
        result=sandbox_result,
    )
    _LOGGER.info(
        "sandbox run ok | cid=%s entry=%s duration_ms=%.3f",
        correlation_id,
        entry,
        sandbox_meta["duration_ms"],
    )
    execution = VmExecution(return_value=execution.return_value, trace=trace)
    _record_orchestrator_telemetry(task, execution)
    return execution


def _resolve_sandbox_options(task: Mapping[str, Any]) -> SandboxOptions:
    payload = task.get("sandbox") if isinstance(task, Mapping) else None
    if isinstance(payload, SandboxOptions):
        return SandboxOptions(
            timeout=payload.timeout,
            memory_limit_mb=payload.memory_limit_mb,
            env_allowlist=payload.env_allowlist,
        )
    if isinstance(payload, Mapping):
        return SandboxOptions().merged(payload)
    return SandboxOptions()


def _record_sandbox_security_event(
    *,
    status: str,
    correlation_id: str,
    entry: str,
    sandbox: SandboxOptions,
    task_id: Any,
    result: SandboxResult[VmExecution] | None = None,
    error_kind: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "correlation_id": correlation_id,
        "entry": entry,
        "status": status,
        "task_id": task_id,
        "timeout": sandbox.timeout,
        "memory_limit_mb": sandbox.memory_limit_mb,
    }
    if error_kind:
        payload["error_kind"] = error_kind
    duration_ms: float | None = None
    if result is not None:
        duration_ms = round(result.telemetry.duration * 1000.0, 3)
        payload["duration_ms"] = duration_ms
        payload["memory_peak_kb"] = result.telemetry.memory_peak_kb
    telemetry_hooks.dispatch(telemetry_hooks.SANDBOX_EVENT, payload)
    try:
        if status == "ok" and duration_ms is not None and result is not None:
            telemetry_metrics.emit(
                "praxis.sandbox.execution_ms",
                duration_ms,
                tags={"entry": entry},
                extra={"correlation_id": correlation_id},
            )
            memory_peak = result.telemetry.memory_peak_kb
            if memory_peak is not None:
                telemetry_metrics.emit(
                    "praxis.sandbox.memory_peak_kb",
                    memory_peak,
                    tags={"entry": entry},
                    extra={"correlation_id": correlation_id},
                )
        else:
            telemetry_metrics.emit(
                "praxis.sandbox.violation",
                1,
                tags={"entry": entry, "status": status, "error": error_kind or status},
                extra={"correlation_id": correlation_id},
            )
    except Exception:  # pragma: no cover - telemetry should never break core execution
        pass


__all__ = ["run", "VmExecution"]


def _record_orchestrator_telemetry(task: Mapping[str, Any], execution: VmExecution) -> None:
    task_id = task.get("id") if isinstance(task, Mapping) else None
    telemetry_hooks.dispatch(
        telemetry_hooks.ORCHESTRATOR_RUN_COMPLETED,
        {
            "task_id": task_id,
            "entry": task.get("entry", "solve"),
            "had_trace": bool(execution.trace),
        },
    )

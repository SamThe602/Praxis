"""Tests for the sandbox utility helper."""

from __future__ import annotations

import os
import socket

import pytest

from packages.utils.sandbox import SandboxExecutionError, SandboxTimeoutError, run_in_sandbox


def test_sandbox_captures_output_and_value() -> None:
    """Standard execution should return value and captured stdout."""

    def payload() -> int:
        print("hello sandbox")
        return 21

    result = run_in_sandbox(payload)
    assert result.value == 21
    assert "hello sandbox" in result.stdout
    assert result.stderr == ""
    assert result.telemetry.duration >= 0.0


def test_sandbox_blocks_network() -> None:
    """Network operations must be rejected inside the sandbox."""

    def payload() -> None:
        socket.create_connection(("127.0.0.1", 9))

    with pytest.raises(SandboxExecutionError) as excinfo:
        run_in_sandbox(payload)

    assert "network access" in str(excinfo.value)
    assert excinfo.value.kind in {"sandbox_error", "RuntimeError"}


def test_sandbox_environment_allowlist() -> None:
    """Only the allowlisted environment variables should leak into the sandbox."""

    os.environ["UNSAFE_TOKEN"] = "secret"

    def payload() -> str | None:
        return os.environ.get("UNSAFE_TOKEN")

    result = run_in_sandbox(payload, env_allowlist=("PATH",))
    assert result.value is None

    del os.environ["UNSAFE_TOKEN"]


def test_sandbox_memory_limit() -> None:
    """Heap allocations exceeding the limit should raise."""

    def payload() -> None:
        _ = bytearray(32 * 1024 * 1024)

    with pytest.raises(SandboxExecutionError) as excinfo:
        run_in_sandbox(payload, memory_limit_mb=1)

    assert excinfo.value.kind in {"MemoryError", "sandbox_error"}
    assert (
        excinfo.value.telemetry.memory_peak_kb is None
        or excinfo.value.telemetry.memory_peak_kb >= 0
    )


def test_sandbox_timeout() -> None:
    """Long-running tasks should trigger a timeout error."""

    def payload() -> None:
        import time

        time.sleep(0.2)

    with pytest.raises(SandboxTimeoutError) as excinfo:
        run_in_sandbox(payload, timeout=0.05)

    assert excinfo.value.timeout == pytest.approx(0.05, rel=0.5)

"""Security regression tests validating sandbox policy enforcement."""

from __future__ import annotations

import socket
import time
import uuid
from pathlib import Path

import pytest

from packages.utils.sandbox import SandboxExecutionError, SandboxTimeoutError, run_in_sandbox


def test_sandbox_blocks_network_calls() -> None:
    """Any network access attempt should be rejected inside the sandbox."""

    def payload() -> None:
        socket.gethostbyname("example.com")

    with pytest.raises(SandboxExecutionError) as excinfo:
        run_in_sandbox(payload)

    assert "network access" in str(excinfo.value)
    assert excinfo.value.kind in {"RuntimeError", "sandbox_error"}


def test_sandbox_denies_writes_outside_root() -> None:
    """Write operations targeting external paths must be blocked."""

    outside_file = Path(__file__).resolve().parent / f"outside-{uuid.uuid4().hex}.txt"

    def payload() -> None:
        with open(outside_file, "w", encoding="utf-8") as handle:
            handle.write("blocked")

    try:
        with pytest.raises(SandboxExecutionError) as excinfo:
            run_in_sandbox(payload)
    finally:
        if outside_file.exists():
            outside_file.unlink()

    assert excinfo.value.kind == "PermissionError"
    assert "outside sandbox root" in str(excinfo.value)


def test_sandbox_enforces_memory_limits() -> None:
    """Allocations exceeding the memory budget should raise a sandbox error."""

    def payload() -> None:
        _ = bytearray(32 * 1024 * 1024)

    with pytest.raises(SandboxExecutionError) as excinfo:
        run_in_sandbox(payload, memory_limit_mb=1)

    assert excinfo.value.kind in {"MemoryError", "sandbox_error"}


def test_sandbox_enforces_timeouts() -> None:
    """Long-running tasks should trigger the timeout guard."""

    def payload() -> None:
        time.sleep(0.2)

    with pytest.raises(SandboxTimeoutError) as excinfo:
        run_in_sandbox(payload, timeout=0.05)

    assert excinfo.value.timeout == pytest.approx(0.05, rel=0.5)

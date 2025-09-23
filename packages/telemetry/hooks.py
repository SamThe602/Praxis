"""Simple hook registry for the telemetry pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterable, Mapping

from . import logger

HookFn = Callable[["HookEvent"], None]

TRANSLATOR_DECODE_COMPLETED = "translator.decode.completed"
SYNTH_SEARCH_COMPLETED = "synthesizer.search.completed"
VM_EXECUTION_COMPLETED = "vm.execution.completed"
ORCHESTRATOR_RUN_COMPLETED = "orchestrator.run.completed"
SANDBOX_EVENT = "sandbox.security.event"


@dataclass(frozen=True)
class HookEvent:
    """Payload passed to registered hook functions."""

    name: str
    payload: Mapping[str, Any]
    timestamp: float


class HookHandle:
    """Disposable handle returned from :func:`register_hook`."""

    def __init__(self, name: str, fn: HookFn) -> None:
        self._name = name
        self._fn = fn
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        unregister_hook(self._name, self._fn)
        self._closed = True

    def __enter__(self) -> "HookHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - trivial
        self.close()


_LOCK = RLock()
_HOOKS: Dict[str, list[HookFn]] = {}
_LOGGER = logger.get_logger("praxis.telemetry.hooks")


def register_hook(name: str, fn: HookFn) -> HookHandle:
    """Register ``fn`` for ``name`` and return a disposable handle."""

    if not isinstance(name, str) or not name:
        raise ValueError("hook name must be a non-empty string")
    if not callable(fn):
        raise TypeError("hook callback must be callable")
    with _LOCK:
        bucket = _HOOKS.setdefault(name, [])
        bucket.append(fn)
    return HookHandle(name, fn)


def unregister_hook(name: str, fn: HookFn) -> None:
    with _LOCK:
        bucket = _HOOKS.get(name)
        if not bucket:
            return
        try:
            bucket.remove(fn)
        except ValueError:
            return
        if not bucket:
            _HOOKS.pop(name, None)


def dispatch(name: str, payload: Mapping[str, Any] | None = None) -> None:
    """Trigger hooks registered for ``name`` with *payload*."""

    event = HookEvent(
        name=name,
        payload=MappingProxyType(dict(payload or {})),
        timestamp=time.time(),
    )
    callbacks: Iterable[HookFn]
    with _LOCK:
        callbacks = list(_HOOKS.get(name, ()))
    for fn in callbacks:
        try:
            fn(event)
        except Exception as exc:  # pragma: no cover - defensive path
            _LOGGER.exception("hook %s failed: %s", name, exc)


def registered_hooks() -> Mapping[str, tuple[HookFn, ...]]:
    with _LOCK:
        return {name: tuple(callbacks) for name, callbacks in _HOOKS.items()}


__all__ = [
    "HookEvent",
    "HookHandle",
    "HookFn",
    "ORCHESTRATOR_RUN_COMPLETED",
    "SANDBOX_EVENT",
    "SYNTH_SEARCH_COMPLETED",
    "TRANSLATOR_DECODE_COMPLETED",
    "VM_EXECUTION_COMPLETED",
    "dispatch",
    "register_hook",
    "registered_hooks",
    "unregister_hook",
]

"""Typed data transfer objects for the Praxis orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from .text_interface import TranslationOutcome


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cache" / "runs"


def _as_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return value if isinstance(value, Path) else Path(value)


def _deep_update(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``base`` with ``updates`` recursively merged."""

    result: dict[str, Any] = {key: value for key, value in base.items()}
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _coerce_float(value: Any, *, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        return float(value)
    except Exception:
        return fallback


def _coerce_int(value: Any, *, fallback: int | None = None) -> int | None:
    if value is None:
        return fallback
    try:
        return int(value)
    except Exception:
        return fallback


@dataclass(slots=True)
class SynthesizerOptions:
    """Configuration knobs for the synthesiser stage."""

    config_path: Path | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SandboxOptions:
    """Execution safeguards enforced when running untrusted payloads."""

    timeout: float | None = None
    memory_limit_mb: int | None = None
    env_allowlist: tuple[str, ...] | None = None

    def to_kwargs(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.timeout is not None:
            payload["timeout"] = float(self.timeout)
        if self.memory_limit_mb is not None:
            payload["memory_limit_mb"] = int(self.memory_limit_mb)
        if self.env_allowlist is not None:
            payload["env_allowlist"] = tuple(self.env_allowlist)
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.timeout is not None:
            payload["timeout"] = self.timeout
        if self.memory_limit_mb is not None:
            payload["memory_limit_mb"] = self.memory_limit_mb
        if self.env_allowlist is not None:
            payload["env_allowlist"] = list(self.env_allowlist)
        return payload

    def merged(self, overrides: Mapping[str, Any] | None) -> "SandboxOptions":
        if not isinstance(overrides, Mapping):
            return SandboxOptions(
                timeout=self.timeout,
                memory_limit_mb=self.memory_limit_mb,
                env_allowlist=self.env_allowlist,
            )
        timeout = _coerce_float(overrides.get("timeout"), fallback=self.timeout)
        memory = _coerce_int(overrides.get("memory_limit_mb"), fallback=self.memory_limit_mb)
        allowlist: tuple[str, ...] | None = self.env_allowlist
        raw_allowlist = overrides.get("env_allowlist")
        if isinstance(raw_allowlist, Sequence) and not isinstance(raw_allowlist, (str, bytes)):
            allowlist = tuple(str(item) for item in raw_allowlist)
        elif raw_allowlist is None:
            allowlist = self.env_allowlist
        return SandboxOptions(timeout=timeout, memory_limit_mb=memory, env_allowlist=allowlist)


@dataclass(slots=True)
class VerifierOptions:
    """Settings passed to the VM verifier."""

    enabled: bool = True
    entrypoint: str = "solve"
    args: tuple[Any, ...] = ()
    limits: dict[str, Any] | None = None
    sandbox: SandboxOptions | None = None


@dataclass(slots=True)
class OutputOptions:
    """Persistence preferences for orchestrator runs."""

    directory: Path = field(default_factory=_default_output_dir)
    save_json: bool = True
    save_yaml: bool = False
    save_markdown: bool = False
    include_trace: bool = True


@dataclass(slots=True)
class TranslatorOptions:
    """Parameters controlling the natural language translation flow."""

    locale: str = "en"
    config: dict[str, Any] = field(default_factory=dict)
    accept_low_confidence: bool = False


@dataclass(slots=True)
class ExplanationOptions:
    """Renderer settings for synthesiser explanations."""

    auto_generate: bool = False
    renderer: str = "markdown"


@dataclass(slots=True)
class OrchestratorConfig:
    """Top-level orchestrator configuration bundle."""

    synthesizer: SynthesizerOptions = field(default_factory=SynthesizerOptions)
    verifier: VerifierOptions = field(default_factory=VerifierOptions)
    output: OutputOptions = field(default_factory=OutputOptions)
    translator: TranslatorOptions = field(default_factory=TranslatorOptions)
    explanation: ExplanationOptions = field(default_factory=ExplanationOptions)
    sandbox: SandboxOptions = field(default_factory=SandboxOptions)
    telemetry: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OrchestratorConfig":
        payload = dict(data or {})
        synthesizer = cls._build_synthesizer_options(payload.get("synthesizer"))
        sandbox_options = cls._build_sandbox_options(payload.get("sandbox"))
        verifier = cls._build_verifier_options(payload.get("verifier"), defaults=sandbox_options)
        output = cls._build_output_options(payload.get("output"))
        translator = cls._build_translator_options(payload.get("translator"))
        explanation = cls._build_explanation_options(payload.get("explanation"))
        telemetry = (
            dict(payload.get("telemetry", {}))
            if isinstance(payload.get("telemetry"), Mapping)
            else {}
        )
        return cls(
            synthesizer=synthesizer,
            verifier=verifier,
            output=output,
            translator=translator,
            explanation=explanation,
            sandbox=sandbox_options,
            telemetry=telemetry,
        )

    def merge(self, overrides: Mapping[str, Any] | None) -> "OrchestratorConfig":
        if not overrides:
            return self
        merged = self.to_dict()
        merged = _deep_update(merged, overrides)
        return OrchestratorConfig.from_mapping(merged)

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthesizer": {
                "config_path": (
                    str(self.synthesizer.config_path) if self.synthesizer.config_path else None
                ),
                "overrides": dict(self.synthesizer.overrides),
            },
            "verifier": {
                "enabled": self.verifier.enabled,
                "entrypoint": self.verifier.entrypoint,
                "args": list(self.verifier.args),
                "limits": (
                    dict(self.verifier.limits)
                    if isinstance(self.verifier.limits, Mapping)
                    else None
                ),
                "sandbox": (self.verifier.sandbox.to_dict() if self.verifier.sandbox else None),
            },
            "output": {
                "directory": str(self.output.directory),
                "save_json": self.output.save_json,
                "save_yaml": self.output.save_yaml,
                "save_markdown": self.output.save_markdown,
                "include_trace": self.output.include_trace,
            },
            "translator": {
                "locale": self.translator.locale,
                "config": dict(self.translator.config),
                "accept_low_confidence": self.translator.accept_low_confidence,
            },
            "explanation": {
                "auto_generate": self.explanation.auto_generate,
                "renderer": self.explanation.renderer,
            },
            "sandbox": self.sandbox.to_dict(),
            "telemetry": dict(self.telemetry),
        }

    @staticmethod
    def _build_synthesizer_options(data: Any) -> SynthesizerOptions:
        if not isinstance(data, Mapping):
            return SynthesizerOptions()
        config_path = _as_path(data.get("config_path"))
        overrides = (
            dict(data.get("overrides", {})) if isinstance(data.get("overrides"), Mapping) else {}
        )
        return SynthesizerOptions(config_path=config_path, overrides=overrides)

    @staticmethod
    def _build_verifier_options(
        data: Any, *, defaults: SandboxOptions | None = None
    ) -> VerifierOptions:
        if not isinstance(data, Mapping):
            return VerifierOptions(sandbox=defaults)
        enabled = bool(data.get("enabled", True))
        entrypoint = str(data.get("entrypoint", "solve"))
        args_raw = data.get("args", ())
        if isinstance(args_raw, Sequence) and not isinstance(args_raw, (str, bytes)):
            args = tuple(args_raw)
        else:
            args = ()
        limits = dict(data.get("limits", {})) if isinstance(data.get("limits"), Mapping) else None
        sandbox_overrides = (
            data.get("sandbox") if isinstance(data.get("sandbox"), Mapping) else None
        )
        base = defaults or SandboxOptions()
        sandbox = base.merged(sandbox_overrides)
        return VerifierOptions(
            enabled=enabled,
            entrypoint=entrypoint,
            args=args,
            limits=limits,
            sandbox=sandbox,
        )

    @staticmethod
    def _build_sandbox_options(data: Any) -> SandboxOptions:
        base = SandboxOptions()
        if not isinstance(data, Mapping):
            return base
        return base.merged(data)

    @staticmethod
    def _build_output_options(data: Any) -> OutputOptions:
        if not isinstance(data, Mapping):
            return OutputOptions()
        directory = _as_path(data.get("directory")) or _default_output_dir()
        save_json = bool(data.get("save_json", True))
        save_yaml = bool(data.get("save_yaml", False))
        save_markdown = bool(data.get("save_markdown", False))
        include_trace = bool(data.get("include_trace", True))
        return OutputOptions(
            directory=directory,
            save_json=save_json,
            save_yaml=save_yaml,
            save_markdown=save_markdown,
            include_trace=include_trace,
        )

    @staticmethod
    def _build_translator_options(data: Any) -> TranslatorOptions:
        if not isinstance(data, Mapping):
            return TranslatorOptions()
        locale = str(data.get("locale", "en"))
        config = dict(data.get("config", {})) if isinstance(data.get("config"), Mapping) else {}
        accept_low_confidence = bool(data.get("accept_low_confidence", False))
        return TranslatorOptions(
            locale=locale, config=config, accept_low_confidence=accept_low_confidence
        )

    @staticmethod
    def _build_explanation_options(data: Any) -> ExplanationOptions:
        if not isinstance(data, Mapping):
            return ExplanationOptions()
        auto_generate = bool(data.get("auto_generate", False))
        renderer = str(data.get("renderer", "markdown"))
        return ExplanationOptions(auto_generate=auto_generate, renderer=renderer)


@dataclass(slots=True)
class SynthesizerReport:
    """Summary returned by the synthesiser stage."""

    status: str
    reason: str
    steps: tuple[str, ...]
    visited: int
    expanded: int
    best_score: float
    pending: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    trace: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "steps": list(self.steps),
            "visited": self.visited,
            "expanded": self.expanded,
            "best_score": self.best_score,
            "pending": list(self.pending),
            "metadata": dict(self.metadata),
            "trace": [dict(item) for item in self.trace],
        }


@dataclass(slots=True)
class ProgramArtifact:
    """Executable payload produced by the synthesiser."""

    spec_id: str
    entrypoint: str
    module: Mapping[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "entrypoint": self.entrypoint,
            "module": dict(self.module),
            "metadata": dict(self.metadata),
            "path": str(self.path) if self.path else None,
        }


@dataclass(slots=True)
class VerificationReport:
    """Outcome returned by the verifier stage."""

    status: str
    entrypoint: str
    result: Any = None
    trace: Mapping[str, Any] | None = None
    error: str | None = None
    detail: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "entrypoint": self.entrypoint,
            "result": self.result,
        }
        if self.trace is not None:
            payload["trace"] = dict(self.trace)
        if self.error is not None:
            payload["error"] = self.error
        if self.detail:
            payload["detail"] = dict(self.detail)
        return payload


@dataclass(slots=True)
class TelemetrySnapshot:
    """Lightweight telemetry capture attached to a run."""

    metrics: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": dict(self.metrics),
            "extras": dict(self.extras),
        }


@dataclass(slots=True)
class SolutionReport:
    """Composite bundle returned by orchestrator runs."""

    spec_id: str
    spec: Mapping[str, Any]
    synthesizer: SynthesizerReport
    program: ProgramArtifact | None
    verifier: VerificationReport
    telemetry: TelemetrySnapshot
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    explanation: str | None = None
    translation: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "spec": dict(self.spec),
            "synthesizer": self.synthesizer.to_dict(),
            "program": self.program.to_dict() if self.program else None,
            "verifier": self.verifier.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "created_at": self.created_at.astimezone(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "explanation": self.explanation,
            "translation": (
                dict(self.translation) if isinstance(self.translation, Mapping) else None
            ),
        }

    def summary(self) -> str:
        return (
            f"spec={self.spec_id} synth={self.synthesizer.status} "
            f"verify={self.verifier.status} steps={len(self.synthesizer.steps)}"
        )


@dataclass(slots=True)
class ExportResult:
    """Information about exported artifacts."""

    paths: tuple[Path, ...]
    formats: tuple[str, ...]
    directory: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "formats": list(self.formats),
            "paths": [str(path) for path in self.paths],
        }


@dataclass(slots=True)
class TextRunResult:
    """Outcome bundle for natural language prompts."""

    translation: "TranslationOutcome"
    solution: SolutionReport | None
    status: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "translation": {
                "prompt": self.translation.prompt,
                "locale": self.translation.locale,
                "status": self.translation.status,
                "confidence": self.translation.confidence,
                "fallback": dict(self.translation.fallback) if self.translation.fallback else None,
            },
            "status": self.status,
            "notes": list(self.notes),
        }
        if self.translation.translation is not None:
            payload["translation"]["spec"] = dict(self.translation.translation.spec)
        if self.solution is not None:
            payload["solution"] = self.solution.to_dict()
        return payload


__all__ = [
    "ExplanationOptions",
    "ExportResult",
    "OrchestratorConfig",
    "OutputOptions",
    "ProgramArtifact",
    "SandboxOptions",
    "SolutionReport",
    "SynthesizerOptions",
    "SynthesizerReport",
    "TelemetrySnapshot",
    "TextRunResult",
    "TranslatorOptions",
    "VerificationReport",
    "VerifierOptions",
    "_deep_update",
]

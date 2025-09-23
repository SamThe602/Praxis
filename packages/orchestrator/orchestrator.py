"""High-level orchestration helpers tying together Praxis subsystems."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from packages.explanations import builder as explanation_builder
from packages.orchestrator import runner as orchestrator_runner
from packages.orchestrator import spec_loader
from packages.orchestrator.spec_loader import StructuredSpec
from packages.synthesizer.interface import DEFAULT_CONFIG_PATH, Synthesizer, build_settings
from packages.synthesizer.search import SearchResult
from packages.synthesizer.state import SynthState
from packages.telemetry import metrics as telemetry_metrics
from packages.utils import config as config_loader
from packages.utils.sandbox import SandboxError, SandboxExecutionError, SandboxTimeoutError
from packages.verifier.diff_checker import DiffConfig, DiffReport, compare
from packages.verifier.runner import VmExecutionError

from .text_interface import translate_prompt
from .types import (
    ExportResult,
    OrchestratorConfig,
    ProgramArtifact,
    SandboxOptions,
    SolutionReport,
    SynthesizerOptions,
    SynthesizerReport,
    TelemetrySnapshot,
    TextRunResult,
    VerificationReport,
    VerifierOptions,
    _deep_update,
)


def load_configuration(
    config_path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> OrchestratorConfig:
    """Return an :class:`OrchestratorConfig` from ``config_path`` and overrides."""

    data: Mapping[str, Any] | None = None
    if config_path is not None:
        data = config_loader.load_config(config_path)
    config = OrchestratorConfig.from_mapping(data)
    return config.merge(overrides)


def run_structured(
    spec_source: StructuredSpec | Mapping[str, Any] | str | Path,
    *,
    config: OrchestratorConfig,
    generate_explanation: bool | None = None,
) -> SolutionReport:
    """Execute the full orchestration pipeline for a structured spec."""

    structured = _coerce_spec(spec_source)
    synthesizer = _prepare_synthesizer(config.synthesizer)
    search_result = synthesizer.run(structured)
    synth_report, candidate = _summarise_search(structured, search_result)
    artifact = _build_program(structured.identifier, candidate, config.verifier.entrypoint)
    verifier_report = _run_verifier(artifact, config.verifier)
    telemetry = _collect_telemetry(
        {
            "spec_id": structured.identifier,
            "synth_status": synth_report.status,
            "verifier_status": verifier_report.status,
        }
    )

    report = SolutionReport(
        spec_id=structured.identifier,
        spec=structured.raw,
        synthesizer=synth_report,
        program=artifact,
        verifier=verifier_report,
        telemetry=telemetry,
    )

    should_explain = (
        generate_explanation
        if generate_explanation is not None
        else config.explanation.auto_generate
    )
    if should_explain:
        report.explanation = explanation_builder.build_explanation(report.to_dict())

    return report


def run_prompt(
    prompt: str,
    *,
    config: OrchestratorConfig,
    locale: str | None = None,
    generate_explanation: bool | None = None,
) -> TextRunResult:
    """Execute the natural language pipeline, including synthesis when possible."""

    translator_locale = locale or config.translator.locale
    translation = translate_prompt(
        prompt, locale=translator_locale, config=config.translator.config
    )
    notes: list[str] = []
    solution: SolutionReport | None = None

    if translation.translation is not None and (
        translation.status == "auto_accepted" or config.translator.accept_low_confidence
    ):
        structured = spec_loader.parse_structured_spec(translation.translation.spec)
        solution = run_structured(
            structured,
            config=config,
            generate_explanation=generate_explanation,
        )
        solution.translation = {
            "prompt": translation.prompt,
            "status": translation.status,
            "confidence": translation.confidence,
            "locale": translation.locale,
        }
    else:
        notes.append("translation_pending_confirmation")
        if translation.fallback:
            notes.append("fallback_available")

    status = "completed" if solution is not None else "pending_confirmation"
    return TextRunResult(
        translation=translation, solution=solution, status=status, notes=tuple(notes)
    )


def explain_report(report: SolutionReport | Mapping[str, Any]) -> str:
    """Return a Markdown explanation for ``report``."""

    payload = report.to_dict() if isinstance(report, SolutionReport) else dict(report)
    return explanation_builder.build_explanation(payload)


def verify_module(
    program: Mapping[str, Any],
    *,
    config: OrchestratorConfig | None = None,
    entry: str | None = None,
    args: Sequence[Any] | None = None,
    limits: Mapping[str, Any] | None = None,
) -> VerificationReport:
    """Run the verifier against ``program`` and return the outcome."""

    base_config = config or OrchestratorConfig.from_mapping(None)
    options = base_config.verifier
    entrypoint = entry or program.get("entry") or options.entrypoint
    program_args = args if args is not None else program.get("args", options.args)
    program_limits = limits if limits is not None else program.get("limits", options.limits)
    module_candidate = (
        program.get("module") if isinstance(program, Mapping) and "module" in program else program
    )
    if not isinstance(module_candidate, Mapping):
        raise TypeError("program must provide a mapping under 'module'")
    module_mapping = dict(module_candidate)

    local_options = VerifierOptions(
        enabled=True,
        entrypoint=entrypoint,
        args=(
            tuple(program_args)
            if isinstance(program_args, Sequence) and not isinstance(program_args, (str, bytes))
            else tuple(options.args)
        ),
        limits=dict(program_limits) if isinstance(program_limits, Mapping) else None,
        sandbox=base_config.sandbox,
    )
    artifact = ProgramArtifact(
        spec_id=str(program.get("spec_id", "ad-hoc")),
        entrypoint=entrypoint,
        module=module_mapping,
    )
    return _run_verifier(artifact, local_options)


def export_report(
    report: SolutionReport | Mapping[str, Any],
    directory: str | Path,
    *,
    formats: Sequence[str] = ("json",),
    basename: str | None = None,
) -> ExportResult:
    """Persist ``report`` under ``directory`` using the requested ``formats``."""

    payload = report.to_dict() if isinstance(report, SolutionReport) else dict(report)
    explanation = payload.get("explanation")
    spec_id = str(payload.get("spec_id", "run"))

    formats = tuple(dict.fromkeys(formats))  # preserve order, remove duplicates
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    base = basename or f"{spec_id}-{now_utc:%Y%m%d%H%M%S}"

    saved: list[Path] = []
    if "json" in formats:
        path = output_dir / f"{base}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        saved.append(path)
    if "yaml" in formats:
        path = output_dir / f"{base}.yaml"
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True, allow_unicode=True)
        saved.append(path)
    if "markdown" in formats:
        md_path = output_dir / f"{base}.md"
        explanation_text = explanation or explanation_builder.build_explanation(payload)
        md_path.write_text(explanation_text, encoding="utf-8")
        saved.append(md_path)

    return ExportResult(paths=tuple(saved), formats=formats, directory=output_dir)


def _coerce_spec(source: StructuredSpec | Mapping[str, Any] | str | Path) -> StructuredSpec:
    if isinstance(source, StructuredSpec):
        return source
    if isinstance(source, Mapping):
        return spec_loader.parse_structured_spec(source)
    return spec_loader.load_spec(source)


def _prepare_synthesizer(options: SynthesizerOptions) -> Synthesizer:
    config_path = options.config_path or DEFAULT_CONFIG_PATH
    config_data = config_loader.load_config(config_path)
    if options.overrides:
        config_data = _deep_update(config_data, options.overrides)
    settings = build_settings(config_data)
    return Synthesizer(settings=settings)


def _summarise_search(
    spec: StructuredSpec,
    result: SearchResult,
) -> tuple[SynthesizerReport, SynthState | None]:
    candidate = result.solution or result.best
    if result.solution is not None:
        status = "solved"
    elif candidate is not None:
        status = "partial"
    else:
        status = "infeasible"
    steps = tuple(candidate.steps) if candidate is not None else ()
    pending = (
        tuple(candidate.pending_requirements) if candidate is not None else tuple(spec.goal_tokens)
    )
    metadata = {
        "reason": result.reason,
        "best_score": result.best_score,
        "solution_found": result.solution is not None,
        "total_goals": len(spec.goal_tokens),
    }
    trace = tuple(_state_to_dict(state) for state in result.trace) if result.trace else ()
    report = SynthesizerReport(
        status=status,
        reason=result.reason,
        steps=steps,
        visited=result.visited,
        expanded=result.expanded,
        best_score=result.best_score,
        pending=pending,
        metadata=metadata,
        trace=trace,
    )
    return report, candidate


def _build_program(
    spec_id: str,
    state: SynthState | None,
    entrypoint: str,
) -> ProgramArtifact | None:
    if state is None:
        return None
    step_count = len(state.steps)
    module = {
        "constants": [],
        "functions": [
            {
                "name": entrypoint,
                "registers": 1,
                "stack_slots": 0,
                "contracts": [],
                "instructions": [
                    {
                        "opcode": "LoadImmediate",
                        "operands": {
                            "kind": "RegImmediate",
                            "value": [0, {"Scalar": {"Int": step_count}}],
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
    metadata = {
        "steps": list(state.steps),
        "covered_requirements": sorted(state.covered_requirements),
        "pending_requirements": list(state.pending_requirements),
        "latency_estimate": state.metadata.get("latency_estimate"),
        "reuse_hits": state.metadata.get("reuse_hits"),
    }
    return ProgramArtifact(spec_id=spec_id, entrypoint=entrypoint, module=module, metadata=metadata)


def _run_verifier(artifact: ProgramArtifact | None, options: VerifierOptions) -> VerificationReport:
    if artifact is None or not options.enabled:
        return VerificationReport(status="skipped", entrypoint=options.entrypoint)
    sandbox_options = options.sandbox or SandboxOptions()
    correlation_id = f"{artifact.spec_id}-{uuid.uuid4().hex}"
    task_payload = {
        "module": dict(artifact.module),
        "entry": options.entrypoint,
        "args": list(options.args),
        "limits": options.limits,
        "id": artifact.spec_id,
        "sandbox": sandbox_options,
        "correlation_id": correlation_id,
    }
    try:
        execution = orchestrator_runner.run(task_payload)
    except SandboxTimeoutError as exc:
        detail = {"message": str(exc), "timeout": exc.timeout}
        if exc.stdout:
            detail["stdout"] = exc.stdout
        if exc.stderr:
            detail["stderr"] = exc.stderr
        return VerificationReport(
            status="error",
            entrypoint=options.entrypoint,
            error="sandbox_timeout",
            detail=detail,
        )
    except SandboxExecutionError as exc:
        error_detail: dict[str, Any] = {"message": str(exc), "kind": exc.kind}
        if exc.stdout:
            error_detail["stdout"] = exc.stdout
        if exc.stderr:
            error_detail["stderr"] = exc.stderr
        if exc.traceback:
            error_detail["traceback"] = list(exc.traceback)
        error_detail["duration_ms"] = round(exc.telemetry.duration * 1000.0, 3)
        error_detail["memory_peak_kb"] = exc.telemetry.memory_peak_kb
        return VerificationReport(
            status="error",
            entrypoint=options.entrypoint,
            error=str(exc.kind),
            detail=error_detail,
        )
    except SandboxError as exc:
        return VerificationReport(
            status="error",
            entrypoint=options.entrypoint,
            error="sandbox_error",
            detail={"message": str(exc)},
        )
    except VmExecutionError as exc:
        detail = {"message": str(exc)}
        if exc.detail is not None:
            detail["detail"] = exc.detail
        return VerificationReport(
            status="error",
            entrypoint=options.entrypoint,
            error=exc.kind,
            detail=detail,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        return VerificationReport(
            status="error",
            entrypoint=options.entrypoint,
            error=exc.__class__.__name__,
            detail={"message": str(exc)},
        )

    diff_detail: dict[str, Any] | None = None
    baseline_payload = _extract_baseline(artifact)
    if baseline_payload is not None:
        candidate_payload = {"module": dict(artifact.module), "entry": options.entrypoint}
        if options.limits is not None:
            candidate_payload["limits"] = options.limits

        def _sandbox_behavioural_runner(
            *,
            module: Mapping[str, Any],
            entry: str,
            args: Sequence[Any],
            limits: Mapping[str, Any] | None = None,
        ):
            return orchestrator_runner.run(
                {
                    "module": module,
                    "entry": entry,
                    "args": list(args or ()),
                    "limits": limits,
                    "id": artifact.spec_id,
                    "sandbox": sandbox_options,
                    "correlation_id": f"{correlation_id}.diff",
                }
            )

        try:
            diff_report = compare(
                candidate_payload,
                baseline_payload,
                config=DiffConfig(
                    entrypoint=baseline_payload.get("entry") or options.entrypoint,
                    limits=options.limits,
                    behavioural_checks=1,
                    behavioural_runner=_sandbox_behavioural_runner,
                ),
            )
        except Exception as diff_exc:  # pragma: no cover - diff stage is best-effort
            diff_detail = {"equivalent": False, "error": str(diff_exc)}
        else:
            diff_detail = _diff_summary(diff_report)

    report_detail = {"diff": diff_detail} if diff_detail is not None else None
    return VerificationReport(
        status="ok",
        entrypoint=options.entrypoint,
        result=execution.return_value,
        trace=execution.trace,
        detail=report_detail,
    )


def _extract_baseline(artifact: ProgramArtifact) -> Mapping[str, Any] | None:
    metadata = artifact.metadata if isinstance(artifact.metadata, Mapping) else {}
    candidates = (
        metadata.get("baseline"),
        metadata.get("baseline_module"),
        metadata.get("reference"),
        metadata.get("reference_module"),
    )
    for entry in candidates:
        if isinstance(entry, Mapping):
            module = entry.get("module", entry)
            if not isinstance(module, Mapping):
                continue
            payload: dict[str, Any] = {"module": module}
            entry_value = entry.get("entry")
            if isinstance(entry_value, str):
                payload["entry"] = entry_value
            limits_value = entry.get("limits")
            if isinstance(limits_value, Mapping):
                payload["limits"] = dict(limits_value)
            return payload
    return None


def _diff_summary(diff: DiffReport) -> dict[str, Any]:
    payload: dict[str, Any] = {"equivalent": diff.equivalent}
    structural: dict[str, Any] = {"equivalent": diff.structural.equivalent}
    if not diff.structural.equivalent:
        structural["differences"] = list(diff.structural.differences)
    payload["structural"] = structural
    if diff.behavioural is not None:
        behavioural: dict[str, Any] = {
            "equivalent": diff.behavioural.equivalent,
            "checks_run": diff.behavioural.checks_run,
        }
        if diff.behavioural.mismatches:
            behavioural["mismatches"] = list(diff.behavioural.mismatches)
        payload["behavioural"] = behavioural
    return payload


def _collect_telemetry(extras: Mapping[str, Any] | None = None) -> TelemetrySnapshot:
    registry = telemetry_metrics.get_registry()
    snapshot = registry.summaries()
    return TelemetrySnapshot(metrics=snapshot, extras=dict(extras or {}))


def _state_to_dict(state: SynthState) -> dict[str, Any]:
    return {
        "steps": list(state.steps),
        "covered": sorted(state.covered_requirements),
        "pending": list(state.pending_requirements),
        "metadata": dict(state.metadata),
    }


__all__ = [
    "export_report",
    "explain_report",
    "load_configuration",
    "run_prompt",
    "run_structured",
    "verify_module",
]

# Verification Stack

The verification layer now executes candidate programs through the native VM so
that dynamic checks, traces, and contract violations are captured in a single
place.  Two Python entry points are available:

* `packages.verifier.runner.verify(program)` – lightweight helper used by the
  verifier to run bytecode modules and collect traces.
* `packages.orchestrator.runner.run(task)` – orchestrator-facing wrapper that
  forwards tasks to the same runtime.

Both helpers delegate to the Rust FFI (`praxis_vm_execute`) which expects the
JSON request described in [`docs/vm_bytecode.md`](vm_bytecode.md).  Errors are
normalised to `VmExecutionError` exceptions containing the native `kind`
identifier and optional `detail` payload.

## Telemetry instrumentation

The runner now emits telemetry on every VM invocation so downstream services
and dashboards can track search quality and runtime health:

* `praxis.vm.latency_ms` – wall clock latency (ms) grouped by entry point.
* `praxis.verifier.smt_calls` – SMT solver requests surfaced by the trace when
  verification engages symbolic back-ends.
* Hooks: `vm.execution.completed` (VM), `orchestrator.run.completed`
  (orchestrator wrapper).  Subscribe via `packages.telemetry.hooks.register_hook`
  to collect structured execution records.

All metrics flow through `packages.telemetry.metrics.emit` which forwards to the
JSONL exporter declared in `configs/logging.yaml`.  Dashboards can reuse
`packages.telemetry.dashboards.build_dashboard()` to obtain a printable summary
of the collected telemetry.

## Runtime integration

1. Callers assemble a bytecode module (via the compiler or fixtures) and provide
   inputs as serialised VM values.
2. The Python runner loads `libpraxis_ffi` (configurable through
   `PRAXIS_FFI_LIB`) and invokes the native interpreter.
3. On success the runner returns a `VmExecution` dataclass containing the result
   value and full execution trace, which downstream passes can analyse for
   coverage or resource accounting.
4. On failure the runner raises `VmExecutionError`, allowing orchestration code
   to surface meaningful diagnostics (instruction limit hit, contract violation,
   etc.).

## Failure propagation

| Error kind             | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `function_not_found`   | Entry point missing from the module.               |
| `invalid_register`     | Bytecode referenced an out-of-range register.      |
| `stack_underflow`      | Stack discipline violated in the bytecode.         |
| `sandbox_error`        | One of the sandbox limits was exceeded.            |
| `contract_violation`   | Runtime contract assertion returned false.         |
| `memory_error`         | Heap limit exceeded or invalid container key.      |
| `builtin_error`        | Intrinsic rejected the provided arguments.         |

Trace and metric information accompanies successful runs regardless of outcome
(for example, even failed contract checks include the executed prefix).

## Explanation outputs

Verifier traces together with synthesiser evidence can be converted into
human-readable narratives using `packages.explanations.builder.build_explanation`.
Provide a dictionary containing `program`, `trace`, and `verifier` entries; the
builder returns Markdown describing the high-level result, key execution steps,
and verification status. Example explanations are available in
`docs/explanation_examples.md`.

## Testing

The Python unit test `tests/python/test_verifier_runner.py` now exercises the
runner and validates its error-handling behaviour.  Integration tests can be
extended by constructing bytecode fixtures and asserting over the returned trace
object without touching the lower-level FFI.

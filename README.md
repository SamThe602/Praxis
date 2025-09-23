# Praxis
A non-LLM reasoning engine that compiles tasks into verifiable programs. Built for proof-carrying cognition.

## Quickstart
1. Install Poetry and the Rust toolchain (stable).
2. Bootstrap the environment:
   ```bash
   poetry install --no-root
   ```
3. Build the native runtime fast path (recommended):
   ```bash
   cargo build --release -p praxis_ffi
   ```
   This produces `target/release/libpraxis_ffi.*`; the Python bridge loads it automatically and
   falls back to the pure-Python analyzers if it is missing.
4. Run a structured spec end-to-end:
   ```bash
   poetry run python scripts/run_task.py tests/python/fixtures/specs/array_reverse.yaml --json
   ```
5. Translate a prompt and synthesise:
   ```bash
   poetry run python scripts/run_text_prompt.py --prompt "Sort the array ascending." --json
   ```

## CLI Highlights
- Structured spec: `poetry run python -m packages.orchestrator.cli structured path/to/spec.yaml --json`
- Natural language prompt: `poetry run python -m packages.orchestrator.cli prompt --prompt "Reverse the array."`
- Export artifacts: `poetry run python -m packages.orchestrator.cli export report.json --dir out/`
- Evaluate suites: `poetry run python scripts/eval_suite.py --suite regression`

All commands accept `--config` to supply a YAML configuration (defaults to `configs/orchestrator/default.yaml`). Use repeated `--set key=value` pairs to override nested config entries.

## Testing & Linting
- Aggregate linters: `poetry run bash scripts/lint_ci.sh`
- Python tests + coverage gate (85% over `packages/`): `poetry run pytest`
- Rust tests: `cargo test -q`
- Rust coverage thresholds: `cargo tarpaulin --locked -p praxis_vm_runtime --fail-under 70` and
  `cargo tarpaulin --locked -p praxis_verifier_native --fail-under 70`

## Evaluation Harness
Evaluation suites live under `configs/eval/`. Each run writes reports to `data/cache/eval_reports/` with an `index.json` summary. Example nightly invocation:
```bash
poetry run python scripts/eval_suite.py --suite arc_suite --suite leetcode_easy
```

## Repository Layout
- `packages/orchestrator/` – CLI, orchestrator core, DTOs
- `scripts/` – Automation entries (`run_task.py`, `run_text_prompt.py`, `eval_suite.py`)
- `configs/` – Orchestrator defaults, evaluation suites, synth configs
- `data/cache/` – Generated telemetry, evaluation reports

See `docs/architecture.md` for subsystem details.

## Native Fast Path
- The Rust crates expose a shared library (`praxis_ffi`) that accelerates static analysis,
  metamorphic relations, and property checkers; build it once with `cargo build --release -p
  praxis_ffi`.
- Python front-ends call through `packages.verifier.native_bridge`, which automatically loads the
  shared library from `target/{release,debug}` or a custom path provided via `PRAXIS_FFI_LIB`.
- When the library is unavailable, the bridge records a telemetry fallback and reverts to the
  pure-Python implementations so workflows continue to function.

## Sandbox Guarantees
- Every VM run executes inside the `Sandbox` with conservative defaults: 100k instructions, 1,024
  stack frames, call-depth 128, 200 ms wall-clock budget, and a 16 MiB tracked heap.
- Limits surface through the FFI (`limits` block on requests) and the orchestrator CLI
  (`--limits '{"instruction_limit": 200000}'`), allowing per-task adjustments while preserving
  deterministic behavior.
- Execution traces always include sandbox and heap metrics so downstream tooling can audit resource
  usage even when a limit trips.

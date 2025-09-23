# Proof Bridge

The proof bridge coordinates bytecode proof obligations across the SMT, Coq,
and Lean backends.  Each obligation is normalised into a
`ProofObligation` instance that captures the backend, goal payload, and optional
assumptions.  The normalised payload is hashed to produce a stable cache key, so
identical goals emitted by different passes deduplicate automatically.

## Execution Flow

1. `packages.verifier.proof_bridge.discharge_obligations` harvests the
   `proof_obligations` list attached to a compiled program.  Duplicate entries
   collapse through `ObligationQueue`, keeping insertion order for telemetry.
2. For each unique obligation the bridge first consults the on-disk cache under
   `data/processed/proofs/`.  Cached hits short-circuit immediately and are
   flagged on the resulting `ProofResult` so downstream consumers can skip
   redundant diagnostics.
3. On cache misses the orchestrator selects the appropriate prover client
   (`SmtClient`, `CoqClient`, or `LeanClient`).  Clients accept injectable
   `runner` callables which makes them trivial to stub in unit tests.  The
   default runners simply return an `unknown` status so callers can decide
   whether a backend is optional or required.
4. Every prover response is converted into a serialisable `ProofResult`,
   persisted to disk, and surfaced via the aggregated `ProofBridgeResult`
   summary.  The summary tracks per-status counters (`proved`, `refuted`,
   `unknown`, `error`) plus the number of cached hits.

## Cache Layout

Cached certificates live at `data/processed/proofs/<sha256>.json`.  The hash is
computed from the backend, obligation payload, and assumptions—metadata such as
user-facing identifiers is excluded so logically equivalent goals reuse the same
entry.  The JSON payload mirrors `ProofResult.to_dict()` and therefore carries
counterexamples, diagnostics, and execution timing when available.

During tests you can redirect the cache location by passing ``cache_dir`` to
`discharge_obligations`, letting the suite use a temporary directory while the
production path remains the project default.

## Testing Strategy

The Python test suite (`tests/python/test_proof_bridge.py`) exercises the bridge
with mocked provers to validate deduplication and cache reuse.  Because the
clients only depend on plain callables they can be patched with lightweight
stubs or even synchronous HTTP shims without touching the orchestrator logic.

# Praxis Program Synthesis VM — Technical Specification (Omega Revision)

## 0. Guiding Principles
- **Trustworthy by construction**: every emitted program is sandboxed, reproducible, and accompanied by machine-checkable evidence.
- **Search before guess**: treat cognition as guided exploration over executable artifacts, not next-token prediction.
- **Composable stack**: DSL, search, guidance, verification, translation, and proof services are separable modules with clear contracts.
- **Frugal compute**: all core training/inference flows must run on a single 24 GB GPU plus commodity CPU; elastic scale-out is optional.
- **Observability-first**: every run emits structured telemetry, traces, provenance hashes, and config digests.
- **Human leverage**: design hooks for human preference signals, curriculum shaping, and knowledge sharing from day one.

## 1. Product Vision & Personas
### Vision
Deliver an intelligence engine that converts natural-language or structured STEM tasks into verified micro-programs with deterministic behavior, beating state-of-the-art LLMs on spec-driven reasoning, correctness, latency, and transparency.

### Personas
- **Research Engineer** – demands fast, verifiably correct algorithmic solutions.
- **Verification Lead** – enforces proof-carrying patches in CI/CD pipelines.
- **Curriculum Designer** – crafts new adversarial tasks to expand capability fronts.
- **Human Feedback Curator** – reviews borderline solves and supplies preference labels.
- **Infra Operator (secondary)** – maintains training/telemetry stack, manages hardware budgets.

### Core Use Cases
1. Competitive programming assistant with guaranteed AC submissions and performance budgets.
2. Verified helper functions for internal pipelines (ETL, analytics) with machine-enforced contracts.
3. Auto-generated proofs or SMT certificates for safety-critical routines.
4. Natural-language tutoring: users describe problems in English; Praxis returns code, verification trace, and explanation.

## 2. Scope & Non-Goals
### In Scope (MVP+Enhancements)
- Typed DSL + bytecode VM with arrays, maps, matrices, graphs, control flow, contracts, resource annotations.
- Hybrid search (heuristics + neural guidance + retrieval reuse) with multi-pass verification (static, dynamic, SMT/Coq optional).
- Natural-language-to-spec translator with constrained decoding and schema validation.
- Preference learning head ingesting human feedback, integrated into PPO reward shaping.
- Latency-aware scoring: optimize for correctness and runtime simultaneously.
- Explanation engine templating verifier traces into structured natural-language outputs.
- CLI + Python API covering text prompts and structured specs.
- Training pipelines: supervised imitation, PPO, meta-curriculum self-play, human feedback fine-tuning.
- Observability, benchmarking, documentation, and knowledge base sharing.

### Out of Scope (until future cycles)
- General open-domain conversation or unconstrained creative writing.
- Arbitrary language synthesis (beyond optional Python transpilation for debugging).
- Full-scale multi-node distributed training (only single-node + optional multiprocessing).
- Real-world robotics/continuous control (future domain packs).

## 3. Architectural Overview
```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Praxis Cognitive Stack                             │
│ ┌────────────┐   ┌────────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│ │  NL Parser │→→│ Spec Loader│→→│ Synthesizer│→→│ VM Runtime│→→│ Verifier │ │
│ └────────────┘   └────────────┘   └──────────┘   └──────────┘   └─────────┘ │
│        ↑              ↓               ↓             ↑           ↘   ↙       │
│  Prompt Cache   Retrieval Engine   Neural Guide   Proof Bridge   Feedback    │
│        │              ↑               ↓             │             ↓         │
│ ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐ ┌──────────┐ │
│ │ Human Loop │←←│ Experience │←←│ Telemetry   │←│ SMT/Coq  │ │ Explainer│ │
│ └────────────┘  │  Logger    │  │ Pipelines  │  └──────────┘ └──────────┘ │
│                 └────────────┘  └────────────┘                             │
└────────────────────────────────────────────────────────────────────────────┘
```

## 4. Component Deep Dive
### 4.1 DSL & AST
- **Primitives**: `int`, `float`, `bool`, `string`, `vector<T>`, `matrix<T>`, `map<K,V>`, `tuple`, `struct`.
- **Control**: `if/else`, `match`, `for`, `while`, bounded recursion, `lambda`, `return`.
- **Annotations**: `@time_bound`, `@cost`, `@pre`, `@post`, `@requires`, `@ensures`, `@smt_hint`.
- **AST Nodes**: `Module`, `FunctionDecl`, `Contract`, `Block`, `Let`, `Assign`, `Loop`, `Conditional`, `MatchArm`, `Call`, `BuiltinCall`, `Lambda`, `Literal`, `BinaryOp`, `UnaryOp`, `Comprehension`.
- **Semantic Checks**: lexical scope, Hindley–Milner inference, contract propagation, resource annotation reconciliation.
- **Bytecode**: stack/register hybrid with structured loop markers, container ops, contract assertions, SMT hook opcodes.
- **Serialization**: canonical JSON + hashed AST IDs for caching, retrieval, embeddings.

### 4.2 Natural-Language Translation Layer
- **Parser**: constrained decoder (small LLaMA/Mixtral) using grammar-based decoding into intermediate representation.
- **Schema Validator**: ensures NL outputs conform to YAML schema; fallback to structured spec prompts.
- **Glossary**: user-extensible mapping of domain terms to DSL constructs.
- **Confidence Scoring**: log-likelihood + constraint satisfaction; low confidence triggers human confirmation or structured form fill.
- **Outputs**: YAML spec + optional example set for synthesizer.

### 4.3 Program Synthesizer
- Beam search (default beam=128) with retrieval priors, neural guidance, and heuristic scoring.
- Node scoring: `score = α·policy + β·heuristic + γ·value – δ·resource_penalty – ε·latency_penalty + ζ·reuse_bonus`.
- Expansion operators: `AddStatement`, `RefineExpression`, `InsertContractGuard`, `InlineSnippet`, `SpecializeLoop`, `ReparameterizeLambda`, `ApplyProofHint`.
- Pruning: static typing, contract violation, e-graph dedupe, partial evaluation, static analyzer rejection (see 4.5).
- Retrieval integration: vector-search over solved programs; candidate snippets inserted via `InlineSnippet` operator.
- Feedback ingestion: counterexamples, SMT countermodels, performance regression hints.

### 4.4 Neural Guidance & Preference Learning
- Transformer encoder (8 layers, hidden 512, heads 12) with rotary positional and type embeddings.
- Inputs: spec tokens, AST tokens, execution sketch vectors, feedback signatures, retrieval context embeddings.
- Heads: policy (256 actions), value (success probability), auxiliary guard predictor, latency regressor, human preference head.
- Training regimes: supervised imitation (2M samples), PPO (clip=0.2), human preference fine-tuning (DPO/Implicit Preferences), auxiliary latency regression.
- Meta-curriculum controller adjusts sampling weights via success rates, novelty, and failure taxonomy.

### 4.5 Verification Stack (Multi-Pass)
1. **Static Analyzer** (`packages/verifier/static_analysis.py`): symbolic execution/abstract interpretation for quick rejection.
2. **VM Execution**: sandboxed run with deterministic trace, coverage metrics.
3. **Property & Metamorphic Tests**: invariants, perturbations, equivalence vs baseline.
4. **Proof Bridge**: optional SMT/Coq verification, proof artifact generation, countermodel retrieval.
- Feedback objects include failure type, minimal counterexample, suggested guard, optional SMT model.

### 4.6 Proof Bridge (SMT/Coq)
- gRPC clients to Z3/CVC5 (SMT) and Lean/Coq via Server (proof automation) for critical contracts.
- Triggered when spec includes `@requires`/`@ensures` with `critical` severity or config demands formal proof.
- Caches proof obligations; stores certificates under `data/processed/proofs/`.
- Countermodels fed back into synthesizer to guide search.

### 4.7 Latency-Aware Execution
- VM instrumentation collects instruction counts, heap usage, loop iterations.
- Latency regressor head predicts expected runtime; penalty incorporated into scoring and RL reward.
- Contracts allow `@time_bound(x)` enforcement; verifier flags overages.

### 4.8 Human Feedback Loop
- Triage queue collects borderline solves (low confidence, multiple guards, large latency) for review.
- CLI/Notebook UI allows rating (good/bad) and textual comments.
- Feedback ingested into preference dataset powering reward model and retrieval prioritization.

### 4.9 Explanation Engine
- Templates program + trace + verifier evidence into structured natural-language explanation.
- Output includes summary, guard rationale, performance report, proof highlights.
- Supports multi-format (Markdown, JSON, CLI) for integration.

### 4.10 Knowledge Base & Retrieval
- Content-addressed solved programs with spec embeddings and telemetry stats.
- Retrieval pipeline computes dense vector (spec+trace) via encoder; stored in FAISS/ScaNN index.
- Snippet marketplace CLI exports/imports KB shards.

## 5. Repository Blueprint
```
Praxis/
├── README.md
├── Spec.md
├── pyproject.toml
├── poetry.lock
├── Cargo.toml
├── Makefile
├── .env.example
├── configs/
│   ├── synth/default.yaml
│   ├── synth/search_profiles/{fast.yaml, thorough.yaml, curriculum.yaml, smt_heavy.yaml}
│   ├── synth/retrieval.yaml
│   ├── training/{supervised.yaml, rl.yaml, preference.yaml}
│   ├── translator/{nl_default.yaml, glossary.yaml}
│   ├── verifier/{static.yaml, smt.yaml}
│   ├── eval/{arc_suite.yaml, leetcode_easy.yaml, stress.yaml, regression.yaml}
│   └── logging.yaml
├── docs/
│   ├── architecture.md
│   ├── dsl.md
│   ├── vm_bytecode.md
│   ├── verifier.md
│   ├── proof_bridge.md
│   ├── translator.md
│   ├── training_playbook.md
│   ├── testing_strategy.md
│   ├── data_generation.md
│   ├── human_feedback.md
│   └── glossary.md
├── scripts/
│   ├── bootstrap_env.sh
│   ├── run_task.py
│   ├── run_text_prompt.py
│   ├── train_supervised.py
│   ├── train_rl.py
│   ├── train_preference.py
│   ├── eval_suite.py
│   ├── export_checkpoint.py
│   ├── seed_curriculum.py
│   ├── ingest_feedback.py
│   ├── sync_kb.py
│   └── lint_ci.sh
├── data/
│   ├── raw/{arc/, leetcode/, natural_prompts/, README.md}
│   ├── processed/{curriculum_v1/, benchmarks/, kb/, proofs/, checkpoints/}
│   └── cache/{vm_traces/, search_runs/, translator/, crashes/, eval_reports/}
├── notebooks/
│   ├── dsl_playground.ipynb
│   ├── nl_translation_eval.ipynb
│   ├── search_diagnostics.ipynb
│   ├── policy_eval.ipynb
│   ├── preference_analysis.ipynb
│   └── telemetry_dashboard.ipynb
├── packages/
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── cli.py
│   │   ├── orchestrator.py
│   │   ├── spec_loader.py
│   │   ├── text_interface.py
│   │   ├── runner.py
│   │   └── types.py
│   ├── translator/
│   │   ├── __init__.py
│   │   ├── decoder.py
│   │   ├── schema_validator.py
│   │   ├── glossary.py
│   │   ├── confidence.py
│   │   └── cache.py
│   ├── dsl/
│   │   ├── __init__.py
│   │   ├── grammar.py
│   │   ├── ast.py
│   │   ├── type_system.py
│   │   ├── serializer.py
│   │   ├── transpiler.py
│   │   └── generators.py
│   ├── synthesizer/
│   │   ├── __init__.py
│   │   ├── search.py
│   │   ├── frontier.py
│   │   ├── expansions.py
│   │   ├── heuristics.py
│   │   ├── pruning.py
│   │   ├── retrieval.py
│   │   ├── feedback.py
│   │   ├── scoring.py
│   │   └── interface.py
│   ├── guidance/
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── embeddings.py
│   │   ├── dataset.py
│   │   ├── curriculum.py
│   │   ├── trainer_supervised.py
│   │   ├── trainer_rl.py
│   │   ├── trainer_preference.py
│   │   └── utils.py
│   ├── verifier/
│   │   ├── __init__.py
│   │   ├── static_analysis.py
│   │   ├── contracts.py
│   │   ├── runner.py
│   │   ├── property_tests.py
│   │   ├── metamorphic.py
│   │   ├── diff_checker.py
│   │   ├── proof_bridge.py
│   │   └── reporters.py
│   ├── proof_bridge/
│   │   ├── __init__.py
│   │   ├── smt_client.py
│   │   ├── coq_client.py
│   │   ├── lean_client.py
│   │   ├── obligations.py
│   │   └── cache.py
│   ├── telemetry/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   ├── hooks.py
│   │   ├── exporters.py
│   │   ├── analyzers.py
│   │   └── dashboards.py
│   ├── explanations/
│   │   ├── __init__.py
│   │   ├── templates.py
│   │   ├── builder.py
│   │   └── renderers.py
│   ├── feedback/
│   │   ├── __init__.py
│   │   ├── queue.py
│   │   ├── annotator_api.py
│   │   ├── storage.py
│   │   └── sampler.py
│   └── utils/
│       ├── __init__.py
│       ├── config.py
│       ├── hashing.py
│       ├── ids.py
│       ├── sandbox.py
│       ├── timers.py
│       ├── serialization.py
│       └── concurrency.py
├── rust/
│   ├── vm_runtime/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, bytecode.rs, compiler.rs, ir.rs, interpreter.rs, memory.rs, trace.rs, builtins.rs, sandbox.rs}
│   ├── verifier_native/
│   │   ├── Cargo.toml
│   │   └── src/{lib.rs, checkers.rs, metamorphic.rs, static.rs}
│   └── ffi/
│       ├── Cargo.toml
│       └── src/lib.rs
├── tests/
│   ├── python/
│   │   ├── test_orchestrator.py
│   │   ├── test_text_interface.py
│   │   ├── test_translator_decoder.py
│   │   ├── test_translator_validator.py
│   │   ├── test_dsl_parser.py
│   │   ├── test_type_system.py
│   │   ├── test_serializer.py
│   │   ├── test_synthesizer_frontier.py
│   │   ├── test_synthesizer_search.py
│   │   ├── test_retrieval.py
│   │   ├── test_guidance_supervised.py
│   │   ├── test_guidance_rl.py
│   │   ├── test_guidance_preference.py
│   │   ├── test_verifier_static.py
│   │   ├── test_verifier_runner.py
│   │   ├── test_proof_bridge.py
│   │   ├── test_explanations.py
│   │   ├── test_feedback_queue.py
│   │   ├── test_telemetry_hooks.py
│   │   ├── test_end_to_end_structured.py
│   │   ├── test_end_to_end_text.py
│   │   └── fixtures/
│   │       ├── specs/{array_reverse.yaml, histogram.yaml, graph_path.yaml, dp_min_cost.yaml, matrix_mult.yaml}
│   │       ├── prompts/{text_sorting.json, text_graph.json}
│   │       └── programs/{array_reverse.dsl, histogram.dsl, graph_path.dsl}
│   └── rust/
│       ├── vm_runtime/integration.rs
│       ├── verifier_native/properties.rs
│       └── verifier_native/static.rs
└── .github/
    ├── workflows/{ci.yaml, nightly.yaml}
    └── ISSUE_TEMPLATE.md
```

## 6. File-Level Responsibilities
*(Full mapping provided in Appendix A with new modules; highlights for new additions:)*
- `packages/orchestrator/text_interface.py`: orchestrates NL parsing, confidence gating, fallback prompts.
- `packages/translator/*.py`: NL decoding pipeline, glossary management, translation cache.
- `packages/synthesizer/scoring.py`: composite scoring function blending policy, heuristics, latency, reuse, preferences.
- `packages/guidance/trainer_preference.py`: DPO/implicit preference training loop using human labels.
- `packages/verifier/static_analysis.py`: symbolic execution/abstract interpretation pass.
- `packages/verifier/proof_bridge.py`: orchestrates proof obligations; interfaces with proof_bridge package.
- `packages/proof_bridge/*.py`: SMT/Coq/Lean clients, caching, obligation lifecycle.
- `packages/explanations/*.py`: build structured textual explanations from traces and verifier reports.
- `packages/feedback/*.py`: manage human review queue, store ratings, sample for training.
- `packages/utils/concurrency.py`: thread/process helpers for translator/synthesizer parallelism.

## 7. Data Schemas & DSL Specification
### 7.1 Problem Spec Schema (YAML/JSON)
*(extends prior schema with `natural_prompt`, `proof_required`, `latency_target`, see docs/translator.md)*

### 7.2 Natural Prompt Schema
```json
{
  "prompt_id": "string",
  "language": "en",
  "user_text": "String description of task",
  "examples": [ {"input": {...}, "output": {...}} ],
  "constraints": ["must run in <100ms", "avoid recursion"],
  "metadata": {"difficulty": "medium", "tags": ["arrays"]}
}
```

### 7.3 Execution Trace Schema
*(unchanged plus contract events, SMT proof references, latency metrics)*

### 7.4 Human Feedback Record
```json
{
  "solution_id": "hash",
  "spec_id": "id",
  "rating": "good" | "bad" | "ambiguous",
  "notes": "optional string",
  "timestamp": "ISO",
  "reviewer": "user handle"
}
```

## 8. Search Algorithm Specifics
- Frontier uses concurrent beams per retrieval cluster; fallback min-heap for exploration.
- Heuristic features: spec coverage, guard debt, AST depth, latency estimate, retrieval similarity.
- Static analysis integration prunes invalid nodes before VM execution.
- SMT countermodels inserted as negative constraints to steer search away from failing regions.
- Parallel expansion uses work-stealing queue with concurrency helpers.
- Partial solution fallback stores best verified attempt if global timeout hit.

## 9. Neural Training & Preference Program
### 9.1 Supervised Stage
- 2 M synthetic triples balanced across difficulty; includes NL→spec pairs for translator pretraining.
- Loss: cross-entropy (policy) + MSE (value) + BCE (guard) + MSE (latency).

### 9.2 RL Stage
- PPO clip 0.2, entropy 0.01, value coeff 0.5, latency penalty scaled by instruction count.
- Meta-curriculum increments tier when success ≥85%; inject adversarial tasks generated from failure taxonomy.

### 9.3 Preference Stage
- Human-labeled pairs feed into implicit preference or DPO loss; reward head fine-tuned.
- Active learning sampler surfaces uncertain solutions for review (low confidence × high impact).

### 9.4 Translator Pretraining
- Mixed objective: translation likelihood + schema compliance + slot accuracy.
- Backed by synthetic NL prompts seeded from spec templates.

## 10. Evaluation Strategy
- **Daily smoke**: structured + text prompts, fast profile.
- **Nightly**: full benchmark (ARC, LeetCode, synthetic), dual profiles (fast/thorough).
- **Proof cadence**: weekly proof-heavy suite to ensure SMT/Coq stability.
- **Explanation QA**: sample explanations rated weekly for clarity via human review.
- **Leaderboards**: telemetry dashboards compare heuristics, retrieval toggles, NLP translator accuracy.

## 11. Telemetry & Observability
- Metrics: `praxis.translator.accuracy`, `praxis.synth.search_nodes`, `praxis.vm.latency_ms`, `praxis.verifier.smt_calls`, `praxis.feedback.pending`, `praxis.rl.reward`, `praxis.pref.loss`.
- Exporters: JSONL, Prometheus, MLflow; optional Grafana dashboards.
- Alert thresholds: NL confidence <0.7 (rolling), SMT failure rate >10%, preference queue >100 pending, latency p95 >10s.

## 12. Testing & Quality Strategy
*(expands prior pyramid)*
- Translator tests (decoder, validator, glossary accuracy, NL confidence thresholds).
- Retrieval tests (embedding similarity, snippet reuse correctness).
- Proof bridge tests with mocked SMT servers, countermodel injection, certificate caching.
- Explanation tests verifying template output vs golden text.
- Human feedback pipeline tests with mocked queues and integration to preference trainer.
- Determinism across text interface end-to-end runs.
- Performance/regression harness includes NL prompts and proof-heavy cases.

## 13. Roadmap (12 Weeks)
| Week | Deliverables | Key Files |
| --- | --- | --- |
| 1 | DSL grammar/AST, VM skeleton, NL translator stub returning structured forms | `packages/dsl/*`, `rust/vm_runtime/src/interpreter.rs`, `packages/translator/decoder.py` |
| 2 | Type system, bytecode lowering, static analysis scaffold | `packages/dsl/type_system.py`, `rust/vm_runtime/src/bytecode.rs`, `packages/verifier/static_analysis.py` |
| 3 | BFS synthesizer baseline, spec loader (structured + text), translator validator | `packages/synthesizer/search.py`, `packages/orchestrator/text_interface.py`, `tests/python/test_translator_validator.py` |
| 4 | VM sandboxing, trace emission, NL translator training data pipeline | `rust/vm_runtime/src/sandbox.rs`, `packages/translator/cache.py` |
| 5 | Neural supervised training, retrieval index pipeline | `packages/guidance/models.py`, `packages/synthesizer/retrieval.py`, `scripts/train_supervised.py` |
| 6 | Guided search integration, latency-aware scoring, initial explanations | `packages/synthesizer/scoring.py`, `packages/explanations/builder.py` |
| 7 | Verifier property/metamorphic + static analyzer + proof bridge MVP | `packages/verifier/property_tests.py`, `packages/proof_bridge/smt_client.py` |
| 8 | Knowledge base marketplace + meta-curriculum + telemetry dashboards | `packages/synthesizer/retrieval.py`, `packages/guidance/curriculum.py`, `packages/telemetry/dashboards.py` |
| 9 | Preference data pipeline + trainer | `packages/feedback/*`, `packages/guidance/trainer_preference.py`, `scripts/train_preference.py` |
| 10 | Full benchmark harness (structured + text) + proof stress suite | `scripts/eval_suite.py`, `configs/eval/regression.yaml` |
| 11 | Docs polish, CLI UX, explanation output, human feedback playbook | `docs/*.md`, `packages/orchestrator/cli.py` |
| 12 | Hardening, kill-criteria review, demo, backlog grooming | `tests/python/test_end_to_end_*`, `docs/roadmap.md` |

## 14. Risk Registry & Mitigations
| Risk | Signal | Mitigation | Owner |
| --- | --- | --- | --- |
| NL translation ambiguity | Low confidence scores, human corrections spike | Active glossary, constrained decoding, fallback to structured prompts | Translator Lead |
| SMT/Coq integration flakiness | Proof timeouts >30%, countermodel mismatches | Cache obligations, selective invocation, mock servers in tests | Proof Lead |
| Retrieval drift | Reuse rate drop, stale snippets | Periodic re-embedding, freshness scoring, human curation of KB | Synth Lead |
| Human feedback scarcity | Pending queue >100, preference loss unstable | Active learning sampler, lightweight UI, reward model fallback | HF Curator |
| Latency regressions | p95 > target | Performance regression tests, adjust scoring weights, optimize VM | VM Lead |

## 15. Operational Considerations
- Poetry + Rust toolchains; Hydra for configs.
- Secrets: translator model keys, SMT endpoints, telemetry tokens (managed via `.env`, GitHub secrets).
- CI matrix includes text-mode tests, SMT mocks, coverage enforcement.
- Hardware budget planning: supervised (~30 GPU-hours), RL (~40–60), preference fine-tune (~10), translator (~10) → ~$45–$60 in cloud spend or personal GPU usage.

## 16. Kill Criteria (Pivot Triggers)
1. Guided + retrieval search fails to beat unguided by ≥15% after Week 6.
2. NL translator accuracy <80% schema compliance after Week 6 despite glossary updates.
3. SMT/Coq verification incorrect or flaky for >3 consecutive releases.
4. Meta-curriculum fails (benchmark success <40% while curriculum >80%).
5. Preference learning introduces regressions (success rate drop >10%) without recoverable tuning.

## 17. Future Extensions
- Domain packs (symbolic math, robotics) with specialized DSL primitives.
- Differentiable interpreter for gradient-guided synthesis.
- Multi-agent solver ensembles (diverse heuristics/co-operative search).
- Federated knowledge sharing (secure snippet exchange).
- On-device quantized inference for edge deployments.

## Appendix A: File Responsibility Matrix (Highlights)
- Detailed mapping of every file to owner + responsibilities stored in `docs/architecture.md` (table generated from this spec).
- Automated script (`scripts/sync_kb.py`) updates knowledge base metadata and documentation tables.

## Appendix B: Test Coverage & Ownership
- Coverage dashboards in `notebooks/telemetry_dashboard.ipynb`.
- Ownership table in `docs/testing_strategy.md` assigns each test suite to responsible engineer.

---
All changes confined to `Spec.md`.

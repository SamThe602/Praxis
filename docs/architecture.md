# Praxis Architecture

## Retrieval Pipeline
- `packages/guidance/embeddings.py` generates deterministic 128-dim hashed embeddings from structured specs; the helper returns both the vector and the tokenised view for debugging.
- `packages/synthesizer/retrieval.py` maintains a JSON knowledge base under `data/processed/kb/index.json`, providing CRUD operations and cosine-similarity search via the in-process `KnowledgeBase` class.
- Retrieval results are merged into the spec through `apply_retrieval`, which augments `reuse_pool` and records a provenance/score payload in `spec.metadata["retrieval"]`.

## Search Reuse Integration
- During `Synthesizer.run` the spec is first enriched with retrieval data; the BFS expansion logic enumerates all snippets in the `reuse_pool` via the `InlineSnippet` operator.
- Expansion metadata carries `reuse_hits`, `reuse_rank`, and `snippet_reuse` so `scoring.composite_score` can reward high-similarity snippets deterministically.
- `ScoreWeights.reuse_bonus` (configurable) multiplies the reuse feature, ensuring snippet-backed branches surface earlier in the frontier ordering.

## Knowledge Base Tooling
- The sync utility (`scripts/sync_kb.py`) offers `export`, `import`, and `list` commands to manage shards. Imports support overwrite semantics, while exports emit portable JSON suitable for sharing or CI caching.
- Set `PRAXIS_KB_PATH` to redirect operations (including tests) to an alternate knowledge base root without touching the default workspace copy.

## Persistence Layout
- Default root: `data/processed/kb/`
  - `index.json`: primary store containing `dimension`, optional metadata, and an array of snippet entries `{spec_id, snippet, embedding, metadata}`.
- Embedding vectors are L2-normalised float lists compatible with simple Annoy/FAISS migrations when scaling beyond the local runner.

## Orchestrator CLI
- `packages/orchestrator/cli.py` wires a `praxis` CLI with commands for structured specs, natural language prompts, verification, explanations, and artifact export.
- Structured runs load YAML specs via `packages/orchestrator/spec_loader`, synthesise programs, verify them with the VM, capture telemetry, and optionally emit Markdown explanations.
- Prompt runs translate text with `packages/orchestrator/text_interface`, fall back to form-fill payloads if confidence drops below the configured threshold, and share the same synthesis/verification/export path when the translation is accepted.
- Configuration mirrors Hydra-style overrides: pass `--config path/to/config.yaml` alongside repeated `--set group.key=value` flags to tweak nested options.

## Evaluation Harness
- `scripts/eval_suite.py` executes suites defined under `configs/eval/`, supporting mixed prompt/spec tasks and per-task overrides (locale, explanation, export formats).
- Results land in `data/cache/eval_reports/<suite>/`, each with an `index.json` summary referencing generated artifacts (JSON/Markdown) and task statuses.
- The nightly CI job reuses the harness to exercise representative suites, producing regression-friendly artifacts.

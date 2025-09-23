# Human Feedback Workflow

The human feedback stack is intentionally lightweight for the unit tests yet
echoes the production dataflow.  All state lives in memory and interactions are
thread-safe to support concurrent CLI usage during development.

## Components
- `packages.feedback.queue` holds pending review items in a priority queue.
- `packages.feedback.annotator_api` exposes helper functions for the tooling and
  scripts, including enqueueing new work and submitting labels.
- `packages.feedback.storage` maintains a normalised list of feedback records
  (solution, spec, rating, metadata) with simple helper methods for inspection.
- `packages.feedback.sampler` retrieves the highest-priority items either by
  peeking (`consume=False`) or by popping them from the queue.

## Typical Loop
1. Generate JSONL feedback artifacts (e.g. model outputs requiring review).
2. Ingest them: `python -m scripts.ingest_feedback path/to/records.jsonl --enqueue`.
3. An annotator process requests work via `annotator_api.request_next_item()`.
4. After reviewing, the annotator calls `annotator_api.submit_feedback(...)` to
   persist the rating and optionally requeue the solution.

The ingestion script accepts `--priority` to bump urgent items.  Stored records
are returned as dataclasses, keeping downstream consumers type-safe and
serialisable.

## Training Integration
Preference training jobs can use `packages.feedback.sampler.sample_for_review()`
to fetch fresh human-labelled examples before reformatting them into pairs for
`train_preference`.  Because the queue is deterministic, repeated calls with the
same parameters yield identical batches—ideal for reproducible unit tests.

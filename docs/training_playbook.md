# Training Playbook

This playbook documents the lightweight, deterministic training loops that back
Praxis' unit tests.  The flows mirror the high-level production stages while
remaining CPU-only and completing within a few seconds.

## Supervised Imitation
- **Config:** `configs/training/supervised.yaml`
- **Entry point:** `python -m scripts.train_supervised`
- **Objective:** cross-entropy for policy logits plus auxiliary value, guard, and
  latency heads.
- **Output:** checkpoint stored in `data/processed/checkpoints/supervised` and a
  metrics JSON summary printed to stdout.

The dataset is generated on the fly with deterministic seeds.  Adjust the
training hyperparameters or model dimensions via the YAML file to explore
alternatives.

## Reinforcement Learning (PPO)
- **Config:** `configs/training/rl.yaml`
- **Entry point:** `python -m scripts.train_rl`
- **Objective:** clipped PPO surrogate with value/entropy terms and auxiliary
  guard/latency heads that model termination.
- **Output:** checkpoint written to `data/processed/checkpoints/rl` and metrics
  emitted to stdout.

The trainer consumes a synthetic trajectory buffer provided by
`packages.guidance.dataset.TrajectoryDataset`.  Modifying the dataset section in
`rl.yaml` allows quick experiments with trajectory length or action space size.

## Preference Optimisation
- **Config:** `configs/training/preference.yaml`
- **Entry point:** `python -m scripts.train_preference`
- **Objective:** DPO-style loss using a temperature parameter (`beta`) while
  monitoring pairwise accuracy.
- **Output:** checkpoint saved under `data/processed/checkpoints/preference` plus
  an accuracy/loss summary.

## Logging & Reproducibility
All three trainers honour the `seed` key in their configs.  The telemetry hooks
are wired through `packages.telemetry.logger`, giving structured log messages on
stderr.  Checkpoints include both model and optimiser state along with the exact
config snapshot, simplifying restarts or audits.

## Curriculum Seeding
- **Entry point:** `python -m scripts.seed_curriculum`
- **Objective:** convert the bundled `.dsl` fixtures into deterministic
  curriculum artifacts inside `data/processed/curriculum_v1`.
- **Output:** a `programs/` directory mirroring the fixtures and accompanying
  `manifest.json`/`metadata.json` files that track hashes, byte sizes, and seed
  information.

The script accepts `--limit`, `--seed`, and `--output` overrides for quick local
experiments.  When invoked without parameters it rewrites the manifest to remain
fully deterministic and idempotent.

## Checkpoint Export
- **Entry point:** `python -m scripts.export_checkpoint`
- **Objective:** gather the latest checkpoint from each trainer directory,
  compute provenance hashes, and copy them under
  `data/processed/checkpoints/exports/` for archival or release packaging.
- **Output:** per-run export directories (e.g. `export_20240905_143000/`) with a
  `manifest.json` describing file hashes plus a top-level `latest.json`
  convenience pointer.

Provide `--root` to target a custom checkpoint tree and `--output` to stage the
bundles elsewhere.  The optional `--seed` flag simply ensures reproducible
ordering for audit trails.

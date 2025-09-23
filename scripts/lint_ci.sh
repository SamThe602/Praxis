#!/usr/bin/env bash
# Aggregates Python and Rust linters/formatters to match CI expectations.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
    printf '[lint-ci] %s\n' "$*"
}

POETRY_BIN="${POETRY_BIN:-$(command -v poetry || true)}"
if [[ -z "${POETRY_BIN}" ]]; then
    log "Poetry not found; run scripts/bootstrap_env.sh first"
    exit 1
fi

PYTHON_TARGETS=()
for path in packages tests/python; do
    if [[ -d "${path}" ]]; then
        PYTHON_TARGETS+=("${path}")
    fi
done

if ((${#PYTHON_TARGETS[@]})); then
    log "Running Ruff"
    "${POETRY_BIN}" run ruff check "${PYTHON_TARGETS[@]}"

    log "Running Black"
    "${POETRY_BIN}" run black --check "${PYTHON_TARGETS[@]}"

    log "Running isort"
    "${POETRY_BIN}" run isort --check-only "${PYTHON_TARGETS[@]}"

    log "Running mypy"
    "${POETRY_BIN}" run mypy "${PYTHON_TARGETS[@]}"
fi

log "Running cargo fmt"
cargo fmt --all -- --check

log "Running cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings

log "Lint suite completed"

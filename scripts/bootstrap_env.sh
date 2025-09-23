#!/usr/bin/env bash
# Bootstraps developer tooling: installs Poetry env, Rust toolchain, and configures pre-commit hooks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
    printf '[bootstrap] %s\n' "$*"
}

ensure_poetry() {
    if command -v poetry >/dev/null 2>&1; then
        return
    fi

    if command -v pipx >/dev/null 2>&1; then
        log "Installing poetry via pipx"
        pipx install --include-deps poetry >/dev/null
    else
        log "pipx not found; installing poetry with pip --user"
        python3 -m pip install --user poetry >/dev/null
    fi
}

ensure_poetry
POETRY_BIN="${POETRY_BIN:-$(command -v poetry || true)}"
if [[ -z "${POETRY_BIN}" ]]; then
    log "Poetry is required but was not installed"
    exit 1
fi

log "Syncing Python dependencies with Poetry"
"${POETRY_BIN}" install

log "Ensuring stable Rust toolchain with rustfmt and clippy"
rustup toolchain install stable --profile minimal
rustup default stable
rustup component add rustfmt clippy

if ! command -v pre-commit >/dev/null 2>&1; then
    if command -v pipx >/dev/null 2>&1; then
        log "Installing pre-commit via pipx"
        pipx install pre-commit >/dev/null
    else
        log "Installing pre-commit inside Poetry environment"
        "${POETRY_BIN}" run pip install pre-commit >/dev/null
    fi
fi

if [[ -f ".pre-commit-config.yaml" ]]; then
    log "Installing pre-commit hooks"
    pre-commit install --install-hooks --hook-type pre-commit
fi

log "Bootstrap complete"

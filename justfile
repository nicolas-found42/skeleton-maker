# Dev tasks. These wrap the commands CI runs. Install just with `uv tool install rust-just`
# (or `brew install just`). `uv run --locked` also fails if uv.lock is stale.

set shell := ["bash", "-euo", "pipefail", "-c"]

# List the recipes.
default:
    @just --list

# Create or update .venv from uv.lock and install the git hooks.
setup:
    uv sync --locked
    uvx prek install

# Ruff check and format check (what the `lint` CI job runs).
lint:
    uv run --locked ruff check .
    uv run --locked ruff format --check .

# Apply ruff formatting.
fmt:
    uv run --locked ruff format .

# Run every local hook over all files (what the `prek` CI job runs).
hooks:
    SKIP=no-commit-to-branch uvx prek run --all-files

# Run the offline test suite. Extra pytest arguments are passed through.
test *args:
    uv run --locked python -m pytest -q {{ args }}

# Everything CI runs locally. Type checking and coverage are added by later issues.
check: hooks lint test

# Audit locked dependencies for known vulnerabilities (experimental in uv).
audit:
    uv audit --locked --preview-features audit-command

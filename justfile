# Dev tasks. These wrap the commands CI runs. Install just with `uv tool install rust-just`
# (or `brew install just`). `uv run --locked` also fails if uv.lock is stale.

set shell := ["bash", "-euo", "pipefail", "-c"]

# List the recipes.
default:
    @just --list

# Bootstrap Python, hooks, just (if absent), npm, and matching Chromium.
setup:
    uv run --locked python -m scripts.setup_dev

# Install the Chromium build pinned by package-lock.json and its Linux libraries.
browser-install:
    npm ci
    if [ "$(uname -s)" = Linux ]; then npm exec -- playwright install --with-deps --only-shell chromium; else npm exec -- playwright install chromium; fi

# Check required tools and confirm the project-pinned Playwright Chromium launches.
doctor:
    uv run --locked python -m scripts.check_dev_environment

# Create the Grounded SAM 2 environment worker outside the repo (downloads about 1 GB of models and libraries).
environment-worker-setup:
    uv run --locked python -m scripts.setup_environment_worker

da3-geometry-setup:
    uv run --locked python -m scripts.setup_da3_geometry_worker

# Run the offline generated-fixture browser regressions.
browser-check:
    uv run --locked python -m scripts.run_browser_checks

# Validate pose feature invariants and TypeSafe schemas without local data or a key.
research-check:
    uv run --locked python -m experiments.pose_classification.research_battery preflight

# Ruff check and format check (what the `lint` CI job runs).
lint:
    uv run --locked ruff check .
    uv run --locked ruff format --check .

# Apply ruff formatting.
fmt:
    uv run --locked ruff format .

# Audit workflows with zizmor (offline mode; the zizmor CI job also runs the online SHA checks).
workflows:
    uvx zizmor@1.30.1 --no-progress .github/workflows/

# Run every local hook over all files (what the `prek` CI job runs).
hooks:
    SKIP=no-commit-to-branch uvx prek run --all-files

# Type check with ty (blocking in CI). The version is pinned in pyproject.toml.
type:
    uv run --locked ty check

# Run the offline test suite with coverage; fails below the floor in pyproject.toml.
# Extra pytest arguments are passed through.
test *args:
    uv run --locked python -m pytest -q --cov --cov-config=pyproject.toml {{ args }}

# Everything CI runs locally.
check: hooks lint type workflows test research-check browser-check

# Audit locked dependencies for known vulnerabilities (experimental in uv).
audit:
    uv audit --locked --preview-features audit-command

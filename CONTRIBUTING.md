# Contributing

Thanks for helping. The full rules are in [`docs/agents/dev-workflow.md`](docs/agents/dev-workflow.md); this page is the short version. Coding agents should read [`AGENTS.md`](AGENTS.md) first.

## Setup

With [uv](https://docs.astral.sh/uv/), Node/npm and ffmpeg available, bootstrap the project tools from a fresh checkout:

```bash
uv run --locked python -m scripts.setup_dev
just doctor  # check required tools and matching project Chromium
just check   # hooks, lint, types, workflow audit, Python tests, research and browser checks
```

The bootstrap installs `just` when absent, the locked Python environment and hooks, pinned npm dependencies, and matching Chromium. See [browser-testing.md](docs/agents/browser-testing.md) for browser setup and diagnostics, and [pose-classification-tooling.md](docs/research/pose-classification-tooling.md) for offline research validation.

## Making a change

1. Open an issue for a feature or a bug (templates are provided). Chores and doc fixes can skip it.
2. Branch from `main` with a [Conventional Branch](https://conventional-branch.github.io) name such as `feature/issue-12-short-description`.
3. Use [Conventional Commits](https://www.conventionalcommits.org) for commits and the PR title. This project's type list is in the workflow doc (it uses `doc`, not `docs`).
4. Open a PR using the template: Summary, Evidence (what you ran and the before and after), and a one-line Merge Danger. `main` only accepts squash merges with a passing `ci-ok` check.

## Ground rules

- Tests run offline and need no API key. A bug fix starts with a failing regression test.
- Never commit secrets (`NVIDIA_API_KEY`, `.env`), model weights or video. They are gitignored.
- Do not lower the coverage floor in `pyproject.toml` to make a PR pass; add tests instead.

To report a security problem, follow [SECURITY.md](SECURITY.md) instead of opening a public issue.

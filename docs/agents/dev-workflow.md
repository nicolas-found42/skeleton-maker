# Development workflow

The rules for changing this repo. They apply equally to humans and to every coding agent. `AGENTS.md` is the short version; this is the reference.

## Flow

1. **Issue.** Features and bugs start from a GitHub issue (use the templates in `.github/ISSUE_TEMPLATE/`). Chores and doc fixes may skip the issue. Agents only start on issues labelled `ready-for-agent`, unless the maintainer says otherwise.
2. **Branch** from `main`.
3. **Implement**, running the checks under [Quality gates](#quality-gates).
4. **Pull request** using `.github/pull_request_template.md`, with `Closes #<issue>` when an issue exists.
5. **CI green**, then **squash merge**. The branch is deleted on merge.

`main` is PR-only. Never push to it directly, never force-push it, never skip hooks with `--no-verify` to get around a failing check.

## Branches

Conventional Branch: `<type>/<description>` in lowercase kebab-case. Types: `feature/` (alias `feat/`), `bugfix/` (alias `fix/`), `hotfix/`, `release/`, `chore/`. Include the issue number when there is one: `feature/issue-12-add-hand-tracking`. Use `chore/` for docs, config and tooling.

## Commits and PR titles

Conventional Commits: `<type>: <description>` or `<type>(<context>): <description>`, lowercase, no trailing period.

This project's type list (use these identifiers exactly):

| Type | Use |
| --- | --- |
| `feat` | New features |
| `fix` | Bug fixes |
| `hotfix` | Emergency bug fixes |
| `refactor` | Large code refactorings |
| `perf` | Performance improvements |
| `test` | Test additions or changes |
| `doc` | Documentation (note: `doc`, not `docs`) |
| `style` | Formatting that does not change meaning |
| `ci` | Continuous integration configuration |
| `dev` | Development environment and tool configuration |
| `chore` | Maintenance, dependency updates, configuration adjustments |
| `sec` | Security fixes or features |
| `revert` | Reversal of a previous commit |
| `merge` | Branch merges |

There is no `build` type. With squash merge the PR title becomes the commit on `main`, so it must follow the same format. Do not add a `Co-Authored-By` trailer to commits or PR descriptions.

## Issues

Use the templates in `.github/ISSUE_TEMPLATE/`: `bug-report.md` for defects, `feature-request.md` for features and chores. They are derived from the `issue-authoring` and `pr` skills, so an agent with those skills and an agent without them produce the same structure. The bug template is the skill's template plus project-specific hints in the Environment and Evidence comments (version and ffmpeg/video details to record, and a reminder to strip `NVIDIA_API_KEY` and `nvapi-` tokens from logs).

A bug report has ten sections: Summary, Impact, Environment, Preconditions, Steps to reproduce, Expected behavior, Actual behavior, Evidence, Acceptance criteria, Scope and developer notes. A feature or chore keeps Summary, Acceptance criteria and Scope, and replaces the reproduction sections with Problem and current workflow, Desired behavior, a Concrete example, Alternatives considered, and Out of scope and dependencies.

Rules for both:

- One independently fixable problem per issue.
- Separate what you observed, what the code suggests, and what you only suspect.
- Acceptance criteria are unchecked, externally observable pass/fail conditions, each checkable by running a command or inspecting a file or page. Never check a box before the fix exists.
- Unknown information is allowed when named with its consequence ("browser version not captured; observed in Chrome desktop; cross-browser untested"). Do not invent facts, success rates or severities.
- Evidence must be readable in the published issue: embedded or pasted, never a local path. Prefer synthetic data and never include secrets.

Labels (the default triage vocabulary, see `triage-labels.md`): `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. The templates apply `needs-triage`.

## Pull requests

Fill in `.github/pull_request_template.md` (derived from the `pr` skill's default template):

- **Summary**: the problem and why it matters, the resulting behaviour, the linked issue, and the smallest before/after view that shows the change: a `diff` block for code, a labelled "Conceptual diff" for behaviour or structure, or a usage example and its result for a wholly new feature. Note tradeoffs and limitations.
- **Evidence**: each behaviour claim with the command or scenario and the observed **before** and **after** result. Before/after claims need an observation from both versions. A source diff shows what was edited, not what it does. Keep the decisive output excerpt in the body. Say plainly what you did not verify, why, and what check is still needed. Failed checks stay visible.
- **Merge Danger**: one line with **Door** (one-way or two-way, and why) and **Blast Radius** (who or what is affected and what could go wrong, including downstream consumers and in-flight branches).

Keep PRs to one concern. A mechanical reformat is its own PR.

## Testing policy

- All tests in `tests/` run offline and need no API key. Stub the NIM and any network call; see `tests/test_nim_stubs.py`.
- Live checks against the hosted NIM are manual, are never run in CI, and never commit a key.
- A bug fix starts with a failing regression test that the fix turns green.
- A feature ships with its tests in the same PR. Test-first (red, green, refactor) is encouraged.
- Property-based tests (`hypothesis`) are for new parsing or serialization code, not a blanket requirement.

## Secrets and data

`NVIDIA_API_KEY` lives in `.env` (gitignored; see `.env.example`). Never commit it, paste it into an issue or PR, or write it to a log. Model weights (`*.pt`, `*.onnx`), media and pipeline outputs are gitignored; do not force-add them.

## Quality gates

The gates below are rolled out issue by issue. The Status column says what is live today; a PR that makes a gate live updates its row.

| Gate | Tool | Status |
| --- | --- | --- |
| Tests | `pytest` (offline) | live |
| Lint and format | `ruff` (line length 100, rules `E,F,W,I,B,UP,SIM,RUF,C4,PT,S`; formatter owns line length) | live: tree is clean (#2), enforced by the `lint` CI job (#3) |
| Workflow security | `zizmor`, SHA-pinned actions, least-privilege permissions | live (#3) |
| Dependency updates | Dependabot with `cooldown` (`github-actions` and `uv`; `ty` and `ruff` are excluded from grouping and bumped deliberately; semver-major bumps and `grpcio`, `grpcio-tools`, `protobuf` are ignored and upgraded by hand) | live (#3, #5) |
| Local hooks | `prek` (commit stage: `ruff`, `typos`, `rumdl`, file hygiene), also run in CI | live (#4) |
| Reproducible installs | `uv.lock`, `uv sync --locked`, `just` recipes | live (#5) |
| Vulnerability audit | `uv audit` (weekly workflow `audit.yml`, non-blocking, experimental) | live (#5) |
| Types | `ty` (version pinned; bump in its own PR), blocking, run by `just type` | live (#6) |
| Coverage | `pytest-cov` with a `fail_under` floor in `pyproject.toml` that only ratchets up | live (#7) |

CI is the backstop: hooks can be skipped locally, CI cannot. Run the same commands CI runs before pushing.

CI jobs (`.github/workflows/ci.yml`): `lint` (ruff check and format), `ty` (type check), `prek` (the local hook config over all files), `zizmor` (workflow audit), `test` (Ubuntu on Python 3.10, 3.11, 3.12 and macOS on 3.12), and `ci-ok`, an aggregate that fails if any other job failed or was skipped. The `main` ruleset requires only `ci-ok`, so the matrix can change without editing the ruleset. Every action is pinned to a full commit SHA with a version comment; Dependabot proposes updates weekly with a 7-day cooldown.

## Everyday commands

Install [just](https://github.com/casey/just) (`uv tool install rust-just` or `brew install just`), then from a fresh clone:

```bash
just setup   # uv sync --locked, and install the git hooks
just check   # hooks + lint + types + workflow audit + tests with coverage: what CI runs, locally
```

Other recipes: `just lint`, `just fmt`, `just hooks`, `just type`, `just workflows`, `just test [pytest args]`, `just audit`.

Dependencies:

- Dev tools live in `[dependency-groups] dev` in `pyproject.toml` (PEP 735); `uv sync` installs them by default. `ruff` and `ty` are pinned exactly; bump either deliberately, in its own PR. Keep the `ruff` pin in step with the `ruff-pre-commit` rev in `.pre-commit-config.yaml`.
- `uv.lock` is committed. Any change to dependencies must include the updated lock (`uv lock`). CI runs `uv lock --check` and `uv sync --locked`, so a stale lock fails the build.
- End users still install with `uv pip install -e ".[download]"`; there is no `dev` extra.
- The weekly `audit` workflow runs `uv audit` (experimental, uv pinned to a version in the workflow) and is not part of `ci-ok`. A red run means someone should look; it never blocks a merge.

## Coverage floor

`pytest --cov` (always on in `just test` and in CI) fails when total coverage drops below `fail_under` in `[tool.coverage.report]` of `pyproject.toml`. Generated gRPC stubs (`skeleton_maker/_gen`) are omitted from measurement.

The floor is a ratchet:

- It only goes up. Never lower it to make a PR pass; add the missing tests instead.
- A PR that adds tests should raise it to the new measured total, rounded down.
- Measure twice, then use the lower number. Total coverage is slightly different on a fresh checkout (the tests build the gRPC stubs, which covers that code) than when `skeleton_maker/_gen` already exists. The first run is what CI sees; the second is what you see on every later local run. At the time the floor was set these were 36.3% and 34.2%, so the floor is 34.
- Do not add or change tests only to inflate the number; coverage that does not assert behaviour is worse than none.

## Local hooks

`prek` (a fast Rust drop-in for `pre-commit`) runs the config in `.pre-commit-config.yaml` at commit time: file hygiene (TOML and YAML validity, merge-conflict markers, large files over 500 KB, end-of-file, trailing whitespace), `ruff` check and format, `typos`, `rumdl` (Markdown) and a guard that refuses commits directly on `main`. `ty` and `pytest` are not hooks because they are slow; CI runs them.

- Once per clone: `prek install` (install prek with `uv tool install prek`, or run it as `uvx prek`).
- Any time: `prek run --all-files`.
- Do not bypass hooks with `--no-verify`. If a hook is wrong, fix the hook config in its own PR.
- Update hook versions with `prek autoupdate --freeze --cooldown-days 7`, then check the diff: `--freeze` stores commit SHAs, and a repo with several tag families (such as `typos`) can resolve to the wrong tag, so confirm each `# frozen:` comment is the release you intend.
- `rumdl` is configured in `.rumdl.toml` (the 80-column rule is off because prose is one paragraph per line).
- For local `git blame`, run `git config blame.ignoreRevsFile .git-blame-ignore-revs` once; GitHub reads that file automatically.

## Later, with triggers

Tools evaluated and deliberately not adopted yet. Adopt one when its trigger fires, in its own issue and PR.

| Tool | Adopt when |
| --- | --- |
| `git-cliff` | the first tagged release or package publish |
| `hypothesis` as a dependency | new parsing or serialization code is added |
| `inline-snapshot` | three or more tests assert on large literals or JSON |
| `diff-cover` | the global coverage floor reaches 60% |
| `mutmut` | core modules (`stage`, `verify`, `bbox`) have stable APIs and coverage is 70% or more |
| `deptry` | a dependency is found unused or missing in review |
| `vulture` | dead code is found in review, or the package passes about 30 modules |
| `tach` | the package passes about 30 modules or import cycles appear |
| `complexipy` | over-long functions keep reaching review |
| `gitleaks` | GitHub push protection is unavailable, or a secret is ever committed |
| `scorecard` | the project gains external users or is promoted |
| `harden-runner` | CI starts using secrets or publishing artifacts |

Rejected: `osv-scanner` and `pip-audit` (redundant with `uv audit` for a single ecosystem), `gh-aw` (heavy, tied to one vendor's tokens), and agent-specific hooks or actions (the repo must work for any agent).

## Repository settings (applied by the maintainer, not by PRs)

These live in GitHub, not in the repo, so a PR cannot change them:

- **`main` ruleset** (name `main`, active): a pull request is required (squash merge only, no approvals needed), the `ci-ok` check must pass, and force-pushes and deletion are blocked. The repository admin role can bypass it, so a maintainer can still fix things directly; agents should not rely on that.
- **Merge settings**: squash only (merge commits and rebase merges are off), the head branch is deleted on merge, and the squash commit uses the PR title with an empty body.
- **Secret scanning with push protection** is on. Dependabot security updates are off.

Because `ci-ok` is the single required check, adding or renaming matrix jobs in `ci.yml` needs no ruleset change, but renaming `ci-ok` itself does. To inspect the ruleset: `gh api repos/nicolas-found42/skeleton-maker/rules/branches/main`.

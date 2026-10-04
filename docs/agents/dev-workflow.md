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

Use the templates in `.github/ISSUE_TEMPLATE/`: `bug-report.md` for defects, `feature-request.md` for features and chores. They are derived from the `issue-authoring` and `pr` skills, so an agent with those skills and an agent without them produce the same structure.

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
| Lint and format | `ruff` (line length 100, rules `E,F,W,I,B,UP,SIM,RUF,C4,PT,S`; formatter owns line length) | configured, tree is clean (#2); CI gate planned (#3) |
| Workflow security | `zizmor`, SHA-pinned actions, least-privilege permissions | planned (#3) |
| Dependency updates | Dependabot with `cooldown` | planned (#3, `uv` ecosystem in #5) |
| Local hooks | `prek` (commit stage: `ruff`, `typos`, `rumdl`, file hygiene) | planned (#4) |
| Reproducible installs | `uv.lock`, `uv sync --locked`, `just` recipes | planned (#5) |
| Vulnerability audit | `uv audit` (weekly, non-blocking, experimental) | planned (#5) |
| Types | `ty` (version pinned; bump in its own PR) | planned (#6) |
| Coverage | `pytest-cov` with a `fail_under` floor that only ratchets up | planned (#7) |

CI is the backstop: hooks can be skipped locally, CI cannot. Run the same commands CI runs before pushing.

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

## Maintainer settings (not done by PRs)

Applied by the maintainer once the required CI check names exist: a `main` ruleset (PR required, required checks, no force-push or deletion, admin bypass kept), squash-only merges with delete-branch-on-merge, and GitHub secret-scanning push protection.

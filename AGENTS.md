# AGENTS.md

Instructions for any coding agent (and human) working in this repository. `CLAUDE.md` imports this file; edit only here.

`skeleton-maker` is a Python 3.10-3.12 CLI that draws 3D body-pose skeletons on video using the NVIDIA 3D Body Pose NIM. See `README.md` for usage.

## How to work here

The full workflow is in [`docs/agents/dev-workflow.md`](docs/agents/dev-workflow.md). Read it before your first change. The short version:

- Features and bugs start from a GitHub issue; agents pick up only issues labelled `ready-for-agent`. Chores and doc fixes may skip the issue.
- Branch from `main` using Conventional Branch names (`feature/`, `bugfix/`, `chore/`, ...). Never push to `main`; open a PR.
- Commit messages and PR titles follow Conventional Commits with this project's type list (`doc`, not `docs`). Do not add a `Co-Authored-By` trailer.
- Use the issue and PR templates in `.github/`. A PR needs evidence (commands run, before/after) and a one-line Merge Danger.
- Tests must run offline and need no API key. A bug fix starts with a failing regression test.
- Never commit secrets (`.env`, `NVIDIA_API_KEY`), model weights or media. They are gitignored.
- Run `just check` before pushing (set up once with `just setup`; see "Everyday commands" in the workflow doc). CI runs the same checks.

If your agent supports skills for writing issues or PRs (for example `issue-authoring` or `pr`), use them as helpers. The templates in `.github/` are the requirement; the skills are optional helpers that produce the same structure.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues (`nicolas-found42/skeleton-maker`) via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-label vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

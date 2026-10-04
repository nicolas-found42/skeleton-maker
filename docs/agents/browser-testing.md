# Browser checks

The character viewer has two offline browser regressions. `just browser-check` creates a temporary 5-second synthetic WebM, generates a synthetic pose fixture with `uv run --locked python -m scripts.make_character_fixture`, serves both on an operating-system-assigned loopback port, and runs `scripts/check_character_browser.cjs` plus `scripts/check_character_edges.cjs` in the repository's pinned Playwright Chromium. The fixture server supports HTTP byte ranges so Chromium can seek the video. Each page sends a sample off-origin fetch through Playwright's route interceptor to prove it is blocked before reaching the network; any other off-origin request fails the test. The runner strips `NVIDIA_API_KEY` from child environments. The server and files are removed after success or failure. The checks make no NIM requests and use no real clip.

## First-time setup

Run `uv run --locked python -m scripts.setup_dev` from the repository root. `uv run` syncs the locked Python environment; the setup command installs hooks, installs `just` with `uv tool` if absent, then installs the pinned npm dependency and its matching Chromium build. On Linux it also installs Playwright's Chromium system dependencies. The command works when `just` is not yet on `PATH`.

For a browser-only reinstall, run `just browser-install`. To diagnose missing `just`, `uv`, Node/npm, ffmpeg, an absent Playwright install, a stale package version, or Chromium launch failure, run `uv run --locked python -m scripts.check_dev_environment` (or `just doctor`). The doctor reports each actionable problem and exits nonzero.

`package.json` and `package-lock.json` pin the Playwright package. Always install browser binaries through that local CLI so Chromium matches the installed package. CI runs the same browser-check runner in a required `browser-test` job and includes it in `ci-ok`.

## Failure investigation

The runner writes all generated files under a temporary directory and chooses its loopback port by binding port `0`; it never occupies a fixed development-server port. A failing regression preserves its subprocess output and still stops the server and removes generated files. Run the two Node scripts directly only when diagnosing one check, with a fixture served locally and the pinned `node_modules` installation available.

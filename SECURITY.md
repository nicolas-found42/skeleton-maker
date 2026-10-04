# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue or PR. Use GitHub's private reporting: open the repository's **Security** tab and choose **Report a vulnerability**. Include what you found, how to reproduce it, and the version or commit.

This is a small project maintained by one person, so responses are best effort and there is no guaranteed timeline.

## Scope

In scope: this repository's code and its CI configuration (for example a way to leak `NVIDIA_API_KEY`, unsafe handling of untrusted video or file paths, or a workflow that lets a pull request run with elevated permissions). The hosted NVIDIA 3D Body Pose service is NVIDIA's; report problems with it to NVIDIA.

Only the latest commit on `main` is supported; the project has no tagged releases yet.

## If you committed or pasted a key

Revoke the key at <https://build.nvidia.com> and create a new one first, then tell the maintainer. Removing a key from git history or an issue does not make it safe again.

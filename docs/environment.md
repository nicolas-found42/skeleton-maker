# Environment scan

`skeleton-maker environment` scans a clip for the things around the people in it (surfaces, objects, vehicles) and writes a versioned manifest. This page documents the command, the manifest and the backend contract. Only the contract and the manifest are in the base package: the perception and geometry models run in a separate worker environment, so a plain install has **no backend**. Until one is installed the command reports that and exits without writing anything. Set up the Grounded SAM 2 worker with `just environment-worker-setup`; see [environment-worker.md](environment-worker.md).

## Command

```bash
skeleton-maker environment demo.skeleton/clip.mp4 --out demo.skeleton/environment.json --geometry off --sample-fps 2
```

| Option | Meaning |
| --- | --- |
| `video` | The conformed clip. If poses are reused later, this must be the clip whose frame ids they describe. A variable-frame-rate file is refused: conform it with `skeleton-maker clip` first. |
| `--out` | Manifest to write (default `<video>.environment.json`). Assets go in `<out stem>.assets/` beside it. |
| `--poses` | Existing pose output (JSON Lines) for this exact clip; reused with no NIM call or credential. See [Reusing poses](#reusing-poses). |
| `--backend` | Installed backend name (default `grounded-sam2-da3`). |
| `--geometry` | `off`: semantics only, geometry status `not_requested`. `auto`: geometry when the backend can produce it, otherwise status `unavailable` and the scan still completes. `required`: exit nonzero unless a valid shared registration (`registered_relative` or `registered_metric`) is produced. |
| `--sample-fps` | Frames per second to scan, 0.05 to 30 and no more than the clip's rate (default 2). The step is rounded to whole source frames. |
| `--cache-dir`, `--no-cache` | Where inference results are cached (default `~/.cache/skeleton-maker/environment`) and a switch to bypass it. See [environment-worker.md](environment-worker.md#cache). |
| `--device` | `auto` (the backend's first reported device) or one the backend reports. An unavailable device is rejected before inference. |

Exit codes: `0` complete, `1` failed, `2` invalid options or input, `3` partial, `130` interrupted. Every option is checked before inference. A failed or interrupted run, or a failed write, leaves any existing manifest and assets byte-identical; the new files replace the old ones only after everything validated. A `partial` run (the backend finished but says it could not cover everything) writes the manifest with `run.status: "partial"` and a `reason`, and exits 3 so a script cannot mistake it for a complete run.

## Manifest

Schema `skeleton-maker.environment/1`. Loaders must reject any other version.

| Field | Contents |
| --- | --- |
| `run` | `status` (`complete`, `partial`, `failed`), `reason` (required unless complete), `created_at`, `tool` |
| `source` | `path`, `sha256`, `width`, `height`, `frame_rate` as `[numerator, denominator]`, `frame_count`, `frame_count_source` (`container` or `duration_estimate`), `duration_s` |
| `processed_frames` | scanned source frames: `frame_id`, `time` as an exact `[numerator, denominator]` of seconds, `time_s` (display float) |
| `frame_range` | `[first, last]` scanned frame id |
| `shots` | visual shots tiling the whole clip: `id`, `first_frame`, `last_frame`, `first_time`, `last_time` (exact `[numerator, denominator]` seconds). Entity ids are scoped by shot and an entity is only observed inside its shot |
| `shot_detection` | `method`, `version`, `parameters`, `pose_stage_shots` (`null` without `--poses`), `boundaries` (`frame_id`, `time`, `distance`, `isolation`, `agreement`) and `pose_only` gaps, see [Shots](#shots) |
| `entities` | `id`, `shot`, `family` (`surface`, `object`, `vehicle`, `person`), `motion` (`static`, `dynamic`, `unknown`), `labels.native` and `labels.normalized`, and for `person` entities `skeleton_id` |
| `observations` | `id`, `entity`, `frame_id` (a processed frame), `bbox` `[x0, y0, x1, y1]` in full decoded source pixels, `mask` (`null` or `{"asset": "<relative path>"}`), `score`, `score_meaning`, `visibility` (`visible`, `occluded`, `absent`, `uncertain`) |
| `poses` | `null` without `--poses`; otherwise `path`, `association`, `frame_count`, `skeletons` (`id`, `first_frame`, `last_frame`, `frames`) and `person_free_ranges` as `[first, last]` frame ids |
| `geometry` | `status` (`not_requested`, `unavailable`, `relative`, `registered_relative`, `registered_metric`), `mode`, `reason` |
| `backend` | `name`, `version`, `checkpoints`, `device` |
| `config` | `requested_labels`, `geometry_mode`, `sample_fps`, `device` |
| `assets` | `path` (relative to `<stem>.assets/`), `sha256`, `bytes` |

Mask assets are single-channel PNG images the size of the source frame (any non-zero pixel is inside). Scoring against human labels is described in [environment-scoring.md](environment-scoring.md).

Loading (`skeleton_maker.envmanifest.load_manifest`) checks the schema version, structure, finite numbers, that observations reference real entities and processed frames, that boxes lie inside the frame, that every asset exists inside the bundle with the recorded hash, and that no path, absolute path or symlink leaves the bundle.

A `registered_relative` geometry result has no solved metric scale and must not be overlaid on the NIM's meter-valued skeletons.

## Shots

The command finds visual shot boundaries from the video itself, so a cut during a person-free stretch is found too, and passes the shots to the backend; the backend does not decide them. Identities are shot-local: an entity names one shot and is only observed inside it, and nothing claims an identity across a cut.

A cut is a frame whose colour histogram is at least 0.40 away (0 to 1) from the previous frame's and at least 3 times farther than every frame pair within 2 frames of it. Requiring that the change stands out from its neighbours is what keeps fades, gradual lighting changes and whip pans (many small changes, or motion without a colour change) from counting as cuts. Detection is coarse to fine: the sampled frames find intervals where the scene changed, and only those are decoded frame by frame to place the cut on its exact first frame. A cut within 3 frames of the previous cut or the end of the clip is treated as a flash and dropped. The parameters are recorded in `shot_detection.parameters`. Dissolves longer than a few frames are, by design, not cuts.

With `--poses`, the shots the character stage derives from the poses are recorded in `pose_stage_shots`. That stage only sees frames with people, so its boundary lies somewhere in the gap between a shot's last frame and the next shot's first. Each visual cut gets an `agreement`: `agrees` (inside such a gap), `visual_only` (inside a pose shot: the tracker kept its ids across the cut), `outside_pose_coverage` (before the first or after the last pose frame; the poses say nothing there) or `no_poses`. A gap with no visual cut in it is listed under `pose_only`. The visual shots are never rewritten to match the poses.

## Reusing poses

```bash
skeleton-maker environment demo.skeleton/clip.mp4 --poses demo.skeleton/pose.json --out demo.skeleton/environment.json
```

The poses must describe the same clip. The NIM emits one record per decoded frame, so the file must have exactly one record per clip frame with ids `0..N-1`, no duplicates, and every detection carries an integer `tracking_id` and a finite `root_pose.translation`, and every `bbox` (`[x, y, width, height]` in pose files, unlike the manifest's `[x0, y0, x1, y1]`) within the frame plus a 25% margin. Anything else exits 2 naming the mismatch and writes nothing.

Pose files carry no fingerprint, so a matching length is not proof of identity: the manifest records `association: "user-supplied"`. If every record carries a `source_sha256` it is compared with the clip's hash: a match records `hash-verified`, a mismatch is rejected.

People are not duplicated into the environment inventory. A backend reports a person as an entity with `family: "person"` and `skeleton_id` naming a `tracking_id` in the pose file (at most one entity per shot and skeleton); an unknown, missing or duplicated id fails the run. Frame ranges where the poses hold nobody are recorded in `poses.person_free_ranges` and are scanned like any other frame. Without `--poses` the command still runs, and `person` entities must not carry a `skeleton_id`.

## Backend contract

Contract id `skeleton-maker.environment-backend/1`. A backend is an object with `name`, `available_devices()`, `supports_geometry()`, `max_frames()`, `cache_identity()` (a dict of everything that changes its answers, or `None` to opt out of caching) and `run(request, assets_dir)`, registered in `skeleton_maker.environment.BACKENDS`. Heavyweight workers run in their own interpreter through `skeleton_maker.envworkers.SubprocessBackend`, which speaks the worker protocol documented in that module (`preflight` and `run`).

Request (`poses` is `null`, or the manifest's `poses` block):

```json
{
  "contract": "skeleton-maker.environment-backend/1",
  "video": "/abs/clip.mp4",
  "device": "cpu",
  "geometry": "auto",
  "requested_labels": [],
  "poses": null,
  "shots": [{"id": "shot-0", "first_frame": 0, "last_frame": 19}, {"id": "shot-1", "first_frame": 20, "last_frame": 39}],
  "source": {"width": 64, "height": 48, "frame_rate": [30, 1], "frame_count": 30},
  "frames": [{"frame_id": 0, "time": [0, 1]}, {"frame_id": 15, "time": [1, 2]}]
}
```

Response: `contract`, `status` (`complete` or `partial` with a `reason`), `backend` (`name`, `version`, `checkpoints`), `device`, `entities`, `observations` and `geometry`, shaped as in the manifest. Masks are files the backend writes under `assets_dir`, referenced by relative path. A response that fails validation, a raised exception or a keyboard interrupt is a failed run; nothing is written.

Tests substitute the inference boundary only, by registering a deterministic fake in `BACKENDS` (see `tests/env_fakes.py`). There is no flag or environment variable that selects a fake.

# Environment scan

`skeleton-maker environment` scans a clip for the things around the people in it (surfaces, objects, vehicles) and writes a versioned manifest. This page documents the command, the manifest and the backend contract. Only the contract and the manifest are in the base package: the perception and geometry models run in a separate worker environment, so a plain install has **no backend**. Until one is installed the command reports that and exits without writing anything.

## Command

```bash
skeleton-maker environment demo.skeleton/clip.mp4 --out demo.skeleton/environment.json --geometry off --sample-fps 2
```

| Option | Meaning |
| --- | --- |
| `video` | The conformed clip. If poses are reused later, this must be the clip whose frame ids they describe. A variable-frame-rate file is refused: conform it with `skeleton-maker clip` first. |
| `--out` | Manifest to write (default `<video>.environment.json`). Assets go in `<out stem>.assets/` beside it. |
| `--backend` | Installed backend name (default `grounded-sam2-da3`). |
| `--geometry` | `off`: semantics only, geometry status `not_requested`. `auto`: geometry when the backend can produce it, otherwise status `unavailable` and the scan still completes. `required`: exit nonzero unless a valid shared registration (`registered_relative` or `registered_metric`) is produced. |
| `--sample-fps` | Frames per second to scan, 0.05 to 30 and no more than the clip's rate (default 2). The step is rounded to whole source frames. |
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
| `shots` | `id`, `first_frame`, `last_frame`; entity ids are scoped by shot |
| `entities` | `id`, `shot`, `family` (`surface`, `object`, `vehicle`, `person`), `motion` (`static`, `dynamic`, `unknown`), `labels.native` and `labels.normalized` |
| `observations` | `id`, `entity`, `frame_id` (a processed frame), `bbox` `[x0, y0, x1, y1]` in full decoded source pixels, `mask` (`null` or `{"asset": "<relative path>"}`), `score`, `score_meaning`, `visibility` (`visible`, `occluded`, `absent`, `uncertain`) |
| `geometry` | `status` (`not_requested`, `unavailable`, `relative`, `registered_relative`, `registered_metric`), `mode`, `reason` |
| `backend` | `name`, `version`, `checkpoints`, `device` |
| `config` | `requested_labels`, `geometry_mode`, `sample_fps`, `device` |
| `assets` | `path` (relative to `<stem>.assets/`), `sha256`, `bytes` |

Loading (`skeleton_maker.envmanifest.load_manifest`) checks the schema version, structure, finite numbers, that observations reference real entities and processed frames, that boxes lie inside the frame, that every asset exists inside the bundle with the recorded hash, and that no path, absolute path or symlink leaves the bundle.

A `registered_relative` geometry result has no solved metric scale and must not be overlaid on the NIM's meter-valued skeletons.

## Backend contract

Contract id `skeleton-maker.environment-backend/1`. A backend is an object with `name`, `available_devices()`, `supports_geometry()` and `run(request, assets_dir)`, registered in `skeleton_maker.environment.BACKENDS`. Heavyweight workers are expected to be invoked from such an adapter as a subprocess.

Request:

```json
{
  "contract": "skeleton-maker.environment-backend/1",
  "video": "/abs/clip.mp4",
  "device": "cpu",
  "geometry": "auto",
  "requested_labels": [],
  "source": {"width": 64, "height": 48, "frame_rate": [30, 1], "frame_count": 30},
  "frames": [{"frame_id": 0, "time": [0, 1]}, {"frame_id": 15, "time": [1, 2]}]
}
```

Response: `contract`, `status` (`complete` or `partial` with a `reason`), `backend` (`name`, `version`, `checkpoints`), `device`, `shots`, `entities`, `observations` and `geometry`, shaped as in the manifest. Masks are files the backend writes under `assets_dir`, referenced by relative path. A response that fails validation, a raised exception or a keyboard interrupt is a failed run; nothing is written.

Tests substitute the inference boundary only, by registering a deterministic fake in `BACKENDS` (see `tests/env_fakes.py`). There is no flag or environment variable that selects a fake.

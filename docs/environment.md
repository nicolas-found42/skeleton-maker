# Environment scan

`skeleton-maker environment` scans a clip for the things around the people in it (surfaces, objects, vehicles) and writes a versioned manifest. This page documents the command, the manifest and the backend contract. Only the contract and the manifest are in the base package: the perception and geometry models run in a separate worker environment, so a plain install has **no backend**. Until one is installed the command reports that and exits without writing anything. Set up the Grounded SAM 2 worker with `just environment-worker-setup`; see [environment-worker.md](environment-worker.md).

## Command

```bash
skeleton-maker environment demo.skeleton/clip.mp4 --out demo.skeleton/environment.json --geometry off --sample-fps 2
```

Add `--poses demo.skeleton/pose.jsonl --viewer demo.skeleton/environment.html` to export a synchronized offline inspection page:

```bash
skeleton-maker environment demo.skeleton/clip.mp4 --poses demo.skeleton/pose.jsonl \
  --out demo.skeleton/environment.json \
  --overlay demo.skeleton/environment-overlay.mp4 --viewer demo.skeleton/environment.html
```

The overlay uses environment observations only on their recorded source frames and draws existing pose records at full frame cadence. The command writes the environment manifest, its assets, the overlay, the viewer page, and the viewer's local bundle only after all requested outputs are prepared.

To run the environment scan after the normal clip, tracking, pose, and skeleton-render stages, opt in with `--environment`:

```bash
skeleton-maker all demo.mp4 --work demo.skeleton \
  --environment --environment-backend grounded-sam2-da3
```

This reuses `demo.skeleton/pose.json` from the same NIM invocation. It keeps the usual `clip.mp4`, `boxes.txt`, `pose.json`, and `overlay.mp4` artifacts and adds `environment.json`, `environment.assets/`, `environment-overlay.mp4`, `environment.html`, and `environment.viewer.assets/`. The optional settings are `--environment-classes`, `--environment-geometry`, `--environment-calibration`, `--environment-sample-fps`, `--environment-device`, `--environment-cache-dir`, and `--environment-no-cache`. `--environment-calibration` accepts the same versioned JSON camera calibration used by the standalone environment command. Without `--environment`, `all` retains the pose-only command behavior.

Keep the generated `environment.viewer.assets/` directory beside the HTML page when moving or sharing the export. It contains a copy of the source video, the viewer's local JavaScript, and copies of the referenced environment masks. The HTML embeds the validated observation, shot, geometry and 2D-pose data. The page makes no hosted-service requests. Its frame slider follows source frame numbers and seeks the video at the manifest's exact frame rate; overlays are drawn only for observations recorded at that frame. Unprocessed frames say “not sampled,” processed frames without an entity observation say “missing observation,” manifest `uncertain` visibility is shown as “low confidence,” and `absent` remains “confirmed absent.” Numeric model scores keep their original `score_meaning` and are not presented as calibrated probabilities. Geometry status is displayed in the header; the page remains useful when registration is unavailable.

| Option | Meaning |
| --- | --- |
| `video` | The conformed clip. If poses are reused later, this must be the clip whose frame ids they describe. A variable-frame-rate file is refused: conform it with `skeleton-maker clip` first. |
| `--out` | Manifest to write (default `<video>.environment.json`). Assets go in `<out stem>.assets/` beside it. |
| `--poses` | Existing pose output (JSON Lines) for this exact clip; reused with no NIM call or credential. See [Reusing poses](#reusing-poses). |
| `--backend` | Installed backend name (default `grounded-sam2-da3`). |
| `--classes` | Extra labels to look for, comma separated, each optionally `label=family` (`surface`, `object` or `vehicle`; default `object`), added to the preset. Example: `--classes "forklift=vehicle,pallet"`. See [Labels](#labels). |
| `--geometry` | `off`: semantics only, geometry status `not_requested`. `auto`: keep semantic results when registration is underconstrained, mark the run partial and explain why; a combined run still publishes its prepared outputs. `required`: exit nonzero unless validated metric registration is produced. |
| `--sample-fps` | Frames per second to scan, 0.05 to 30 and no more than the clip's rate (default 2). The step is rounded to whole source frames. |
| `--cache-dir`, `--no-cache` | Where inference results are cached (default `~/.cache/skeleton-maker/environment`) and a switch to bypass it. See [environment-worker.md](environment-worker.md#cache). |
| `--device` | `auto` (the backend's first reported device) or one the backend reports. An unavailable device is rejected before inference. |
| `--calibration` | Optional versioned JSON with camera intrinsics/distortion and measured scene anchors. Geometry records it; only camera intrinsics and lens distortion affect DA3 inference. |
| `--viewer` | Optional offline HTML viewer path. Its `<stem>.viewer.assets/` companion must stay beside the page; the source clip is copied into that bundle. |
| `--overlay` | Optional MP4 combining masks, bounds, labels, and supplied skeleton poses on the source timeline. Environment observations appear only on their recorded source frames. |

Exit codes: `0` completed, `1` failed, `2` invalid options or input, `3` incomplete perception, `130` interrupted. Every option is checked before inference. A failed or interrupted run, or a failed write, leaves any existing manifest and assets byte-identical; the new files replace the old ones only after everything validated. When perception completes but optional geometry cannot be registered, `run.status` is `partial`, `run.perception_status` remains `complete`, and `run.reason` explains the geometry abstention. The command exits 0 because semantic processing and artifact publication completed; the partial status means spatial registration is unavailable. The combined command publishes the prepared clip, pose, overlay, manifest and viewer. An incomplete perception response exits 3, with `run.perception_status: "partial"`; required geometry failures remain nonzero.

## Manifest

Schema `skeleton-maker.environment/1`. Loaders must reject any other version.

| Field | Contents |
| --- | --- |
| `run` | `status` (`complete`, `partial`, `failed`), `perception_status` (`complete`, `partial`, `failed`), `reason` (required unless complete), `created_at`, `tool` |
| `source` | `path`, `sha256`, `width`, `height`, `frame_rate` as `[numerator, denominator]`, `frame_count`, `frame_count_source` (`container` or `duration_estimate`), `duration_s` |
| `processed_frames` | scanned source frames: `frame_id`, `time` as an exact `[numerator, denominator]` of seconds, `time_s` (display float) |
| `frame_range` | `[first, last]` scanned frame id |
| `shots` | visual shots tiling the whole clip: `id`, `first_frame`, `last_frame`, `first_time`, `last_time` (exact `[numerator, denominator]` seconds). Entity ids are scoped by shot and an entity is only observed inside its shot |
| `shot_detection` | `method`, `version`, `parameters`, `pose_stage_shots` (`null` without `--poses`), `boundaries` (`frame_id`, `time`, `distance`, `isolation`, `agreement`) and `pose_only` gaps, see [Shots](#shots) |
| `entities` | `id`, `shot`, `family` (`surface`, `object`, `vehicle`, `person`), `motion` (`static`, `dynamic`, `unknown`), `labels` (`requested`, `native`, `normalized`, `status`, optional `candidates`), and for `person` entities `skeleton_id` |
| `observations` | `id`, `entity`, `frame_id` (a processed frame), `bbox` `[x0, y0, x1, y1]` in full decoded source pixels, `mask` (`null` or `{"asset": "<relative path>"}`), `score`, `score_meaning`, `visibility` (`visible`, `occluded`, `absent`, `uncertain`) |
| `poses` | `null` without `--poses`; otherwise `path`, `association`, `frame_count`, `skeletons` (`id`, `first_frame`, `last_frame`, `frames`) and `person_free_ranges` as `[first, last]` frame ids |
| `geometry` | `status` (`not_requested`, `unavailable`, `relative-camera-frame`, legacy `relative`, `registered_relative`, `registered_metric`), `mode`, `reason`, plus camera frames, coordinate convention and scale provenance; registered statuses include a versioned `registration` and scorer-facing `evaluation` |
| `backend` | `name`, `version`, `checkpoints`, `device` |
| `relations` | `contained_in` relations (a `door` or `window` inside a `wall`), each with `child`, `parent`, `frames` and `evidence` (`bbox`) |
| `config` | `requested_labels` (the user's `--classes`), `label_vocabulary` (`label`, `family`, `source` `preset` or `user`), `geometry_mode`, `sample_fps`, `device` |
| `assets` | `path` (relative to `<stem>.assets/`), `sha256`, `bytes` |

Mask assets are single-channel PNG images the size of the source frame (any non-zero pixel is inside). Scoring against human labels is described in [environment-scoring.md](environment-scoring.md).

### Camera calibration and measured references

Calibration uses schema `skeleton-maker.geometry-reference/1`. `image_size` must exactly match the decoded source clip. `camera.intrinsics` is a finite 3×3 pixel-space matrix with positive focal lengths and last row `[0, 0, 1]`; optional distortion is `none` or OpenCV Brown-Conrady with 4, 5, 8, 12 or 14 coefficients. Anchors name a source `frame_id`, source-frame pixel, measured `position_m`, role, stable ID and `uncertainty_m`. Measured dimensions name two pixel endpoints, a positive `length_m`, role, stable ID and uncertainty. At least three non-collinear `fit` anchors and a separate `check` anchor are required for metric fitting; check anchors and dimensions are never used to fit the transform. Required mode also needs these inputs before semantic inference when an independent DA3 worker supplies geometry.

```json
{
  "schema": "skeleton-maker.geometry-reference/1",
  "image_size": [1920, 1080],
  "camera": {
    "intrinsics": [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]],
    "distortion": {"model": "none", "coefficients": []}
  },
  "anchors": [
    {"frame_id": 0, "pixel": [320, 640], "position_m": [0, 0, 0], "role": "reference"}
  ],
  "measured_dimensions": [
    {
      "id": "door-height",
      "a": {"frame_id": 0, "pixel": [900, 200]},
      "b": {"frame_id": 0, "pixel": [900, 1000]},
      "length_m": 2.0,
      "role": "check"
    }
  ]
}
```

Loading (`skeleton_maker.envmanifest.load_manifest`) checks the schema version, structure, finite numbers, that observations reference real entities and processed frames, that boxes lie inside the frame, that every asset exists inside the bundle with the recorded hash, and that no path, absolute path or symlink leaves the bundle. Geometry registration validation checks transform direction, units, homogeneous matrices, proper rotations, independent fit/check IDs, check uncertainty, exact evaluation frame/point IDs, and explicit meter units on withheld dimensions.

A `registered_relative` geometry result has no solved metric scale and must not be overlaid on the NIM's meter-valued skeletons. `--geometry required` accepts only `registered_metric`.

Metric registration records `registration.scene_transforms` from each shot-local DA3 relative scene frame to the measured world frame, plus one `registration.camera_transforms` per sampled source frame from raw NIM camera meters to that world frame. DA3 extrinsics are world-to-camera. The NIM transform inverts that camera pose and maps the NIM camera basis into the DA3 basis; its meter-valued joint vectors are never multiplied by the DA3 scene scale, and root translation is not added a second time. The character stage's people-derived floor/up/center transforms are not used. The manifest records `root_translation_applied: false` and `stage_transform_applied: false` on camera transforms. Independent check-anchor projections and withheld dimension predictions are stored by ID in `geometry.evaluation`; only passing checks can produce `registered_metric`.

DA3-Small geometry starts as `relative-camera-frame`: depth and camera poses use relative model scale, and each shot has its own camera reference frame. Given sufficient independent camera and scene measurements, the adapter can add `registered_metric` transforms after fit and withheld-check validation. Without them, geometry remains unregistered and meter-valued skeletons are not overlaid as if they shared a world frame. Depth assets preserve DA3's processed pixel grid and declare `depth_pixel_space: "processed_frame_pixels"`; `preprocessing.undistorted_source_to_processed` and its inverse give the continuous x/y resize transforms for point coordinates. The adapter checks those full homogeneous maps against the actual source and processed dimensions and checks the source-space intrinsics against calibration and any lens correction. Dynamic and unknown-motion masks are resampled onto the depth grid with OpenCV `INTER_NEAREST` (legacy nearest-neighbor sampling); its discrete pixel selection can differ at integer boundaries from applying an exact inverse scale to each output index. The continuous coordinate matrices and the raster sampling convention describe separate operations. The resampled masks invalidate the selected depth pixels; that mechanism does not establish semantic mask accuracy. Tracks remain in the environment inventory and are listed under `geometry.excluded_dynamic_entities`; validated bounding boxes are used when masks are absent. With `--poses`, NIM skeletons remain in the pose tracks and are listed separately in `geometry.excluded_skeletons`; their clipped source-frame boxes invalidate depth pixels for sampled frames. Without `--poses`, `geometry.skeleton_exclusion.status` is `unavailable` and names why person-box exclusion could not be applied. `geometry.static_fusion_entities` names observed static environment IDs eligible for downstream static geometry. Existing manifests using the earlier `relative` status remain valid.

## Labels

The scan looks for a vocabulary of labels: the preset below plus anything given with `--classes`. The preset is a starting point, **not a promise**: a label in the list is searched for, and finding it is not guaranteed. Accuracy is measured with `environment-score` (see [environment-scoring.md](environment-scoring.md)), never assumed. People are not environment labels; the skeletons describe them.

| Family | Preset labels |
| --- | --- |
| Surfaces and structural elements | `wall`, `floor`, `ceiling`, `ground`, `road`, `stairs`, `door`, `window` |
| Objects | `chair`, `table`, `sofa`, `bed`, `desk`, `shelf`, `cabinet`, `bench`, `stool`, `bottle`, `cup`, `bowl`, `plate`, `knife`, `bag`, `backpack`, `box`, `bucket`, `ladder`, `hammer`, `drill`, `laptop`, `phone`, `monitor`, `lamp`, `book`, `pot` |
| Vehicles | `car`, `truck`, `bus`, `bicycle`, `motorcycle` |

Every entity carries one label record:

| Field | Meaning |
| --- | --- |
| `requested` | The vocabulary label the detection was prompted under, or `null` when none could be chosen. |
| `native` | The phrase the backend returned, kept as it came. |
| `normalized` | The vocabulary label, or `unknown`. |
| `status` | `matched`: the phrase is one vocabulary label (or a known alias of one). `ambiguous`: the phrase spans several vocabulary labels and `candidates` lists them. `unknown`: it matches none. |

A label that cannot be placed stays explicit: an `ambiguous` or `unknown` entity has `normalized: "unknown"` and is kept with its mask, as a per-frame `object`. It is never forced into a class and never counted as one. Aliases such as `couch` for `sofa` are applied only when the target label is in the vocabulary.

A `door` or `window` is its own surface entity. When its box lies inside a wall's box in a frame, the manifest adds a `contained_in` relation for those frames; neither mask is changed, and the relation is evidence from boxes only.

Masks are checked when a backend answers: each must be a readable single-channel PNG the size of the source frame, non-empty, and inside its observation's box. A visible non-person observation must have a mask; an absent one must not.

DA3-Small geometry is recorded as `relative-camera-frame` unless independent metric references validate a shared transform. Depth assets preserve DA3's native processed grid; the manifest records actual per-frame dimensions, independent x/y resize maps, and source-space intrinsics. Dynamic and unknown-motion tracks remain in the environment inventory but are listed under `geometry.excluded_dynamic_entities`; their masks or validated bounds invalidate corresponding depth pixels. `geometry.static_fusion_entities` names observed static IDs eligible for downstream static geometry. Existing manifests using the earlier `relative` status remain valid.

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
  "label_vocabulary": [{"label": "wall", "family": "surface", "source": "preset"}, {"label": "forklift", "family": "vehicle", "source": "user"}],
  "poses": null,
  "shots": [{"id": "shot-0", "first_frame": 0, "last_frame": 19}, {"id": "shot-1", "first_frame": 20, "last_frame": 39}],
  "source": {"width": 64, "height": 48, "frame_rate": [30, 1], "frame_count": 30},
  "frames": [{"frame_id": 0, "time": [0, 1]}, {"frame_id": 15, "time": [1, 2]}]
}
```

Response: `contract`, `status` (`complete` or `partial` with a `reason`), `backend` (`name`, `version`, `checkpoints`), `device`, `entities`, `observations` and `geometry`, shaped as in the manifest. Masks are files the backend writes under `assets_dir`, referenced by relative path. A response that fails validation, a raised exception or a keyboard interrupt is a failed run; nothing is written.

Tests substitute the inference boundary only, by registering a deterministic fake in `BACKENDS` (see `tests/env_fakes.py`). There is no flag or environment variable that selects a fake.

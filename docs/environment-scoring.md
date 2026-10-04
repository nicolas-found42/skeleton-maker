# Scoring environment scans

`skeleton-maker environment-score` grades environment manifests against human annotations and reports every gate separately. There is no combined score: `all_gates_passed` is true only when each gate is, so a strong floor result cannot hide a failed ceiling.

```bash
skeleton-maker environment-score --predictions runs/ --annotations labels/ --out score.json
```

`--split` picks the clips the gates score (`heldout` by default, or `development` or `all`). Corpus coverage always looks at every annotation. Exit codes: `0` all gates passed, `1` a gate failed, `2` invalid input (malformed annotation or manifest, missing directory, empty corpus). The report is written atomically.

Ground truth is human labelling only. Model predictions and Jev judgments are never ground truth. The targets below were fixed before any held-out evaluation; changing one needs an explicit spec change with its rationale. Scorer version: `2` (in the report as `scorer_version`). Version 2 adds annotation `scope` and `provenance`; the targets are unchanged. Rationale: the maintainer approved non-human annotation on 2026-10-04, and the independent camera and depth measurements that registration needs come from motion-capture datasets that carry no masks. Declaring a clip's scope keeps those clips out of the mask rules instead of padding them with empty labels.

## Inputs

A predictions directory of environment manifests (`*.json`, loaded with full validation), joined to annotations by `source.sha256`. A clip with an annotation but no manifest scores as all misses and fails coverage. Incomplete prediction runs also fail coverage, except when perception completed and only geometry is partial with an honest unregistered status (`not_requested`, `unavailable`, `relative` or `relative-camera-frame`).

Mask assets in manifests and annotations are single-channel PNG images the size of the source frame; any non-zero pixel is inside. Only observations with `visibility: "visible"` and a mask count as detections.

### Annotation file

One JSON file per clip, schema `skeleton-maker.environment-annotations/1`, with masks referenced by path relative to the file (no absolute paths, no `..`, no symlinks that leave the directory).

```json
{
  "schema": "skeleton-maker.environment-annotations/1",
  "clip": "kitchen-02",
  "source_sha256": "<64 hex chars of the exact clip>",
  "split": "heldout",
  "width": 1280,
  "height": 720,
  "frame_rate": [30, 1],
  "tags": ["indoor", "tripod", "person_free"],
  "aliases": {"couch": "sofa"},
  "review": {"second_reviewer": "name", "disagreements_resolved": true},
  "scope": "semantic",
  "provenance": {
    "kind": "published_dataset",
    "source": "VIPSeg",
    "citation": "Miao et al., CVPR 2022",
    "license": "non-commercial research",
    "quality_control": "what the dataset's authors did to check the labels"
  },
  "shots": [{"first_frame": 0, "last_frame": 299}],
  "frames": [
    {
      "frame_id": 45,
      "surfaces": [{"class": "floor", "mask": "masks/kitchen-02-45-floor.png"}],
      "negative_classes": ["ceiling"],
      "instances": [
        {"id": "chair-1", "family": "object", "class": "chair", "visibility": "visible", "mask": "masks/c1.png"},
        {"id": "chair-2", "family": "object", "class": "chair", "visibility": "occluded", "mask": "masks/c2.png"}
      ],
      "ignore": [{"mask": "masks/ambiguous-45.png"}]
    }
  ],
  "tracking_intervals": [{"first_frame": 30, "last_frame": 120}],
  "geometry": {
    "eligible_frames": [30, 45],
    "control_points": [{"frame_id": 30, "id": "door-corner", "xy": [412.0, 380.5]}],
    "dimensions": [{"id": "door-height", "meters": 2.03, "uncertainty_m": 0.01, "withheld": true}]
  }
}
```

- `scope` is `semantic` (default: masks, classes, identities) or `geometry` (camera and scene measurements only). A geometry-scope clip needs a `geometry` block and carries no surfaces, instances, ignore regions, negative classes or tracking intervals. The per-clip semantic rules below, and the clip totals, count semantic clips only; geometry-scope clips feed the registration, metric-scale, underconstrained and geometry-coverage checks.
- `provenance` names where labels came from: `human` or `published_dataset`, with `source`, `citation`, `license` and `quality_control`. For a held-out semantic clip, the second-review check accepts a `review` block or `provenance.kind` `published_dataset`; the report says which clips rely on the published dataset's own quality control. Whether that is as strong as a named second reviewer is the maintainer's call, so the check states it rather than hiding it.
- `split` is `development` or `heldout`; clips from one capture session must not appear on both sides.
- `surfaces` hold `wall`, `floor` and `ceiling` (the union of their pixels in the frame). `negative_classes` lists structural classes known to be absent from the frame; a class cannot be both.
- `instances` are `object` or `vehicle` with a stable `id` across the frame sequence and `visibility` `visible`, `occluded` or `absent` (absent needs no mask). `aliases` map model labels onto the annotation's class names.
- `ignore` regions are ambiguous areas. `tracking_intervals` must lie inside one shot: identities are never scored across a cut.
- A tag list drives corpus coverage; the required tags are in `TARGETS["required_tags"]`.

### Registration evidence in a manifest

Registration and scale are scored from an optional `geometry.evaluation` block in the manifest, written by the geometry stage: `frames` (`frame_id`, `registered`, `control_points` as `{id, xy}`: the predicted pixel position of each independently annotated control point) and `dimensions` (`{id, meters}`). A frame counts as validly registered only if `registered` is true and every annotated control point of that frame has a predicted position.

## Gates

| Gate | Rule |
| --- | --- |
| `objects`, `vehicles` | Class-correct, one-to-one mask matches at IoU at least 0.50 (greedy, best IoU first). Precision at least 0.80 and recall at least 0.70, separately. A class with at least 10 annotated instances needs recall at least 0.50. |
| `walls`, `floors`, `ceilings` | Pixel IoU pooled over positive frames at least 0.65, and at least 0.50 in every positive clip. On class-negative frames a predicted region over 1% of the image is a false-positive frame; the rate must be at most 5%. `structural_surfaces` is the AND of the three. |
| `identity_objects`, `identity_vehicles` | IDF1 at least 0.75 over the annotated tracking intervals, with ids assigned by an optimal one-to-one match per interval. Reports id switches, fragmentation (a matched track lost and re-found) and misses. |
| `registration` | Coverage of eligible frames at least 0.80, median reprojection error at most 0.5% and 95th percentile at most 2.0% of the image diagonal. Percentiles use the nearest-rank method. Abstained frames count against coverage. |
| `metric_scale` | Every withheld dimension within 10% relative error, and the manifest's geometry status is `registered_metric`. The annotation's `uncertainty_m` is reported with it. |
| `underconstrained` | Every clip tagged `underconstrained` reports `not_requested`, `unavailable`, `relative` or `relative-camera-frame` geometry, never a registration. |
| `corpus_coverage` | At least 12 semantic clips (6 development, 6 held-out), plus any geometry-scope clips; 10 annotated frames and a 2-second tracking interval per held-out semantic clip; second review (or a published dataset's quality control) on every held-out semantic clip; complete prediction runs; at least 2 held-out clips with positive annotations for each of objects, vehicles, walls, floors and ceilings, and at least one class-negative frame for each structural class; at least 6 object and 3 vehicle subtypes; every required scene tag; 2 held-out translating-camera clips with eligible frames and control points; one withheld metric dimension. |

## Empty sets, ignored regions and aggregation

- A zero denominator is never a pass. No annotated instances, no predictions (undefined precision), no positive pixels, no class-negative frames, no eligible frames and no withheld dimensions each fail their gate with a stated reason.
- Occluded instances and `ignore` regions are neither detections nor misses. A prediction that matches an occluded instance, or has at least half its area inside an ignore region, is dropped; ignored pixels are removed from both sides of every IoU.
- A frame the scan did not process counts as all misses (annotated instances are false negatives, structural positives score zero overlap) and is listed under `inputs.clips.<clip>.frames_unscanned`; unscanned class-negative frames are not counted.
- Counts and IoU pool over all scored clips; per-clip values are reported alongside. Nothing is averaged across gates.

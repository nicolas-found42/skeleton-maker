# Environment evaluation corpus, version 2

Written 2026-10-04 for #64 (corpus) and #66 (held-out run). It replaces the frozen YouTube split in `environment-corpus-manifest.json` as the scored corpus, because no person annotated the YouTube clips and the maintainer approved non-human annotation sources instead. The YouTube manifest stays as the record of what was frozen and why.

Nothing here comes from this project's models or from Jev as ground truth. Jev only ranked candidates inside a role. Every label is a published label, converted by a script in `scripts/`, and every measurement is a published measurement.

## Sources

| Source | Used for | Terms |
| --- | --- | --- |
| [VIPSeg](https://github.com/VIPSeg-dataset/VIPSeg-Dataset) (CVPR 2022) | 12 semantic clips: wall, floor and ceiling masks, object and vehicle instance masks with identities | Non-commercial research use only. Human-checked, machine-assisted labels; test labels are withheld, so all clips come from its train and val splits |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) `freiburg1_xyz`, `freiburg1_rpy` | 2 geometry clips: camera positions from an eight-camera motion-capture system, Kinect depth | CC BY 4.0 |

The first source gives masks and no camera truth. The second gives camera truth and no masks. That is why the scorer has two clip scopes (`semantic`, `geometry`) and a `provenance` field; see [environment-scoring.md](../environment-scoring.md).

## How the clips were chosen

1. VIPSeg statistics came from the label files only (`scripts/build_vipseg_corpus.py stats`). A video passed when its frame size is even, its frame spacing is uniform, it has at least 14 annotated frames and at most 5% unlabelled pixels.
2. Role filters, written before ranking. Indoor structure: wall, floor and ceiling in 60% of frames, at least two object classes, people in at most 30% of frames. Indoor objects: wall and floor in 60% of frames, at least three object classes. Outdoor vehicles: at least two vehicle instances, no ceiling, little wall.
3. Camera motion was measured per candidate (phase correlation of consecutive frames). A video below 0.3 px per frame at 320 px wide counts as a fixed camera.
4. Jev ranked each role's top ten (`noul` per video, one call per role and split). Held-out clips come from VIPSeg val, development clips from train, so the two sets share no video.
5. The top-ranked videos were taken, with two changes made by a stated coverage rule, not by preference. A required scene tag that no top-ranked clip carried was filled from the highest Jev-ranked candidate that carried it:
   - `vehicle_stationary`: development vehicle slot 6 became `577_IzGj3UWB83M` (fixed camera, parked and moving vehicles; Jev 0.72, tied with two others, fitness breaks the tie).
   - `edited_cut`: development object slot 3 became `971_y_XHCtfTt1Y` (a shot cut at frame 21 of 46; the only candidate with a cut).
   - A fixed-camera held-out vehicle clip was required by the corpus plan; `2381_zHtCNhABlLw` is the highest-ranked fixed-camera val candidate (Jev 0.72 in the fixed-camera set).
6. Jev rankings that disagreed between sets were kept as they came (`2381` scored 0.32 among 10 mixed candidates and 0.72 among four fixed ones). No ranking was repeated to improve a score.

## The corpus

| Clip | Split | VIPSeg video | Camera shift px | Tags |
| --- | --- | --- | --- | --- |
| vip-h1-indoor-structure | held-out | 360_5Y1GpL768Sk | 1.218 | indoor, occlusion, person_free |
| vip-h2-indoor-structure | held-out | 2303_hzYaY23ijn0 | 3.459 | indoor, occlusion, person_free |
| vip-h3-indoor-objects | held-out | 950_OgypP-5jWJ4 | 0.066 | handled_object, indoor, occlusion, tripod, underconstrained |
| vip-h4-vehicles | held-out | 2289_cnZjjNYsrtw | 9.635 | occlusion, outdoor |
| vip-h5-vehicles | held-out | 154_6-pyJNd1h3I | 0.398 | occlusion, outdoor |
| vip-h6-vehicles-fixed | held-out | 2381_zHtCNhABlLw | 0.099 | occlusion, outdoor, person_free, tripod, underconstrained, vehicle_moving |
| vip-d1-indoor-structure | development | 980_ZIngSGeILZ4 | 2.662 | indoor, occlusion, person_free |
| vip-d2-indoor-structure | development | 989_ga6HgsrRVFE | 3.497 | indoor, occlusion, person_free |
| vip-d3-indoor-objects | development | 971_y_XHCtfTt1Y | 5.541 | edited_cut, handled_object, indoor, occlusion |
| vip-d4-vehicles | development | 779_aNNsqEq4FnQ | 10.963 | outdoor |
| vip-d5-vehicles | development | 162_G4K8AjNIVPA | 0.893 | occlusion, outdoor |
| vip-d6-vehicles | development | 577_IzGj3UWB83M | 0.111 | occlusion, outdoor, tripod, underconstrained, vehicle_moving, vehicle_stationary |
| tum-fr1-xyz | held-out | rgbd_dataset_freiburg1_xyz | n/a | indoor, translating |
| tum-fr1-rpy | held-out | rgbd_dataset_freiburg1_rpy | n/a | indoor, motion_blur, translating |
| tum-fr2-rpy | development | rgbd_dataset_freiburg2_rpy | n/a | indoor, pan, underconstrained |

Each VIPSeg clip is at most 16 consecutive annotated frames (about 3 s at 5 fps), scaled to 640 px wide, with masks scaled by nearest neighbour. Both limits come from measurement on the 16 GiB reference machine, not from taste:

- A native 2560 × 1440 clip made the Grounded SAM 2 worker grow to 21 GB and swap; the run was stopped after 3 of 15 frames in 14 minutes and its log is kept.
- At 640 × 360 the real pipeline took 748 s for 6 frames (about 2 min a frame) with a 985 MB peak resident size. At that speed the full 46-frame clips would not finish in a working day.
- The DA3 geometry worker accepts at most 16 sampled frames per shot, so a 16-frame window also lets it run on these clips.

A window starts at the first frame unless the video has a cut; then it is the first window that keeps the cut at least three frames from either end (the shot detector misses a cut on the last frame) and still holds a two-second run inside one shot.

The TUM clips keep the full 30 fps colour video (723, 798 and 901 frames, 640 × 480) and are scanned at 0.5 fps, so the geometry worker sees 13, 14 and 16 frames. A clip is cut to at most 16 samples for the same reason.

TUM anchors and dimension. Four fit frames and four check frames are chosen on the 60-frame grid; each frame gives corners on flat valid-depth pixels, back-projected through the measured pose. The check dimension is the distance between two measured check points (1.64 m for xyz, 1.85 m for rpy, 2.41 m for fr2-rpy) and is withheld from the pipeline. Depth and pose agree to a median of 1.9 cm (xyz), 2.5 cm (rpy) and 0.9 cm (fr2-rpy) when points seen in two frames are compared. The anchor uncertainty is bounded at 3 cm per point, and the builder refuses a sequence whose median disagreement is above it.

## Tags: measured, not assumed

Each tag follows a rule in the converter, and a tag is absent when the rule did not fire.

| Tag | Rule |
| --- | --- |
| indoor / outdoor | ceiling in at least 2 frames at 2% of the image, or outdoor stuff classes at 15% |
| person_free | no person label in any frame |
| edited_cut | the project's own shot detector found a cut in the encoded clip |
| occlusion | a vehicle or object instance of at least 0.2% of the frame split into two or more components |
| handled_object | an object instance touching a person that is larger than it |
| tripod, underconstrained, vehicle_moving, vehicle_stationary | camera shift below 0.3 px; vehicle motion only on fixed-camera clips (step at least 1.5% of the width a frame is moving, at most 0.3% is stationary) |
| translating | mocap path extent of at least 0.3 m |
| pan | camera turns at least 20° and moves under 0.3 m (`tum-fr2-rpy`: 52° at 0.173 m) |
| motion_blur | at least 5% of poses turn at 100° a second or faster. Checked by eye on `freiburg1_rpy` frame 429 (185° a second, visibly smeared) beside frame 244 (3° a second, sharp). A sharpness-ratio test was tried first and flagged 45% of the slow `freiburg1_xyz`, so it was dropped |

The 100° a second threshold was set after looking at one frame pair, so treat it as a proxy checked on one example, not a validated blur detector.

## Open gaps and rejected inputs

- `pan` was not covered when the corpus was frozen: `freiburg1_rpy` turns 115° but its path extent is 0.328 m, just above the 0.3 m rule, so it is tagged `translating`, and the refused `freiburg1_360` turns 180° but moves 0.85 m. The 0.3 m rule was fixed before the data was seen and was not changed. The maintainer approved one more download; candidate ground-truth trajectories were measured with the converter's own rule before anything was chosen. `freiburg2_rpy` qualified (52° at 0.173 m on the truncated 30 s window, depth/pose agreeing to 0.9 cm) and joined development. `freiburg3_sitting_rpy` also measures as `pan` (extent 0.2999 m, a 0.0001 m margin — the tag flips if the sequence is re-recorded or remeasured) and needs the Freiburg 3 Xtion calibration with a different depth scale, so it was rejected on the margin; the fr1/fr2 `_validation` twins publish no ground truth, and every other fr2 quasi-static sequence moves 1.8–2.8 m.
- `freiburg1_room` was built and refused: depth and pose disagree by a median 6.4 cm against the 3 cm bound. `freiburg1_360` was refused for the same reason. Their outputs are kept outside Git as rejected inputs.
- The second review for held-out semantic clips is VIPSeg's own documented quality control (expert checking of machine-propagated masks), not a named second reviewer. The scorer states which clips rely on it. The maintainer accepted this in place of a reviewer; it is a weaker guarantee than #64 first asked for.
- `tum-fr2-rpy` is the first development geometry clip, so geometry behavior can now be exercised on development data; `tum-fr1-xyz` and `tum-fr1-rpy` stay held-out. The development split was frozen with six semantic clips; the maintainer approved this addition.
- `motion_blur`, `translating` and `pan` cannot be measured on VIPSeg clips, so no semantic clip carries them.
- The sports-hall dimension candidate from [environment-independent-references.md](environment-independent-references.md) was not used. TUM supplies the measured dimension.
- Whether the perception models saw VIPSeg frames in training is unknown. VIPSeg videos come from YouTube, and a model trained on web video may have seen them. Scores here measure agreement with VIPSeg labels, not generalisation to unseen footage.
- The VIPSeg label ontology and the project's labels differ. Model labels map to VIPSeg classes through the `aliases` field of each annotation (chair and stool to `chair_or_seat`, bottle and cup to `bottle_or_cup`, and so on). Jev chose the mapping before any pipeline run; bucket (0.56) and monitor (0.64) were the least certain. People, animals, doors, windows and VIPSeg stuff classes the scorer does not grade are ignore regions, so a model that reports them is not penalised.

## Reproduce

```sh
python -m scripts.build_vipseg_corpus convert VIPSeg/ VIDEO OUT --name NAME --split heldout --max-width 640 --max-frames 16
python -m scripts.build_tum_reference SEQUENCE OUT --name NAME --split heldout --sample-step 60 --max-samples 16
python -m scripts.hash_corpus CORPUS > hashes.json
python -m scripts.run_environment_corpus CORPUS RUNS --split development
skeleton-maker environment-score --predictions RUNS --annotations CORPUS/annotations --split heldout
```

The corpus files stay outside Git (`~/Documents/datasets/skeleton-maker/corpus-v2`). The tree hash of the corpus used for a result is recorded with the result.

# What to do with the skeletons — brainstorm, 2026-10-03

Scope: what to build on top of `pose.json` (77 joints, metric 3D, 2D keypoints + confidence,
joint rotations, root pose, tracking ids). Method: GitHub + awesome-list search, with Jev
(`typesafe/jev-1.13` via OpenRouter) used to triage, score and test. ~7,000 Jev calls in total.

**Read this first — what is and is not established**

- Every Jev *motion* experiment below ran on **synthetic** 16-joint skeletons with feature digests
  I wrote, not on real Nova-77 output. They show how Jev behaves when the evidence is in the
  digest; they do not show it works on real NIM data. That needs one real clip (see "Next").
- Jev's *search/triage* judgments were useful but noisy (examples in "Where Jev was wrong").
- Repo facts (stars, licence, last push) were read from the GitHub API on 2026-10-03.
  arXiv entries marked † come from awesome-list READMEs and were not individually opened.

## Recommended order

1. **Exporters (BVH / glTF / C3D) + a Blender importer.** Table stakes; many downstream tools
   consume these. Our edge over single-person estimators (e.g. `squall01337/mixamo-llm-mocap`,
   GVHMR-based): multi-person with stable ids and per-joint rotations already in the file.
2. **A features + Jev layer: natural-language motion search and event cutting.** The
   experiments below say this is promising *if* built as many atomic questions composed in code.
3. **Per-failure-mode quality gate** (jitter, id teleport, limb fly-away, L/R swap) before
   anything trusts the data.
4. **Retargeting to humanoids / avatars** (GMR, VRM). Check each target's expected input
   skeleton before committing; not verified here.

Skip or gate behind explicit consent: gait re-identification, aggression/weapon alerts.
Jev's composite ranked "aggression/weapon-draw alerts" 7th of 60 (mean of two question
variants); the composite has no ethics weight beyond one risk question, and I would not follow it.

## Hidden gems (verified to exist; low stars, active)

| Repo | ★ | Last push | Licence | Why it is interesting |
|---|---|---|---|---|
| [posecode-dev/posecode](https://github.com/posecode-dev/posecode) | 117 | 2026-10-03 | Apache-2.0 | Text language + validator + Three.js renderer for inspectable 3D movement; a human/LLM-editable layer over skeletons |
| [squall01337/mixamo-llm-mocap](https://github.com/squall01337/mixamo-llm-mocap) | 335 | 2026-09-26 | unspecified | Video → Mixamo rig via GVHMR + Blender FK; the closest existing competitor for step 1 |
| [maosika-ai/reshot](https://github.com/maosika-ai/reshot) | 46 | 2026-09-28 | Apache-2.0 | "Copy the shot, not the actors": video → depth/OpenPose skeleton → re-cast in generative video |
| [yedp123/ComfyUI-Yedp-Action-Director](https://github.com/yedp123/ComfyUI-Yedp-Action-Director) | 380 | 2026-06-01 | MIT | Mixamo FBX → ControlNet passes; BVH/FBX export would plug straight in |
| [sign-language-processing/pose](https://github.com/sign-language-processing/pose) | 114 | 2026-09-19 | MIT | `.pose` file format + viewer/augmentation lib; a ready-made container and tooling |
| [emilzawistowski/drumming-sonification](https://github.com/emilzawistowski/drumming-sonification) | 0 | 2026-09-10 | MIT | Joint motion → MIDI/OSC; the most unusual use found |
| [TaatiTeam/stgcn_parkinsonism_prediction](https://github.com/TaatiTeam/stgcn_parkinsonism_prediction) | 35 | 2025-07-31 | unspecified | Clinical gait scores from skeleton trajectories |
| [aleflabo/MoCoDAD](https://github.com/aleflabo/MoCoDAD) | 92 | 2026-05-17 | MIT | Skeleton-based video anomaly detection (ICCV) |
| [opensim-org/opensim-fitter](https://github.com/opensim-org/opensim-fitter) | 6 | 2026-10-02 | Apache-2.0 | Fits OpenSim kinematics to mocap/video-derived data → biomechanics without a lab |
| [opencap-org/opencap-processing](https://github.com/opencap-org/opencap-processing) | 154 | 2026-09-10 | Apache-2.0 | Processing utilities for video-based biomechanics |
| [digitalworlds/UPose](https://github.com/digitalworlds/UPose) | 18 | 2026-03-23 | MIT | Joint rotations from pose → Unity avatar stream (we already have rotations) |
| [JGL/TrackOSC](https://github.com/JGL/TrackOSC) | 26 | 2026-09-25 | MIT | Streams body-tracking results over OSC to creative-coding tools |
| [Vivixiao980/tennis-video-coach-report](https://github.com/Vivixiao980/tennis-video-coach-report) | 27 | 2026-06-30 | MIT | Pose → rally segmentation → PDF coaching report |
| [mzniu/classroom_attention](https://github.com/mzniu/classroom_attention) | 48 | 2026-07-13 | Apache-2.0 | Classroom engagement from pose; also the privacy question in miniature |
| [AmmarkoV/SAM3DBody-cpp](https://github.com/AmmarkoV/SAM3DBody-cpp) | 681 | 2026-10-01 | MIT | Multi-person 3D body from one camera; a comparison/cross-check for NIM output |

Bigger, not hidden, but directly relevant: [YanjieZe/GMR](https://github.com/YanjieZe/GMR)
(2,741★, retargeting to humanoids), [nv-tlabs/vid2player3d](https://github.com/nv-tlabs/vid2player3d)
(345★, tennis skills from broadcast video), [sh-akira/VirtualMotionCapture](https://github.com/sh-akira/VirtualMotionCapture)
(827★, MIT). HumanX ([arXiv 2602.02473](https://arxiv.org/abs/2602.02473), checked) learns humanoid
interaction skills from human video. A surprising neighbour: `ruvnet/RuView` (96k★) senses presence and vital signs from
WiFi with no video, a privacy-preserving contrast to the anonymisation idea.

## Awesome lists

Found on GitHub but **absent from context-awesome** (checked with its `browse_awesome_lists`):
`Zilize/awesome-text-to-motion`, `SUZ-tsinghua/awesome-humanoid-ball-sports`,
`we-dance/awesome-dance`, `AwesomeDance/AwesomeDance`, `AtomScott/awesome-sports-analytics`,
`Foruck/Awesome-Human-Motion`, `YanjieZe/awesome-humanoid-robot-learning`,
`BNU-IVC/Awesome-Gait-Recognition`, and the sign-language family
(`VIPL-SLP/Awesome-Sign-Language-Processing`, `ZechengLi19/Awesome-Sign-Language`, ...).
Present in context-awesome but **empty there** (0 items): `akirosingh/awesome-climbing-web`,
`freekatz/awesome-embodied-ai-datasets`. Best present ones: `modenaxe/awesome-biomechanics`
(467 items), `derikon/awesome-human-motion`, `wentaol86/awesome-human-video-generation`.

The humanoid-ball-sports list is a small, fresh cluster of 2026 work converting human motion
into robot skills (tennis, badminton, soccer†). It is the most "new category" signal found.

## Jev experiments (what was run, what it showed)

| # | Question | Result |
|---|---|---|
| 1 | Triage 92 awesome lists for downstream-use value | Sensible ordering; biomechanics, gait, dance, sign language and humanoid lists rose, UI "skeleton loader" lists were excluded |
| 2–3 | Score 1,946 repos (644 from list READMEs, 1,302 from 125 readme-scoped searches) | 59 candidates after filters (consumes skeleton data, <300–500★, pushed since 2025); fitness rep counters correctly scored as unsurprising |
| 4 | Rank 60 ideas; rerun with reworded questions and with the idea description removed | Single dimensions agree across rewording (r=0.71–0.90); the **composite ranking does not**: Spearman 0.66 (reworded), 0.74 (no description), top-10 overlap only 3–8 of 10. Treat the ranking as a shortlist generator, not an ordering |
| 5 | Name the action in a 2 s synthetic window, 10 actions × 20 | One 10-way Choice: 52–79% depending on digest layout, **34–40% on noisy input, with confident errors** (in the first noisy run only 4% of answers at ≥0.9 confidence were correct; with a better digest the same bucket was 87%, so confidence says little when the input quality shifts). One yes/no question per property composed by code rules: 87–89%; 74–76% on the noisy input |
| 5b–d | Debug the failures | Four separate causes, none of them "the model can't": (a) digest lacked end-of-window values and body-size normalisation, (b) a compound question (`seated` = knees + hips + still) answered 0.46; split into three atomic questions → 20/20, (c) my rule order checked squat before sit, (d) smoothing flattened a 2 Hz wave below my amplitude threshold |
| 6 | Free-text motion search, 8 queries × 3 paraphrases, 200 windows | Macro ROC-AUC 0.948. Concrete queries (airborne, on the floor, arm thrust) AUC 1.0 with no paraphrase spread; abstract ones vary: "seated" 0.54–1.0 depending on wording |
| 7 | Quality gate: 5 injected failure modes vs 200 clean windows | A dedicated yes/no per failure mode: AUC 0.97–1.0, 100% detected. A single generic "is this unreliable?" question false-alarmed on 33% of clean windows (every run, jump and punch window: fast motion is not a defect). The `flat_depth` detector false-alarmed on ~39% of clean windows, because standing people are genuinely shallow in depth |

Design lessons these support (on synthetic data): ask atomic questions and compose in code;
design the digest as carefully as the questions; never gate on one generic confidence or
"unreliable" question; normalise by body size; per-failure-mode detectors with their own thresholds.

### Where Jev was wrong in the search work

- A paper on muscular fatigue passed the "uses body motion data" filter; its input is sEMG,
  not motion ([arXiv 2602.15684](https://arxiv.org/abs/2602.15684), opened and checked).
- DDPM and PPO paper entries from a humanoid-motion list scored as "genuinely unusual" uses of
  motion data (they are generic ML references). About 7 of the top 40 items in the
  absent-list pass were noise like this.
- The first novelty probe rated a vertical-leap leaderboard 0.92 "genuinely new"; later runs
  scored fitness-style ideas low. Absolute levels drift; compare within one run.

## Next

1. Run one real clip through the NIM and repeat experiments 5–7 on actual Nova-77 digests.
   `skeleton_maker/nova77.py` now maps the canonical joints used by the character stage;
   its left/right assignment is inferred per track. Use that map and retain the assignment
   uncertainty when building features.
2. Prototype `skeleton-maker export --format bvh|gltf` first; it is useful regardless of what
   the Jev experiments turn out to mean.
3. Reproduce: scripts and cached Jev answers are in the session scratchpad
   (`.../scratchpad/exp*.py`, `research/*.json`); say the word and they move into `experiments/`.

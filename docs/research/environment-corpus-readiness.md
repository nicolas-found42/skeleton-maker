# Environment corpus readiness for #64 and #66

Status on 2026-10-04: neither #64 nor #66 is satisfied. This report records what could be prepared without a person, what was checked, and which human-produced artifacts are still missing. It adds no ground truth and claims no accuracy. Model predictions, Jev judgments and the contact-sheet observations below are never annotations.

## What was verified

- All 12 candidate files in `environment-youtube-development/` match the SHA-256 digests in [the source inventory](environment-source-candidates.json). All 120 proposal frames match the digests in `annotation-proposal-20261004/frame-proposal.json` (ledger SHA-256 `ea69d893ddb905b59e47242880e07810bec21fb3c33f76cb21ced136b467ab17`; no mismatches).
- Prior exposure to model output: the only files in `work/` and `docs/` that contain any of the 12 candidate digests are the source inventory, its working copy `work/jev-run/corpus-inventory.json`, and two Jev gate-input files (`work/jev-run/docs-gate.json`, `work/resume-jev/final-gate-inputs/batch-01.json`). No environment manifest or run ledger contains one. The only real-model runs used the Street traffic excerpt (`_7sCYyxw4Ic`, SHA-256 `38c01a6a…`) and the local `cooking.mp4` clip (SHA-256 `8a3ed3b6…`). Neither is among the 12 candidates, so neither can populate the held-out side and both are development-exposed. This test matches exact bytes only: a re-encoded copy of a candidate would not be detected.
- Human exposure is not zero: the candidate frames were viewed during source selection and again for this report (contact sheets, ten frames per clip). A reviewer should treat held-out frame selection as made by the midpoint-strata rule in the proposal ledger, not by anyone's viewing.
- Shot boundaries from the repository's classical detector (`skeleton_maker.shots`, method `color-histogram-isolated-peak`, version 1, 5 samples per second, no environment model). Every clip has at least one shot of 2 seconds or more, which the shot-local tracking interval needs. Cut frames: `RGQSkVRiw_A` 134, 402, 507, 882, 949, 1004; `AQfaV49HCdU` 104, 283; `R33oeO9TMcc` 21, 203, 285, 346; `3R_H8mbPwzg` 75; the other eight clips have none. The full output is `annotation-proposal-20261004/shot-audit.json` outside Git (SHA-256 `0177c667c36e6d733b53dbf0dfbbf108a07f41e41ea6eedfe035bd66d8d49942`). Cuts are a detector proposal; the annotator confirms them.

## Observations from contact sheets (unreviewed, not labels)

| Clip | Family | Seen in the ten proposal frames |
| --- | --- | --- |
| `Avg9pGlqwf0` | indoor | Dorm room, a person on camera throughout, camera turning across the room, bed, lamp, shelves, bunk bed |
| `RGQSkVRiw_A` | indoor | Dutch apartment, painted ceiling beams, tile floor, sofa, chair, lamp, no person seen, handheld tilts |
| `_dOzjhTsR4A` | indoor | Small hallway, camera appears fixed for the first frames, checkered floor, cabinets, plant, a shoe handled by the person |
| `AQfaV49HCdU` | indoor | Conference room, long table, chairs, windows, ceiling lights, gimbal-style moves, on-screen text overlay |
| `R33oeO9TMcc` | indoor | Low-resolution (400 × 224) lobby, ceiling visible, black pillarbox bars in some frames, a person only at the start |
| `3R_H8mbPwzg` | indoor | Kitchen, then a cut to a garden with people; 480 × 270 at 10 fps |
| `QfnC5TucOEo` | outdoor | Moving camera in congested road traffic under an overpass: cars, vans, trucks, a bus |
| `B_Rtsa3Z7oM` | outdoor | Moving camera on an open road, a pickup truck ahead, some motion blur |
| `fA8gI34Pv-s` | outdoor | Fixed elevated camera, dense stationary and slow traffic at dusk |
| `x6zdwIBdLLM` | outdoor | Moving wide-angle camera at an intersection, vans and cars, motion blur |
| `wzI-Sm0CshA` | outdoor | Moving camera with windscreen edges and dashboard occluding, a bus |
| `53rmMpz3WcI` | outdoor | Moving camera with mirrors and handlebars occluding, trucks and cars |

Outdoor road clips under overpasses show large dark structures overhead; an annotator will have to decide whether those are ignore regions or ceiling negatives, since the known ceiling failure on outdoor frames lives there.

## Requirement status for #64

| Requirement | Status |
| --- | --- |
| 12 clips, 6 development and 6 held-out, split by capture session | 12 candidates from 12 different uploaders exist, so session independence is plausible. The split is unassigned and unfrozen. A person must confirm capture sessions and freeze it. |
| Rights and source manifest, cleared for the intended use | Source URLs, uploaders and reported licences are recorded in the inventory. That is metadata. A person must clear third-party content (faces, trademarks, music) and record the decision per clip. |
| Coverage: families, subtypes, motion conditions, geometry | See the proposed split below. Coverage is plausible from the pixels on the contact sheets but has never been established by labels. |
| Independent annotation of masks, classes, identities, ignored regions, visibility, cuts, geometry eligibility | Missing. No annotation exists. |
| Second reviewer on all held-out category and identity labels and a sample of masks | Missing. No reviewer was supplied. |
| At least ten selected evaluation frames per held-out clip, selected before examining output | Prepared: ten midpoint-strata frames per clip, created 2026-10-04T08:26:53Z, before any candidate prediction exists. The ledger is unreviewed, and the human exposure above applies. |
| Two translating-camera held-out clips with independent control observations | Candidate clips exist. No control observations exist. |
| One measured scene dimension, withheld from scale fitting, with stated uncertainty | Missing. No candidate source provides a measurement. Someone must measure a scene, or supply a reference scan or a documented standard dimension with its uncertainty. |
| Hashes recorded, media outside Git | Met for inputs and proposal frames. Annotations do not exist yet. |

## Proposed split (not frozen)

This is a proposal made from the contact sheets, chosen so the held-out side can meet the scorer's coverage gates. Freezing it, or choosing another, is the person's decision.

- Held-out: `RGQSkVRiw_A` and `AQfaV49HCdU` (walls, floors, ceilings, objects, cuts, person-free), `_dOzjhTsR4A` (handled object, fixed camera), `fA8gI34Pv-s` (fixed camera, stationary vehicles, underconstrained), `QfnC5TucOEo` and `x6zdwIBdLLM` (translating camera, vehicles, motion blur, occlusion).
- Development: `Avg9pGlqwf0`, `R33oeO9TMcc`, `3R_H8mbPwzg`, `B_Rtsa3Z7oM`, `wzI-Sm0CshA`, `53rmMpz3WcI`.

Against the scorer's `TARGETS`: walls, floors and ceilings would each have two positive held-out clips, objects four, vehicles three, translating clips two, and tripod, pan, edited cut, person-free and underconstrained tags each at least one clip. Tags and subtype counts are unconfirmed until a person labels them. The scorer's coverage gate counts tags over all 12 clips, but its "complete prediction runs" check requires a prediction manifest for every clip, development clips included, so #66 must run all 12.

## What a person must still supply

1. Capture-session and rights confirmation per clip, and a frozen split.
2. Annotations in the `skeleton-maker.environment-annotations/1` format for every clip: masks, classes and aliases, instance identities, visibility, ignore regions, shot boundaries, geometry eligibility, with at least ten frames and one shot-local interval of two or more seconds per held-out clip.
3. A second reviewer for every held-out label and a documented sample of masks, with disagreements resolved.
4. Control point observations for the two translating clips.
5. One measured dimension in a held-out clip with its uncertainty, withheld from scale fitting.

After 1 to 5 exist and validate, #66 freezes the development configuration, hashes inputs, annotations, configuration and scorer, runs all 12 clips once, and reports every gate separately, including failures. The known outdoor ceiling failure stays in the record.

## Judgments

`jev ask` (model `typesafe/jev-1.13-20260917`) rated these propositions as unlikely: that the agent can fully satisfy #64 without a human artifact (0.01), that agent-drawn masks count as independent human annotation (0.02), and that Creative Commons metadata alone clears every third-party element (0.09). It chose preparation over self-annotation (0.99). These judgments are advisory and agree with the issue text; the checks above are the evidence.

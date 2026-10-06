# Independent references for #64 without a person annotating

Written 2026-10-04. The question: can the independent annotations, camera controls and measured dimension that #64 needs come from published sources instead of a person drawing them? This report cites primary pages and records what the sources do and do not give. Nothing was downloaded from the datasets below, and no label in this repository comes from a model or from Jev.

## Annotation: published human-checked sets

| Source | What it gives | Access and terms | What it lacks |
| --- | --- | --- | --- |
| [VIPSeg](https://github.com/VIPSeg-dataset/VIPSeg-Dataset) (CVPR 2022) | Panoptic masks per frame. "Stuff" classes use the class ID; "thing" classes use `category_id × 100 + instance_id`. The category file includes wall, floor, ceiling, car, bus and truck. 3,536 videos, 84,750 annotated frames, 232 indoor and outdoor scenes. | Google Drive and Baidu links. "The data is released for non-commercial research purpose only." The README does not give file sizes or the licences of the source videos. | No camera poses, no measured scale. |
| [TUM RGB-D](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) | Kinect RGB-D video at 640 × 480, 30 Hz, with camera trajectory from "a high-accuracy motion-capture system with eight high-speed tracking cameras (100 Hz)". | "licensed under a Creative Commons 4.0 Attribution License (CC BY 4.0)". The archives are direct downloads without a login: `rgbd_dataset_freiburg1_xyz.tgz` answered with HTTP 200 and 448,204,271 bytes. Listed sizes: fr1/xyz 0.47 GB, fr1/rpy 0.42 GB, fr1/room 0.83 GB. | No semantic labels. Scenes are small indoor offices. |
| [ARKitScenes](https://github.com/apple/ARKitScenes) | iPad LiDAR RGB-D scans, camera poses, laser-scanner depth for a subset, and manually labelled oriented 3D boxes for furniture. | Apple software licence ("personal, non-commercial, non-exclusive"); the README does not quote the data terms. | Boxes, not masks. Indoor only. |
| [ScanNet](https://github.com/ScanNet/ScanNet), [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) | Closest single fit: instance labels with consistent IDs, camera poses and laser or RGB-D geometry. | ScanNet needs a signed terms-of-use form; KITTI-360 needs registration and a stated purpose. Both are account or agreement steps that a person must complete. | Not available to an agent. |

VIPSeg's masks are not hand-drawn from scratch. The paper says annotators double-check machine output, propagate masks with a model, and repeat refinement until the result is satisfactory. Instance identities come from a multiple-object tracker with human correction of wrongly associated instances. Those labels are independent of this repository's models, but they are human-checked machine-assisted labels, and the passages retrieved from the paper do not describe a separate second review of every clip (Jev rated the whole statement 0.42, so check the full paper before relying on it). Whether that meets "a second reviewer checks all held-out category and identity labels" is a decision for the maintainer, because it changes the text of #64.

## Measured dimension: published standards

A published standard is a reference only when the object in the clip follows it.

- [FIBA Official Basketball Rules 2024, equipment](https://assets.fiba.basketball/image/upload/documents-corporate-fiba-official-rules-2024-official-basketball-rules-and-basketball-equipment.pdf): backboards measure 1,800 mm (plus at most 30 mm) horizontally and 1,050 mm (plus at most 20 mm) vertically; the top of the ring is 3,050 mm (plus or minus at most 6 mm) above the floor; the ring inside diameter is 450 to 459 mm. The [NBA rule](https://official.nba.com/rule-no-1-court-dimensions-equipment/) gives a 6 ft by 3.5 ft backboard.
- [Commission Regulation (EU) 1299/2014, section 4.2.4.1](https://www.legislation.gov.uk/eur/2014/1299/annex/division/4/2020-01-31/data.xht?view=snippet&wrap=true) (retained in UK law): "European standard nominal track gauge shall be 1 435 mm." Gauge tolerances sit in the maintenance limits, not in that sentence.

Forty-nine more Creative Commons clips (court markings, road markings, containers, rail, parking, plates and signs) were downloaded to `environment-pool-20261004-reference/` outside Git, classified by Jev from titles, and reviewed on contact sheets. Most were unusable: promotional slides, stock footage with text, or views without a measurable standard object. Two stand out.

- `JroNis3xZPM` (school sports hall, 12 s, translating camera): a basketball backboard and ring are visible above a court with several overlapping line sets. It would give two independent published values, backboard width (fit) and ring height (withheld). The clip does not say where it was filmed or whether the hoop meets FIBA, and the ring sits on what looks like a handball goal frame. Jev rated compliance risk 0.83. The values are valid only if a person confirms the hoop type.
- `DH6WNj3XnEk` (fixed view down a UK railway line): standard gauge is a hard standard, but it gives one dimension, so nothing is left to withhold, and a fixed camera is underconstrained for scale. Jev rated a second published dimension 0.04.

Jev chose `JroNis3xZPM` as the best fit (0.79, none adequate 0.21), and its own compliance rating shows the gap. No candidate clip yet has a verified measured dimension, so the metric-scale requirement of #64 stays open.

## Status

- Annotations for the frozen 12 clips: still missing. An agent cannot supply them under the rule that predictions and Jev are not ground truth.
- If VIPSeg replaces the pixel annotation, #64's text changes (clips would come from the dataset, not the frozen 12) and the held-out split must be re-made from its own splits with the exposure rule applied. Registration and metric scale would then need TUM RGB-D or a verified reference clip.
- Control points: none exist. TUM RGB-D's motion-capture trajectory is the only independent camera truth found without an account.
- Source licences: all 100 pool and 49 reference clips report Creative Commons Attribution on YouTube, accepted by the maintainer for personal evaluation. VIPSeg and ARKitScenes are non-commercial only; TUM is CC BY 4.0.

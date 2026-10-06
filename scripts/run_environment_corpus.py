# SPDX-License-Identifier: MIT
"""Run the real environment pipeline over a corpus, once per clip, and record how it went.

The configuration is fixed in this file and in the clip's own annotation, never chosen per clip:
every clip is scanned at the rate that visits every annotated frame, on the same device, with the
cache off so a recorded wall time is a real one. A clip with a geometry reference is given it
through ``--calibration``. One record per clip captures the exact command, exit code, wall time,
peak resident memory and the SHA-256 of the manifest, so a result can be audited and a failed run
stays on the record.

    python -m scripts.run_environment_corpus CORPUS OUT --split development
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

DEVICE = "mps"
GEOMETRY = "auto"
SPLITS = ("development", "heldout")
#: ``/usr/bin/time -l`` prints this on macOS.
_PEAK_RSS = re.compile(r"^\s*(\d+)\s+maximum resident set size", re.MULTILINE)


class RunError(RuntimeError):
    """The corpus is not laid out as expected."""


def sample_fps(annotation: dict, build_report: dict | None) -> float:
    """The scan rate that lands on every annotated frame (semantic) or anchor frame (geometry)."""
    numerator, denominator = annotation["frame_rate"]
    rate = numerator / denominator
    if annotation["scope"] == "geometry":
        if build_report is None:
            raise RunError(f"{annotation['clip']}: a geometry clip needs its build report")
        return rate / build_report["parameters"]["sample_step"]
    return rate


def command(corpus: Path, annotation: dict, out: Path, build_report: dict | None) -> list[str]:
    name = annotation["clip"]
    reference = corpus / "references" / f"{name}.geometry-reference.json"
    argv = [
        str(Path(sys.executable).with_name("skeleton-maker")),
        "environment",
        str(corpus / "clips" / f"{name}.mp4"),
        "--out",
        str(out / f"{name}.json"),
        "--sample-fps",
        f"{sample_fps(annotation, build_report):g}",
        "--device",
        DEVICE,
        "--geometry",
        GEOMETRY,
        "--no-cache",
    ]
    if reference.is_file():
        argv += ["--calibration", str(reference)]
    return argv


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_one(corpus: Path, annotation: dict, out: Path) -> dict:
    name = annotation["clip"]
    report_path = corpus / "build-reports" / f"{name}.build-report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else None
    argv = command(corpus, annotation, out, report)
    log = out / f"{name}.log"
    started = time.time()
    with log.open("w") as handle:
        completed = subprocess.run(  # noqa: S603 - fixed argument list, no shell
            ["/usr/bin/time", "-l", *argv], stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    wall = time.time() - started
    peak = _PEAK_RSS.search(log.read_text(errors="replace"))
    record = {
        "clip": name,
        "split": annotation["split"],
        "command": argv,
        "exit_code": completed.returncode,
        "wall_seconds": round(wall, 1),
        "peak_rss_bytes": int(peak.group(1)) if peak else None,
        "manifest_sha256": _sha256(out / f"{name}.json"),
        "log": str(log),
    }
    (out / f"{name}.run-record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("corpus", type=Path, help="a directory with clips/, annotations/ ...")
    parser.add_argument("out", type=Path)
    parser.add_argument("--split", choices=(*SPLITS, "all"), required=True)
    parser.add_argument("--only", action="append", default=[], help="a clip name; repeatable")
    args = parser.parse_args(argv)
    annotations = sorted((args.corpus / "annotations").glob("*.json"))
    if not annotations:
        print(f"error: no annotations under {args.corpus / 'annotations'}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    failed = 0
    for path in annotations:
        annotation = json.loads(path.read_text())
        if args.split != "all" and annotation["split"] != args.split:
            continue
        if args.only and annotation["clip"] not in args.only:
            continue
        if (args.out / f"{annotation['clip']}.run-record.json").exists():
            print(f"skip {annotation['clip']}: already run; a result is never overwritten")
            continue
        record = run_one(args.corpus, annotation, args.out)
        print(f"{record['clip']}: exit {record['exit_code']} in {record['wall_seconds']} s")
        failed += record["exit_code"] != 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

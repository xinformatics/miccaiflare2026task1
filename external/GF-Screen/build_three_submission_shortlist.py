#!/usr/bin/env python3
"""Build the auditable three-slot manifest from the completed priority stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    metrics: Path
    archive: Path
    gate: Path


def partition_available(
    specs: list[CandidateSpec],
) -> tuple[list[CandidateSpec], list[dict[str, str]]]:
    """Separate complete candidates from auditable, unavailable entries.

    A historical selection may legitimately have no hidden archive because its
    save gate kept the anchor.  Such an entry must not prevent newly qualified
    candidates from being ranked, but it also must not disappear silently.
    """

    available: list[CandidateSpec] = []
    unavailable: list[dict[str, str]] = []
    for spec in specs:
        assets = {
            "metrics": spec.metrics,
            "submission": spec.archive,
            "gate": spec.gate,
        }
        missing = [name for name, path in assets.items() if not path.is_file()]
        if missing:
            unavailable.append(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "missing": ",".join(missing),
                    "metrics": str(spec.metrics.resolve()),
                    "submission": str(spec.archive.resolve()),
                    "gate": str(spec.gate.resolve()),
                    "status": "EXCLUDE_UNAVAILABLE_ASSET",
                }
            )
        else:
            available.append(spec)
    return available, unavailable


def selected_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one selected row, found {len(rows)}")
    return rows[0]


def threshold_tag(value: str) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def resolve_generated(value: str, stage_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else stage_root / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(project: Path) -> list[CandidateSpec]:
    gf = project / "external" / "GF-Screen"
    submissions = project / "submissions"
    long_root = gf / "public_sweeps" / "long_focus_dual_20260820" / "prompted_ranking"
    wise_root = gf / "public_sweeps" / "long_focus_wise_dual_20260820" / "prompted_ranking"
    guided_root = (
        project
        / "external"
        / "BiomedParse"
        / "public_sweeps"
        / "automatic_text_proposals_20260820"
        / "prompted_ranking"
    )
    biomedparse = project / "external" / "BiomedParse"

    long = selected_row(long_root / "selected.tsv")
    long_tag = (
        f"{long['variant']}_{long['checkpoint_tag']}_l{threshold_tag(long['lesion_threshold'])}"
        f"_mm{int(float(long['min_mm3']))}"
    )
    wise = selected_row(wise_root / "selected.tsv")
    wise_tag = (
        f"longwise_{wise['variant']}_{wise['checkpoint_tag']}"
        f"_l{threshold_tag(wise['lesion_threshold'])}_mm{int(float(wise['min_mm3']))}"
    )
    guided = selected_row(guided_root / "selected.tsv")
    specs: list[CandidateSpec] = []
    # Late-run model descendants are eligible only if their deployed,
    # prompt-confirmed masks beat the immutable champion.  KEEP_ANCHOR and
    # FALLBACK rows are retained as evidence but must not consume a scarce
    # hidden-validation slot.
    if long.get("prompt_decision") == "PROMOTE":
        specs.append(CandidateSpec(
            name=f"long20_{long_tag}",
            family=f"long_{long['variant']}",
            metrics=resolve_generated(long["prompt_metrics"], gf),
            archive=submissions / f"gf_long20_{long_tag}_prompt_or_e050.zip",
            gate=resolve_generated(long["prompt_robust"], gf),
        ))
    if wise.get("prompt_decision") == "PROMOTE":
        specs.append(CandidateSpec(
            name=wise_tag,
            family=f"wise_{wise['variant']}",
            metrics=resolve_generated(wise["prompt_metrics"], gf),
            archive=submissions / f"gf_{wise_tag}_prompt_or_e050.zip",
            gate=resolve_generated(wise["prompt_robust"], gf),
        ))
    specs.append(CandidateSpec(
            name=f"biomedparse_{guided['score_tag']}",
            family=f"guided_{guided.get('selected_mode', guided['variant'])}",
            metrics=resolve_generated(guided["prompt_metrics"], biomedparse),
            archive=submissions / f"gf_biomedparse_{guided['score_tag']}_prompt_or_e050.zip",
            gate=resolve_generated(guided["prompt_robust"], biomedparse),
        ))
    nnunet_selected = (
        gf
        / "public_sweeps"
        / "nnunet_guided_focus_20260820"
        / "prompted_ranking"
        / "selected.tsv"
    )
    if nnunet_selected.is_file():
        nnunet = selected_row(nnunet_selected)
        proposal_min = int(float(nnunet["proposal_min_voxels"]))
        minimum = int(float(nnunet["min_mm3"]))
        focus = f"{float(nnunet['lesion_threshold']):.2f}"
        support_tag = nnunet["checkpoint_tag"]
        tag = f"nnunet_guided_{support_tag}_v{proposal_min}_l{focus}_mm{minimum}"
        specs.append(
            CandidateSpec(
                name=tag,
                family="nnunet_guided_focus",
                metrics=resolve_generated(nnunet["prompt_metrics"], gf),
                archive=submissions / f"gf_{tag}_prompt_or_e050.zip",
                gate=resolve_generated(nnunet["prompt_robust"], gf),
            )
        )
    for sidecar in sorted((project / "external").glob("**/submission_candidate.tsv")):
        with sidecar.open(newline="") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        required = {"name", "family", "metrics", "submission", "gate"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(f"invalid submission sidecar: {sidecar}")
        for row in rows:
            specs.append(
                CandidateSpec(
                    name=row["name"],
                    family=row["family"],
                    metrics=resolve_generated(row["metrics"], sidecar.parent),
                    archive=resolve_generated(row["submission"], sidecar.parent),
                    gate=resolve_generated(row["gate"], sidecar.parent),
                )
            )
    unique: list[CandidateSpec] = []
    seen_names: set[str] = set()
    seen_archives: set[Path] = set()
    for spec in specs:
        archive = spec.archive.resolve()
        if spec.name in seen_names or archive in seen_archives:
            continue
        seen_names.add(spec.name)
        seen_archives.add(archive)
        unique.append(spec)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument(
        "--ranker",
        type=Path,
        default=Path(__file__).with_name("rank_submission_shortlist.py"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    specs, unavailable = partition_available(discover(project))
    if not specs:
        raise ValueError("no complete submission candidates are available")
    candidate_paths = {spec.archive.resolve() for spec in specs}
    prior_hashes = {
        sha256(path)
        for path in (project / "submissions").glob("*.zip")
        if path.resolve() not in candidate_paths
    }
    submitted_registry = project / "submissions" / "codabench_submitted_sha256.tsv"
    if submitted_registry.is_file():
        with submitted_registry.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                digest = row.get("sha256", "").strip().lower()
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError(f"{submitted_registry}: invalid SHA-256 {digest!r}")
                prior_hashes.add(digest)

    args.output.mkdir(parents=True, exist_ok=True)
    unavailable_manifest = args.output / "unavailable.tsv"
    with unavailable_manifest.open("w", newline="", encoding="utf-8") as stream:
        fields = [
            "name", "family", "missing", "metrics", "submission", "gate", "status"
        ]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(unavailable)
    manifest = args.output / "candidates.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["name", "family", "metrics", "submission", "gate", "submitted"],
            delimiter="\t",
        )
        writer.writeheader()
        for spec in specs:
            writer.writerow(
                {
                    "name": spec.name,
                    "family": spec.family,
                    "metrics": str(spec.metrics.resolve()),
                    "submission": str(spec.archive.resolve()),
                    "gate": str(spec.gate.resolve()),
                    "submitted": str(sha256(spec.archive) in prior_hashes).lower(),
                }
            )

    subprocess.run(
        [
            sys.executable,
            str(args.ranker),
            "--manifest",
            str(manifest),
            "--anchor",
            str(args.anchor),
            "--output-dir",
            str(args.output),
            "--count",
            "3",
            "--expected-cases",
            "50",
            "--expected-masks",
            "100",
            "--hidden-anchor",
            str(project / "submissions" / "gf_prompt_or_e050_confirm_d0_norescue.zip"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()

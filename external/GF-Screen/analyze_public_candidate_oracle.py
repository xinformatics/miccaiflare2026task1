#!/usr/bin/env python3
"""Measure case-level complementarity across public prediction metrics.

This is a diagnostic upper bound, not a deployable selector: it uses public
labels to choose the best candidate independently for each case.  It answers
whether the existing prediction pool contains enough complementary signal to
justify learning a label-free router, or whether new proposal coverage is
required.  Only complete 50-case metric tables are admitted and exact metric
duplicates are collapsed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean


EXCLUDED_PATH_TOKENS = (
    "incomplete",
    "invalid",
    "gt_proposal",
    "focus_oracle",
)


@dataclass(frozen=True)
class Candidate:
    path: Path
    dsc: tuple[float, ...]
    nsd: tuple[float, ...]

    @property
    def mean_dsc(self) -> float:
        return fmean(self.dsc)

    @property
    def mean_nsd(self) -> float:
        return fmean(self.nsd)


def read_table(path: Path, expected_names: tuple[str, ...]) -> Candidate | None:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        return None
    if len(rows) != len(expected_names):
        return None
    by_name = {row.get("Name", ""): row for row in rows}
    if set(by_name) != set(expected_names):
        return None
    try:
        dsc = tuple(float(by_name[name]["Lesion_DSC"]) for name in expected_names)
        nsd = tuple(float(by_name[name]["Lesion_NSD"]) for name in expected_names)
    except (KeyError, TypeError, ValueError):
        return None
    if any(not 0.0 <= value <= 1.0 for value in dsc + nsd):
        return None
    return Candidate(path=path.resolve(), dsc=dsc, nsd=nsd)


def digest(candidate: Candidate) -> str:
    payload = json.dumps([candidate.dsc, candidate.nsd], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_per_case(candidates: list[Candidate]) -> tuple[list[int], float, float]:
    selected: list[int] = []
    dsc_values: list[float] = []
    nsd_values: list[float] = []
    for case_index in range(len(candidates[0].dsc)):
        winner = max(
            range(len(candidates)),
            key=lambda index: (
                candidates[index].dsc[case_index],
                candidates[index].nsd[case_index],
            ),
        )
        selected.append(winner)
        dsc_values.append(candidates[winner].dsc[case_index])
        nsd_values.append(candidates[winner].nsd[case_index])
    return selected, fmean(dsc_values), fmean(nsd_values)


def greedy_portfolio(anchor: Candidate, candidates: list[Candidate], limit: int) -> list[dict]:
    portfolio = [anchor]
    remaining = [candidate for candidate in candidates if candidate.path != anchor.path]
    result: list[dict] = []
    for step in range(1, limit + 1):
        if step > 1:
            winner = max(
                remaining,
                key=lambda candidate: select_per_case(portfolio + [candidate])[1:],
                default=None,
            )
            if winner is None:
                break
            portfolio.append(winner)
            remaining.remove(winner)
        selected, dsc, nsd = select_per_case(portfolio)
        result.append(
            {
                "size": len(portfolio),
                "added": str(portfolio[-1].path),
                "oracle_dsc": dsc,
                "corresponding_nsd": nsd,
                "cases_not_anchor": sum(index != 0 for index in selected),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--portfolio-size", type=int, default=10)
    args = parser.parse_args()

    with args.anchor.open(newline="", encoding="utf-8") as stream:
        anchor_rows = list(csv.DictReader(stream))
    expected_names = tuple(row["Name"] for row in anchor_rows)
    if len(expected_names) != 50 or len(set(expected_names)) != 50:
        raise ValueError("anchor must contain exactly 50 unique public cases")
    anchor = read_table(args.anchor, expected_names)
    if anchor is None:
        raise ValueError("anchor metrics are invalid")

    paths: set[Path] = set()
    for root in args.roots:
        paths.update(path for path in root.rglob("*.csv") if "metrics" in path.name)
    excluded = [
        path for path in paths
        if any(token in str(path).lower() for token in EXCLUDED_PATH_TOKENS)
    ]
    admitted: list[Candidate] = []
    invalid = 0
    for path in sorted(paths - set(excluded)):
        candidate = read_table(path, expected_names)
        if candidate is None:
            invalid += 1
        else:
            admitted.append(candidate)
    admitted.append(anchor)

    # Preserve the strongest representative when multiple files contain the
    # same rounded organizer metrics. Path order provides deterministic ties.
    # Keep the explicitly supplied anchor as the representative of its digest;
    # otherwise lexical path ordering could make the report point at an
    # equivalent sweep copy rather than the authoritative deployed anchor.
    anchor_digest = digest(anchor)
    unique: dict[str, Candidate] = {anchor_digest: anchor}
    for candidate in sorted(admitted, key=lambda item: str(item.path)):
        unique.setdefault(digest(candidate), candidate)
    candidates = sorted(
        unique.values(), key=lambda item: (item.mean_dsc, item.mean_nsd), reverse=True
    )
    anchor = unique[anchor_digest]

    selected, oracle_dsc, oracle_nsd = select_per_case(candidates)
    best_single = candidates[0]
    pair_rows = []
    for candidate in candidates:
        pair_selected, pair_dsc, pair_nsd = select_per_case([anchor, candidate])
        pair_rows.append(
            {
                "path": str(candidate.path),
                "candidate_dsc": candidate.mean_dsc,
                "candidate_nsd": candidate.mean_nsd,
                "oracle_dsc": pair_dsc,
                "corresponding_nsd": pair_nsd,
                "switched_cases": sum(index == 1 for index in pair_selected),
            }
        )
    pair_rows.sort(
        key=lambda row: (row["oracle_dsc"], row["corresponding_nsd"]), reverse=True
    )

    winner_counts: dict[str, int] = {}
    for index in selected:
        key = str(candidates[index].path)
        winner_counts[key] = winner_counts.get(key, 0) + 1
    report = {
        "warning": (
            "Label-informed diagnostic upper bound only; per-case choices must "
            "never be used for hidden inference or submission."
        ),
        "anchor": {
            "path": str(anchor.path),
            "DSC": anchor.mean_dsc,
            "NSD": anchor.mean_nsd,
        },
        "scan": {
            "metric_files_found": len(paths),
            "excluded_by_path": len(excluded),
            "invalid_or_incomplete": invalid,
            "unique_complete_candidates": len(candidates),
        },
        "best_single_public": {
            "path": str(best_single.path),
            "DSC": best_single.mean_dsc,
            "NSD": best_single.mean_nsd,
        },
        "all_candidate_case_oracle": {
            "DSC": oracle_dsc,
            "corresponding_NSD": oracle_nsd,
            "winning_candidate_counts": sorted(
                winner_counts.items(), key=lambda item: item[1], reverse=True
            ),
        },
        "best_anchor_pair_oracle": pair_rows[0],
        "greedy_case_oracle_portfolio": greedy_portfolio(
            anchor, candidates, args.portfolio_size
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "top_candidates.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(("rank", "DSC", "NSD", "metrics"))
        for rank, candidate in enumerate(candidates[:100], 1):
            writer.writerow((rank, candidate.mean_dsc, candidate.mean_nsd, candidate.path))
    with (args.output_dir / "anchor_pair_oracles.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(pair_rows[:100])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

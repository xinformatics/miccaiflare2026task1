"""Auditable sampling weights for macro case/cohort candidate-refiner training."""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def semantic_source_group(source_name: str) -> str:
    """Collapse parity-partitioned lung cells into one proposal family."""

    source_name = str(source_name).strip()
    if not source_name:
        raise ValueError("source_name is required for source-group balancing")
    return "lung_cells" if source_name.startswith("lung_cell") else source_name


def hierarchical_sample_weights(
    rows: list[dict[str, str]],
    temperature: float,
    balance_source_groups: bool = False,
) -> np.ndarray:
    """Interpolate label-balanced and label/cohort/case-balanced sampling.

    At temperature zero this exactly reproduces the v2 label-balanced sampler.
    At one, probability mass is balanced by label, then by observed cohort,
    then by case, and finally uniformly across that case's patches. Intermediate
    values geometrically interpolate the two distributions.
    """

    if not rows or not np.isfinite(temperature) or not 0 <= temperature <= 1:
        raise ValueError("rows must be nonempty and temperature must be in [0, 1]")
    parsed: list[tuple[int, str, str, str]] = []
    for row in rows:
        label = int(row["label"])
        cohort, case = str(row["cohort"]), str(row["case"])
        if label not in (0, 1) or not cohort or not case:
            raise ValueError("each row requires binary label, cohort, and case")
        source_group = (
            semantic_source_group(row.get("source_name", ""))
            if balance_source_groups
            else ""
        )
        parsed.append((label, cohort, case, source_group))
    labels = sorted({label for label, _, _, _ in parsed})
    if labels != [0, 1]:
        raise ValueError("both labels must be represented")
    label_counts = Counter(label for label, _, _, _ in parsed)
    # Preserve the original v2 vector bit-for-bit at the default operating
    # point. Avoiding an unnecessary renormalization also preserves the seeded
    # WeightedRandomSampler stream for already-running controlled folds.
    if temperature == 0:
        return np.asarray(
            [0.5 / label_counts[label] for label, _, _, _ in parsed],
            dtype=np.float64,
        )
    cohorts_by_label: dict[int, set[str]] = defaultdict(set)
    cases_by_cell: dict[tuple[int, str], set[str]] = defaultdict(set)
    source_groups_by_case: dict[tuple[int, str, str], set[str]] = defaultdict(set)
    patches_by_source_case = Counter(parsed)
    for label, cohort, case, source_group in parsed:
        cohorts_by_label[label].add(cohort)
        cases_by_cell[(label, cohort)].add(case)
        source_groups_by_case[(label, cohort, case)].add(source_group)

    weights = []
    for label, cohort, case, source_group in parsed:
        label_balanced = 0.5 / label_counts[label]
        hierarchical = 0.5 / (
            len(cohorts_by_label[label])
            * len(cases_by_cell[(label, cohort)])
            * len(source_groups_by_case[(label, cohort, case)])
            * patches_by_source_case[(label, cohort, case, source_group)]
        )
        weight = label_balanced ** (1.0 - temperature) * hierarchical**temperature
        weights.append(weight)
    result = np.asarray(weights, dtype=np.float64)
    result /= result.sum()
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("invalid sampling weights")
    return result


def sampling_audit(
    rows: list[dict[str, str]],
    temperature: float,
    balance_source_groups: bool = False,
) -> dict[str, object]:
    weights = hierarchical_sample_weights(rows, temperature, balance_source_groups)
    cell_mass: dict[tuple[int, str], float] = defaultdict(float)
    case_mass: dict[tuple[int, str, str], float] = defaultdict(float)
    source_cell_mass: dict[tuple[int, str, str], float] = defaultdict(float)
    source_group_cell_mass: dict[tuple[int, str, str], float] = defaultdict(float)
    source_names = [str(row.get("source_name", "")).strip() for row in rows]
    if any(source_names) and not all(source_names):
        raise ValueError("source_name must cover every row or no rows in a sampling audit")
    for row, weight in zip(rows, weights):
        label, cohort, case = int(row["label"]), str(row["cohort"]), str(row["case"])
        cell_mass[(label, cohort)] += float(weight)
        case_mass[(label, cohort, case)] += float(weight)
        source = str(row.get("source_name", "")).strip()
        if source:
            source_cell_mass[(label, cohort, source)] += float(weight)
            source_group_cell_mass[
                (label, cohort, semantic_source_group(source))
            ] += float(weight)
    return {
        "temperature": float(temperature),
        "balance_source_groups": bool(balance_source_groups),
        "patches": len(rows),
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
        "max_to_min_patch_weight": float(weights.max() / weights.min()),
        "cell_mass": [
            {"label": label, "cohort": cohort, "probability": probability}
            for (label, cohort), probability in sorted(cell_mass.items())
        ],
        "source_cell_mass": [
            {
                "label": label,
                "cohort": cohort,
                "source": source,
                "probability": probability,
            }
            for (label, cohort, source), probability in sorted(
                source_cell_mass.items()
            )
        ],
        "source_group_cell_mass": [
            {
                "label": label,
                "cohort": cohort,
                "source_group": source_group,
                "probability": probability,
            }
            for (label, cohort, source_group), probability in sorted(
                source_group_cell_mass.items()
            )
        ],
        "case_probability_range": {
            "minimum": float(min(case_mass.values())),
            "maximum": float(max(case_mass.values())),
        },
    }

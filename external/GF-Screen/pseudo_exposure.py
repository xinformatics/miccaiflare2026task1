#!/usr/bin/env python3
"""Choose an integer pseudo-dataset repeat nearest a target case exposure."""

from __future__ import annotations


def exposure_fraction(
    positive_cases: int,
    labeled_cases: int,
    repeat: int,
) -> float:
    """Return the pseudo-case fraction after repeating the positive cohort."""
    if positive_cases < 1:
        raise ValueError("positive_cases must be positive")
    if labeled_cases < 1:
        raise ValueError("labeled_cases must be positive")
    if repeat < 1:
        raise ValueError("repeat must be positive")
    repeated = positive_cases * repeat
    return repeated / (labeled_cases + repeated)


def select_integer_repeat(
    positive_cases: int,
    labeled_cases: int,
    target_fraction: float,
    minimum: int = 1,
    maximum: int = 8,
) -> int:
    """Select the bounded integer repeat with minimum absolute target error.

    Ties prefer the smaller repeat. This avoids the large discontinuity from
    always rounding upward near an integer boundary.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must lie strictly between zero and one")
    if minimum < 1 or maximum < minimum:
        raise ValueError("invalid repeat bounds")
    candidates = range(minimum, maximum + 1)
    return min(
        candidates,
        key=lambda repeat: (
            abs(
                exposure_fraction(positive_cases, labeled_cases, repeat)
                - target_fraction
            ),
            repeat,
        ),
    )

#!/usr/bin/env python3
"""Predeclared large-lesion extension of disconnected consensus screening.

This reuses the geometry validation, exact organizer metric, and immutable-base
logic from ``screen_disconnected_guided_additions``.  Its finite grid tests
whether two independent models can safely recover medium/large lesions that
the established <=2 cm3 rescue cannot represent.
"""

from __future__ import annotations

import itertools
import math

import screen_disconnected_guided_additions as implementation


def main() -> None:
    # Keep the search compact and clinically interpretable.  The 2 cm3 entry
    # is the deployed reference; larger caps test missed dominant lesions.
    max_mm3 = (2_000.0, 10_000.0, 30_000.0, 100_000.0)
    support_fraction = (0.25, 0.50)
    support_mm3 = (100.0, 500.0)
    implementation.CONFIGS = tuple(
        implementation.Config(
            min_mm3=0.0,
            max_mm3=maximum,
            min_distance_mm=0.0,
            min_support_fraction=fraction,
            min_support_mm3=minimum_support,
            support_dilation_mm=math.inf,
        )
        for maximum, fraction, minimum_support in itertools.product(
            max_mm3, support_fraction, support_mm3
        )
    )
    implementation.main()


if __name__ == "__main__":
    main()

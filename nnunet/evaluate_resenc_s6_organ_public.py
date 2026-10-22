#!/usr/bin/env python3
"""Score frozen organ-target ResEnc deployments on fixed public data."""

from __future__ import annotations

import nnunet.select_resenc_s6_public as base_selector
from nnunet.select_resenc_s6_organ_public import load_policy


base_selector.load_policy = load_policy

import nnunet.evaluate_resenc_s6_public as base_evaluator  # noqa: E402


base_evaluator.load_policy = load_policy


if __name__ == "__main__":
    base_evaluator.main()

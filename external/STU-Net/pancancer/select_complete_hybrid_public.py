"""Select a complete-hybrid endpoint under immutable DSC and NSD gates."""

from __future__ import annotations

from collections.abc import Sequence


DSC_GATE = 0.6366639257651584
NSD_FLOOR = 0.5347520263176905
ALLOWED_CONFIGURATIONS = frozenset(
    {"stunet_standalone", "gf_anchor_confirm_either_p70"}
)


def select_candidate(rows: Sequence[dict]) -> tuple[dict, bool]:
    normalized = []
    seen = set()
    for source in rows:
        row = dict(source)
        epoch = int(row["epoch"])
        configuration = str(row["configuration"])
        dsc = float(row["Lesion_DSC"])
        nsd = float(row["Lesion_NSD"])
        identity = (epoch, configuration)
        if epoch not in (40, 80) or configuration not in ALLOWED_CONFIGURATIONS:
            raise ValueError(f"invalid complete-hybrid candidate: {identity}")
        if identity in seen:
            raise ValueError(f"duplicate complete-hybrid candidate: {identity}")
        seen.add(identity)
        if not (0.0 <= dsc <= 1.0 and 0.0 <= nsd <= 1.0):
            raise ValueError(f"non-finite or invalid metrics: {identity}")
        row.update(epoch=epoch, configuration=configuration, Lesion_DSC=dsc, Lesion_NSD=nsd)
        row["public_slot_gate"] = dsc >= DSC_GATE and nsd >= NSD_FLOOR
        normalized.append(row)
    expected = {(epoch, configuration) for epoch in (40, 80) for configuration in ALLOWED_CONFIGURATIONS}
    if seen != expected:
        raise ValueError(f"candidate inventory mismatch: missing={sorted(expected-seen)}")
    qualified = [row for row in normalized if row["public_slot_gate"]]
    selected = max(
        qualified or normalized,
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
    )
    return selected, bool(qualified)

"""Shared helpers. Defined once so the three call sites cannot drift apart."""


def source_of(case_id: str) -> str:
    """Map a case id to its source cohort.

    Case ids end in a per-case token that starts with a digit, but how many
    underscore-separated parts precede it varies by cohort:

        Adrenal-ACC-Ki67-Seg_001              -> Adrenal-ACC-Ki67-Seg
        Chest_LUNA25_0171                     -> Chest_LUNA25
        Pancreas_PanTrack_045-20241109        -> Pancreas_PanTrack
        WholeBody_DeepLesion_000908-02-01-016 -> WholeBody_DeepLesion
        FLARE23Ts_0001                        -> FLARE23Ts

    So strip trailing digit-leading parts rather than a fixed number of them.
    """
    parts = case_id.split("_")
    while len(parts) > 1 and parts[-1][:1].isdigit():
        parts.pop()
    return "_".join(parts)

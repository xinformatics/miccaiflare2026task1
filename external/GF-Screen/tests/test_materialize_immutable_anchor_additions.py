import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from materialize_immutable_anchor_additions import materialize_case


def save(path: Path, value: np.ndarray, affine=None):
    nib.save(
        nib.Nifti1Image(
            value.astype(np.uint8),
            np.eye(4) if affine is None else affine,
        ),
        str(path),
    )


def run(base, candidate, supports, **overrides):
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        base_path, candidate_path = root / "base.nii.gz", root / "candidate.nii.gz"
        output_path = root / "output.nii.gz"
        save(base_path, base)
        save(candidate_path, candidate)
        support_paths = []
        for index, support in enumerate(supports):
            path = root / f"support{index}.nii.gz"
            save(path, support)
            support_paths.append(path)
        config = dict(
            min_mm3=1.0,
            max_mm3=1000.0,
            min_support_fraction=0.25,
            min_support_mm3=1.0,
            min_support_models=1,
        )
        config.update(overrides)
        rows = materialize_case(
            base_path, candidate_path, support_paths, output_path, **config
        )
        output = np.asanyarray(nib.load(str(output_path)).dataobj).astype(bool)
        return output, rows


def test_preserves_anchor_and_adds_only_confirmed_disconnected_component():
    shape = (10, 10, 10)
    base = np.zeros(shape, bool)
    base[1:3, 1:3, 1:3] = True
    candidate = np.zeros(shape, bool)
    candidate[5:7, 5:7, 5:7] = True
    candidate[8:10, 1:3, 1:3] = True
    support = np.zeros(shape, bool)
    support[5:7, 5:7, 5:7] = True
    output, rows = run(base, candidate, [support])
    assert np.all(output[base])
    assert np.all(output[5:7, 5:7, 5:7])
    assert not np.any(output[8:10, 1:3, 1:3])
    assert sum(int(row["accepted"]) for row in rows) == 1


def test_rejects_candidate_component_that_touches_anchor():
    shape = (8, 8, 8)
    base = np.zeros(shape, bool)
    base[1:3, 1:3, 1:3] = True
    candidate = np.zeros(shape, bool)
    candidate[2:5, 2:5, 2:5] = True
    support = candidate.copy()
    output, rows = run(base, candidate, [support])
    assert np.array_equal(output, base)
    assert rows[0]["overlaps_anchor"] == 1
    assert rows[0]["accepted"] == 0


def test_requires_declared_number_of_independent_votes():
    shape = (8, 8, 8)
    base = np.zeros(shape, bool)
    candidate = np.zeros(shape, bool)
    candidate[3:6, 3:6, 3:6] = True
    first = candidate.copy()
    second = np.zeros(shape, bool)
    output, rows = run(
        base, candidate, [first, second], min_support_models=2
    )
    assert not np.any(output)
    assert rows[0]["support_votes"] == 1
    assert rows[0]["accepted"] == 0

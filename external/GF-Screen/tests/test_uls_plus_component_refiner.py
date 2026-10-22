import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from argparse import Namespace

import numpy as np
import SimpleITK as sitk


SCRIPT = Path(__file__).resolve().parents[1] / "run_uls_plus_component_refiner.py"
SPEC = importlib.util.spec_from_file_location("uls_plus_component_refiner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONSENSUS_SCRIPT = Path(__file__).resolve().parents[1] / "materialize_uls_consensus.py"
CONSENSUS_SPEC = importlib.util.spec_from_file_location(
    "materialize_uls_consensus", CONSENSUS_SCRIPT
)
CONSENSUS = importlib.util.module_from_spec(CONSENSUS_SPEC)
assert CONSENSUS_SPEC.loader is not None
CONSENSUS_SPEC.loader.exec_module(CONSENSUS)


def test_paste_crop_clips_lower_and_upper_padding():
    crop = np.ones((4, 5, 6), dtype=bool)
    pasted = MODULE.paste_crop(crop, (5, 6, 7), [-2, 2, -1])
    # start_xyz [-2, 2, -1] becomes zyx [-1, 2, -2]. The valid region is
    # crop[1:4, 0:4, 2:6] pasted at full[0:3, 2:6, 0:4].
    expected = np.zeros((5, 6, 7), dtype=bool)
    expected[0:3, 2:6, 0:4] = True
    assert np.array_equal(pasted, expected)


def test_interior_voxel_is_inside_and_prefers_component_core():
    component = np.zeros((9, 9, 9), dtype=bool)
    component[2:7, 2:7, 2:7] = True
    click = MODULE.interior_voxel(component, (1.0, 1.0, 1.0))
    assert component[click]
    assert click == (4, 4, 4)


def test_sparse_crop_mapping_matches_dense_paste():
    crop = np.zeros((4, 5, 6), dtype=bool)
    crop[0:4, 1:5, 1:6] = True
    shape = (5, 6, 7)
    start_xyz = [-2, 2, -1]
    indices, bbox = CONSENSUS.crop_indices_and_bbox(crop, shape, start_xyz)
    dense = MODULE.paste_crop(crop, shape, start_xyz)
    assert np.array_equal(indices, np.flatnonzero(dense.ravel()))
    coordinates = np.argwhere(dense)
    assert bbox is not None
    assert np.array_equal(bbox[0], coordinates.min(axis=0))
    assert np.array_equal(bbox[1], coordinates.max(axis=0) + 1)


def test_consensus_bbox_overlap_guard():
    first = (np.array([0, 0, 0]), np.array([2, 2, 2]))
    touching = (np.array([2, 0, 0]), np.array([4, 2, 2]))
    overlapping = (np.array([1, 1, 1]), np.array([3, 3, 3]))
    assert not CONSENSUS.boxes_overlap(first, touching)
    assert CONSENSUS.boxes_overlap(first, overlapping)


def test_spawned_python_resolves_workspace_identity_resampler():
    overlay = SCRIPT.parents[1] / "ULS-Plus/python_overlay"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(overlay)
    code = (
        "from nnunetv2.preprocessing.resampling.utils import "
        "recursive_find_resampling_fn_by_name as find; "
        "print(find('no_resampling_data_or_seg_to_shape').__name__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.strip().endswith("no_resampling_data_or_seg_to_shape")


def test_materialize_keeps_zero_proposal_case_in_output(tmp_path):
    proposals = tmp_path / "proposals"
    base = tmp_path / "base"
    output = tmp_path / "output"
    predicted = tmp_path / "predicted"
    for directory in (proposals, base, predicted):
        directory.mkdir()

    case = "FLARE-Val_0001"
    reference = sitk.GetImageFromArray(np.zeros((5, 6, 7), dtype=np.uint8))
    reference.SetSpacing((1.0, 1.5, 2.0))
    sitk.WriteImage(reference, str(proposals / f"{case}.nii.gz"), True)
    base_array = np.zeros((5, 6, 7), dtype=np.uint8)
    base_array[2, 3, 4] = 1
    base_image = sitk.GetImageFromArray(base_array)
    base_image.CopyInformation(reference)
    sitk.WriteImage(base_image, str(base / f"{case}.nii.gz"), True)

    args = Namespace(
        output=output,
        proposals=proposals,
        predicted_crops=predicted,
        base=base,
        case_id=None,
        case_limit=None,
        min_seed_coverage=0.25,
        min_refined_overlap=0.25,
        min_volume_ratio=0.1,
        max_volume_ratio=10.0,
    )
    MODULE.materialize(args, [])

    result = sitk.ReadImage(str(output / f"{case}.nii.gz"))
    assert MODULE.same_geometry(result, reference)
    assert np.array_equal(sitk.GetArrayFromImage(result), base_array)

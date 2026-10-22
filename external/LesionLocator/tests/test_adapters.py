import binascii
import struct
import zlib

import nibabel as nib
import numpy as np

from extract_single_zip_entry import extract
from merge_instance_predictions import case_name, merge_case


def test_extract_split_deflated_entry(tmp_path):
    payload = (b"lesion-locator-checkpoint" * 4096) + b"end"
    compressed = zlib.compress(payload)[2:-4]
    name = b"fold_0/checkpoint_final.pth"
    header = struct.pack(
        "<4s5H3I2H",
        b"PK\x03\x04",
        20,
        8,
        8,
        0,
        0,
        0,
        0,
        0,
        len(name),
        0,
    )
    complete = header + name + compressed
    split = len(complete) // 3
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_bytes(complete[:split])
    second.write_bytes(complete[split:])
    output = tmp_path / "checkpoint.pth"
    extract(
        [first, second],
        output,
        len(compressed),
        len(payload),
        binascii.crc32(payload) & 0xFFFFFFFF,
    )
    assert output.read_bytes() == payload


def test_merge_modes_and_codabench_name(tmp_path):
    references = tmp_path / "references"
    instances = tmp_path / "instances"
    anchors = tmp_path / "anchors"
    references.mkdir()
    instances.mkdir()
    anchors.mkdir()
    affine = np.diag([1.2, 1.3, 2.5, 1.0])
    reference = references / "case_001_0000.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 8)), affine), reference)

    one = np.zeros((8, 8, 8), dtype=np.uint8)
    one[1:3, 1:3, 1:3] = 1
    two = np.zeros_like(one)
    two[5:7, 5:7, 5:7] = 1
    nib.save(nib.Nifti1Image(one, affine), instances / "case_001_0000_lesion_1.nii.gz")
    nib.save(nib.Nifti1Image(two, affine), instances / "case_001_0000_lesion_2.nii.gz")

    anchor_data = np.zeros_like(one)
    anchor_data[2:6, 2:6, 2:6] = 1
    anchor = anchors / "case_001.nii.gz"
    nib.save(nib.Nifti1Image(anchor_data, affine), anchor)

    assert case_name(reference) == "case_001"
    refined = np.asarray(merge_case(reference, instances, None, "refiner").dataobj)
    union = np.asarray(merge_case(reference, instances, anchor, "union").dataobj)
    intersection = np.asarray(merge_case(reference, instances, anchor, "intersection").dataobj)
    assert refined.sum() == 16
    assert union.sum() > refined.sum()
    assert intersection.sum() == 2

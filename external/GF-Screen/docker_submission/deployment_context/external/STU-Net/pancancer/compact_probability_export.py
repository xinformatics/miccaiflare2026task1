"""nnU-Net export callback that writes the deployed probability format once."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)


def compact_lesion_probability(
    probabilities: np.ndarray,
    output_nifti: Path,
    output_truncated: Path,
) -> Path:
    """Write one uint16 lesion channel in the saved NIfTI's axis order."""

    probabilities = np.asarray(probabilities)
    if probabilities.ndim != 4 or probabilities.shape[0] != 2:
        raise ValueError("binary probability export must have shape [2,D,H,W]")
    reference = nib.load(str(output_nifti))
    lesion = np.asarray(probabilities[1], dtype=np.float32)
    if lesion.shape == reference.shape[::-1]:
        lesion = lesion.transpose(2, 1, 0)
    elif lesion.shape != reference.shape:
        raise ValueError(
            f"probability/NIfTI geometry mismatch: {lesion.shape} != {reference.shape}"
        )
    segmentation = np.asanyarray(reference.dataobj).astype(bool, copy=False)
    if not np.array_equal(lesion > 0.5, segmentation):
        raise ValueError("direct compact probability threshold/orientation mismatch")
    quantized = np.rint(np.clip(lesion, 0.0, 1.0) * 65535.0).astype(np.uint16)
    destination = output_truncated.with_suffix(".prob16.npz")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, lesion_probability_uint16=quantized)
        with np.load(temporary) as check:
            restored = check["lesion_probability_uint16"]
            if restored.dtype != np.uint16 or not np.array_equal(restored, quantized):
                raise ValueError("direct compact probability verification failed")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def export_prediction_from_logits_compact(
    predicted_array_or_file: Union[np.ndarray, torch.Tensor],
    properties_dict: dict,
    configuration_manager: ConfigurationManager,
    plans_manager: PlansManager,
    dataset_json_dict_or_file: Union[dict, str],
    output_file_truncated: str,
    save_probabilities: bool = False,
    num_threads_torch: int = 8,
) -> None:
    """Drop-in exporter that avoids the redundant two-channel float archive."""

    if isinstance(dataset_json_dict_or_file, str):
        dataset_json_dict_or_file = load_json(dataset_json_dict_or_file)
    label_manager = plans_manager.get_label_manager(dataset_json_dict_or_file)
    converted = convert_predicted_logits_to_segmentation_with_correct_shape(
        predicted_array_or_file,
        plans_manager,
        configuration_manager,
        label_manager,
        properties_dict,
        return_probabilities=save_probabilities,
        num_threads_torch=num_threads_torch,
    )
    del predicted_array_or_file
    if save_probabilities:
        segmentation_final, probabilities_final = converted
    else:
        segmentation_final = converted
        probabilities_final = None
    output_truncated = Path(output_file_truncated)
    output_nifti = Path(
        output_truncated.as_posix() + dataset_json_dict_or_file["file_ending"]
    )
    reader_writer = plans_manager.image_reader_writer_class()
    reader_writer.write_seg(segmentation_final, str(output_nifti), properties_dict)
    if probabilities_final is not None:
        compact_lesion_probability(
            np.asarray(probabilities_final), output_nifti, output_truncated
        )

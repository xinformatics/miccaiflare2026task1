#!/usr/bin/env python3
"""Resume nnU-Net preprocessing without deleting complete case outputs."""

import argparse
import multiprocessing
from pathlib import Path
from time import sleep

from batchgenerators.utilities.file_and_folder_operations import load_json
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets
from tqdm import tqdm


def case_outputs(output_stem: Path) -> tuple[Path, Path, Path]:
    return (
        Path(f"{output_stem}.b2nd"),
        Path(f"{output_stem}_seg.b2nd"),
        Path(f"{output_stem}.pkl"),
    )


def case_is_complete(output_stem: Path) -> bool:
    expected = case_outputs(output_stem)
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default="501")
    parser.add_argument("-c", "--configuration", default="3d_fullres")
    parser.add_argument("-plans_name", default="nnUNetResEncUNetLPlans")
    parser.add_argument("-np", "--num-processes", type=int, default=8)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="report complete/remaining cases without deleting or preprocessing",
    )
    args = parser.parse_args()

    dataset_name = maybe_convert_to_dataset_name(args.dataset)
    raw_dir = Path(nnUNet_raw) / dataset_name
    preprocessed_dir = Path(nnUNet_preprocessed) / dataset_name
    plans_manager = PlansManager(preprocessed_dir / f"{args.plans_name}.json")
    configuration_manager = plans_manager.get_configuration(args.configuration)
    dataset_json = load_json(preprocessed_dir / "dataset.json")
    output_dir = preprocessed_dir / configuration_manager.data_identifier
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = get_filenames_of_train_images_and_targets(str(raw_dir), dataset_json)
    remaining_cases = []
    partial_files_removed = 0
    for case_id in dataset:
        output_stem = output_dir / case_id
        if case_is_complete(output_stem):
            continue
        remaining_cases.append(case_id)
        # Blosc2 refuses to overwrite an existing array. Remove only artifacts
        # belonging to incomplete cases; complete triplets remain untouched.
        for path in case_outputs(output_stem):
            if path.exists():
                if args.verify_only:
                    continue
                path.unlink()
                partial_files_removed += 1
    complete_count = len(dataset) - len(remaining_cases)
    print(
        f"Found {complete_count}/{len(dataset)} complete cases; "
        f"preprocessing {len(remaining_cases)} remaining cases with "
        f"{args.num_processes} workers. Removed {partial_files_removed} "
        "partial artifacts.",
        flush=True,
    )
    if args.verify_only:
        if partial_files_removed:
            raise AssertionError("verify-only mode must not remove partial files")
        return
    if not remaining_cases:
        return

    preprocessor = DefaultPreprocessor(verbose=False)
    jobs = []
    with multiprocessing.get_context("spawn").Pool(args.num_processes) as pool:
        original_workers = list(pool._pool)
        for case_id in remaining_cases:
            case = dataset[case_id]
            jobs.append(
                pool.starmap_async(
                    preprocessor.run_case_save,
                    ((
                        str(output_dir / case_id),
                        case["images"],
                        case["label"],
                        plans_manager,
                        configuration_manager,
                        dataset_json,
                    ),),
                )
            )

        pending = list(range(len(jobs)))
        with tqdm(total=len(jobs)) as progress:
            while pending:
                if not all(worker.is_alive() for worker in original_workers):
                    raise RuntimeError(
                        "A preprocessing worker died, most likely from insufficient RAM."
                    )
                finished = [index for index in pending if jobs[index].ready()]
                for index in finished:
                    jobs[index].get()
                    progress.update()
                pending = [index for index in pending if index not in finished]
                sleep(0.1)


if __name__ == "__main__":
    main()

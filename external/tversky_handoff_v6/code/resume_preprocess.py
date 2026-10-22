"""Resume preprocessing without destroying completed work.

nnUNetv2_preprocess opens with

    if isdir(output_directory): shutil.rmtree(output_directory)

so re-running it after an interruption throws away everything already done. At
~12.6 MB and ~0.9 s per case over 17,575 cases that is hours of work and >200GB.

This reproduces DefaultPreprocessor.run() minus the rmtree, skipping cases that are
already complete. nnUNetDatasetBlosc2.save_case writes data.b2nd -> _seg.b2nd ->
.pkl in that order, so the .pkl is the completion marker: if it exists, all three
landed. Cases with a partial triple are cleaned up and redone, because
blosc2.asarray refuses to write over an existing urlpath.
"""
import argparse
import multiprocessing
import os
import pickle
from pathlib import Path

from tqdm import tqdm


def is_complete(out_base: Path) -> bool:
    pkl = out_base.with_suffix(".pkl")
    if not pkl.is_file():
        return False
    if not out_base.with_suffix(".b2nd").is_file():
        return False
    if not Path(str(out_base) + "_seg.b2nd").is_file():
        return False
    try:                                  # a truncated pkl means it died mid-write
        with open(pkl, "rb") as f:
            pickle.load(f)
    except Exception:
        return False
    return True


def clean_partial(out_base: Path):
    for p in (out_base.with_suffix(".b2nd"), Path(str(out_base) + "_seg.b2nd"),
              out_base.with_suffix(".pkl")):
        if p.exists():
            p.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="501")
    ap.add_argument("-c", "--config", default="3d_fullres_S4D2W32")
    ap.add_argument("-p", "--plans", default="nnUNetPlans")
    ap.add_argument("-np", "--num-processes", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets

    ds_name = maybe_convert_to_dataset_name(a.dataset)
    plans_file = join(nnUNet_preprocessed, ds_name, a.plans + ".json")
    plans_manager = PlansManager(plans_file)
    cfg = plans_manager.get_configuration(a.config)
    dataset_json = load_json(join(nnUNet_preprocessed, ds_name, "dataset.json"))

    out_dir = Path(nnUNet_preprocessed) / ds_name / cfg.data_identifier
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = get_filenames_of_train_images_and_targets(join(nnUNet_raw, ds_name),
                                                        dataset_json)

    todo, done, partial = [], 0, 0
    for k in dataset.keys():
        base = out_dir / k
        if is_complete(base):
            done += 1
            continue
        if base.with_suffix(".b2nd").exists() or base.with_suffix(".pkl").exists():
            partial += 1
            if not a.dry_run:
                clean_partial(base)
        todo.append(k)

    print(f"dataset       : {ds_name}  config {a.config}")
    print(f"output        : {out_dir}")
    print(f"total cases   : {len(dataset)}")
    print(f"already done  : {done}")
    print(f"partial (redo): {partial}")
    print(f"TO PROCESS    : {len(todo)}", flush=True)
    if a.dry_run or not todo:
        return

    pp = DefaultPreprocessor(verbose=False)
    with multiprocessing.get_context("spawn").Pool(a.num_processes) as p:
        results = [
            p.apply_async(pp.run_case_save,
                          (str(out_dir / k), dataset[k]["images"], dataset[k]["label"],
                           plans_manager, cfg, dataset_json))
            for k in todo
        ]
        failed = []
        for k, r in tqdm(list(zip(todo, results)), total=len(todo)):
            try:
                r.get()
            except Exception as e:
                failed.append((k, str(e)[:160]))

    print(f"\ncompleted. failures: {len(failed)}")
    for k, e in failed[:20]:
        print(f"  {k}: {e}")
    final = sum(1 for k in dataset.keys() if is_complete(out_dir / k))
    print(f"cases complete on disk: {final}/{len(dataset)}")


if __name__ == "__main__":
    main()

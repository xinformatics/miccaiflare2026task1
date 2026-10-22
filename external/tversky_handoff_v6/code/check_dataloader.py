"""Prove the training dataloader actually delivers lesion voxels.

Foreground is ~0.07% of voxels. If nnU-Net's foreground oversampling were broken --
bad class_locations, mis-parsed labels, wrong label value -- the model would see
almost no positive examples and could never learn, and a 3.5-day run would be
wasted producing exactly the all-background output a 10-epoch model produces
legitimately. The two failure modes look identical early on, so measure instead of
assuming.

nnU-Net's default oversample_foreground_percent is 0.33, applied per batch: the last
ceil(0.33 * batch_size) samples of each batch are forced to be centred on a
foreground voxel. With batch_size 2 that means >=1 of every 2 patches should contain
lesion, so expect roughly 50%+ of sampled patches to be positive -- far above the
~0.07% that random cropping would give.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 40

    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

    ds = maybe_convert_to_dataset_name(501)
    folder = join(nnUNet_preprocessed, ds)
    plans = PlansManager(join(folder, "nnUNetPlans.json"))
    cfg = plans.get_configuration("3d_fullres_S4D2W32")
    dataset_json = load_json(join(folder, "dataset.json"))
    lm = plans.get_label_manager(dataset_json)

    preproc = join(folder, cfg.data_identifier)
    ds_class = infer_dataset_class(preproc)          # nnUNetDatasetBlosc2 here
    ids = ds_class.get_identifiers(preproc)          # moved onto the class in 2.6
    print(f"cases in preprocessed folder : {len(ids)}")
    print(f"patch size                   : {cfg.patch_size}")
    print(f"batch size                   : {cfg.batch_size}")
    print(f"labels                       : {dataset_json['labels']}")
    print(f"foreground labels            : {lm.foreground_labels}")


    dataset = ds_class(preproc, ids)

    dl = nnUNetDataLoader(
        dataset, cfg.batch_size, cfg.patch_size, cfg.patch_size,
        label_manager=lm, oversample_foreground_percent=0.33,
        sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=False,
    )
    print(f"oversample_foreground_percent: 0.33 "
          f"-> expect >=1 of every {cfg.batch_size} patches to contain lesion\n")

    pos, tot, fracs = 0, 0, []
    for i in range(n_batches):
        b = next(dl)
        seg = b["target"]
        seg = seg[0] if isinstance(seg, (list, tuple)) else seg
        seg = np.asarray(seg)
        for s in seg:
            n_fg = int((s > 0).sum())
            tot += 1
            pos += n_fg > 0
            fracs.append(n_fg / s.size)
        if i < 3:
            print(f"  batch {i}: shapes data{tuple(np.asarray(b['data']).shape)} "
                  f"seg{tuple(seg.shape)}  fg per sample "
                  f"{[int((s>0).sum()) for s in seg]}")

    fr = np.array(fracs)
    print(f"\npatches sampled           : {tot}")
    print(f"patches containing lesion : {pos}  ({pos/tot:.1%})")
    print(f"mean fg fraction in patch : {fr.mean():.5f}  "
          f"(whole-volume baseline ~0.0007)")
    print(f"max  fg fraction in patch : {fr.max():.5f}")

    ok = pos / tot >= 0.30
    print("\nVERDICT:", "OK -- oversampling is delivering lesion voxels" if ok else
          "BROKEN -- the model is barely seeing foreground; do not start the long run")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

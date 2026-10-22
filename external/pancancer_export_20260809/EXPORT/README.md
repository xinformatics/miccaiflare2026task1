# Continue the Tversky run past epoch 59

Everything needed to pick up where this machine stopped. The run that produced our
best result — **0.4249 hidden DSC** — finished its 60-epoch schedule; this package
continues from those weights.

---

## What you have

```
weights/
  checkpoint_ep0059_weights_only.pth   <- continuation point (weights only, 21MB)
  checkpoint_ep0051_weights_only.pth   <- BEST: 0.4249 hidden / 0.4699 public
code/
  nnUNetTrainer_FLARE_Tversky.py   trainer + Tversky loss + continuation variants
  nnUNetTrainer_FLARE.py           epoch/LR variants, perform_actual_validation override
  test_tversky.py                  loss correctness test — RUN THIS FIRST
  03_finetune_plans.py             creates the 3d_fullres_S4D2W32_ft config
  predict.py evaluate.py SurfaceDice.py make_submission.py
  checkpoint_snapshotter.py        keeps per-epoch checkpoints (save_every=1 overwrites)
  sweep_epochs.sh                  score many epochs on public val
plans/
  nnUNetPlans.json                 all configs incl. 3d_fullres_S4D2W32_ft
  dataset_fingerprint.json         the 46-min computation — do not redo
  dataset.json
docs/
  TVERSKY_FINDING.md               READ FIRST — the result and the public-val trap
  RESULTS.md HANDOFF.md TVERSKY_RUN.md
```

**Note on the weights:** these are stripped to `network_weights` only (21 MB vs 42 MB).
`-pretrained_weights` reads nothing else (`load_pretrained_weights.py:23`), and this
was verified to load into the `3d_fullres_S4D2W32_ft` config with 0 missing / 0
unexpected keys. Optimizer state is deliberately absent — continuation uses a fresh
schedule anyway, so resuming optimizer momentum would be wrong here.

**Not included:** the 800 GB raw corpus and the 203 GB preprocessed cache. Copy the
cache from `/mnt/yadav/PancancerCTSeg/nnUNet_preprocessed/Dataset501_PanCancer/` or
regenerate it (~5 h + 46 min fingerprint) using `HANDOFF.md` §7.2. It is
**spacing-specific only** — valid across any architecture, patch, batch, LR or epoch
change.

---

## Why you cannot just `--c`

The poly schedule is `lr = init_lr * (1 - epoch/num_epochs)^0.9`. At epoch 59 of 60 it
has decayed to ~0. Resuming with `--c` would train at zero learning rate and change
nothing.

Continuation = a **fresh, shorter, lower-LR schedule** initialised from the ep59
weights via `-pretrained_weights`, which loads network weights only (not optimizer
state) — exactly what is wanted here.

**Start LOW.** The single most expensive mistake on this project was restarting a
decayed schedule at full LR: it knocked the model out of its minimum and public val
fell 0.4725 → 0.3614. Do not repeat that at this stage.

---

## Setup

```bash
# 1. environment (nnU-Net >= 2.5 required for the blosc2 cache backend)
python -m venv --system-site-packages .venv
.venv/bin/pip install nnunetv2==2.6.2

# 2. register the trainers (nnU-Net discovers by class name, any .py in that dir)
TD=$(.venv/bin/python -c 'import nnunetv2.training.nnUNetTrainer as m,os;print(os.path.dirname(m.__file__))')
cp code/nnUNetTrainer_FLARE.py code/nnUNetTrainer_FLARE_Tversky.py "$TD/"

# 3. VERIFY THE LOSS — seconds, and catches a silently-wrong implementation
.venv/bin/python code/test_tversky.py     # must print ALL CHECKS PASSED

# 4. point nnU-Net at the data
export nnUNet_raw=...  nnUNet_preprocessed=...  nnUNet_results=...
# plans/ contents go to $nnUNet_preprocessed/Dataset501_PanCancer/
```

`test_tversky.py` asserts that Tversky at α=β=0.5 reproduces
`MemoryEfficientSoftDiceLoss` to 1e-6. That validates the whole TP/FP/FN derivation —
a wrong implementation still trains and still produces plausible losses, so this is
the only cheap way to know it is right.

---

## Run

```bash
CK=weights/checkpoint_ep0059_weights_only.pth

nnUNetv2_train 501 3d_fullres_S4D2W32_ft all \
    -tr nnUNetTrainer_FT_Tversky_Cont_Lr5e5 \
    -pretrained_weights $CK
```

| trainer | epochs | LR | use when |
|---|---|---|---|
| **`..._Cont_Lr5e5`** | 100 | 5e-5 | **default** — half the LR that produced 0.4249 |
| `..._Cont_Lr2e5` | 100 | 2e-5 | conservative polish, minimal drift risk |
| `..._Cont_Lr1e4_Long` | 400 | 1e-4 | a machine with real compute; untested |

All keep α=0.3, β=0.7, oversample 0.5, `save_every=1`.

Also start the snapshotter, or per-epoch weights are lost — `save_every=1` writes
every epoch but always to the same filename:

```bash
python code/checkpoint_snapshotter.py \
    --model-dir $nnUNet_results/Dataset501_PanCancer/nnUNetTrainer_FT_Tversky_Cont_Lr5e5__nnUNetPlans__3d_fullres_S4D2W32_ft/fold_all
```

---

## Evaluate — and do not trust public val to rank

Measured public→hidden retention across five pairs: **0.47, 0.55, 0.70, 0.80, 0.90.**
No stable conversion, and the *ordering is not preserved*. Public val showed the
Tversky gain as +2.8% when the true hidden gain was +28.8%.

Use public val only to discard obviously-bad checkpoints; decide on the leaderboard.

```bash
bash code/sweep_epochs.sh "10 20 30 40 50 60 70 80 90 100"     # shortlist

python code/predict.py -i validation/Validation-Hidden-Images -o preds/epN \
    --model-dir <snapshot_dir> -f all --checkpoint checkpoint_latest.pth
python code/make_submission.py --pred preds/epN --out submission_epN.zip
```

`make_submission.py` writes a **flat** zip. Zipping the folder instead produces
`folder/FLARE-Val_0001.nii.gz` and Codabench fails with
"Submission volume not found".

---

## Reference numbers to beat

| model | public DSC | hidden DSC |
|---|---|---|
| organisers' baseline | 0.4570 | 0.3300 |
| fine-tune @1e-3 ep19 | — | 0.3800 |
| **Tversky ep51** | 0.4699 | **0.4249** |

Leaderboard: 0.71 / 0.59 / 0.53 / 0.49. **0.4249 is ~5th**; ~0.49 reaches 4th.

Efficiency of this model, measured (budget: ≤4096 MB, ≤60 s/case):
**2521 MB peak, 7.7 s mean, 20.4 s max.** Eval hardware is ~half this GPU's speed, so
double the times — still safe. Efficiency is **half** the ranking weight.

---

## Highest-value next steps

1. **Score the un-submitted epochs you already have.** ep27 (7/50 zeros — the best
   zero-count) and ep35 and ep59 are packaged but unscored. Cheap, and public val
   cannot rank them.
2. **Scale.** Everything here is 60 epochs of a 5.5M model on one consumer GPU. The
   FLARE25 winner used **5000 epochs on 8× RTX 4090**. Tversky at scale is untested
   and is the biggest untried lever — `-num_gpus N`, no new code.
3. **Two-stage detection → segmentation** (paper #16's 0.59 recipe). Still
   unimplemented. Better motivated now: if FN-weighting alone bought +28.8%, explicit
   detection should buy more.
4. **Sweep β.** 0.7 works; 0.85 and 0.6 untested. One variable, cheap.
5. **Ablate the LR** (`nnUNetTrainer_FT_Tversky_Lr1e3`) — loss and LR are currently
   confounded, which matters for a writeup.

## Do not redo — measured and rejected

| idea | result |
|---|---|
| Mirroring TTA | DSC 0.4725 → **0.4673 (worse)** at 1.64× the time |
| VISTA3D | 0.1140 zero-shot; 5 lesion classes only; 872 MB blows the VRAM budget |
| ResEnc 107M from scratch | 0.1011 hidden after 161 epochs; needs far more compute |
| Fine-tune at LR 1e-3 | resets a decayed schedule to full and destroys the minimum |
| Ranking checkpoints by public val | retention 0.47–0.90; ordering not preserved |

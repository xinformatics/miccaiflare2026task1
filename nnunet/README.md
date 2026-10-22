# nnU-Net v2 CT segmentation setup

This setup uses dataset ID `501` and the `nnUNetPlannerResEncL` 3D full-resolution
preset. Source images are never copied: `prepare_dataset.py` creates symlinks in
`nnUNet_raw/Dataset501_PanCancerCT`.

## Data split

- `imagesTr/labelsTr`: only the 17,575 challenge training cases.
- Fold 0: a deterministic, source-stratified 95/5 internal split made only from
  those training cases.
- Local public validation: 50 labeled cases, used only by
  `evaluate_public.sh` after training.
- Codabench Development phase: 100 hidden-validation images, used by
  `submit_hidden_validation.sh` to create the leaderboard ZIP.
- Inference only (`imagesTs`): those 50 public-validation and 100 Codabench
  Development images, plus 172 healthy/no-lesion cases. No masks are fabricated
  for these cases.

The public validation labels never enter `nnUNet_raw`; they are used only by
`evaluate_public.sh` after inference. This prevents them from influencing the
dataset fingerprint or model optimization and follows the challenge rule:
"Please do not use the public validation set for model training."

The training labels are partial: only a primary lesion is labeled and other
real lesions may appear as background. This standard nnU-Net run is therefore a
baseline; it may learn to suppress unlabeled metastases. Healthy scans also
remain evaluation-only, making false-positive performance an important public
and hidden-test check.

## Run

Planning and preprocessing is deliberately separate from training because the
raw dataset is about 803 GiB and preprocessing will be long-running:

```bash
./nnunet/plan_and_preprocess.sh
./nnunet/train_2gpu.sh
./nnunet/evaluate_public.sh
./nnunet/submit_hidden_validation.sh
```

Resume an interrupted run with:

```bash
./nnunet/train_2gpu.sh --c
```

For a one-epoch two-GPU smoke run:

```bash
./nnunet/run_1epoch.sh
```

On a fresh workspace this performs the required planning/preprocessing first,
then launches `train_2gpu.sh --epochs 1`. Combined output is saved in
`logs/nnunet_1epoch.log` and the W&B run name defaults to
`resencL-fold0-1epoch`.

The training command exposes GPUs 0 and 1 and starts two DDP ranks. Results go
to `nnUNet_results/Dataset501_PanCancerCT`. Internal fold metrics are written by
nnU-Net; challenge public-validation metrics are written to
`predictions/public_fold0/summary.json`.

## Weights & Biases tracking

The training entry point preserves stock nnU-Net checkpoints while logging from
DDP rank 0 only. It tracks training/validation loss, online foreground Dice and
EMA Dice, learning rate, epoch duration, rank-0 peak CUDA memory, W&B system
metrics for both visible GPUs, best internal Dice, final internal validation,
and aggregate public-validation metrics.

Logging defaults to offline mode so missing credentials or network access never
blocks training. Runs are stored in `wandb/`. For a live dashboard:

```bash
source nnunet/env.sh
wandb login
WANDB_MODE=online ./nnunet/train_2gpu.sh
```

To upload an offline run later:

```bash
source nnunet/env.sh
wandb sync wandb/offline-run-*
```

Defaults can be overridden with `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_NAME`,
`WANDB_GROUP`, and `WANDB_MODE`. The run ID is saved beside the checkpoints, so
`--c` resumes the same W&B run and `evaluate_public.sh` attaches challenge
metrics to it. Checkpoints are not uploaded as artifacts by default.

## Codabench hidden-validation submission

After fold 0 has a final checkpoint, run:

```bash
./nnunet/submit_hidden_validation.sh
```

This partitions the 100 hidden-validation volumes between GPU 0 and GPU 1,
runs both nnU-Net inference processes concurrently, validates every prediction,
and creates:

```text
submissions/pancancer_hidden_fold0.zip
```

The archive matches the organizer's baseline submission: exactly 100 binary
NIfTI masks at the ZIP root, named `FLARE-Val_0001.nii.gz` through
`FLARE-Val_0100.nii.gz`. The script rejects missing/extra cases, `_0000`
prediction names, non-binary labels, corrupt NIfTI files, and image/prediction
shape or affine mismatches. It does not include probability maps or metadata.

If inference already completed and only the archive needs rebuilding or
rechecking, run:

```bash
./nnunet/submit_hidden_validation.sh --package-only
```

Inference overwrites old masks by default, preventing predictions from an older
checkpoint from silently entering a new submission. To resume an interrupted
inference with the same checkpoint, use
`CONTINUE_PREDICTION=1 ./nnunet/submit_hidden_validation.sh`.

Upload the resulting ZIP under **My Submissions -> Development Phase** on the
[Codabench competition](https://www.codabench.org/competitions/7149/#/participate-tab).
Codabench owns the hidden labels, so its returned Lesion DSC and NSD are the
quality measurements for these cases; local W&B cannot calculate those scores.
The public-validation script continues to provide per-run local/W&B quality
tracking before spending a leaderboard submission.

To run nnU-Net commands manually, first load the project-local paths:

```bash
source nnunet/env.sh
```

Fingerprinting defaults to 4 workers and preprocessing to 8 workers. Higher
counts of 22 and 16 both caused the OS to kill a worker after exhausting RAM.
These values can be overridden with
`NNUNET_FINGERPRINT_PROCESSES` and `NNUNET_PREPROCESS_PROCESSES`.

The native labeled arrays would occupy roughly 3.5 TiB as float32 image plus
mask data. nnU-Net 2.6.4 uses chunked compressed `b2nd` preprocessing and does
not unpack those arrays at training startup. Even so, the preprocessing script
requires at least 1 TiB free by default; monitor the project filesystem while it
runs.

Seven PanTS cases (`00000259`, `00005731`, `00006447`, `00006466`, `00007130`,
`00007151`, and `00007558`) have image/mask affine-header differences. Their
shapes and spacing match, and voxel-space overlays were checked and found to be
aligned. nnU-Net's integrity check will report these as non-fatal warnings.

Thirteen training CTs and ten masks also contain non-orthogonal NIfTI sforms
(gantry tilt/shear). `env.sh` enables ITK's permissive-sform reader so these
matched arrays can be processed without modifying the source challenge files.
ITK will print warnings for those headers; they are expected.

## Challenge implications

- The final test labels contain all lesions even though training masks contain
  only the primary lesion. A competitive follow-up should investigate
  conservative pseudo-labeling or positive-unlabeled learning without touching
  public/hidden validation labels.
- The test set contains healthy cases, so connected-component and confidence
  postprocessing must be selected using permitted development data and checked
  for whole-volume false positives.
- Public datasets/models are permitted only under the challenge's forum and
  July 31 disclosure rule. Do not add a new external source without confirming
  that compliance requirement.
- ResEnc-L is a performance baseline, not a deployment solution for the stated
  4 GB inference-memory target. Submission optimization should be measured
  separately after segmentation quality is established.

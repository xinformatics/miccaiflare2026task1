# Pan-cancer CT Segmentation — MICCAI 2026 FLARE Task 1

Whole-body cancer lesion segmentation in CT. Binary task (background / tumor),
17,575 labelled training cases across 28 source cohorts.

## Scoring rules that drive every design decision

From `Information.md` and the organisers' own metric code
(`code/SurfaceDice.py`, vendored verbatim from
[JunMa11/FLARE](https://github.com/JunMa11/FLARE) `FLARE24/`):

| Rule | Consequence |
|---|---|
| NSD tolerance = **2 mm** | fixed, not a tunable |
| `max(pred) > 1` → DSC 0, NSD 0 | output **must** be uint8 {0,1} |
| empty GT + empty pred → **1.0** | healthy cases are free points… |
| empty GT + *any* FP → **0.0 / 0.0** | …or a total loss. 172 healthy val cases |
| non-empty GT + empty pred → 0 / 0 | missing a lesion entirely is unrecoverable |
| **DSC < 0.2 → NSD forced to 0** | sub-threshold cases lose both metrics |
| >60 s on a case → DSC 0 | hard cliff |
| GPU memory ≤ **4096 MB** → perfect efficiency score | half the ranking is efficiency |

Ranking = mean of per-case ranks over 500 test cases × 4 metrics. **Efficiency is
half the ranking weight**, not a tiebreaker.

## Competitive landscape (from `2025papers/`)

| Solution | DSC | NSD | Max GPU | Time/case | GPU penalty¹ |
|---|---|---|---|---|---|
| **VISTA3D zero-shot** (measured here) | **0.1140** | **0.0506** | — | ~7 s | — |
| Provided baseline docker | 0.2534 | 0.2440 | — | — | — |
| Template submission on hidden leaderboard | **0.33** | — | — | — | — |
| **Organisers' baseline docker, measured on OUR public val** | **0.4725** | **0.3671** | **1601 MB** | 9.3 s mean | **0** |
| **This pipeline (measured)** | TBD | TBD | **2468 MB** | **12.5 s mean / 29 s max** | **0** |
| #9 "A Lightweight and Effective nnU-Net" | 0.5348 | 0.4384 | 6904 MB | ~18 s | 2808 MB |
| #16 "Towards Accurate and Efficient Pan-cancer Screening" | **0.59** | **0.52** | 14044 MB | 23–31 s | 9948 MB |
| **This repo (target)** | ≥0.53 | ≥0.44 | **<4096 MB** | **<10 s** | **0** |

¹ `GPU Memory = max(0, peak − 4096MB)` (`Information.md:50`).

**Both published leaders overshoot the 4 GB tolerance.** Ranking averages per-case
ranks across DSC, NSD, runtime *and* GPU memory — efficiency is half the weight. So
the strategy is: maximise accuracy **subject to** staying under 4096 MB, where we take
a perfect efficiency rank and they do not.

DSC 0.9 is not attainable here — that is an organ-segmentation number. The
leaderboard leader is at 0.59. Lesion boundaries are ambiguous, inter-rater variance
is high, and training labels mark **only the primary lesion** while the test set
labels *all* lesions.

### What the papers give us

- **#9 (winner, 53.48%)** — patch **128×160×160** @ spacing **2.4×1.6×1.6mm**
  (expands coverage from 96×126×126mm to 307×256×256mm); 5000 epochs, SGD, lr 1e-3
  poly, Dice+CE, batch 2. Reports **45.2M params** while claiming a "4-layer"
  network — strong evidence they hit the config bug below and trained the deep net
  unknowingly. They excluded the 5,000 DeepLesion cases because those labels are
  **MedSAM2-generated pseudo-labels**, training on 5,328 verified cases only.
  *We train on the full 17,575 instead* — their pool was 10,328, ours is 70% larger,
  so the noisy-label tradeoff is not the same one they faced.
- **#16 (leader, 59%)** — two-stage: a 3D ResNet-18 classifier rejects sub-volumes
  with no lesion, then SwinUNETR segments only the candidates. Cuts false positives
  *and* compute. Region-specific HU windows: chest `[-900, 650]`, abdomen
  `[-175, 250]`. No postprocessing.
- Not relevant to Task 1: #15 (CT→PET UDA), #12 (Task 2 CPU organ seg), #1/#3/#6
  (VLM tasks), #2/#5/#10 (MRI/PET domain adaptation).

### Public -> hidden calibration (two MEASURED pairs, from actual leaderboard uploads)

| model | public DSC | hidden DSC | ratio |
|---|---|---|---|
| organisers' baseline | 0.4725 | 0.3300 | 0.70 |
| our fine-tune @ ep149 | 0.3560 | 0.2857 | 0.80 |

hidden ~= 0.70-0.80 x public. It is a RATIO, not a fixed offset -- an early single-
point read of "~0.14 lower" was wrong and made the target look harder than it is.
**To beat 0.33 hidden, aim for public >= 0.42-0.47.**

Our model loses less going public -> hidden than the baseline does, plausibly because
it trains on 17,575 cases spanning more cancer types than the baseline's 5,328.

### Public val OVERSTATES hidden val — calibrate against this

The organisers' own baseline docker, run locally through `code/predict.py`:

| set | DSC | source |
|---|---|---|
| **our public val (50 FLARE23Ts)** | **0.4725** | measured here |
| hidden leaderboard | 0.33 | template submission score |
| hidden ("Validation") | 0.2534 | `Information.md:100` |

The same model drops **~0.15–0.20 from public to hidden**. So a public-val reading of
~0.50 likely means ~0.33–0.35 hidden. Read our own curve accordingly.

This runs *opposite* to paper #16 (56.4 public -> 59 hidden), which suggests the
hidden set was refreshed for MICCAI 2026. Trust this measured offset over the
papers'.

Its efficiency is also already inside budget (1601 MB, 9.3 s/case), so beating it
means beating it on **accuracy**, not on efficiency.

### Zero-shot foundation-model floor (`code/run_vista3d.py`)

VISTA3D (MONAI, CVPR 2025 — 11,454 volumes, 127 classes, automatic class-prompted)
on the 50 public val cases, prompting its five lesion classes (23 lung, 24
pancreatic, 26 hepatic, 27 colon, 128 bone) and unioning them to binary:

| | DSC | NSD |
|---|---|---|
| VISTA3D zero-shot | **0.1140** | **0.0506** |

Distribution: 16/50 scored exactly 0, 22 in (0, 0.2), 12 ≥0.2, only 1 ≥0.5
(best 0.869). NSD collapses further than DSC because the `DSC<0.2 → NSD=0` rule
zeroes 38 of the 50.

Two honest caveats: VISTA3D has **no kidney, adrenal, oesophageal, endometrial or
lymph-node tumor class**, so it is structurally unable to fire on much of a
whole-body pan-cancer set; and it was never trained on this label definition. Read
0.1140 as "generic foundation model, no task training" — a floor, not a ceiling.
The useful conclusions are that task-specific training is required (it lands at
~45% of the provided baseline and ~21% of the winner), and that this run exercised
the full predict → binarise → geometry-match → score chain on real predictions.

## Current run: fine-tune from the organisers' baseline

`3d_fullres_S4D2W32_ft` (5.47M params) initialised from the FLARE25 baseline
checkpoint via `-pretrained_weights`, 500 epochs, reusing the existing 203GB cache
(`code/03_finetune_plans.py` verifies the weights load with 0 missing/0 unexpected
keys — the only edit needed was the last stride, `[2,2,2]` -> `[1,2,2]`).

| epoch | public DSC | public NSD | note |
|---|---|---|---|
| — | 0.4725 | 0.3671 | baseline we initialised from |
| 50 | 0.3614 | 0.2438 | dip: LR reset to full 1e-3 + domain shift |
| 100 | 0.4335 | 0.3243 | +0.072 in 50 epochs, re-converging |

The dip is expected and was worth understanding rather than panicking over: the
baseline was itself trained at `initial_lr=1e-3` for 2000 epochs, so its poly
schedule had decayed to ~0. Starting a fresh 1e-3 schedule resets LR to full and
moves the model out of its minimum before it re-converges on our preprocessing
(2.4x1.6x1.6mm and our own normalisation constants, both different from the
baseline's 3x2x2). `nnUNetTrainer_FT_Lr1e4` / `_Lr3e4` exist as conservative
alternatives if a future run needs to preserve the minimum instead.

Monitored by `code/val_watcher.py` -> `code/val_curve.csv`. Note the watcher costs
~11% (790s per evaluation vs 7300s of training per 50 epochs) because it contends
with training for the GPU, not the ~3% predicted from standalone timings.

## Findings

**1. The published winner config is silently broken on nnU-Net ≥ 2.5.**
FLARE24/25 publish `3d_fullres_S4D2W32` using pre-2.5 keys
(`UNet_base_num_features`, `n_conv_per_stage_encoder`, `pool_op_kernel_sizes`)
nested under `inherits_from: 3d_fullres`. In `plans_handler.py`, inheritance does
`base_config.update(configuration)`, so the parent's `architecture` dict survives —
and the old-key reconstruction path only fires when `architecture` is *absent*.
Loading their plans verbatim builds a **31.20M-param 6-stage** network, not the
documented 5.6M 4-stage one. Verified by parameter count.

`code/arch.py` re-expresses the config in the new `architecture` format;
`code/02_make_plans.py` asserts the result really is 4-stage / 5.6M.

**2. Architecture sweep** (`code/sweep_configs.py`), inference VRAM = net + logit map
at *resampled* resolution:

| config | params | train VRAM | inf VRAM | max time |
|---|---|---|---|---|
| S4W32 @3.0/2.0/2.0 p96 | 5.60M | 3363 MB | 866 MB | 2.2 s |
| **S4W32 @2.4/1.6/1.6 p128** | **5.60M** | **6971 MB** | **1737 MB** | **4.6 s** |
| S5W32 @2.4/1.6/1.6 p128 | 16.55M | 7076 MB | 1842 MB | 4.7 s |
| S6W32 @2.4/1.6/1.6 p128 | 31.20M | 7167 MB | 1984 MB | 4.8 s |

Chosen: **S4W32 @ 2.4/1.6/1.6, patch 128×160×160** — the winner's preprocessing
(which is what drove their accuracy) with the genuinely lightweight net.

**MEASURED (10-epoch checkpoint, 50 public val cases, `code/predict.py`):**
peak inference VRAM **2468 MB** against the 4096 MB budget, time/case mean 12.5 s,
p50 11.9 s, p90 17.8 s, max 29.0 s, **0 cases over the 60 s cliff** — with GPU export
still enabled, so `--cpu-export` remains in reserve. Both published leaders forfeit
the efficiency half of the ranking (penalties 2808 MB and 9948 MB); we do not.
Caveat: measured with a model predicting all background. Sliding-window cost is
identical for a trained model, but connected-component postprocessing adds a little.
Eval hardware is a Quadro RTX 5000 (~half a 4060 Ti), so expect roughly 2x these
times there; p90 ~36 s is safe, the 29 s worst case is the one to watch, and
`--step 0.7` is the lever if it gets close.

**3. The 2GB-vs-7GB gap is the real efficiency lever.** The sweep says 1984 MB for
the config #9 actually trained; they report 6904 MB. The difference is nnU-Net
resampling logits back to **original** resolution *on device* — a 512×512×361 volume
× 2 classes × fp32 is ~757 MB before allocator overhead. `predict.py --cpu-export`
moves that step to CPU. These sweep numbers are therefore a **lower bound**; the
number that counts is `predict.py`'s measured peak, which is instrumented and must
be checked against 4096 MB before the long training run.

**3. Partial labels cut against training-set fidelity.** Only the primary lesion is
annotated in training, but *all* lesions are annotated in the test set. Suppressing
confident secondary-lesion predictions to please training labels therefore hurts the
real score. Public-val DSC is a **floor**, not an unbiased estimate.

## Layout

```
code/
  env.sh                  paths + venv (source this first)
  00_build_raw.py         symlinked nnU-Net raw dataset (never copies the 800GB)
  01_integrity_scan.py    header/geometry/label scan over all 17,575 pairs
  arch.py                 S4D2W32 in the new plans format  <- the fix
  02_make_plans.py        inject the config, assert 4-stage/5.6M
  make_subset.py          stratified subset for fast end-to-end validation
  measure_footprint.py    preprocessed-cache size -> go/no-go on full data
  bench_arch.py           VRAM + runtime vs the 4GB / 60s budget
  predict.py              inference + connected-component postprocessing
  evaluate.py             official metric, public-val + healthy set
  SurfaceDice.py          vendored verbatim from the organisers
  nnUNetTrainer_FLARE.py  Epoch{500,1000,2000,5000}_Lr1e3 variants
  run_pipeline.sh         stage runner
docker/                   submission container (CUDA 11.8 base — see below)
```

## Running

```bash
source code/env.sh

bash code/run_pipeline.sh raw          # symlink raw dataset
bash code/run_pipeline.sh integrity    # scan for malformed headers
bash code/run_pipeline.sh subset 400   # stratified subset

# dry run the whole chain on the subset first (~1-2h)
bash code/run_pipeline.sh sub-plan
bash code/run_pipeline.sh sub-preproc  # prints the disk go/no-go
bash code/run_pipeline.sh sub-train

# full run
bash code/run_pipeline.sh plan         # fingerprint + plan (hours, reads 800GB)
bash code/run_pipeline.sh preproc      # blosc2 .b2nd cache (hours)
bash code/run_pipeline.sh train nnUNetTrainer_Epoch2000_Lr1e3

# evaluate
bash code/run_pipeline.sh predict
bash code/run_pipeline.sh predict-healthy
bash code/run_pipeline.sh eval
```

## Operational trap: systemd-oomd kills long jobs on this box

`systemd-oomd` killed the terminal cgroup three times during preprocessing —
Claude Code and all 277 child processes each time — while **90–100 GB of RAM was
free**. It is not the kernel OOM killer: it watches PSI *memory pressure*, and
`user@1000.service` here is set `ManagedOOMMemoryPressure=kill` at a 50% limit.
Streaming 800 GB of CT through the page cache while writing 216 GB to a spinning
disk keeps the kernel in sustained reclaim, which pushed PSI to 79%.

What does **not** help:

| Attempt | Why it fails |
|---|---|
| `nohup` | blocks SIGHUP; oomd sends SIGKILL to the whole cgroup |
| `setsid` | detaches the *process group*; oomd targets **cgroups** |
| `MemoryHigh=<N>` | **counterproductive** — capping the cgroup forces reclaim *inside* it, generating the exact PSI signal oomd watches. Also throttled throughput to 7 cases in 4 min |

What works: run the job as its own transient systemd unit with **no memory cap**,
modest worker count, and `Restart=on-failure` — combined with resumable scripts
(`resume_preprocess.py`, `train_resilient.sh`) so a kill costs seconds, not hours.
Throughput at `-np 4` measured 0.97 cases/s, indistinguishable from `-np 12`: the
bottleneck is disk I/O, not workers.

Permanent fix (needs sudo): `sudo systemctl disable --now systemd-oomd`. The kernel
OOM killer remains as the real safety net.

```bash
# launch training so it survives oomd kills and session restarts
systemd-run --user --unit=pancancer-train --service-type=exec \
  -p Restart=on-failure -p RestartSec=30 -p StartLimitIntervalSec=0 \
  --working-directory=$PROJ code/train_resilient.sh
journalctl --user -u pancancer-train -f
```

## Environment

Isolated venv at `.venv` (`--system-site-packages`, reuses the existing
torch 2.8.0+cu128). nnU-Net needs `blosc2>=3.0` but base conda has 2.7.0, so
installing into base would have broken it.

nnU-Net 2.6.2 stores preprocessed data as **blosc2 `.b2nd`** (ZSTD, memory-mapped,
no `.npz`→`.npy` unpack step), which is what makes 17,575 cases viable on the
~127 GB free.

## Hardware note

Local GPU is an RTX 4060 Ti (16 GB); the evaluation desktop is an RTX 5000 (16 GB),
28 GB RAM, 12 CPU cores, **driver 510.39.01 / CUDA 11.6**. The submission image is
pinned to a CUDA 11.8 base for minor-version compatibility — a cu12x image would not
start on that driver.

## Data provenance

- Training: `train_label/imagesTr_part{1,2}` + `labelsTr_part{1,2}`, 17,575 cases,
  28 cohorts (largest: `WholeBody_DeepLesion` 5000, `Chest_LUNA25` 4049).
- `validation/Validation-Public-{Images,Labels}` — 50 FLARE23Ts cases. **Never
  train on these** (`Information.md:30`).
- `validation/Validation-Hidden-Images` — 100 FLARE-Val, labels withheld.
- `validation/HealthyImages-noLesion` — 172 cases (PANCREAS 82, CHAOS 40, other 50).
  Used **only** for false-positive evaluation; they sit under `validation/`, so
  training on them is not clearly permitted.

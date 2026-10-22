#!/usr/bin/env bash
# Pan-cancer CT segmentation -- full pipeline.
#
#   bash code/run_pipeline.sh <stage>
#
# Stages are resumable and can be run independently. Heavy stages print elapsed time.
set -euo pipefail
cd "$(dirname "$0")/.."
source code/env.sh

STAGE="${1:-help}"
DS_FULL=501
DS_SUB=502
CFG=3d_fullres_S4D2W32
NPROC=14

t0=$(date +%s)
banner() { echo; echo "=== $* ==="; date '+%F %T'; }
elapsed() { echo "--- elapsed $(( ($(date +%s) - t0) / 60 )) min"; }

case "$STAGE" in

raw)          banner "build symlinked raw dataset"
              $PY code/00_build_raw.py ;;

integrity)    banner "integrity scan (17,575 pairs)"
              $PY code/01_integrity_scan.py 2>&1 | tee code/integrity_scan.log ;;

subset)       banner "stratified subset -> Dataset${DS_SUB}"
              $PY code/make_subset.py --n "${2:-400}" --id $DS_SUB --name Sample ;;

# ---------- subset dry run: validates the whole chain in ~1-2h ----------
sub-plan)     banner "clone 501 fingerprint+plans -> subset (no second 800GB read)"
              $PY code/clone_plans.py --src $DS_FULL --src-name PanCancer \
                                      --dst $DS_SUB --dst-name Sample
              elapsed ;;

sub-preproc)  banner "preprocess subset + measure footprint"
              nnUNetv2_preprocess -d $DS_SUB -c $CFG -np $NPROC -plans_name nnUNetPlans
              $PY code/measure_footprint.py --dataset $DS_SUB --name Sample
              elapsed ;;

# De-risk the 75-hour run: 10 epochs (~25 min) produces a real checkpoint, then we
# exercise predict.py end-to-end and measure ACTUAL inference VRAM against the
# 4096MB budget. The sweep numbers are a lower bound; this is the number that counts.
smoke)        banner "10-epoch smoke train on FULL data"
              nnUNetv2_train $DS_FULL $CFG all -tr nnUNetTrainer_Epoch10_Lr1e3
              banner "smoke predict on public val (measures real VRAM + time)"
              $PY code/predict.py -i validation/Validation-Public-Images \
                  -o "$PREDS/smoke" -d $DS_FULL -c $CFG \
                  -tr nnUNetTrainer_Epoch10_Lr1e3 ${SMOKE_EXTRA:-}
              banner "smoke score (expect near-zero DSC -- we are checking plumbing)"
              $PY code/evaluate.py --pred "$PREDS/smoke"
              elapsed ;;

# ---------- full run ----------
plan)         banner "fingerprint + plan (FULL 17,575 cases -- hours)"
              nnUNetv2_extract_fingerprint -d $DS_FULL -np $NPROC
              nnUNetv2_plan_experiment -d $DS_FULL
              $PY code/02_make_plans.py
              elapsed ;;

preproc)      banner "preprocess FULL dataset (hours, writes blosc2 .b2nd)"
              nnUNetv2_preprocess -d $DS_FULL -c $CFG -np $NPROC -plans_name nnUNetPlans
              $PY code/measure_footprint.py --dataset $DS_FULL --name PanCancer
              elapsed ;;

train)        banner "train ${2:-nnUNetTrainer_Epoch2000_Lr1e3} on FULL"
              nnUNetv2_train $DS_FULL $CFG all -tr "${2:-nnUNetTrainer_Epoch2000_Lr1e3}"
              elapsed ;;

train-resume) banner "resume training"
              nnUNetv2_train $DS_FULL $CFG all -tr "${2:-nnUNetTrainer_Epoch2000_Lr1e3}" --c
              elapsed ;;

# ---------- evaluation ----------
predict)      banner "predict public val"
              $PY code/predict.py -i validation/Validation-Public-Images \
                  -o "$PREDS/public" -d $DS_FULL -c $CFG \
                  -tr "${2:-nnUNetTrainer_Epoch2000_Lr1e3}" --min-mm3 "${3:-0}" ;;

predict-healthy) banner "predict healthy (false-positive check)"
              $PY code/predict.py -i validation/HealthyImages-noLesion \
                  -o "$PREDS/healthy" -d $DS_FULL -c $CFG \
                  -tr "${2:-nnUNetTrainer_Epoch2000_Lr1e3}" --min-mm3 "${3:-0}" ;;

eval)         banner "score public val + healthy"
              $PY code/evaluate.py --pred "$PREDS/public"
              $PY code/evaluate.py --pred "$PREDS/healthy" --healthy ;;

bench)        banner "architecture vs 4GB / 60s budget"
              $PY code/bench_arch.py ;;

*) sed -n '1,12p' "$0"; echo
   echo "stages: raw integrity subset | sub-plan sub-preproc |"
   echo "        plan preproc smoke train train-resume |"
   echo "        predict predict-healthy eval bench" ;;
esac

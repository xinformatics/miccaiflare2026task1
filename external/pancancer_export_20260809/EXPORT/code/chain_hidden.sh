#!/usr/bin/env bash
# Wait for the public-val evaluation to finish, then predict the hidden set with the
# SAME snapshotted checkpoint and package a submission, then resume training.
#
# Kept in a file rather than inlined into `bash -c` on purpose: an inline version
# containing the string "predict.py" makes `pgrep -f predict.py` match the waiting
# shell itself, which deadlocked an earlier attempt at exactly this.
set -uo pipefail
cd "$(dirname "$0")/.."
source code/env.sh

SNAP="$PREDS/eval_snapshot"
OUT="$PREDS/hidden_resenc"
CFG=3d_fullres_ResEncS6
TR=nnUNetTrainer_Epoch500_Lr1e3

echo "$(date '+%F %T') waiting for public-val evaluation to finish..."
until grep -q "LESION-BEARING" code/eval_now.log 2>/dev/null; do sleep 20; done
echo "$(date '+%F %T') public val done:"
grep -A5 "LESION-BEARING" code/eval_now.log | tail -5

EP=$(nnUNet_compile=0 $PY -c "
import torch;print(torch.load('$SNAP/fold_all/checkpoint_best.pth',map_location='cpu',weights_only=False).get('current_epoch','?'))" 2>/dev/null | tail -1)
echo "$(date '+%F %T') snapshot is epoch $EP"

# eval_checkpoint.sh resumes training when it finishes; stop it again so the hidden
# pass gets the whole GPU. At 107M params, inference alongside training starves and
# hangs -- observed: 2146MB grabbed, 0.9% CPU, zero cases written in 52 minutes.
if systemctl --user is-active --quiet pancancer-train; then
    echo "$(date '+%F %T') pausing training for hidden pass"
    systemctl --user stop pancancer-train
    sleep 10
fi

rm -rf "$OUT"
$PY code/predict.py -i validation/Validation-Hidden-Images -o "$OUT" \
    --model-dir "$SNAP" -f all --checkpoint checkpoint_best.pth
echo "$(date '+%F %T') packaging"
$PY code/make_submission.py --pred "$OUT" --out "$PROJ/submission_resenc_ep${EP}.zip"

echo "$(date '+%F %T') resuming training"
systemctl --user reset-failed pancancer-train 2>/dev/null || true
systemd-run --user --unit=pancancer-train --service-type=exec --quiet \
  -p Restart=on-failure -p RestartSec=30 -p StartLimitIntervalSec=0 \
  -E nnUNet_raw="$nnUNet_raw" -E nnUNet_preprocessed="$nnUNet_preprocessed" \
  -E nnUNet_results="$nnUNet_results" -E nnUNet_compile=1 -E nnUNet_n_proc_DA=8 \
  -E MKL_THREADING_LAYER=GNU -E OMP_NUM_THREADS=1 \
  -E DS=501 -E CFG=$CFG -E TR=$TR -E FOLD=all \
  --working-directory="$PROJ" code/train_resilient.sh
sleep 5
echo "$(date '+%F %T') training: $(systemctl --user is-active pancancer-train)"

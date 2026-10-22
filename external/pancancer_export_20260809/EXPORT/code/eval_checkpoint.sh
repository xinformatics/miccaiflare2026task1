#!/usr/bin/env bash
# Score the current training checkpoint on public val, safely.
#
# Why not concurrently: at 107M params, training holds ~12.8GB of the 16.4GB card and
# ResEnc inference needs ~3.3GB. Running val_watcher alongside starved it -- predict.py
# grabbed 2146MB, then sat at 0.9% CPU for 52 minutes without writing a single case,
# while also slowing training 25% (51 min per evaluation vs 201 min per 50 epochs).
# This worked at 5.6M params; it does not at 107M.
#
# So: stop training, evaluate at full GPU (~15 min), resume from checkpoint. nnU-Net
# checkpoints every 50 epochs and train_resilient.sh resumes with --c, so the only
# cost is the epochs since the last checkpoint plus ~1 min of compile warmup.
#
#   bash code/eval_checkpoint.sh [checkpoint_latest.pth|checkpoint_best.pth]
set -uo pipefail
cd "$(dirname "$0")/.."
source code/env.sh

CKPT_NAME="${1:-checkpoint_latest.pth}"
CFG="${CFG:-3d_fullres_ResEncS6}"
TR="${TR:-nnUNetTrainer_Epoch500_Lr1e3}"
M="$nnUNet_results/Dataset501_PanCancer/${TR}__nnUNetPlans__${CFG}/fold_all"
SNAP="$PREDS/eval_snapshot"
OUT="$PREDS/eval_public"

[ -f "$M/$CKPT_NAME" ] || { echo "no checkpoint at $M/$CKPT_NAME"; exit 1; }

WAS_ACTIVE=$(systemctl --user is-active pancancer-train 2>/dev/null || true)
EP=$(nnUNet_compile=0 $PY -c "
import torch;print(torch.load('$M/$CKPT_NAME',map_location='cpu',weights_only=False).get('current_epoch','?'))" 2>/dev/null | tail -1)
echo "=== evaluating $CKPT_NAME (epoch $EP)"

# snapshot BEFORE stopping, so we score exactly what we inspected
rm -rf "$SNAP"; mkdir -p "$SNAP/fold_all"
cp "$M/$CKPT_NAME" "$SNAP/fold_all/"
cp "$M/../plans.json" "$M/../dataset.json" "$SNAP/" 2>/dev/null || \
  cp "$(dirname "$M")/plans.json" "$(dirname "$M")/dataset.json" "$SNAP/"

if [ "$WAS_ACTIVE" = "active" ]; then
    echo "=== pausing training to free VRAM"
    systemctl --user stop pancancer-train
    sleep 10
fi

rm -rf "$OUT"
$PY code/predict.py -i validation/Validation-Public-Images -o "$OUT" \
    --model-dir "$SNAP" -f all --checkpoint "$CKPT_NAME"
$PY code/evaluate.py --pred "$OUT" --csv "$PREDS/metrics_${CFG}_ep${EP}.csv" | tail -8

if [ "$WAS_ACTIVE" = "active" ]; then
    echo "=== resuming training from checkpoint"
    systemctl --user reset-failed pancancer-train 2>/dev/null || true
    systemd-run --user --unit=pancancer-train --service-type=exec --quiet \
      -p Restart=on-failure -p RestartSec=30 -p StartLimitIntervalSec=0 \
      -E nnUNet_raw="$nnUNet_raw" -E nnUNet_preprocessed="$nnUNet_preprocessed" \
      -E nnUNet_results="$nnUNet_results" -E nnUNet_compile=1 -E nnUNet_n_proc_DA=8 \
      -E MKL_THREADING_LAYER=GNU -E OMP_NUM_THREADS=1 \
      -E DS=501 -E CFG="$CFG" -E TR="$TR" -E FOLD=all \
      --working-directory="$PROJ" code/train_resilient.sh
    sleep 5
    echo "training: $(systemctl --user is-active pancancer-train)"
fi

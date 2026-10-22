#!/usr/bin/env bash
# Score a set of per-epoch checkpoints on public val and report the peak.
#
# The fine-tuning optimum is sharp -- measured 0.33 hidden at ep17 and 0.38 at ep19 --
# so the answer is which epoch, not just whether the run worked. Uses public val
# because we hold those labels and code/evaluate.py reproduces Codabench to within
# rounding (predicted 0.2857 / 0.1011 / 0.0743 vs actual 0.29 / 0.10 / 0.07).
#
# Prints mean DSC/NSD AND the count of DSC=0 cases. That count is the real test of
# the Tversky change: 16/50 cases scored exactly zero before, and if the FN-weighted
# loss worked, that number falls. A better mean with an unchanged zero-count would
# mean the loss change did not address the failure mode it was built for.
#
#   bash code/sweep_epochs.sh "15 19 23 27 31 35 39 45 51 59"
set -uo pipefail
cd "$(dirname "$0")/.."
source code/env.sh

EPOCHS="${1:-15 19 23 27 31 35 39 45 51 59}"
TR="${TR:-nnUNetTrainer_FT_Tversky_Lr1e4}"
CFG="${CFG:-3d_fullres_S4D2W32_ft}"
M="$nnUNet_results/Dataset501_PanCancer/${TR}__nnUNetPlans__${CFG}/fold_all"
SUM="$PROJ/code/epoch_sweep_${TR}.csv"

[ -f "$SUM" ] || echo "epoch,dsc,nsd,dsc0_cases,over02_cases" > "$SUM"
echo "model dir: $M"
echo "epochs   : $EPOCHS"

for E in $EPOCHS; do
    CK=$(printf "%s/epochs/checkpoint_ep%04d.pth" "$M" "$E")
    if [ ! -f "$CK" ]; then echo "-- ep$E: no checkpoint, skipping"; continue; fi
    grep -q "^$E," "$SUM" && { echo "-- ep$E: already scored"; continue; }

    SNAP="$PREDS/sweep_snapshot"; OUT="$PREDS/sweep_ep$E"
    rm -rf "$SNAP" "$OUT"; mkdir -p "$SNAP/fold_all"
    cp "$CK" "$SNAP/fold_all/checkpoint_latest.pth"
    cp "$M/../plans.json" "$M/../dataset.json" "$SNAP/" 2>/dev/null || \
      cp "$(dirname "$M")/plans.json" "$(dirname "$M")/dataset.json" "$SNAP/"

    echo; echo "=== epoch $E ==="
    $PY code/predict.py -i validation/Validation-Public-Images -o "$OUT" \
        --model-dir "$SNAP" -f all --checkpoint checkpoint_latest.pth \
        > "$PROJ/code/sweep_ep$E.log" 2>&1
    R=$($PY code/evaluate.py --pred "$OUT" --csv "$PREDS/metrics_sweep_ep$E.csv" 2>/dev/null | tail -8)
    D=$(echo "$R" | grep "^DSC  mean" | awk '{print $3}')
    N=$(echo "$R" | grep "^NSD  mean" | awk '{print $3}')
    Z=$(echo "$R" | grep "cases DSC=0" | sed 's/.*DSC=0 : \([0-9]*\).*/\1/')
    O=$(echo "$R" | grep "cases DSC=0" | sed 's/.*DSC>0.2: \([0-9]*\).*/\1/')
    echo "ep$E: DSC $D  NSD $N  zeros $Z/50  above0.2 $O/50"
    echo "$E,$D,$N,$Z,$O" >> "$SUM"
    rm -rf "$OUT"          # keep the disk sane; metrics csv retains per-case detail
done

echo; echo "================ SWEEP SUMMARY ================"
column -s, -t < "$SUM"
echo
echo "baseline for comparison : DSC 0.4570  zeros 7/50   (organisers' model, our plans)"
echo "prev fine-tune @1e-3    : DSC 0.3560  zeros 16/50  (-> 0.2857 hidden)"
BEST=$(tail -n +2 "$SUM" | sort -t, -k2 -rn | head -1)
echo "best by DSC: epoch $(echo $BEST|cut -d, -f1) -> DSC $(echo $BEST|cut -d, -f2) zeros $(echo $BEST|cut -d, -f4)"

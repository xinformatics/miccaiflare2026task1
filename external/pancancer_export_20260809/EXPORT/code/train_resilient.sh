#!/usr/bin/env bash
# Resumable training wrapper, for use as a systemd unit ExecStart.
#
# The 2000-epoch run is ~75h. systemd-oomd on this box kills cgroups on memory
# PRESSURE (not exhaustion) -- it took the terminal down three times during
# preprocessing while 90-100GB was free. A kill mid-training costs up to 50 epochs
# (nnU-Net checkpoints every 50), so the unit is set to auto-restart and this script
# decides whether to start fresh or resume.
#
# nnUNetv2_train --c resumes from checkpoint_latest.pth, but FAILS if no checkpoint
# exists yet, so --c cannot simply be hardcoded into the unit.
set -uo pipefail
cd "$(dirname "$0")/.."
source code/env.sh

DS=${DS:-501}
CFG=${CFG:-3d_fullres_S4D2W32}
TR=${TR:-nnUNetTrainer_Epoch2000_Lr1e3}
FOLD=${FOLD:-all}

CKPT="$nnUNet_results/Dataset${DS}_PanCancer/${TR}__nnUNetPlans__${CFG}/fold_${FOLD}/checkpoint_latest.pth"

ARGS=("$DS" "$CFG" "$FOLD" -tr "$TR")
if [ -f "$CKPT" ]; then
    # -pretrained_weights is deliberately NOT passed on resume: the initialisation
    # is already baked into checkpoint_latest.pth, and passing both would discard
    # the fine-tuning done so far.
    echo "$(date '+%F %T') resuming from $CKPT"
    ARGS+=(--c)
else
    echo "$(date '+%F %T') no checkpoint at $CKPT -- starting fresh"
    if [ -n "${PRETRAINED:-}" ]; then
        [ -f "$PRETRAINED" ] || { echo "PRETRAINED not found: $PRETRAINED"; exit 1; }
        echo "$(date '+%F %T') initialising from $PRETRAINED"
        ARGS+=(-pretrained_weights "$PRETRAINED")
    fi
fi

echo "nnUNetv2_train ${ARGS[*]}"
exec nnUNetv2_train "${ARGS[@]}"

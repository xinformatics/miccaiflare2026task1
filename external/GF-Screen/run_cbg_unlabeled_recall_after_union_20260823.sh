#!/usr/bin/env bash
set -euo pipefail

# Recall-oriented positive-unlabeled continuation of the CBG checkpoint that
# supports gf_conf_cbg41_e283. It never treats an empty teacher output as a
# negative label and cannot pre-empt the active union-native refiner pipeline.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
upstream="${root}/public_sweeps/candidate_refiner_union_native_saved_20260823/UNION_NATIVE_SAVE_ONLY_TERMINAL.json"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
work="${root}/public_sweeps/gf_unlabeled_teacher_cbg_20260823"
deployed="${root}/public_sweeps/gf_glance_unlabeled_cbg_deployed_20260823"
controller_log="${root}/runs/run_cbg_unlabeled_recall_after_union_20260823.log"

mkdir -p "${work}" "${deployed}"
exec 9>"${root}/runs/run_cbg_unlabeled_recall_after_union_20260823.lock"
flock -n 9 || { echo "CBG unlabeled-recall controller already active" >&2; exit 1; }
[[ -s "${deployed}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
[[ -s "${checkpoint}" ]] || { echo "missing CBG checkpoint: ${checkpoint}" >&2; exit 2; }

while [[ ! -s "${upstream}" ]]; do
  printf '%s waiting for union-native save-only terminal\n' "$(date -Ins)" | tee -a "${controller_log}"
  sleep 30
done

bash "${root}/run_cbg_focus_tta_current_20260823.sh" \
  >> "${controller_log}" 2>&1

UNLABELED_CHECKPOINT="${checkpoint}" \
UNLABELED_UPSTREAM="${upstream}" \
UNLABELED_WORK="${work}" \
UNLABELED_ARM_BALANCED="gf_glance_ul_cbg_bal_ep283_2ep_20260823" \
UNLABELED_ARM_CONSERVATIVE="gf_glance_ul_cbg_con_ep283_2ep_20260823" \
UNLABELED_DEPLOYED="${deployed}" \
UNLABELED_START_EPOCH=283 \
UNLABELED_MAX_EPOCHS=285 \
UNLABELED_EVAL_EPOCHS="284 285" \
UNLABELED_SKIP_LEGACY_SAVE=1 \
EVALUATE_ALL_EPOCHS=1 \
bash "${root}/run_dual_unlabeled_glance_after_allpartial_20260822.sh" \
  >> "${controller_log}" 2>&1

[[ -s "${deployed}/PUBLIC_ONLY_COMPLETE.txt" ]]
bash "${root}/evaluate_cbg_unlabeled_recall_additions_public_20260823.sh" \
  >> "${controller_log}" 2>&1
date -Ins > "${work}/CBG_UNLABELED_RECALL_COMPLETE.txt"

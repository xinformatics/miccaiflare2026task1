#!/usr/bin/env bash
set -euo pipefail

# Serialize all GPU stages after the already-running full-coverage watcher.
# Unlike a PID daisy chain, a non-zero stage stops this queue instead of
# silently releasing every downstream watcher. The sentinel PID must not exist;
# it tells child scripts that serialization has already been handled here.
upstream_pid="${WAIT_PID:-889242}"
sentinel_pid="${NO_WAIT_SENTINEL_PID:-4194303}"
if kill -0 "${sentinel_pid}" 2>/dev/null; then
  echo "NO_WAIT_SENTINEL_PID unexpectedly exists: ${sentinel_pid}" >&2
  exit 1
fi

while kill -0 "${upstream_pid}" 2>/dev/null; do
  sleep 30
done

run_stage() {
  local label="$1"
  shift
  echo "[$(date --iso-8601=seconds)] starting ${label}"
  WAIT_PID="${sentinel_pid}" bash "$@"
  echo "[$(date --iso-8601=seconds)] completed ${label}"
}

# The GT-proposal oracle recovers only 4/239 additional lesions and caps the
# current Focus model at public DSC 0.5970. Improve Focus first; Glance is then
# trained with the selected long-run Focus as its reward/segmentation model.
component_root="public_sweeps/gf_component_uniform_dual_20260820"
if [[ "${SKIP_COMPONENT_STAGE:-0}" == "1" ]]; then
  echo "[$(date --iso-8601=seconds)] component_uniform deliberately pruned after checkpoint 283"
elif [[ -f "${component_root}/summary.tsv" \
   && -f "${component_root}/robust_selection.json" ]]; then
  echo "[$(date --iso-8601=seconds)] component_uniform already evaluated; skipping stage"
else
  run_stage component_uniform ./run_dual_component_uniform_after_fullcoverage.sh
fi
run_stage fullcoverage_submission ./materialize_fullcoverage_promotion.sh
run_stage bbox_pseudo_positive ./run_dual_bbox_pseudo_positive_after_glance.sh
run_stage source_sampling ./run_dual_cohort_balance_after_pseudo.sh
run_stage safe_cancer_negatives ./run_dual_safe_cancer_negatives_after_proposal.sh
run_stage long_focus ./run_long_best_focus_after_queue.sh
run_stage long_focus_wise ./run_long_focus_wise_after_queue.sh
run_stage biomedparse_guided ../BiomedParse/run_public_audit_before_focus_queue.sh
run_stage glance_objectives ./run_dual_glance_objective_after_fullcoverage.sh
run_stage swin_late ../../nnunet/run_swin_late_checkpoint_audit_after_gf.sh
run_stage nnunet_guided_focus ./run_nnunet_guided_focus_after_swin.sh
run_stage asymmetric_support_rescue ./run_asymmetric_support_rescue_after_nnunet.sh
run_stage focus_tta ./run_dual_focus_tta_after_safe_negatives.sh
run_stage healthy_audit ./run_champion_healthy_audit_after_queue.sh
run_stage context_materialization ./materialize_context24_best_after_queue.sh
run_stage final_submission_shortlist ./finalize_three_submission_shortlist.sh

echo "[$(date --iso-8601=seconds)] downstream queue complete"

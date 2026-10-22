#!/usr/bin/env bash
set -euo pipefail

# CPU-only exact-707 patch-OOF test of a predeclared two-family contour route.
# Classification remains globally selected. No public/hidden data or packaging.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
training_complete="${train}/TRAINING_COMPLETE.json"
training_skipped="${train}/TRAINING_SKIPPED.json"
fold0_source_audits="${train}/FOLD0_SOURCE_AUDITS_COMPLETE.json"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
coupled_summary="${train}/five_fold_oof_summary.json"
decoupled_dir="${train}/decoupled_oof"
decoupled_terminal="${decoupled_dir}/TERMINAL.json"
decoupled_summary="${decoupled_dir}/five_fold_oof_summary.json"
output="${train}/source_routed_oof"
summary="${output}/five_fold_oof_summary.json"
full_route_summary="${output}/five_fold_oof_full_route_summary.json"
partial_route_selection="${output}/PARTIAL_ROUTE_SELECTION.json"
coupled_gate="${output}/COUPLED_GATE.json"
route_gate="${output}/SOURCE_ROUTE_GATE.json"
terminal="${output}/TERMINAL.json"

fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v4_source_routed_oof_20260824.lock"
flock -n 9 || { echo "source-routed exact-707 OOF controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${training_complete}" && ! -s "${training_skipped}" ]]; do
  printf '%s waiting for exact-707 five-fold training terminal\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${training_skipped}" ]]; then
  "${python_bin}" - "${training_skipped}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_GLOBAL_CONTOUR",
 "reason":"primary exact-707 training skipped","source":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
while [[ ! -s "${decoupled_terminal}" ]]; do
  printf '%s waiting for global decoupled OOF terminal\n' "$(date -Ins)"
  sleep 30
done
while [[ ! -s "${fold0_source_audits}" ]]; do
  printf '%s waiting for fold-0 scored-epoch source audit terminal\n' "$(date -Ins)"
  sleep 30
done
for required in "${patches}/COMPLETE.txt" "${patches}/index.csv" \
  "${patches}/AUDIT.json" "${coupled_summary}" "${decoupled_summary}"; do
  [[ -s "${required}" ]]
done

ensure_source_checkpoint() {
  local fold="$1" source_group="$2" tag="$3"
  local run selection_dir checkpoint prediction stem epoch audit
  local checkpoint_epoch prediction_epoch
  local -a audits=() audit_args=()
  run="$(fold_dir "${fold}")"
  checkpoint="${run}/checkpoint_best_${tag}_segmentation.pt"
  prediction="${run}/validation_predictions_best_${tag}_segmentation.npz"
  selection_dir="${output}/source_checkpoint_selection/fold${fold}"
  mkdir -p "${selection_dir}"
  shopt -s nullglob
  for checkpoint_epoch in "${run}"/checkpoint_epoch_*.pt; do
    stem="$(basename "${checkpoint_epoch}" .pt)"
    epoch="${stem##*_}"
    prediction_epoch="${run}/validation_predictions_epoch_${epoch}.npz"
    [[ -s "${prediction_epoch}" ]] || continue
    audit="${selection_dir}/source_epoch_${epoch}.json"
    if [[ ! -s "${audit}" ]]; then
      "${python_bin}" -u "${root}/audit_candidate_refiner_fold_sources.py" \
        --index "${patches}/index.csv" --prediction "${prediction_epoch}" \
        --checkpoint "${checkpoint_epoch}" --folds 5 --fold "${fold}" \
        --seed 20260823 --epoch "$((10#${epoch}))" --output "${audit}" \
        > "${selection_dir}/source_epoch_${epoch}.log" 2>&1
    fi
    audits+=("${audit}")
  done
  shopt -u nullglob
  [[ "${#audits[@]}" -gt 0 ]] || {
    echo "no frozen scored artifacts for ${source_group} checkpoint selection: fold ${fold}" >&2
    return 3
  }
  for audit in "${audits[@]}"; do audit_args+=(--audit "${audit}"); done
  "${python_bin}" -u "${root}/select_candidate_refiner_source_checkpoint.py" \
    "${audit_args[@]}" --source-group "${source_group}" \
    --checkpoint-output "${checkpoint}" --prediction-output "${prediction}" \
    --output "${selection_dir}/SELECTION_${tag}.json" \
    > "${selection_dir}/select_${tag}.log" 2>&1
}
for fold in 0 1 2 3 4; do
  ensure_source_checkpoint "${fold}" currentgf currentgf
  ensure_source_checkpoint "${fold}" epoch105 epoch105
  ensure_source_checkpoint "${fold}" autopet_weak autopet
  ensure_source_checkpoint "${fold}" lndb_weak lndb
  ensure_source_checkpoint "${fold}" lung_cells lung
done

full_source_route_args=(
  --source-route epoch105=epoch105
  --source-route currentgf=currentgf
  --source-route autopet_weak=autopet
  --source-route lndb_weak=lndb
)
for cell in 0 1 2 3 4 5 6 7; do
  full_source_route_args+=(--source-route "lung_cell${cell}=lung")
done
source_group_args=(
  --source-group mature=epoch105,currentgf
  --source-group weak_sources=autopet_weak,lndb_weak
  --source-group lung_cells=lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7
)

full_summary_args=()
for fold in 0 1 2 3 4; do
  run="$(fold_dir "${fold}")"
  fold_output="${output}/fold${fold}"
  mkdir -p "${fold_output}"
  class_checkpoint="${run}/checkpoint_best_classification.pt"
  current_checkpoint="${run}/checkpoint_best_currentgf_segmentation.pt"
  epoch_checkpoint="${run}/checkpoint_best_epoch105_segmentation.pt"
  autopet_checkpoint="${run}/checkpoint_best_autopet_segmentation.pt"
  lndb_checkpoint="${run}/checkpoint_best_lndb_segmentation.pt"
  lung_checkpoint="${run}/checkpoint_best_lung_segmentation.pt"
  class_predictions="${run}/validation_predictions_best_classification.npz"
  current_predictions="${run}/validation_predictions_best_currentgf_segmentation.npz"
  epoch_predictions="${run}/validation_predictions_best_epoch105_segmentation.npz"
  autopet_predictions="${run}/validation_predictions_best_autopet_segmentation.npz"
  lndb_predictions="${run}/validation_predictions_best_lndb_segmentation.npz"
  lung_predictions="${run}/validation_predictions_best_lung_segmentation.npz"
  composed="${fold_output}/validation_predictions_full_source_route.npz"
  audit="${fold_output}/FULL_SOURCE_COMPOSITION_AUDIT.json"
  for required in "${class_checkpoint}" "${current_checkpoint}" \
    "${epoch_checkpoint}" "${autopet_checkpoint}" "${lndb_checkpoint}" \
    "${lung_checkpoint}" "${class_predictions}" "${current_predictions}" \
    "${epoch_predictions}" "${autopet_predictions}" "${lndb_predictions}" \
    "${lung_predictions}"; do
    [[ -s "${required}" ]]
  done
  "${python_bin}" -u "${root}/compose_candidate_refiner_source_routed_validation_predictions.py" \
    --index "${patches}/index.csv" \
    --classification "${class_predictions}" \
    --classification-checkpoint "${class_checkpoint}" \
    --contour "currentgf=${current_predictions}" \
    --contour "epoch105=${epoch_predictions}" \
    --contour "autopet=${autopet_predictions}" \
    --contour "lndb=${lndb_predictions}" \
    --contour "lung=${lung_predictions}" \
    --contour-checkpoint "currentgf=${current_checkpoint}" \
    --contour-checkpoint "epoch105=${epoch_checkpoint}" \
    --contour-checkpoint "autopet=${autopet_checkpoint}" \
    --contour-checkpoint "lndb=${lndb_checkpoint}" \
    --contour-checkpoint "lung=${lung_checkpoint}" \
    "${full_source_route_args[@]}" --output "${composed}" --audit "${audit}" \
    > "${fold_output}/compose_full_route.log" 2>&1
  full_summary_args+=(--checkpoint "${class_checkpoint}")
  full_summary_args+=(--prediction "${composed}")
  full_summary_args+=(--source-composition-audit "${audit}")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${full_summary_args[@]}" \
  --output "${full_route_summary}" "${source_group_args[@]}" \
  > "${output}/summarize_full_route.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_partial_source_routes.py" \
  --baseline "${decoupled_summary}" --candidate "${full_route_summary}" \
  --minimum-case-dice-gain 0.002 --minimum-cohort-dice-gain 0.001 \
  --expected-cases 707 --output "${partial_route_selection}" \
  > "${output}/select_partial_routes.log" 2>&1

# Compose the deployable hybrid: independently qualified source routes use
# their source-selected epoch, while every other source uses the global B32
# segmentation checkpoint.  The emitted specification contains every and only
# the routes actually used, so provenance validation remains fail-closed.
route_specs_text=$("${python_bin}" "${root}/candidate_refiner_route_selection.py" \
  --selection "${partial_route_selection}") || {
    echo "invalid partial source-route selection" >&2; exit 3;
  }
mapfile -t route_specs <<< "${route_specs_text}"
hybrid_summary_args=()
for fold in 0 1 2 3 4; do
  run="$(fold_dir "${fold}")"
  fold_output="${output}/fold${fold}"
  class_checkpoint="${run}/checkpoint_best_classification.pt"
  class_predictions="${run}/validation_predictions_best_classification.npz"
  contour_args=(); contour_checkpoint_args=(); hybrid_source_args=()
  for specification in "${route_specs[@]:1}"; do
    IFS=$'\t' read -r kind name value family <<< "${specification}"
    if [[ "${kind}" == ROUTE ]]; then
      [[ "${family}" == primary ]] || { echo "unexpected route family: ${family}" >&2; exit 3; }
      checkpoint="${run}/${value}"
      case "${name}" in
        global) prediction="${run}/validation_predictions_best_segmentation.npz";;
        currentgf) prediction="${run}/validation_predictions_best_currentgf_segmentation.npz";;
        epoch105) prediction="${run}/validation_predictions_best_epoch105_segmentation.npz";;
        autopet) prediction="${run}/validation_predictions_best_autopet_segmentation.npz";;
        lndb) prediction="${run}/validation_predictions_best_lndb_segmentation.npz";;
        lung) prediction="${run}/validation_predictions_best_lung_segmentation.npz";;
        *) echo "unexpected partial route: ${name}" >&2; exit 3;;
      esac
      [[ -s "${checkpoint}" && -s "${prediction}" ]] || {
        echo "missing hybrid route artifact: ${name}" >&2; exit 3;
      }
      contour_args+=(--contour "${name}=${prediction}")
      contour_checkpoint_args+=(--contour-checkpoint "${name}=${checkpoint}")
    elif [[ "${kind}" == SOURCE ]]; then
      hybrid_source_args+=(--source-route "${name}=${value}")
    else
      echo "invalid partial route specification: ${specification}" >&2; exit 3
    fi
  done
  composed="${fold_output}/validation_predictions_source_routed.npz"
  audit="${fold_output}/SOURCE_COMPOSITION_AUDIT.json"
  "${python_bin}" -u "${root}/compose_candidate_refiner_source_routed_validation_predictions.py" \
    --index "${patches}/index.csv" --classification "${class_predictions}" \
    --classification-checkpoint "${class_checkpoint}" \
    "${contour_args[@]}" "${contour_checkpoint_args[@]}" \
    "${hybrid_source_args[@]}" --output "${composed}" --audit "${audit}" \
    > "${fold_output}/compose_hybrid.log" 2>&1
  hybrid_summary_args+=(--checkpoint "${class_checkpoint}")
  hybrid_summary_args+=(--prediction "${composed}")
  hybrid_summary_args+=(--source-composition-audit "${audit}")
done
"${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
  --index "${patches}/index.csv" "${hybrid_summary_args[@]}" \
  --output "${summary}" "${source_group_args[@]}" \
  > "${output}/summarize_hybrid.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_decoupled_oof.py" \
  --baseline "${coupled_summary}" --candidate "${summary}" \
  --output "${coupled_gate}" > "${output}/select_vs_coupled.log" 2>&1
"${python_bin}" -u "${root}/select_candidate_refiner_source_routed_oof.py" \
  --baseline "${decoupled_summary}" --candidate "${summary}" \
  --minimum-dice-gain 0.001 \
  --output "${route_gate}" > "${output}/select_source_route.log" 2>&1
"${python_bin}" - "${training_complete}" "${summary}" "${coupled_gate}" \
  "${route_gate}" "${partial_route_selection}" "${full_route_summary}" \
  "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
upstream,summary,coupled_gate,route_gate,partial,full_summary,output=map(Path,sys.argv[1:])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
coupled=json.loads(coupled_gate.read_text()); route=json.loads(route_gate.read_text())
partial_payload=json.loads(partial.read_text())
qualified=(coupled["decision"]=="QUALIFY_DECOUPLED_FULL_VOLUME" and
           route["decision"]=="QUALIFY_SOURCE_ROUTED_FULL_VOLUME" and
           partial_payload["decision"]=="QUALIFY_GROUP_SELECTED_SOURCE_ROUTE")
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE",
 "decision":"QUALIFY_SOURCE_ROUTED_FULL_VOLUME" if qualified else "KEEP_GLOBAL_CONTOUR",
 "upstream":str(upstream.resolve()),"summary":str(summary.resolve()),
 "summary_sha256":sha(summary),"coupled_gate":coupled,"source_route_gate":route,
 "partial_route_selection":str(partial.resolve()),
 "partial_route_selection_sha256":sha(partial),
 "partial_route_payload":partial_payload,
 "full_route_summary":str(full_summary.resolve()),
 "full_route_summary_sha256":sha(full_summary),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output); print(json.dumps(payload,indent=2))
PY

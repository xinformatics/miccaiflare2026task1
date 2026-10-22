#!/usr/bin/env bash
set -euo pipefail

# Accelerate the already-qualified exact-707 frozen composite on GPU 0 while
# its authoritative controller runs folds 1/3 on GPU 1. This script creates
# only the same hash-validated inference caches consumed by that controller.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/analysis/focus_composite_fullvolume_oof_20260824"
primary="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
composite="${root}/analysis/focus_composite_routes_20260824/DECISION.json"
runs="${root}/runs"
lock="${runs}/accelerate_focus_composite_gpu0_20260825.lock"

mkdir -p "${runs}" "${output}"
exec 9>"${lock}"
flock -n 9 || { echo "composite GPU-0 accelerator already active" >&2; exit 1; }

for required in "${primary}/FULL_VOLUME_OOF_COMPLETE.json" "${composite}"; do
  [[ -s "${required}" ]] || { echo "missing prerequisite: ${required}" >&2; exit 2; }
done
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${composite}")
[[ "${decision}" == QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME ]] \
  || { echo "composite route is not qualified" >&2; exit 2; }

wait_for_gpu0() {
  local used util
  while :; do
    read -r used util < <(nvidia-smi --id=0 \
      --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
      | tr -d ' ' | tr ',' ' ')
    if [[ "${used}" =~ ^[0-9]+$ && "${util}" =~ ^[0-9]+$ \
        && "${used}" -le 4096 && "${util}" -le 20 ]]; then
      printf '%s GPU 0 admitted: memory=%s MiB utilization=%s%%\n' \
        "$(date -Ins)" "${used}" "${util}"
      return 0
    fi
    printf '%s waiting for conservatively available GPU 0: memory=%s MiB utilization=%s%%\n' \
      "$(date -Ins)" "${used}" "${util}"
    sleep 30
  done
}

build_args() {
  local fold="$1" specification kind name value expected_hash observed alias
  local small="${runs}/candidate_refiner_v4_cell707_b32_t05_fold${fold}_20260823/checkpoint_best_segmentation.pt"
  [[ -s "${small}" ]] || return 2
  ROUTE_ARGS=(--candidate-contour-route "b32=${small}")
  VALIDATOR_ARGS=(--focus-routed-input-mode soft_lung_seed)
  TTA_ARGS=()
  mapfile -t specifications < <("${python_bin}" - "${composite}" "${fold}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); fold=int(sys.argv[2])
row={int(item["fold"]):item for item in p["fold_records"]}[fold]
for name,route in sorted(row["routes"].items()):
 if name!="b32": print("FOCUS\t%s\t%s\t%s"%(name,route["checkpoint"],route["checkpoint_sha256"]))
for source,route in sorted(p["source_to_route"].items()): print(f"SOURCE\t{source}\t{route}\t-")
PY
  )
  for specification in "${specifications[@]}"; do
    IFS=$'\t' read -r kind name value expected_hash <<< "${specification}"
    if [[ "${kind}" == FOCUS ]]; then
      [[ -s "${value}" ]] || return 2
      mkdir -p "${output}/checkpoint_aliases/fold${fold}"
      alias="${output}/checkpoint_aliases/fold${fold}/${name}.pt"
      [[ -e "${alias}" ]] || ln "${value}" "${alias}"
      observed=$(sha256sum "${alias}" | awk '{print $1}')
      [[ "${observed}" == "${expected_hash}" ]] || return 2
      ROUTE_ARGS+=(--focus-contour-route "${name}=${alias}")
      [[ "${name}" == *_tta_xy ]] && TTA_ARGS+=(--tta-route "${name}")
    elif [[ "${kind}" == SOURCE ]]; then
      ROUTE_ARGS+=(--source-contour-route "${name}=${value}")
    else
      return 2
    fi
  done
  SMALL="${small}"
}

valid_cache() {
  local work="$1"
  "${python_bin}" "${root}/validate_candidate_refiner_inference_marker.py" \
    --marker "${work}/INFERENCE_COMPLETE.json" --checkpoint "${SMALL}" \
    "${ROUTE_ARGS[@]}" "${VALIDATOR_ARGS[@]}" >/dev/null 2>&1 \
    && "${python_bin}" "${root}/validate_focus_tta_xy_marker.py" \
      --work "${work}" "${TTA_ARGS[@]}" >/dev/null 2>&1
}

for fold in 0 2 4; do
  work="${output}/fold${fold}/work_bbox24"
  [[ -s "${work}/manifest.json" && -s "${work}/PREPARE_COMPLETE.txt" ]] \
    || { echo "missing prepared fold ${fold} work cache" >&2; exit 2; }
  build_args "${fold}"
  if valid_cache "${work}"; then
    printf '%s fold %s cache already complete\n' "$(date -Ins)" "${fold}"
    continue
  fi
  wait_for_gpu0
  CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR="${output}/matplotlib" \
    "${python_bin}" -u "${root}/apply_candidate_refiner_focus_tta_xy.py" infer \
      --work "${work}" --checkpoint "${SMALL}" "${ROUTE_ARGS[@]}" \
      --batch-size 4 --workers 2 --resume \
      > "${output}/fold${fold}.accelerator_gpu0.infer.log" 2>&1
  valid_cache "${work}" || { echo "invalid accelerated fold ${fold} cache" >&2; exit 2; }
  printf '%s fold %s accelerated cache complete\n' "$(date -Ins)" "${fold}"
done

date -Ins > "${output}/GPU0_ACCELERATOR_COMPLETE.txt"

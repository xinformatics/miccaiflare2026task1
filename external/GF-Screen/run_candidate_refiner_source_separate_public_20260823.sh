#!/usr/bin/env bash
set -euo pipefail

# Fixed public qualification for the source-separated transfer arm. Thresholds
# and checkpoint family come exclusively from held-out 210-case OOF. This
# script cannot read hidden data, package a ZIP, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
transfer="${root}/analysis/candidate_refiner_v3_source_separate_transfer_oof_20260823"
transfer_complete="${transfer}/TRANSFER_OOF_COMPLETE.json"
calibration="${transfer}/deployment_calibration.json"
selection="${root}/analysis/candidate_refiner_v2_training_20260822/checkpoint_selection.json"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
epoch105="${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions"
currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/candidate_refiner_v3_source_separate_20260823"
work="${output}/work_all_bbox24"
complete="${output}/PUBLIC_ONLY_COMPLETE.txt"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_source_separate_public_20260823.lock"
flock -n 9 || { echo "source-separated public controller already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${transfer_complete}" ]]; do
  printf '%s waiting for source-separated transfer OOF\n' "$(date -Ins)"
  sleep 30
done
decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${transfer_complete}")
if [[ "${decision}" != QUALIFY_FOR_FIXED_PUBLIC ]]; then
  printf '%s %s\n' "$(date -Ins)" "${decision}" > "${output}/OOF_DID_NOT_QUALIFY.txt"
  date -Ins > "${complete}"
  exit 0
fi
for required in "${calibration}" "${selection}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${epoch105}" "${currentgf}" "${base}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done

if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${images}" --proposal "epoch105=${epoch105}" \
    --proposal "currentgf=${currentgf}" --component-policy source_separate \
    --base "${base}" --work "${work}" --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${output}/prepare.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
  > "${output}/audit_cache.log" 2>&1

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
checkpoint_args=(); checkpoint_paths=()
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  checkpoint_args+=(--checkpoint "${checkpoint}"); checkpoint_paths+=("${checkpoint}")
done
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
needs_inference=0
"${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint_paths[@]}" <<'PY' || needs_inference=1
import json,sys
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(path).resolve()) for path in sys.argv[2:]]
if not marker.is_file() or json.loads(marker.read_text()).get("checkpoints",[])!=expected: raise SystemExit(1)
PY
if [[ "${needs_inference}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${output}/infer.log" 2>&1
fi

classification=$("${python_bin}" -c 'import json,sys;print(f"{float(json.load(open(sys.argv[1]))[\"classification_threshold\"]):.2f}")' "${calibration}")
segmentation=$("${python_bin}" -c 'import json,sys;print(f"{float(json.load(open(sys.argv[1]))[\"segmentation_threshold\"]):.2f}")' "${calibration}")
deployment_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("deployment_mode","refined"))' "${calibration}")
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"
for deployment in anchor standalone; do
  tag="ss_${deployment}_${deployment_mode}_c${classification/./p}_s${segmentation/./p}"
  prediction="${output}/${tag}"; empty_arg=()
  [[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" materialize \
    --work "${work}" --base "${base}" --output "${prediction}" "${empty_arg[@]}" \
    --classification-threshold "${classification}" --segmentation-threshold "${segmentation}" \
    --mode "${deployment_mode}" --min-seed-coverage 0 --min-refined-overlap 0 \
    --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
    > "${prediction}.materialize.log" 2>&1
  metrics="${prediction}.metrics.csv"; score_log="${prediction}.score.log"; robust="${prediction}.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
    --gt "${labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" --anchor "${anchor_metrics}" \
    --candidate "${metrics}" --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  gate=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation}" "${deployment_mode}" "${deployment}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${gate}" >> "${output}/summary.tsv"
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr; } > "${output}/summary_by_dsc.tsv"
date -Ins > "${complete}"
cat "${output}/summary_by_dsc.tsv"

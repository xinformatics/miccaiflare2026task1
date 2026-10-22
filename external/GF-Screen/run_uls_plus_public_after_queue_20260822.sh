#!/usr/bin/env bash
set -euo pipefail

# Public-only evaluation of the released ULS+ interactive lesion segmenter as
# an automatic component refiner. Rich16 Focus=0.20 components provide interior
# click points; no public label is used to prepare or infer VOIs. The small
# confirmation grid is evaluated only after the current submission archives
# are safely preserved. This controller cannot infer hidden cases or make ZIPs.

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
upstream_marker="${root}/public_sweeps/gf_hybrid_cross_deployed_20260822/SAVED_BACKUPS_COMPLETE.txt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
proposals="${root}/public_sweeps/gf_rich16_comp281_low_thresholds_20260821/inference/lesion_0.20"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${root}/public_sweeps/uls_plus_rich20_refiner_20260822"
work="${output}/work"
adapter="${root}/run_uls_plus_component_refiner.py"

for required in "${upstream_marker}" "${anchor_metrics}" "${adapter}" \
    "${project}/external/ULS-Plus/weights/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres/fold_all/checkpoint_best.pth"; do
  if [[ "${required}" == "${upstream_marker}" ]]; then
    continue
  fi
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${proposals}" "${base}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 2
  }
done

mkdir -p "${output}" "${work}"
exec 9>"${output}.lock"
flock -n 9 || { echo "ULS+ public refiner is already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0

while [[ ! -s "${upstream_marker}" ]]; do
  printf '%s waiting for hybrid submission archive completion\n' "$(date -Ins)"
  sleep 30
done
while :; do
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]]; then
    break
  fi
  printf '%s waiting for GPU 1: %s MiB\n' "$(date -Ins)" "${used1}"
  sleep 30
done

export MPLCONFIGDIR="${work}/matplotlib"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
common=(
  --images "${images}"
  --proposals "${proposals}"
  --work-dir "${work}"
  --output "${output}/placeholder"
  --min-seed-mm3 20
  --max-seed-mm3 200000
  --max-components-per-case 128
  --preprocessing-workers 2
  --export-workers 2
)

if [[ ! -s "${work}/manifest.json" ]]; then
  "${python_bin}" -u "${adapter}" --stage prepare "${common[@]}" \
    > "${output}/prepare.log" 2>&1
fi
manifest_cases=$("${python_bin}" -c \
  'import json,sys; print(len(set(x["case"] for x in json.load(open(sys.argv[1])))))' \
  "${work}/manifest.json")
[[ "${manifest_cases}" -eq 50 ]] || { echo "manifest has ${manifest_cases}/50 cases" >&2; exit 2; }

if [[ ! -s "${work}/INFERENCE_COMPLETE.txt" ]]; then
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${adapter}" --stage infer "${common[@]}" \
    --device cuda > "${output}/inference.log" 2>&1
  predicted=$(find "${work}/predicted_crops" -maxdepth 1 -name '*.nii.gz' | wc -l)
  expected=$("${python_bin}" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "${work}/manifest.json")
  [[ "${predicted}" -eq "${expected}" ]] || {
    echo "ULS+ predictions incomplete: ${predicted}/${expected}" >&2
    exit 2
  }
  date -Ins > "${work}/INFERENCE_COMPLETE.txt"
fi

printf 'variant\tmode\tseed_coverage\trefined_overlap\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' \
  > "${output}/summary.tsv"
for mode in only union; do
  for seed in 0.05 0.25 0.50; do
    for overlap in 0.05 0.25 0.50; do
      seed_tag=${seed/./p}
      overlap_tag=${overlap/./p}
      tag="${mode}_s${seed_tag}_o${overlap_tag}"
      candidate="${output}/${tag}"
      materialize_args=(
        --stage materialize
        "${common[@]}"
        --output "${candidate}"
        --min-seed-coverage "${seed}"
        --min-refined-overlap "${overlap}"
        --min-volume-ratio 0.1
        --max-volume-ratio 10
      )
      [[ "${mode}" == union ]] && materialize_args+=(--base "${base}")
      "${python_bin}" -u "${adapter}" "${materialize_args[@]}" \
        > "${candidate}.materialize.log" 2>&1
      [[ $(find "${candidate}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
      metrics="${candidate}.metrics.csv"
      score_log="${candidate}.score.log"
      robust="${candidate}.robust.json"
      "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
        --pred "${candidate}" --gt "${labels}" --csv "${metrics}" --workers 4 \
        > "${score_log}" 2>&1
      "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
        --anchor "${anchor_metrics}" --candidate "${metrics}" \
        --anchor-name champion_0p7006 --candidate-name "uls_plus_${tag}" \
        --output "${robust}" > "${candidate}.robust.log"
      dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
      nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
      combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f", (d+n)/2}')
      decision=$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${tag}" "${mode}" "${seed}" "${overlap}" "${dsc}" "${nsd}" \
        "${combined}" "${metrics}" "${robust}" "${decision}" >> "${output}/summary.tsv"
    done
  done
done

{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

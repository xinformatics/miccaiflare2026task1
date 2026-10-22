#!/usr/bin/env bash
set -euo pipefail

# Public-only, case-grouped diagnostic for a frozen Glance appearance
# representation. It never trains on hidden data, packages a ZIP, or uploads.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
features="${root}/analysis/component_glance_embedding_cv_20260822/features.npz"
features_manifest="${root}/analysis/component_glance_embedding_cv_20260822/features.json"
anatomy="${root}/public_sweeps/totalseg_fast_public_20260822"
anatomy_complete="${root}/analysis/component_classifier_anatomy_cv_20260822/COMPLETE.txt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
candidate="${root}/public_sweeps/gf_rich16_comp281_low_thresholds_20260821/inference/lesion_0.20"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
basic="${root}/analysis/component_deep_fpr_cv_20260822"
with_anatomy="${root}/analysis/component_deep_fpr_anatomy_cv_20260822"

mkdir -p "${basic}" "${with_anatomy}"
exec 9>"${root}/analysis/component_deep_fpr_cv_20260822.lock"
flock -n 9 || { echo "deep component CV controller already active" >&2; exit 1; }

while [[ ! -s "${features}" || ! -s "${features_manifest}" ]]; do
  printf '%s waiting for frozen Glance component embeddings\n' "$(date -Ins)"
  sleep 10
done

common=(
  --images "${images}" --labels "${labels}" --base "${base}"
  --candidate "${candidate}"
  --support "seed40=${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.40"
  --support "seed55=${root}/public_sweeps/gf_rich16_comp281_threshold_bracket_20260821/inference/lesion_0.55"
  --support "richfull=${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821/gf_rich16_full_comp050_ep280_lr1e5_3ep_20260821/checkpoint_epoch_0281/inference/lesion_0.60"
  --support "whole_prompt=${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
  --support "tumor_prompt=${project}/external/BiomedParse/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
  --support "nnunet=${root}/public_sweeps/complementary_guided_union_20260821/nnunet_new2k_s25_sd10"
  --component-embeddings "${features}" --seed 20260822
)

run_analysis() {
  local output="$1" name="$2"; shift 2
  [[ -s "${output}/COMPLETE.txt" ]] && return
  MPLCONFIGDIR=/tmp/gf_fpr_mpl OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    "${python_bin}" -u "${root}/analyze_component_classifier_cv.py" \
      "${common[@]}" "$@" --output-dir "${output}" \
      > "${output}/run.log" 2>&1
  [[ $(find "${output}/predictions_oof" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${output}/predictions_oof" --gt "${labels}" \
    --csv "${output}/metrics.csv" --workers 4 > "${output}/score.log" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${output}/metrics.csv" \
    --anchor-name champion_0p7006 --candidate-name "${name}" \
    --output "${output}/robust.json" > "${output}/robust.log"
  date -Ins > "${output}/COMPLETE.txt"
}

run_analysis "${basic}" component_deep_grouped_oof \
  --component-table "${root}/analysis/component_classifier_cv_20260822/components_oof.csv"

while [[ ! -s "${anatomy_complete}" ]]; do
  printf '%s waiting for anatomy diagnostic before deep+anatomy CV\n' "$(date -Ins)"
  sleep 10
done
run_analysis "${with_anatomy}" component_deep_anatomy_grouped_oof \
  --anatomy-dir "${anatomy}" \
  --component-table "${root}/analysis/component_classifier_anatomy_cv_20260822/components_oof.csv"

"${python_bin}" - "${basic}/report.json" "${with_anatomy}/report.json" \
  "${root}/analysis/component_deep_fpr_cv_20260822/comparison.json" <<'PY'
import json
import sys
from pathlib import Path

basic_path, anatomy_path, output_path = map(Path, sys.argv[1:])
basic = json.loads(basic_path.read_text(encoding="utf-8"))
anatomy = json.loads(anatomy_path.read_text(encoding="utf-8"))
payload = {
    "warning": "Public-only grouped CV; not a deployable hidden classifier.",
    "basic_deep_oof_DSC": basic["oof_macro_DSC"],
    "basic_deep_gain": basic["oof_macro_DSC_gain"],
    "deep_anatomy_oof_DSC": anatomy["oof_macro_DSC"],
    "deep_anatomy_gain": anatomy["oof_macro_DSC_gain"],
    "best": max(
        (("deep", basic["oof_macro_DSC"]), ("deep_anatomy", anatomy["oof_macro_DSC"])),
        key=lambda item: item[1],
    )[0],
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
date -Ins > "${root}/analysis/component_deep_fpr_cv_20260822/ALL_COMPLETE.txt"

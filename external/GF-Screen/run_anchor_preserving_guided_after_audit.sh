#!/usr/bin/env bash
set -euo pipefail

# Preserve every voxel from the verified prompt-confirmed champion and add only
# regions recovered by the best prompt-guided candidate.  This is deliberately
# evaluated as a separate deployment candidate: proposal guidance may improve
# recall, but it is not allowed to erase an already accepted champion lesion.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${biomed}/public_sweeps/automatic_text_proposals_20260820"
selected="${audit}/prompted_ranking/selected.tsv"
anchor_public="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/predictions/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf"
anchor_hidden="${audit}/hidden_prompt_or_e050_confirmation/candidates/prompt_or_e050_confirm_d0_norescue"
anchor_metrics="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
rescue_root="${biomed}/public_sweeps/component_confidence_rescue_20260820"
rescue_public="${rescue_root}/focus075_tumor_mean_e010_mm1000_10000"
rescue_hidden="${rescue_root}/hidden_focus075_tumor_mean_e010_mm1000_10000/predictions"
labels="${project}/validation/Validation-Public-Labels"
hidden_images="${project}/validation/Validation-Hidden-Images"
output="${root}/public_sweeps/anchor_preserving_guided_20260821"

[[ -s "${audit}/AUDIT_COMPLETE.json" && -s "${selected}" ]] || {
  echo "guided audit is not complete: ${audit}" >&2
  exit 1
}
for directory in "${anchor_public}" "${anchor_hidden}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -gt 0 ]] || {
    echo "missing anchor masks: ${directory}" >&2
    exit 1
  }
done

selection="$(${python_bin} - "${selected}" <<'PY'
import csv, pathlib, sys
with pathlib.Path(sys.argv[1]).open(newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if len(rows) != 1:
    raise SystemExit(f"expected one selected row, found {len(rows)}")
row = rows[0]
print(row["score_tag"], row["prompt_predictions"], sep="\t")
PY
)"
IFS=$'\t' read -r score_tag guided_public <<< "${selection}"
if [[ ! -d "${guided_public}" ]]; then
  if [[ -d "${biomed}/${guided_public}" ]]; then
    guided_public="${biomed}/${guided_public}"
  elif [[ -d "${root}/${guided_public}" ]]; then
    guided_public="${root}/${guided_public}"
  else
    echo "missing selected public predictions: ${guided_public}" >&2
    exit 1
  fi
fi
candidate_tag="biomedparse_${score_tag}"
guided_hidden="${audit}/hidden_${candidate_tag}/prompt_confirmed"
for pair in "${guided_public}:50" "${guided_hidden}:100" \
            "${anchor_public}:50" "${anchor_hidden}:100" \
            "${rescue_public}:50" "${rescue_hidden}:100"; do
  directory="${pair%:*}"
  expected="${pair##*:}"
  actual=$(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l)
  [[ "${actual}" -eq "${expected}" ]] || {
    echo "${directory}: expected ${expected} masks, found ${actual}" >&2
    exit 1
  }
done

mkdir -p "${output}/public" "${output}/hidden" "${project}/submissions"
"${python_bin}" "${root}/fuse_and_filter_masks.py" \
  --pred-a "${anchor_public}" --pred-b "${guided_public}" \
  --fusion union --min-mm3 0 --workers 8 --output "${output}/public" \
  > "${output}/public.union.log" 2>&1
"${python_bin}" "${root}/evaluate_public_parallel.py" \
  --pred "${output}/public" --gt "${labels}" \
  --csv "${output}/anchor_guided_union.metrics.csv" --workers 16 \
  > "${output}/anchor_guided_union.score.log" 2>&1
"${python_bin}" "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${output}/anchor_guided_union.metrics.csv" \
  --anchor-name prompt_champion --candidate-name anchor_guided_union \
  --output "${output}/anchor_guided_union.robust.json" \
  > "${output}/anchor_guided_union.robust.log"

"${python_bin}" "${root}/fuse_and_filter_masks.py" \
  --pred-a "${anchor_hidden}" --pred-b "${guided_hidden}" \
  --fusion union --min-mm3 0 --workers 8 --output "${output}/hidden" \
  > "${output}/hidden.union.log" 2>&1
submission="${project}/submissions/gf_anchor_guided_add.zip"
"${python_bin}" "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/hidden" \
  --output "${submission}" --expected-count 100 \
  > "${output}/package.log" 2>&1
unzip -t "${submission}" > "${output}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"

# The user's 0.6990/0.6286 confidence-rescue model is a near-tie parent with a
# different error set.  Preserve it independently and add the identical guided
# regions, allowing Codabench to resolve which parent transfers better.
mkdir -p "${output}/public_rescue" "${output}/hidden_rescue"
"${python_bin}" "${root}/fuse_and_filter_masks.py" \
  --pred-a "${rescue_public}" --pred-b "${guided_public}" \
  --fusion union --min-mm3 0 --workers 8 --output "${output}/public_rescue" \
  > "${output}/public_rescue.union.log" 2>&1
"${python_bin}" "${root}/evaluate_public_parallel.py" \
  --pred "${output}/public_rescue" --gt "${labels}" \
  --csv "${output}/rescue_guided_union.metrics.csv" --workers 16 \
  > "${output}/rescue_guided_union.score.log" 2>&1
"${python_bin}" "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${output}/rescue_guided_union.metrics.csv" \
  --anchor-name prompt_champion --candidate-name rescue_guided_union \
  --output "${output}/rescue_guided_union.robust.json" \
  > "${output}/rescue_guided_union.robust.log"
"${python_bin}" "${root}/fuse_and_filter_masks.py" \
  --pred-a "${rescue_hidden}" --pred-b "${guided_hidden}" \
  --fusion union --min-mm3 0 --workers 8 --output "${output}/hidden_rescue" \
  > "${output}/hidden_rescue.union.log" 2>&1
rescue_submission="${project}/submissions/gf_rescue_guided_add.zip"
"${python_bin}" "${project}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${output}/hidden_rescue" \
  --output "${rescue_submission}" --expected-count 100 \
  > "${output}/package_rescue.log" 2>&1
unzip -t "${rescue_submission}" > "${output}/zip_test_rescue.log"
sha256sum "${rescue_submission}" > "${rescue_submission}.sha256"

printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\t%s\t%s\t%s\t%s\n%s\t%s\t%s\t%s\t%s\n' \
  anchor_guided_additions anchor_preserving_guided \
  "${output}/anchor_guided_union.metrics.csv" "${submission}" \
  "${output}/anchor_guided_union.robust.json" \
  rescue_guided_additions rescue_preserving_guided \
  "${output}/rescue_guided_union.metrics.csv" "${rescue_submission}" \
  "${output}/rescue_guided_union.robust.json" \
  > "${output}/submission_candidate.tsv"
printf '%s\n%s\n' "${submission}" "${rescue_submission}"

#!/usr/bin/env bash
set -euo pipefail

# Public-only, pretrained-target prompt audit.  Prompt names are taken from
# BiomedParse's shipped target metadata, not selected from public labels.  The
# immutable champion may only receive disconnected components supported by
# both BiomedParse and GF-Screen.  No hidden inference or ZIP is produced.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
inputs="${root}/public_a160_240_inputs"
checkpoint="${gf}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
base="${gf}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
raw_gf="${gf}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
output="${root}/public_sweeps/organ_specific_rescue_20260821"

mkdir -p "${output}"
exec 9>"${output}.lock"
if ! flock -n 9; then
  echo "organ-specific rescue audit is already active" >&2
  exit 1
fi
for directory in "${images}" "${labels}" "${inputs}" "${base}" "${raw_gf}"; do
  expected='*.nii.gz'
  [[ "${directory}" == "${inputs}" ]] && expected='*.npz'
  [[ $(find "${directory}" -maxdepth 1 -name "${expected}" | wc -l) -eq 50 ]] || {
    echo "expected 50 public files: ${directory}" >&2
    exit 1
  }
done

infer_bank() {
  local gpu="$1" bank="$2"; shift 2
  local npz="${output}/${bank}_npz" nifti="${output}/${bank}_nifti"
  local prompt_args=()
  for prompt in "$@"; do prompt_args+=(--prompt "${prompt}"); done
  mkdir -p "${npz}" "${nifti}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
    ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz}" \
    "${prompt_args[@]}" --device cuda --slice-batch-size 8 \
    --existence-threshold 0.50 --continue-prediction \
    > "${output}/${bank}.inference.log" 2>&1
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
    --images "${images}" --predictions "${npz}" --output "${nifti}" \
    --segmentation-key segs_e050 --continue-prediction \
    > "${output}/${bank}.convert.log" 2>&1
  [[ $(find "${nifti}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
}

infer_bank 0 solid_organ \
  "liver tumor" "kidney tumor" "pancreas tumor" "colon tumor" "abdomen tumor" & p0=$!
infer_bank 1 thoracic "lung tumor" "lung nodule" & p1=$!
status=0
wait "${p0}" || status=$?
wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

organ_union="${output}/organ_union_nifti"
mkdir -p "${organ_union}"
"${python_bin}" -u "${gf}/fuse_and_filter_masks.py" \
  --pred-a "${output}/solid_organ_nifti" --pred-b "${output}/thoracic_nifti" \
  --fusion union --min-mm3 0 --workers 4 --output "${organ_union}" \
  > "${output}/organ_union.log" 2>&1

# Standalone scores are diagnostics; they do not select or alter the champion.
"${python_bin}" -u "${gf}/evaluate_public_grid.py" --gt "${labels}" \
  --output-dir "${output}/standalone_metrics" --workers 8 \
  --candidate "solid_organ=${output}/solid_organ_nifti" \
  --candidate "thoracic=${output}/thoracic_nifti" \
  --candidate "organ_union=${organ_union}" \
  > "${output}/standalone_score.log" 2>&1

# Re-run Focus only where organ-specific BiomedParse proposals request crops.
guided_root="${output}/gf_guided_organ_union"
mkdir -p "${guided_root}"
if ! CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${guided_root}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.40,0.50,0.60 \
    --proposal_mask_path "${organ_union}" --proposal_min_voxels 500 \
    --focus_batch_size 16 --workers 2 --skip_existing \
    > "${output}/gf_guided.inference.log" 2>&1; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${gf}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${guided_root}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds 0.40,0.50,0.60 \
    --proposal_mask_path "${organ_union}" --proposal_min_voxels 500 \
    --focus_batch_size 8 --workers 2 --skip_existing \
    >> "${output}/gf_guided.inference.log" 2>&1
fi

screen() {
  local tag="$1" candidate="$2" support="$3"
  "${python_bin}" -u "${gf}/screen_large_supported_additions.py" screen \
    --base "${base}" --candidate "${candidate}" --support "${support}" \
    --gt "${labels}" --output "${output}/screen_${tag}" --workers 3 \
    > "${output}/screen_${tag}.log" 2>&1
}

screen biomed_candidate_gf_support "${organ_union}" "${raw_gf}" & s0=$!
screen gf_candidate_biomed_support "${raw_gf}" "${organ_union}" & s1=$!
screen guided_l040 "${guided_root}/lesion_0.40" "${organ_union}" & s2=$!
screen guided_l050 "${guided_root}/lesion_0.50" "${organ_union}" & s3=$!
screen guided_l060 "${guided_root}/lesion_0.60" "${organ_union}" & s4=$!
status=0
for pid in "${s0}" "${s1}" "${s2}" "${s3}" "${s4}"; do
  wait "${pid}" || status=$?
done
[[ "${status}" -eq 0 ]] || exit "${status}"

printf 'direction\t' > "${output}/summary.tsv"
head -1 "${output}/screen_biomed_candidate_gf_support/summary.tsv" >> "${output}/summary.tsv"
for tag in biomed_candidate_gf_support gf_candidate_biomed_support \
           guided_l040 guided_l050 guided_l060; do
  tail -n +2 "${output}/screen_${tag}/summary.tsv" | sed "s/^/${tag}\t/" \
    >> "${output}/summary.tsv"
done
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k11,11gr -k9,9gr
} > "${output}/summary_by_combined.tsv"
head -21 "${output}/summary_by_combined.tsv"
printf '%s\n' 'PUBLIC-ONLY AUDIT COMPLETE: no hidden masks or ZIP generated.' \
  > "${output}/PUBLIC_ONLY_COMPLETE.txt"

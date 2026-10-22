#!/usr/bin/env bash
set -euo pipefail

# Apply the already validated, exact BiomedParse prompt-OR component verifier
# to the best remote-pseudo-positive GF candidate.  This is deliberately a
# second gate: the active training controller compares unconfirmed GF masks to
# the unconfirmed GF anchor, whereas the deployed local champion includes
# prompt confirmation.  Hidden inference is authorized only if the combined
# candidate also clears the paired prompt-champion gate.

wait_pid="${WAIT_PID:?set WAIT_PID to the remote-pseudo-positive controller PID}"
gpu_wait_pid="${GPU_WAIT_PID:-}"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
gf_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
workspace="$(cd "${gf_root}/../.." && pwd)"
biomed_root="${workspace}/external/BiomedParse"
public_images="${PUBLIC_IMAGES:-${workspace}/validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-${workspace}/validation/Validation-Public-Labels}"
hidden_images="${HIDDEN_IMAGES:-${workspace}/validation/Validation-Hidden-Images}"
sweep="${gf_root}/public_sweeps/gf_remote_pseudo_positive_dual_20260820"
followup="${sweep}/prompt_followup"
public_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

for required in \
  "${sweep}/summary_by_combined.tsv" \
  "${prompt_config}" \
  "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing required file: ${required}" >&2; exit 1; }
done
[[ $(find "${public_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
[[ $(find "${hidden_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}

best_line=$(sed -n '2p' "${sweep}/summary_by_combined.tsv")
IFS=$'\t' read -r best_variant best_tag best_checkpoint best_focus best_min best_dsc best_nsd best_combined <<< "${best_line}"
[[ -n "${best_checkpoint}" ]] || { echo "empty best candidate" >&2; exit 1; }

threshold_dir="lesion_$(printf '%.2f' "${best_focus}")"
score_tag="${best_variant}_${best_tag}_${threshold_dir}_mm${best_min}"
if [[ "${best_min}" -gt 0 ]]; then
  public_base="${sweep}/scored/${score_tag}"
else
  public_base="${sweep}/${best_variant}/${best_tag}/${threshold_dir}"
fi
[[ $(find "${public_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 selected public GF masks in ${public_base}" >&2
  exit 1
}

candidate_tag="${best_variant}_${best_tag}_l${best_focus/./p}_mm${best_min}_prompt_or_e050"
public_prompt="${followup}/public_${candidate_tag}"
mkdir -p "${public_prompt}"
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${public_base}" --support "${public_support}" \
  --config "${prompt_config}" --output "${public_prompt}" \
  > "${followup}/public_materialize.log" 2>&1
[[ $(find "${public_prompt}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 prompt-confirmed public masks" >&2
  exit 1
}

candidate_metrics="${followup}/public_prompt.metrics.csv"
"${python_bin}" -u "${gf_root}/evaluate_public_parallel.py" \
  --pred "${public_prompt}" --gt "${public_labels}" \
  --csv "${candidate_metrics}" --workers 4 \
  > "${followup}/public_prompt.score.log" 2>&1
"${python_bin}" -u "${gf_root}/select_checkpoint_robust.py" \
  --anchor "${prompt_anchor}" --candidate "${candidate_metrics}" \
  --anchor-name prompt_confirmed_champion --candidate-name "${candidate_tag}" \
  --output "${followup}/robust_vs_prompt_champion.json" \
  > "${followup}/robust_vs_prompt_champion.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${followup}/robust_vs_prompt_champion.json")
printf '%s\n' "${best_line}" > "${followup}/selected_base_configuration.tsv"
if [[ "${decision}" != "PROMOTE" ]]; then
  echo "Prompt-confirmed remote-pseudo-positive candidate did not clear the prompt-champion gate."
  exit 0
fi

# The repaired downstream queue starts another GPU stage immediately after the
# remote-pseudo-positive controller exits. Public prompt materialization and
# scoring above are CPU-only and may proceed at once; a promoted candidate's
# hidden inference must wait for the enclosing queue to release both GPUs.
if [[ -n "${gpu_wait_pid}" ]]; then
  while kill -0 "${gpu_wait_pid}" 2>/dev/null; do
    sleep 30
  done
fi

hidden_root="${followup}/hidden_${candidate_tag}"
mkdir -p "${hidden_root}/shard0" "${hidden_root}/shard1" \
  "${hidden_root}/raw" "${hidden_root}/filtered" "${hidden_root}/prompt_confirmed"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES="${shard}" "${python_bin}" -u "${gf_root}/val_GF_FLARE.py" \
    --test_data_path "${hidden_images}" \
    --save_prediction_path "${hidden_root}/shard${shard}" \
    --trained_pth "${gf_root}/${best_checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_threshold "${best_focus}" \
    --focus_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    > "${hidden_root}/shard${shard}.log" 2>&1 &
  if [[ "${shard}" -eq 0 ]]; then hidden_pid0=$!; else hidden_pid1=$!; fi
done
status=0
wait "${hidden_pid0}" || status=$?
wait "${hidden_pid1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"
cp "${hidden_root}/shard0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/shard1"/*.nii.gz "${hidden_root}/raw/"
[[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden GF masks" >&2
  exit 1
}

if [[ "${best_min}" -gt 0 ]]; then
  "${python_bin}" -u "${gf_root}/fuse_and_filter_masks.py" \
    --pred-a "${hidden_root}/raw" --fusion a --min-mm3 "${best_min}" \
    --output "${hidden_root}/filtered" > "${hidden_root}/filter.log" 2>&1
else
  cp "${hidden_root}/raw"/*.nii.gz "${hidden_root}/filtered/"
fi
"${python_bin}" -u "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/filtered" --support "${hidden_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden masks" >&2
  exit 1
}

submission="${workspace}/submissions/gf_${candidate_tag}.zip"
"${python_bin}" -u "${workspace}/nnunet/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100 \
  > "${hidden_root}/package.log" 2>&1
unzip -t "${submission}" > "${hidden_root}/zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
echo "Prompt-confirmed remote-pseudo-positive submission ready: ${submission}"

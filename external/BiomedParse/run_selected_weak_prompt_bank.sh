#!/usr/bin/env bash
set -euo pipefail

# Usage: run_selected_weak_prompt_bank.sh INPUTS IMAGES OUTPUT PLAN BANK EXPECTED GPU
# The plan was frozen on complete training data. This runner never reads labels.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
[[ "$#" -eq 7 ]] || { echo "expected INPUTS IMAGES OUTPUT PLAN BANK EXPECTED GPU" >&2; exit 2; }
inputs="$1"; images="$2"; output="$3"; plan="$4"; bank="$5"; expected="$6"; gpu="$7"
[[ "${bank}" == autopet || "${bank}" == lndb ]] || { echo "invalid bank" >&2; exit 2; }
[[ "${expected}" =~ ^[0-9]+$ && "${expected}" -ge 1 ]] || { echo "invalid expected count" >&2; exit 2; }
[[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid GPU" >&2; exit 2; }
for required in "${inputs}" "${images}" "${plan}"; do
  [[ -e "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 2; }
done

npz="${output}/predictions_npz"
nifti="${output}/predictions_nifti"
complete="${output}/PROMPT_BANK_COMPLETE.json"
mkdir -p "${output}" "${npz}" "${nifti}"

threshold=$("${python_bin}" -c \
  'import json,sys;print(float(json.load(open(sys.argv[1]))["existence_threshold"]))' "${plan}")
threshold_tag=$("${python_bin}" -c \
  'import sys;print(f"{round(100*float(sys.argv[1])):03d}")' "${threshold}")
disable_nms=$("${python_bin}" -c \
  'import json,sys;print(int(bool(json.load(open(sys.argv[1]))["disable_nms"])))' "${plan}")
mapfile -t prompts < <("${python_bin}" - "${plan}" "${bank}" <<'PY'
import json,sys
for prompt in json.load(open(sys.argv[1],encoding="utf-8"))["prompts"][sys.argv[2]]:
    if "\n" in prompt: raise SystemExit("invalid prompt")
    print(prompt)
PY
)
[[ "${#prompts[@]}" -gt 0 ]]
prompt_args=(); for prompt in "${prompts[@]}"; do prompt_args+=(--prompt "${prompt}"); done
nms_args=(); [[ "${disable_nms}" -eq 1 ]] && nms_args+=(--disable-nms)

if [[ $(find "${npz}" -maxdepth 1 -name '*.npz' | wc -l) -ne "${expected}" ]]; then
  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz}" \
      "${prompt_args[@]}" --device cuda --slice-batch-size 8 "${nms_args[@]}" \
      --existence-threshold "${threshold}" --continue-prediction \
      > "${output}/inference.log" 2>&1; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
      ../miccai_inference.py --input-dir "${inputs}" --output-dir "${npz}" \
      "${prompt_args[@]}" --device cuda --slice-batch-size 4 "${nms_args[@]}" \
      --existence-threshold "${threshold}" --continue-prediction \
      >> "${output}/inference.log" 2>&1
  fi
fi
[[ $(find "${npz}" -maxdepth 1 -name '*.npz' | wc -l) -eq "${expected}" ]]

if [[ $(find "${nifti}" -maxdepth 1 -name '*.nii.gz' | wc -l) -ne "${expected}" ]]; then
  "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
    --images "${images}" --predictions "${npz}" --output "${nifti}" \
    --segmentation-key "segs_e${threshold_tag}" --continue-prediction \
    > "${output}/convert.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_selected_weak_prompt_bank.py" \
  --inputs "${inputs}" --images "${images}" --predictions "${npz}" \
  --nifti "${nifti}" --plan "${plan}" --bank "${bank}" \
  --expected "${expected}" --output "${complete}" > "${output}/audit.log" 2>&1
cat "${complete}"

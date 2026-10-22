#!/usr/bin/env bash
set -euo pipefail

# Correctly re-evaluate the late checkpoints of the completed pretrained
# SwinUNETR nnU-Net run. Inline training-time public inference was known to be
# unreliable; post-hoc checkpoint inference through epoch 51 is authoritative.
wait_pid="${WAIT_PID:-866683}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
source "${script_dir}/env.sh"

dataset="Dataset501_PanCancer"
plans="nnUNetPlans"
configuration="3d_fullres_MONAI_SwinUNETR_f48_pretrained"
fold="all"
model_dir="${nnUNet_results}/${dataset}/nnUNetTrainer__${plans}__${configuration}/fold_${fold}"
public_images="${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Images"
public_labels="${NNUNET_PROJECT_ROOT}/validation/Validation-Public-Labels"
hidden_images="${NNUNET_PROJECT_ROOT}/validation/Validation-Hidden-Images"
root="${NNUNET_PROJECT_ROOT}/predictions/flare25_finetune_late_checkpoints"
anchor_csv="${NNUNET_PROJECT_ROOT}/predictions/flare25_finetune_by_epoch/epoch_0050/challenge_metrics.csv"
evaluator="${NNUNET_PROJECT_ROOT}/handoff_bundle/code/evaluate.py"
selector="${NNUNET_PROJECT_ROOT}/external/GF-Screen/select_checkpoint_robust.py"
gf_evaluator="${NNUNET_PROJECT_ROOT}/external/GF-Screen/evaluate_public_parallel.py"
biomed_root="${NNUNET_PROJECT_ROOT}/external/BiomedParse"
prompt_config="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_anchor="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/scored/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.metrics.csv"
public_prompt_support="${biomed_root}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
hidden_prompt_support="${biomed_root}/public_sweeps/automatic_text_proposals_20260820/hidden_prompt_or_e050_confirmation/prompt_or"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done
for required in "${prompt_config}" "${prompt_anchor}"; do
  [[ -f "${required}" ]] || { echo "missing prompt-confirmation asset: ${required}" >&2; exit 1; }
done
[[ $(find "${public_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public prompt-support masks" >&2
  exit 1
}
[[ $(find "${hidden_prompt_support}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 hidden prompt-support masks" >&2
  exit 1
}
mkdir -p "${root}" "${NNUNET_PROJECT_ROOT}/submissions"

predict_public() {
  local gpu="$1"
  local epoch="$2"
  local checkpoint="checkpoint_epoch_${epoch}.pth"
  local output="${root}/epoch_${epoch}"
  if [[ ! -f "${model_dir}/${checkpoint}" ]]; then
    echo "Missing checkpoint: ${model_dir}/${checkpoint}" >&2
    return 1
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" nnUNetv2_predict \
    -i "${public_images}" -o "${output}/predictions" -d 501 \
    -tr nnUNetTrainer -p "${plans}" -c "${configuration}" -f "${fold}" \
    -chk "${checkpoint}" -npp 2 -nps 2 --disable_tta --continue_prediction \
    > "${output}/inference.log" 2>&1
  "${python_bin}" "${evaluator}" \
    --pred "${output}/predictions" --gt "${public_labels}" \
    --csv "${output}/challenge_metrics.csv" > "${output}/evaluation.log" 2>&1
  "${python_bin}" - "${output}/challenge_metrics.csv" "${output}/metrics_summary.json" <<'PY'
import csv, json, sys
import numpy as np
rows = list(csv.DictReader(open(sys.argv[1], newline="")))
dsc = np.asarray([float(row["Lesion_DSC"]) for row in rows])
nsd = np.asarray([float(row["Lesion_NSD"]) for row in rows])
summary = {
    "case_count": len(rows),
    "mean_dsc": float(dsc.mean()),
    "median_dsc": float(np.median(dsc)),
    "mean_nsd": float(nsd.mean()),
    "median_nsd": float(np.median(nsd)),
    "zero_dsc_cases": int((dsc == 0).sum()),
}
open(sys.argv[2], "w").write(json.dumps(summary, indent=2) + "\n")
PY
}

predict_public 0 0095 &
pid95=$!
predict_public 1 0098 &
pid98=$!
wait "${pid95}"
wait "${pid98}"
predict_public 0 0099

"${python_bin}" - "${root}" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("epoch_*/metrics_summary.json")):
    data = json.loads(path.read_text())
    rows.append((
        path.parent.name,
        data["mean_dsc"], data["mean_nsd"],
        (data["mean_dsc"] + data["mean_nsd"]) / 2,
        data["zero_dsc_cases"],
    ))
rows.sort(key=lambda row: row[3], reverse=True)
with (root / "summary_by_combined.tsv").open("w") as stream:
    stream.write("checkpoint\tDSC\tNSD\tcombined\tzero_DSC\n")
    for row in rows:
        stream.write("%s\t%.6f\t%.6f\t%.6f\t%d\n" % row)
PY
cat "${root}/summary_by_combined.tsv"

manifest="${root}/candidate_manifest.tsv"
printf 'variant\tcheckpoint_tag\tcheckpoint\tlesion_threshold\tmin_mm3\tscore_tag\tpredictions\tmetrics\n' \
  > "${manifest}"
while IFS=$'\t' read -r candidate_epoch _; do
  epoch_number="${candidate_epoch#epoch_}"
  printf 'swin\t%s\t%s/checkpoint_epoch_%s.pth\t0.50\t0\tswin_%s\t%s/%s/predictions\t%s/%s/challenge_metrics.csv\n' \
    "${candidate_epoch}" "${model_dir}" "${epoch_number}" "${candidate_epoch}" \
    "${root}" "${candidate_epoch}" "${root}" "${candidate_epoch}" >> "${manifest}"
done < <(tail -n +2 "${root}/summary_by_combined.tsv")
prompt_rank_root="${root}/prompted_ranking"
"${python_bin}" -u "${NNUNET_PROJECT_ROOT}/external/GF-Screen/rank_prompted_manifest.py" \
  --manifest "${manifest}" --output "${prompt_rank_root}" \
  --labels "${public_labels}" --prompt-support "${public_prompt_support}" \
  --prompt-config "${prompt_config}" --prompt-anchor "${prompt_anchor}" \
  --fusion-script "${biomed_root}/biomedparse_fusion.py" \
  --evaluator "${NNUNET_PROJECT_ROOT}/external/GF-Screen/evaluate_prompted_grid.py" \
  --selector "${selector}" --workers 8 > "${root}/prompted_ranking.log" 2>&1
best_line=$(sed -n '2p' "${prompt_rank_root}/selected.tsv")
IFS=$'\t' read -r _ best_epoch _ _ _ _ _ best_csv public_prompt prompt_metrics \
  prompt_robust team_decision best_dsc best_nsd best_combined deployment_tier \
  <<< "${best_line}"
"${python_bin}" "${selector}" \
  --anchor "${anchor_csv}" --candidate "${best_csv}" \
  --anchor-name swin_epoch50 --candidate-name "swin_${best_epoch}" \
  --output "${root}/robust_selection.json" > "${root}/robust_selection.log"
decision=$(awk -F'"' '/"decision":/ {print $4}' "${root}/robust_selection.json")
if [[ "${decision}" != "PROMOTE" ]]; then
  echo "Late Swin checkpoint did not robustly improve epoch 50 in raw space; retaining fallback status."
fi
cp "${prompt_robust}" "${root}/robust_selection_vs_team_champion.json"
if [[ "${team_decision}" != "PROMOTE" ]]; then
  echo "Late Swin did not clear the deployed team gate; packaging as an explicitly ranked fallback."
fi

epoch_number="${best_epoch#epoch_}"
checkpoint="checkpoint_epoch_${epoch_number}.pth"
hidden_root="${root}/hidden_${best_epoch}"
mkdir -p "${hidden_root}/part0" "${hidden_root}/part1" \
  "${hidden_root}/raw" "${hidden_root}/prompt_confirmed"
common=(
  -i "${hidden_images}" -d 501 -tr nnUNetTrainer -p "${plans}"
  -c "${configuration}" -f "${fold}" -chk "${checkpoint}"
  -npp 2 -nps 2 -num_parts 2 --disable_tta
)
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict "${common[@]}" \
  -part_id 0 -o "${hidden_root}/part0" > "${hidden_root}/part0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 nnUNetv2_predict "${common[@]}" \
  -part_id 1 -o "${hidden_root}/part1" > "${hidden_root}/part1.log" 2>&1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
cp "${hidden_root}/part0"/*.nii.gz "${hidden_root}/raw/"
cp "${hidden_root}/part1"/*.nii.gz "${hidden_root}/raw/"
[[ $(find "${hidden_root}/raw" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 combined hidden Swin masks" >&2
  exit 1
}
"${python_bin}" "${biomed_root}/biomedparse_fusion.py" materialize \
  --base "${hidden_root}/raw" --support "${hidden_prompt_support}" \
  --config "${prompt_config}" --output "${hidden_root}/prompt_confirmed" \
  > "${hidden_root}/prompt_materialize.log" 2>&1
[[ $(find "${hidden_root}/prompt_confirmed" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 100 ]] || {
  echo "expected 100 prompt-confirmed hidden Swin masks" >&2
  exit 1
}
submission="${NNUNET_PROJECT_ROOT}/submissions/swin_${best_epoch}_prompt_or_e050.zip"
"${python_bin}" "${script_dir}/package_codabench.py" \
  --images "${hidden_images}" --pred-dir "${hidden_root}/prompt_confirmed" \
  --output "${submission}" --expected-count 100
unzip -t "${submission}" > "${submission}.zip_test.log"
sha256sum "${submission}" > "${submission}.sha256"
printf 'name\tfamily\tmetrics\tsubmission\tgate\n%s\tswin_direct\t%s\t%s\t%s\n' \
  "swin_${best_epoch}" "${prompt_metrics}" "${submission}" \
  "${root}/robust_selection_vs_team_champion.json" \
  > "${root}/submission_candidate.tsv"
echo "Late Swin submission ready: ${submission}"

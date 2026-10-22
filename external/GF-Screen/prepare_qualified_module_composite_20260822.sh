#!/usr/bin/env bash
set -euo pipefail

# Compose independently save-qualified Focus and Glance modules. This script
# performs no inference and cannot create or upload a submission. A missing
# qualified manifest means that module falls back to the immutable base.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
base="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
focus_saved="${root}/public_sweeps/gf_focus_allpartial_saved_20260822"
glance_saved="${root}/public_sweeps/gf_glance_boxloc_saved_20260822"
cbg_box_saved="${root}/public_sweeps/gf_glance_cbg_box_saved_20260822"
output="${root}/checkpoint_compositions/qualified_focus_glance_20260822"
selection="${output}/selection.json"
merged="${output}/model.pt"

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "qualified module composition already active" >&2; exit 1; }
[[ -s "${output}/COMPOSITION_COMPLETE.txt" ]] && exit 0
for marker in "${focus_saved}/SAVE_ONLY_COMPLETE.txt" \
    "${glance_saved}/SAVE_ONLY_COMPLETE.txt"; do
  [[ -s "${marker}" ]] || { echo "upstream save stage is incomplete: ${marker}" >&2; exit 2; }
done
[[ -s "${base}" ]] || { echo "missing immutable base: ${base}" >&2; exit 2; }

glance_manifest="${glance_saved}/saved_boxloc_submission.tsv"
cbg_box_manifest="${cbg_box_saved}/saved_cbg_box_submission.tsv"
if [[ -s "${cbg_box_manifest}" ]]; then
  if [[ ! -s "${glance_manifest}" ]] || "${python_bin}" -c \
      'import csv,sys
def score(p):
 with open(p,newline="") as f:r=next(csv.DictReader(f,delimiter="\t"))
 return float(r["DSC"])+float(r["NSD"])
raise SystemExit(0 if score(sys.argv[1])>score(sys.argv[2]) else 1)' \
      "${cbg_box_manifest}" "${glance_manifest}"; then
    glance_manifest="${cbg_box_manifest}"
  fi
fi

"${python_bin}" "${root}/select_qualified_module_sources.py" \
  --base "${base}" \
  --focus-manifest "${focus_saved}/saved_focus_submission.tsv" \
  --glance-manifest "${glance_manifest}" \
  --output "${selection}" > "${output}/selection.log"

json_checkpoint() {
  "${python_bin}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]]["checkpoint"])' \
    "${selection}" "$1"
}
focus_checkpoint=$(json_checkpoint focus)
glance_checkpoint=$(json_checkpoint glance)
"${python_bin}" "${root}/merge_focus_glance_checkpoints.py" \
  --focus "${focus_checkpoint}" --glance "${glance_checkpoint}" \
  --drop-training-box-head --output "${merged}" \
  > "${output}/merge.log"
"${python_bin}" "${root}/checkpoint_state_digest.py" "${merged}" \
  > "${output}/model_state.sha256"
"${python_bin}" -c \
  'import sys,torch; x=torch.load(sys.argv[1],map_location="cpu",weights_only=False); s=x["state_dict"]; assert s and all(not k.startswith("glance_model.box_layer.") for k in s); assert any(k.startswith("glance_model.") for k in s); assert any(k.startswith("segmentation_model.") for k in s); print(len(s))' \
  "${merged}" > "${output}/schema_check.log"
date -Ins > "${output}/COMPOSITION_COMPLETE.txt"
printf '%s\n' "${merged}"

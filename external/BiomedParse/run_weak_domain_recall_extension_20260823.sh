#!/usr/bin/env bash
set -euo pipefail

# Second-stage, training-only proposal-recall experiment. This controller runs
# only when the conservative weak-domain prompt bank fails to provide the
# high-headroom bbox+24 oracle gate. Prompt banks, thresholds, and NMS policy are written
# before inference. Complete labels are read only after every mask is frozen.
# There is no public, hidden, packaging, or upload path in this script.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${gf}/analysis/fpr_training_mining_20260822/manifest.json"
inputs="${root}/analysis/weak_domain_biomedparse_20260823"
initial="${root}/analysis/weak_domain_biomedparse_enrichment_20260823"
base="${gf}/analysis/candidate_refiner_v2_multisource_20260823/proposal_union"
labels="${gf}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
base_report="${gf}/analysis/candidate_bbox_context_oracle_epoch105_currentgf_train210_20260823/report.json"
output="${root}/analysis/weak_domain_biomedparse_recall_extension_20260823"
thresholds=(010 020 030 050)

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "weak-domain recall extension already active" >&2; exit 1; }
[[ -s "${output}/EXTENSION_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${initial}/ENRICHMENT_COMPLETE.txt" ]]; do
  printf '%s waiting for initial weak-domain enrichment\n' "$(date -Ins)"
  sleep 30
done

initial_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' \
  "${initial}/selection.json")
if [[ "${initial_decision}" == "SUFFICIENT_PROPOSAL_HEADROOM" ]]; then
  date -Ins > "${output}/SKIPPED_INITIAL_HEADROOM_SUFFICIENT.txt"
  date -Ins > "${output}/EXTENSION_COMPLETE.txt"
  exit 0
fi

for required in "${manifest}" "${inputs}/INPUTS_COMPLETE.txt" \
  "${initial}/selection.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${inputs}/autopet/inputs" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
[[ $(find "${inputs}/lndb/inputs" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]

# This immutable declaration is created before inference and before labels are
# accessed by an evaluator. It makes the search space auditable.
"${python_bin}" - "${output}/PREDECLARED_SEARCH.json" <<'PY'
import datetime,json,sys
from pathlib import Path
payload={
  "created_at":datetime.datetime.now().astimezone().isoformat(),
  "labels_read":False,
  "nms":False,
  "existence_thresholds":[0.10,0.20,0.30,0.50],
  "prompt_banks":{
    "autopet":[
      "whole body lesion","tumor","malignant tumor","PET-avid lesion",
      "lymph node metastasis","bone metastasis","liver metastasis",
      "lung metastasis"
    ],
    "lndb":[
      "pulmonary nodule","lung nodule","lung mass","lung tumor","lung cancer"
    ],
  },
  "selection_rule":"fewest mean tiles among variants within 0.005 macro bbox+24 DSC of the best",
  "required_ceiling_for_0p80_path":0.875,
  "required_target_cohort_DSC":0.85,
  "useful_gain_gate":{
    "minimum_global_DSC_gain":0.03,
    "minimum_autoPETCT_DSC_gain":0.05,
    "minimum_LNDb_DSC_gain":0.05,
  },
  "rationale":"The old 0.90 all-or-nothing gate demands about 86% recovery of the two targeted cohorts' full recoverable deficit; preserve substantial useful gains for controlled 707-case training.",
}
Path(sys.argv[1]).write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
      && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
  printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

infer_bank() {
  local gpu="$1" tag="$2"; shift 2
  local prompt_args=() prompt destination="${output}/${tag}_npz"
  for prompt in "$@"; do prompt_args+=(--prompt "${prompt}"); done
  mkdir -p "${destination}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
    ../miccai_inference.py --input-dir "${inputs}/${tag}/inputs" \
    --output-dir "${destination}" "${prompt_args[@]}" --device cuda \
    --slice-batch-size 8 --disable-nms \
    --existence-threshold 0.10 --existence-threshold 0.20 \
    --existence-threshold 0.30 --existence-threshold 0.50 \
    --continue-prediction > "${output}/${tag}.inference.log" 2>&1
}
infer_bank 0 autopet \
  "whole body lesion" "tumor" "malignant tumor" "PET-avid lesion" \
  "lymph node metastasis" "bone metastasis" "liver metastasis" \
  "lung metastasis" & p0=$!
infer_bank 1 lndb \
  "pulmonary nodule" "lung nodule" "lung mass" "lung tumor" "lung cancer" & p1=$!
status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

for tag in autopet lndb; do
  [[ $(find "${output}/${tag}_npz" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
  for threshold in "${thresholds[@]}"; do
    destination="${output}/${tag}_e${threshold}_nifti"
    mkdir -p "${destination}"
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${inputs}/${tag}/images" --predictions "${output}/${tag}_npz" \
      --output "${destination}" --segmentation-key "segs_e${threshold}" \
      --continue-prediction > "${output}/${tag}_e${threshold}.convert.log" 2>&1
    [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 30 ]]
  done
done

initial_threshold=$("${python_bin}" -c \
  'import json,sys;print("{:03d}".format(round(100*float(json.load(open(sys.argv[1]))["selected"]["existence_threshold"]))))' \
  "${initial}/selection.json")
initial_base="${initial}/enriched_e${initial_threshold}"
[[ $(find "${initial_base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]

# Each candidate is the exact-210 initial selected enrichment plus sparse
# overlays from the broader bank at one frozen threshold.
for threshold in "${thresholds[@]}"; do
  enriched="${output}/enriched_plus_initial_e${threshold}"
  marker="${output}/enriched_plus_initial_e${threshold}.json"
  "${python_bin}" -u "${gf}/fuse_sparse_overlays.py" \
    --manifest "${manifest}" --base "${initial_base}" \
    --overlay "${output}/autopet_e${threshold}_nifti" \
    --overlay "${output}/lndb_e${threshold}_nifti" \
    --output "${enriched}" --marker "${marker}" --workers 4 \
    > "${output}/enriched_plus_initial_e${threshold}.log" 2>&1
done

# The proposal search is now frozen. Labels are opened only below to measure
# achievable bbox+24 coverage; the labels are never copied into predictions.
for threshold in "${thresholds[@]}"; do
  oracle="${output}/oracle_e${threshold}"
  "${python_bin}" -u "${gf}/analyze_candidate_bbox_context_oracle.py" \
    --gt "${labels}" --proposal "${output}/enriched_plus_initial_e${threshold}" \
    --output "${oracle}" --min-mm3 5 --max-mm3 1000000000000 \
    --tile-stride-mm 72 --margins-mm 24 --workers 4 --expected-count 210 \
    --manifest "${manifest}" > "${output}/oracle_e${threshold}.log" 2>&1
done

"${python_bin}" - "${output}" "${initial_threshold}" "${base_report}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); initial_threshold=sys.argv[2]; base_path=Path(sys.argv[3]); rows=[]
base=json.loads(base_path.read_text(encoding="utf-8"))
base_cohorts=base["cohort_summary"]
baseline={
  "macro_bbox24_DSC":float(base["margins"]["24"]["macro_dsc"]),
  "autoPETCT_bbox24_DSC":float(base_cohorts["abdomen:WholeBody_autoPETCT"]["margins"]["24"]["macro_dsc"]),
  "LNDb_bbox24_DSC":float(base_cohorts["chest:Chest_LNDb"]["margins"]["24"]["macro_dsc"]),
}
for threshold in (0.10,0.20,0.30,0.50):
    tag=f"{int(round(threshold*100)):03d}"
    report=json.loads((root/f"oracle_e{tag}"/"report.json").read_text())
    margin=report["margins"]["24"]; cohorts=report["cohort_summary"]
    rows.append({
      "existence_threshold":threshold,
      "macro_bbox24_DSC":float(margin["macro_dsc"]),
      "mean_tiles_per_case":float(margin["mean_tiles_per_case"]),
      "autoPETCT_bbox24_DSC":float(cohorts["abdomen:WholeBody_autoPETCT"]["margins"]["24"]["macro_dsc"]),
      "LNDb_bbox24_DSC":float(cohorts["chest:Chest_LNDb"]["margins"]["24"]["macro_dsc"]),
      "proposal":str((root/f"enriched_plus_initial_e{tag}").resolve()),
      "report":str((root/f"oracle_e{tag}"/"report.json").resolve()),
    })
best_dsc=max(row["macro_bbox24_DSC"] for row in rows)
near=[row for row in rows if row["macro_bbox24_DSC"] >= best_dsc-0.005]
selected=min(near,key=lambda row:(row["mean_tiles_per_case"],-row["macro_bbox24_DSC"]))
selected["global_DSC_gain"]=selected["macro_bbox24_DSC"]-baseline["macro_bbox24_DSC"]
selected["autoPETCT_DSC_gain"]=selected["autoPETCT_bbox24_DSC"]-baseline["autoPETCT_bbox24_DSC"]
selected["LNDb_DSC_gain"]=selected["LNDb_bbox24_DSC"]-baseline["LNDb_bbox24_DSC"]
high_headroom=(
  selected["macro_bbox24_DSC"] >= 0.875
  and selected["autoPETCT_bbox24_DSC"] >= 0.85
  and selected["LNDb_bbox24_DSC"] >= 0.85
)
useful_gain=(
  selected["global_DSC_gain"] >= 0.03
  and selected["autoPETCT_DSC_gain"] >= 0.05
  and selected["LNDb_DSC_gain"] >= 0.05
)
decision=("SUFFICIENT_PROPOSAL_HEADROOM" if high_headroom else
          "USEFUL_PROPOSAL_GAIN" if useful_gain else
          "MORE_PROPOSAL_ENRICHMENT_REQUIRED")
payload={
  "warning":"Complete-label proposal-ceiling selection only; never use oracle masks for inference.",
  "initial_selected_existence_threshold":int(initial_threshold)/100,
  "selection_rule":"fewest mean tiles among variants within 0.005 macro DSC of the best",
  "baseline":baseline,
  "required_ceiling_for_0p80_path":0.875,
  "required_target_cohort_DSC":0.85,
  "useful_gain_gate":{
    "minimum_global_DSC_gain":0.03,
    "minimum_autoPETCT_DSC_gain":0.05,
    "minimum_LNDb_DSC_gain":0.05,
  },
  "candidates":rows,
  "selected":selected,
  "decision":decision,
  "labels_read_after_proposals_frozen":True,
}
(root/"selection.json").write_text(json.dumps(payload,indent=2)+"\n")
print(json.dumps(payload,indent=2))
PY
date -Ins > "${output}/EXTENSION_COMPLETE.txt"
cat "${output}/selection.json"

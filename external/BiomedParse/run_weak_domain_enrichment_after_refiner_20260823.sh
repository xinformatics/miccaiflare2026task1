#!/usr/bin/env bash
set -euo pipefail

# Training-only proposal-recall experiment for the two cohorts that limit the
# 210-case refiner ceiling. Prompt banks and thresholds are fixed before labels
# are read. This script has no public, hidden, packaging, or upload path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
gf="${project}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${gf}/analysis/fpr_training_mining_20260822/manifest.json"
inputs="${root}/analysis/weak_domain_biomedparse_20260823"
input_complete="${inputs}/INPUTS_COMPLETE.txt"
base="${gf}/analysis/candidate_refiner_v2_multisource_20260823/proposal_union"
labels="${gf}/analysis/candidate_refiner_v2_multisource_oof_20260823/staging/all/labels"
base_report="${gf}/analysis/candidate_bbox_context_oracle_epoch105_currentgf_train210_20260823/report.json"
output="${root}/analysis/weak_domain_biomedparse_enrichment_20260823"
thresholds=(030 050 070)

mkdir -p "${output}"
exec 9>"${output}.lock"
flock -n 9 || { echo "weak-domain BiomedParse enrichment already active" >&2; exit 1; }
[[ -s "${output}/ENRICHMENT_COMPLETE.txt" ]] && exit 0
while [[ ! -s "${input_complete}" ]]; do
  printf '%s waiting for audited weak-domain inputs\n' "$(date -Ins)"
  sleep 30
done
[[ $(find "${inputs}/autopet/inputs" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
[[ $(find "${inputs}/lndb/inputs" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 210 ]]
[[ -s "${base_report}" ]]

# Only autoPETCT and LNDb (60/210 cases) change in this experiment. Perfect
# proposals for both targeted cohorts cap the macro bbox+24 DSC near 0.922, so
# the old 0.90 gate requires recovering about 86% of the entire recoverable
# deficit. Freeze a demanding but attainable high-headroom target and a
# separate useful-gain route before inference so a strong incremental proposal
# bank is not discarded merely because it misses the high-headroom target.
"${python_bin}" - "${base_report}" "${output}/PREDECLARED_SEARCH.json" <<'PY'
import datetime,json,sys
from pathlib import Path
base_path,output=map(Path,sys.argv[1:])
base=json.loads(base_path.read_text(encoding="utf-8"))
cohorts=base["cohort_summary"]
target_names=("abdomen:WholeBody_autoPETCT","chest:Chest_LNDb")
baseline=float(base["margins"]["24"]["macro_dsc"])
cases=int(base["cases"])
recoverable=sum(
    int(cohorts[name]["cases"])*(1.0-float(cohorts[name]["margins"]["24"]["macro_dsc"]))
    for name in target_names
)/cases
payload={
  "created_at":datetime.datetime.now().astimezone().isoformat(),
  "labels_read":False,
  "baseline_report":str(base_path.resolve()),
  "baseline_macro_bbox24_DSC":baseline,
  "theoretical_max_if_targeted_cohorts_are_perfect":baseline+recoverable,
  "fraction_of_recoverable_deficit_required_by_old_0p90_gate":(0.90-baseline)/recoverable,
  "existence_thresholds":[0.30,0.50,0.70],
  "required_ceiling_for_0p80_path":0.875,
  "required_target_cohort_DSC":0.85,
  "useful_gain_gate":{
    "minimum_global_DSC_gain":0.03,
    "minimum_autoPETCT_DSC_gain":0.05,
    "minimum_LNDb_DSC_gain":0.05,
  },
  "accepted_scale_out_decisions":["SUFFICIENT_PROPOSAL_HEADROOM","USEFUL_PROPOSAL_GAIN"],
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_for_gpu_set() {
  while :; do
    local used0 used1
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ ]]; then
      if [[ "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then
        printf '0 1\n'; return
      elif [[ "${used0}" -le 4096 ]]; then
        printf '0\n'; return
      elif [[ "${used1}" -le 4096 ]]; then
        printf '1\n'; return
      fi
    fi
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}" >&2
    sleep 30
  done
}

infer_bank() {
  local gpu="$1" tag="$2"; shift 2
  local prompt_args=() prompt destination="${output}/${tag}_npz"
  for prompt in "$@"; do prompt_args+=(--prompt "${prompt}"); done
  mkdir -p "${destination}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${root}/run_official_env.sh" \
    ../miccai_inference.py --input-dir "${inputs}/${tag}/inputs" \
    --output-dir "${destination}" "${prompt_args[@]}" --device cuda \
    --slice-batch-size 8 --existence-threshold 0.30 \
    --existence-threshold 0.50 --existence-threshold 0.70 \
    --continue-prediction > "${output}/${tag}.inference.log" 2>&1
}
read -r -a free_gpus <<< "$(wait_for_gpu_set)"
if [[ "${#free_gpus[@]}" -ge 2 ]]; then
  infer_bank "${free_gpus[0]}" autopet "malignant lesion" "metastasis" "cancer" "abnormal mass" & p0=$!
  infer_bank "${free_gpus[1]}" lndb "lung tumor" "lung nodule" & p1=$!
  status=0; wait "${p0}" || status=$?; wait "${p1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
else
  infer_bank "${free_gpus[0]}" autopet "malignant lesion" "metastasis" "cancer" "abnormal mass"
  infer_bank "${free_gpus[0]}" lndb "lung tumor" "lung nodule"
fi

for tag in autopet lndb; do
  [[ $(find "${output}/${tag}_npz" -maxdepth 1 -name '*.npz' | wc -l) -eq 30 ]]
  for threshold in "${thresholds[@]}"; do
    destination="${output}/${tag}_e${threshold}_nifti"; mkdir -p "${destination}"
    "${root}/run_official_env.sh" ../miccai_nifti_adapter.py convert \
      --images "${inputs}/${tag}/images" --predictions "${output}/${tag}_npz" \
      --output "${destination}" --segmentation-key "segs_e${threshold}" \
      --continue-prediction > "${output}/${tag}_e${threshold}.convert.log" 2>&1
    [[ $(find "${destination}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 30 ]]
  done
done

for threshold in "${thresholds[@]}"; do
  enriched="${output}/enriched_e${threshold}"
  marker="${output}/enriched_e${threshold}.json"
  "${python_bin}" -u "${gf}/fuse_sparse_overlays.py" \
    --manifest "${manifest}" --base "${base}" \
    --overlay "${output}/autopet_e${threshold}_nifti" \
    --overlay "${output}/lndb_e${threshold}_nifti" \
    --output "${enriched}" --marker "${marker}" --workers 4 \
    > "${output}/enriched_e${threshold}.log" 2>&1
  oracle="${output}/oracle_e${threshold}"
  "${python_bin}" -u "${gf}/analyze_candidate_bbox_context_oracle.py" \
    --gt "${labels}" --proposal "${enriched}" --output "${oracle}" \
    --min-mm3 5 --max-mm3 1000000000000 --tile-stride-mm 72 \
    --margins-mm 24 --workers 4 --expected-count 210 --manifest "${manifest}" \
    > "${output}/oracle_e${threshold}.log" 2>&1
done

"${python_bin}" - "${output}" "${base_report}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]); base_path=Path(sys.argv[2]); rows=[]
base=json.loads(base_path.read_text(encoding="utf-8"))
base_cohorts=base["cohort_summary"]
baseline={
  "macro_bbox24_DSC":float(base["margins"]["24"]["macro_dsc"]),
  "autoPETCT_bbox24_DSC":float(base_cohorts["abdomen:WholeBody_autoPETCT"]["margins"]["24"]["macro_dsc"]),
  "LNDb_bbox24_DSC":float(base_cohorts["chest:Chest_LNDb"]["margins"]["24"]["macro_dsc"]),
}
for threshold in (0.30,0.50,0.70):
    tag=f"{int(round(threshold*100)):03d}"
    report=json.loads((root/f"oracle_e{tag}"/"report.json").read_text())
    margin=report["margins"]["24"]
    cohorts=report["cohort_summary"]
    rows.append({
        "existence_threshold":threshold,
        "macro_bbox24_DSC":float(margin["macro_dsc"]),
        "mean_tiles_per_case":float(margin["mean_tiles_per_case"]),
        "autoPETCT_bbox24_DSC":float(cohorts["abdomen:WholeBody_autoPETCT"]["margins"]["24"]["macro_dsc"]),
        "LNDb_bbox24_DSC":float(cohorts["chest:Chest_LNDb"]["margins"]["24"]["macro_dsc"]),
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
    "selection_rule":"fewest mean tiles among thresholds within 0.005 macro DSC of the best",
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
date -Ins > "${output}/ENRICHMENT_COMPLETE.txt"
cat "${output}/selection.json"

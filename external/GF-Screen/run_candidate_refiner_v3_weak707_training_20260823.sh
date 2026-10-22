#!/usr/bin/env bash
set -euo pipefail

# Five-fold, from-scratch scaled refiner training on the balanced 707-case
# complete-label set. The four learned/prompt sources and eight promoted
# parity-separated lung-cell sources remain independent components. This
# controller has no public/hidden inference, packaging, submission, or upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse/analysis/weak_domain_biomedparse_707_selected_20260823"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
epoch105="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050"
epoch105_checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
currentgf="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
currentgf_checkpoint="${root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt"
currentgf_complete="${currentgf}/PROPOSALS_COMPLETE.json"
weak_complete="${biomed}/PROPOSALS_COMPLETE.json"
weak_skipped="${biomed}/INSUFFICIENT_HEADROOM.json"
prompt_ablation="${project}/external/BiomedParse/analysis/weak_domain_broad_only_ablation_20260823/DECISION.json"
margin_decision="${root}/analysis/weak707_bbox_margin_headroom_20260823/DECISION.json"
cell_scale="${root}/analysis/lung_cell707_scale_20260823"
cell_decision="${cell_scale}/DECISION.json"
cells="${cell_scale}/cells48_all707"
work="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
patches="${work}/patches_bbox24_ss_min5_v4_cell48"
provenance_audit="${work}/proposal_provenance_audit.json"
source_audit="${work}/proposal_sources_audit.json"
audit="${patches}/AUDIT.json"
oof="${work}/five_fold_oof_summary.json"
selection="${work}/checkpoint_selection.json"
complete="${work}/TRAINING_COMPLETE.json"
skipped="${work}/TRAINING_SKIPPED.json"
expected=707

fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' "${root}" "$1"; }

mkdir -p "${work}" "${root}/runs"
exec 9>"${root}/runs/run_candidate_refiner_v3_weak707_training_20260823.lock"
flock -n 9 || { echo "weak707 candidate-refiner training already active" >&2; exit 1; }
[[ -s "${complete}" || -s "${skipped}" ]] && exit 0
while [[ ! -s "${weak_complete}" || ! -s "${margin_decision}" \
    || ! -s "${currentgf_complete}" || ! -s "${cell_decision}" ]]; do
  if [[ -s "${weak_skipped}" ]]; then break; fi
  printf '%s waiting for selected 707 weak proposals and context margin\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${weak_skipped}" ]]; then
  "${python_bin}" - "${weak_skipped}" "${skipped}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={
  "created_at":datetime.now().astimezone().isoformat(),
  "decision":"SKIP_TRAINING",
  "reason":"weak proposal source did not meet the predeclared headroom gate",
  "source":str(source.resolve()),
  "source_payload":json.loads(source.read_text(encoding="utf-8")),
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
  exit 0
fi
for required in "${manifest}" "${epoch105}/PROPOSALS_COMPLETE.json" \
  "${currentgf_complete}" "${weak_complete}" "${margin_decision}" \
  "${prompt_ablation}" "${cell_decision}" "${cells}/PROPOSALS_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
prompt_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${prompt_ablation}")
[[ "${prompt_decision}" == BROAD_BANK_SUFFICIENT ]] || {
  echo "scaled weak sources omit a non-redundant initial prompt bank" >&2
  exit 2
}
bbox_margin=$("${python_bin}" - "${margin_decision}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
value=int(payload.get("effective_margin_mm", payload["baseline_margin_mm"] if payload["decision"]=="KEEP_MARGIN_24" else payload["selected"]["margin_mm"]))
if value not in {16,24,32,48,72,96}: raise SystemExit("invalid frozen bbox margin")
print(value)
PY
)
autopet=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["banks"]["autopet"]["complete_source"])' \
  "${weak_complete}")
lndb=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["banks"]["lndb"]["complete_source"])' \
  "${weak_complete}")
cell_gate=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${cell_decision}")
[[ "${cell_gate}" == PROMOTE_LUNG_CELLS_707 ]] || {
  echo "exact-707 lung-cell gate did not promote: ${cell_gate}" >&2
  exit 2
}
cell_dirs=()
for colour in 0 1 2 3 4 5 6 7; do cell_dirs+=("${cells}/colour${colour}"); done
for directory in "${epoch105}" "${currentgf}" "${autopet}" "${lndb}" "${cell_dirs[@]}"; do
  [[ -d "${directory}" ]] || { echo "missing proposal directory: ${directory}" >&2; exit 2; }
done

"${python_bin}" - "${manifest}" "${epoch105}/PROPOSALS_COMPLETE.json" \
  "${currentgf_complete}" "${weak_complete}" "${epoch105_checkpoint}" \
  "${currentgf_checkpoint}" "${epoch105}" "${currentgf}" "${autopet}" \
  "${lndb}" "${cell_decision}" "${cells}/PROPOSALS_COMPLETE.json" \
  "${provenance_audit}" <<'PY'
import hashlib,json,os,sys
from datetime import datetime
from pathlib import Path

(manifest,epoch_marker,current_marker,weak_marker,epoch_checkpoint,current_checkpoint,
 epoch_dir,current_dir,autopet_dir,lndb_dir,cell_decision_path,cell_marker_path,
 output)=map(Path,sys.argv[1:])
def load(path): return json.loads(path.read_text(encoding="utf-8"))
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
def check_gf(marker_path,checkpoint,directory,gate,focus):
    payload=load(marker_path)
    window=payload.get("window",payload.get("window_hu"))
    checks={
      "cases":payload.get("cases")==707,
      "checkpoint":payload.get("checkpoint")==str(checkpoint.resolve()),
      "checkpoint_sha256":payload.get("checkpoint_sha256")==sha(checkpoint),
      "output":payload.get("output",str(directory.resolve()))==str(directory.resolve()),
      "window":list(window or [])==[-175,250],
      "glance_threshold":float(payload.get("glance_threshold",-1))==gate,
      "lesion_threshold":float(payload.get("lesion_threshold",-1))==focus,
      "labels_read":payload.get("labels_read") is False,
    }
    if not all(checks.values()):
        raise SystemExit(f"GF proposal provenance mismatch {marker_path}: {[k for k,v in checks.items() if not v]}")
    return {"marker":str(marker_path.resolve()),"checkpoint":str(checkpoint.resolve()),
            "checkpoint_sha256":sha(checkpoint),"checks":checks}
epoch=check_gf(epoch_marker,epoch_checkpoint,epoch_dir,0.90,0.50)
current=check_gf(current_marker,current_checkpoint,current_dir,0.50,0.50)
weak=load(weak_marker)
weak_checks={
  "cases":weak.get("cases")==707,
  "existence_threshold":float(weak.get("existence_threshold",-1))==0.50,
  "disable_nms":weak.get("disable_nms") is True,
  "labels_read":weak.get("labels_read") is False,
  "public_or_hidden_data_read":weak.get("public_or_hidden_data_read") is False,
  "autopet_path":Path(weak["banks"]["autopet"]["complete_source"]).resolve()==autopet_dir.resolve(),
  "lndb_path":Path(weak["banks"]["lndb"]["complete_source"]).resolve()==lndb_dir.resolve(),
}
if not all(weak_checks.values()):
    raise SystemExit(f"weak proposal provenance mismatch: {[k for k,v in weak_checks.items() if not v]}")
cell_decision=load(cell_decision_path); cell_marker=load(cell_marker_path)
cell_checks={
  "decision":cell_decision.get("decision")=="PROMOTE_LUNG_CELLS_707",
  "cases":cell_marker.get("cases")==707,
  "sources":cell_marker.get("sources")==8,
  "method":cell_marker.get("method")=="filled_48mm_occupied_lung_cells_in_eight_parity_sources",
  "target_spacing_mm":float(cell_marker.get("parameters",{}).get("target_spacing_mm",-1))==2.5,
  "cell_stride_mm":float(cell_marker.get("parameters",{}).get("cell_stride_mm",-1))==48.0,
  "maximum_extent":float(cell_marker.get("maximum_component_axis_extent_mm",1e9))<=48.000001,
  "labels_read":cell_marker.get("labels_read") is False,
  "public_or_hidden_data_read":cell_marker.get("public_or_hidden_data_read") is False,
}
if not all(cell_checks.values()):
    raise SystemExit(f"lung-cell provenance mismatch: {[k for k,v in cell_checks.items() if not v]}")
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "manifest":str(manifest.resolve()),"manifest_sha256":sha(manifest),
 "source_names":["epoch105","currentgf","autopet_weak","lndb_weak"]
                +[f"lung_cell{i}" for i in range(8)],
 "epoch105":epoch,"currentgf":current,
 "weak":{"marker":str(weak_marker.resolve()),"checks":weak_checks},
 "lung_cells":{"decision":str(cell_decision_path.resolve()),
                "marker":str(cell_marker_path.resolve()),"checks":cell_checks},
 "labels_read":False,"public_or_hidden_data_read":False,"status":"PASS",
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
print(json.dumps(payload,indent=2))
PY

"${python_bin}" - "${manifest}" "${expected}" "${epoch105}" "${currentgf}" \
  "${autopet}" "${lndb}" "${cell_dirs[@]}" <<'PY'
import json,sys
from pathlib import Path
manifest=Path(sys.argv[1]); expected=int(sys.argv[2]); sources=list(map(Path,sys.argv[3:]))
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
names={str(item["prediction_name"]) for item in items}
if len(items)!=expected or len(names)!=expected or any(item.get("complete_label") is not True for item in items):
    raise SystemExit("707 manifest coverage/completeness check failed")
for source in sources:
    observed={path.name for path in source.glob("*.nii.gz")}
    if observed!=names:
        raise SystemExit(f"proposal coverage mismatch: {source} {len(observed)}/{len(names)}")
PY

cell_source_args=()
cell_proposal_args=()
for colour in 0 1 2 3 4 5 6 7; do
  cell_source_args+=(--source "lung_cell${colour}=${cells}/colour${colour}")
  cell_proposal_args+=(--proposal "lung_cell${colour}=${cells}/colour${colour}")
done
if [[ ! -s "${source_audit}" ]]; then
  "${python_bin}" -u "${root}/audit_training_proposal_sources.py" \
    --manifest "${manifest}" --source "epoch105=${epoch105}" \
    --source "currentgf=${currentgf}" --source "autopet_weak=${autopet}" \
    --source "lndb_weak=${lndb}" "${cell_source_args[@]}" \
    --expected-count "${expected}" \
    --workers 8 --output "${source_audit}" > "${work}/audit_proposal_sources.log" 2>&1
fi

export MPLCONFIGDIR="${work}/matplotlib"
export TMPDIR="/tmp/gfref_weak707_20260823"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
if [[ ! -s "${patches}/COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/prepare_candidate_refiner_patches.py" \
    --manifest "${manifest}" --proposal "epoch105=${epoch105}" \
    --proposal "currentgf=${currentgf}" \
    --proposal "autopet_weak=${autopet}" --proposal "lndb_weak=${lndb}" \
    "${cell_proposal_args[@]}" \
    --component-policy source_separate --compact-source-components \
    --output "${patches}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 \
    --bbox-margin-mm "${bbox_margin}" --tile-stride-mm 72 \
    --max-negatives-per-case 64 --min-negatives-per-source 2 \
    --max-negative-tiles 1 --workers 5 --resume \
    > "${work}/prepare_patches.log" 2>&1
fi
if [[ ! -s "${audit}" ]]; then
  "${python_bin}" -u "${root}/audit_candidate_refiner_patches.py" \
    --index "${patches}/index.csv" --summary "${patches}/summary.json" \
    --output "${audit}" --workers 8 > "${work}/audit_patches.log" 2>&1
fi
"${python_bin}" - "${audit}" "${patches}/index.csv" <<'PY'
import hashlib,json,sys
from pathlib import Path
audit,index=map(Path,sys.argv[1:])
payload=json.loads(audit.read_text(encoding="utf-8"))
checks={
 "index":payload.get("index")==str(index.resolve()),
 "index_sha256":payload.get("index_sha256")==hashlib.sha256(index.read_bytes()).hexdigest(),
 "patches":payload.get("patches")==75567,
 "exact_file_coverage":payload.get("exact_file_coverage") is True,
 "all_prompts_nonempty":payload.get("all_prompts_nonempty") is True,
 "all_values_finite":payload.get("all_values_finite") is True,
 "targets":payload.get("all_targets_binary_and_label_consistent") is True,
 "shape":payload.get("shape")==[3,64,64,64],
 "spacing":payload.get("spacing")==[1.5,1.5,1.5],
}
failed=[name for name,passed in checks.items() if not passed]
if failed: raise SystemExit(f"invalid exact-707 patch audit: {failed}")
PY

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
wait_any_gpu() {
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    free0=0; free1=0
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free0=1
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}
wait_any_gpu

train_fold() {
  local gpu="$1" fold="$2" output
  output="$(fold_dir "${fold}")"
  mkdir -p "${output}"
  [[ -s "${output}/COMPLETE.txt" && -s "${output}/checkpoint_best.pt" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/train_candidate_refiner.py" \
    --index "${patches}/index.csv" --output "${output}" \
    --folds 5 --fold "${fold}" --epochs 30 --batch-size 8 --workers 2 \
    --lr 3e-4 --weight-decay 1e-4 --base-channels 32 \
    --sampling-temperature 0.5 --samples-per-epoch 8192 --seed 20260823 \
    --segmentation-threshold-max 0.95 --validation-interval 3 \
    --validation-batch-size 32 \
    --wandb-project pancancer-ct \
    --wandb-name "candidate_refiner_v4_cell707_b32_t05_fold${fold}" \
    --resume > "${output}/train.log" 2>&1
}
train_sequence() {
  local gpu="$1"; shift
  local fold
  for fold in "$@"; do train_fold "${gpu}" "${fold}"; done
}
train_remaining_dynamic() {
  # Admit a new fold whenever either GPU becomes available. This avoids
  # pinning all remaining folds to the device that happened to finish fold 0
  # first while another training service releases the second GPU later.
  local -a remaining=("$@")
  local -a job_pid=("" "") job_fold=("" "")
  local gpu pid used fold status=0
  while [[ ${#remaining[@]} -gt 0 || -n "${job_pid[0]}" || -n "${job_pid[1]}" ]]; do
    for gpu in 0 1; do
      pid="${job_pid[${gpu}]}"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || status=$?
        [[ "${status}" -eq 0 ]] || return "${status}"
        job_pid[${gpu}]=""; job_fold[${gpu}]=""
      fi
    done
    if [[ ${#remaining[@]} -gt 0 ]]; then
      for gpu in 0 1; do
        [[ ${#remaining[@]} -gt 0 ]] || break
        [[ -z "${job_pid[${gpu}]}" ]] || continue
        used=$(gpu_memory_mib "${gpu}")
        if [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]]; then
          fold="${remaining[0]}"; remaining=("${remaining[@]:1}")
          train_fold "${gpu}" "${fold}" &
          job_pid[${gpu}]=$!; job_fold[${gpu}]="${fold}"
          printf '%s dispatched exact-707 fold %s to GPU %s\n' \
            "$(date -Ins)" "${fold}" "${gpu}"
        fi
      done
    fi
    [[ ${#remaining[@]} -eq 0 && -z "${job_pid[0]}" && -z "${job_pid[1]}" ]] || sleep 30
  done
}
# Dynamically dispatch from the beginning. If only one device is free now,
# fold 0 starts there immediately; fold 1 can claim the other device as soon
# as an unrelated job releases it, without waiting for fold 0 to finish.
train_remaining_dynamic 0 1 2 3 4

selection_args=()
for strategy_spec in \
  'composite checkpoint_best.pt validation_predictions_best.npz' \
  'segmentation checkpoint_best_segmentation.pt validation_predictions_best_segmentation.npz' \
  'cohort_segmentation checkpoint_best_cohort_segmentation.pt validation_predictions_best_cohort_segmentation.npz' \
  'classification checkpoint_best_classification.pt validation_predictions_best_classification.npz'; do
  read -r strategy checkpoint_name prediction_name <<< "${strategy_spec}"
  strategy_summary="${work}/five_fold_oof_summary_${strategy}.json"
  oof_args=()
  for fold in 0 1 2 3 4; do
    output="$(fold_dir "${fold}")"
    oof_args+=(--checkpoint "${output}/${checkpoint_name}")
    oof_args+=(--prediction "${output}/${prediction_name}")
  done
  "${python_bin}" -u "${root}/summarize_candidate_refiner_oof.py" \
    --index "${patches}/index.csv" "${oof_args[@]}" --output "${strategy_summary}" \
    --source-group lung_cells=lung_cell0,lung_cell1,lung_cell2,lung_cell3,lung_cell4,lung_cell5,lung_cell6,lung_cell7 \
    > "${work}/five_fold_oof_summary_${strategy}.log" 2>&1
  selection_args+=(--candidate "${strategy}=${strategy_summary}")
done
"${python_bin}" -u "${root}/select_candidate_refiner_oof_strategy.py" \
  "${selection_args[@]}" --output "${selection}" --selected-summary "${oof}" \
  > "${work}/checkpoint_selection.log" 2>&1

"${python_bin}" - "${manifest}" "${weak_complete}" "${currentgf_complete}" \
  "${prompt_ablation}" "${margin_decision}" "${cell_decision}" "${bbox_margin}" \
  "${provenance_audit}" "${source_audit}" "${audit}" "${oof}" "${selection}" "${complete}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
manifest,weak_proposals,currentgf_proposals,prompt_ablation,margin_path,cell_decision=map(Path,sys.argv[1:7]); margin=int(sys.argv[7])
provenance_audit,source_audit,audit,oof,selection,output=map(Path,sys.argv[8:])
payload={
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "cases":707,
  "folds":5,
  "architecture":"3D residual U-Net candidate refiner, base_channels=32",
  "parameters":"approximately 9.49M",
  "training":"from scratch",
  "epochs":30,
  "samples_per_epoch":8192,
  "total_weighted_samples_per_fold":245760,
  "compute_matching":"approximately matches the successful 210-case native B32 optimizer-update budget",
  "batch_size_per_gpu":8,
  "optimizer":"AdamW",
  "learning_rate":3e-4,
  "weight_decay":1e-4,
  "scheduler":"5-epoch linear warmup then cosine decay",
  "sampling_temperature":0.5,
  "segmentation_threshold_max":0.95,
  "segmentation_validation_thresholds":[0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95],
  "component_policy":"source_separate",
  "bbox_margin_mm":margin,
  "bbox_margin_decision":str(margin_path.resolve()),
  "proposal_sources":["epoch105","currentgf","autopet_weak","lndb_weak"]
                     +[f"lung_cell{i}" for i in range(8)],
  "manifest":str(manifest.resolve()),
  "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
  "proposal_markers":{
    "weak":str(weak_proposals.resolve()),
    "currentgf":str(currentgf_proposals.resolve()),
  },
  "prompt_redundancy_ablation":str(prompt_ablation.resolve()),
  "lung_cell_scale_decision":str(cell_decision.resolve()),
  "proposal_provenance_audit":str(provenance_audit.resolve()),
  "proposal_source_audit":str(source_audit.resolve()),
  "patch_audit":str(audit.resolve()),
  "patch_oof":str(oof.resolve()),
  "checkpoint_selection":str(selection.resolve()),
  "public_or_hidden_data_read":False,
  "submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload,indent=2))
PY
cat "${oof}"

#!/usr/bin/env bash
set -euo pipefail

# Training-only transfer diagnostic for the selected weak-domain proposal
# enrichment. The already trained v2 checkpoints are applied case-held-out;
# no retraining, public/hidden data, packaging, or upload occurs here. The
# BiomedParse proposal operating point was selected from complete training-label
# ceiling diagnostics, so this is explicitly not claimed as fully nested OOF.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v2_training_20260822"
training_complete="${train}/TRAINING_COMPLETE.txt"
selection="${train}/checkpoint_selection.json"
patch_oof="${train}/five_fold_oof_summary.json"
index="${train}/patches_bbox24_min5_v5/index.csv"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
multisource_oof="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
multisource_calibration="${multisource_oof}/deployment_calibration.json"
multisource_save="${root}/public_sweeps/candidate_refiner_v2_multisource_saved_20260823/SAVE_ONLY_COMPLETE.txt"
biomed="${project}/external/BiomedParse/analysis"
initial="${biomed}/weak_domain_biomedparse_enrichment_20260823"
extension="${biomed}/weak_domain_biomedparse_recall_extension_20260823"
output="${root}/analysis/candidate_refiner_v2_weak_transfer_oof_20260823"
staging="${output}/staging"
calibration="${output}/deployment_calibration.json"
complete="${output}/TRANSFER_OOF_COMPLETE.txt"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v2_weak_transfer_oof_20260823.lock"
flock -n 9 || { echo "weak-enriched transfer OOF controller already active" >&2; exit 1; }
[[ -s "${complete}" ]] && exit 0
while [[ ! -s "${training_complete}" || ! -s "${multisource_save}" \
    || ! -s "${extension}/EXTENSION_COMPLETE.txt" ]]; do
  printf '%s waiting for v2 training, multisource saver, and weak-domain gates\n' "$(date -Ins)"
  sleep 30
done
for required in "${selection}" "${patch_oof}" "${index}" "${manifest}" \
  "${multisource_calibration}" "${initial}/selection.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

# Resolve one frozen exact-210 proposal stream. Prefer the extension when it
# ran; otherwise the extension was skipped because the initial bank was already
# sufficient. The source selection JSON is label-informed and recorded as such.
selected_proposal=$("${python_bin}" - "${initial}" "${extension}" \
  "${output}/proposal_selection.json" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
initial,extension,output=map(Path,sys.argv[1:])
if (extension/"selection.json").is_file():
    source="extension"
    report_path=extension/"selection.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    proposal=Path(report["selected"]["proposal"])
else:
    source="initial"
    report_path=initial/"selection.json"
    report=json.loads(report_path.read_text(encoding="utf-8"))
    threshold=float(report["selected"]["existence_threshold"])
    proposal=initial/f"enriched_e{round(100*threshold):03d}"
proposal=proposal.resolve(strict=True)
files=sorted(proposal.glob("*.nii.gz"))
if len(files)!=210: raise SystemExit(f"expected exact 210 proposals, found {len(files)}")
digest=hashlib.sha256()
for path in files:
    digest.update(path.name.encode()); digest.update(b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
payload={
  "created_at":datetime.now().astimezone().isoformat(),
  "source_stage":source,
  "source_selection":str(report_path.resolve()),
  "source_decision":report["decision"],
  "proposal":str(proposal),
  "cases":210,
  "proposal_sha256_inventory":digest.hexdigest(),
  "proposal_masks_generated_without_labels":True,
  "proposal_hyperparameter_selected_with_complete_training_labels":True,
  "public_or_hidden_data_read":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(proposal)
PY
)

"${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
  --index "${index}" --manifest "${manifest}" --proposals "${selected_proposal}" \
  --output "${staging}" --folds 5 --seed 20260822 \
  > "${output}/stage.log" 2>&1
[[ -s "${staging}/STAGING_COMPLETE.json" ]]

checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' \
  "${selection}")
prepare_fold() {
  local fold="$1" fold_stage="${staging}/fold${fold}" work
  work="$(fold_work "${fold}")"
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${fold_stage}/images" --proposal "weak=${fold_stage}/proposals" \
      --base "${fold_stage}/proposals" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 \
      --tile-stride-mm 72 > "${output}/fold${fold}.prepare.log" 2>&1
  fi
}
prepare_sequence() { local fold; for fold in "$@"; do prepare_fold "${fold}"; done; }
prepare_sequence 0 2 4 & prep0=$!
prepare_sequence 1 3 & prep1=$!
status=0; wait "${prep0}" || status=$?; wait "${prep1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

inference_matches() {
  local work="$1" checkpoint="$2"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" <<'PY'
import json,sys
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(sys.argv[2]).resolve())]
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text())
digest=__import__("hashlib").sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
valid=(payload.get("checkpoints",[])==expected
       and payload.get("checkpoint_sha256",[])==[digest]
       and payload.get("contour_checkpoints",[])==[]
       and payload.get("contour_checkpoint_sha256",[])==[])
raise SystemExit(0 if valid else 1)
PY
}

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
infer_fold() {
  local gpu="$1" fold="$2" work checkpoint
  work="$(fold_work "${fold}")"; checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; return 2; }
  inference_matches "${work}" "${checkpoint}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${checkpoint}" --batch-size 8 --workers 2 \
    > "${output}/fold${fold}.infer.log" 2>&1
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }

all_inference_cached=1
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  if [[ ! -s "${checkpoint}" ]] || ! inference_matches "${work}" "${checkpoint}"; then
    all_inference_cached=0
  fi
done
if [[ "${all_inference_cached}" -eq 0 ]]; then
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    free0=0; free1=0
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 4096 ]] && free0=1
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 4096 ]] && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
  if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
    infer_sequence 0 0 2 4 & infer0=$!
    infer_sequence 1 1 3 & infer1=$!
    status=0; wait "${infer0}" || status=$?; wait "${infer1}" || status=$?
    [[ "${status}" -eq 0 ]] || exit "${status}"
  elif [[ "${free0}" -eq 1 ]]; then
    infer_sequence 0 0 1 2 3 4
  else
    infer_sequence 1 0 1 2 3 4
  fi
else
  printf '%s reusing five provenance-matched inference caches; skipping GPU wait\n' "$(date -Ins)"
fi

gpu_phase_args=()
for fold in 0 1 2 3 4; do
  gpu_phase_args+=(--work "$(fold_work "${fold}")")
  gpu_phase_args+=(--checkpoint "$(fold_dir "${fold}")/${checkpoint_name}")
done
"${python_bin}" -u "${root}/write_candidate_refiner_gpu_phase_marker.py" \
  "${gpu_phase_args[@]}" --output "${output}/GPU_PHASE_COMPLETE.json" \
  > "${output}/gpu_phase_complete.log" 2>&1

audit_cache_valid() {
  local work="$1"
  "${python_bin}" - "${work}/AUDIT.json" "${work}/INFERENCE_COMPLETE.json" <<'PY'
import json,sys
from pathlib import Path
audit,marker=map(Path,sys.argv[1:])
if not audit.is_file() or not marker.is_file() or audit.stat().st_mtime < marker.stat().st_mtime:
    raise SystemExit(1)
payload=json.loads(audit.read_text(encoding="utf-8"))
valid=(payload.get("exact_file_coverage") is True
 and payload.get("all_prompts_nonempty") is True
 and payload.get("all_values_finite_and_normalized") is True
 and int(payload.get("cases",0))>0 and int(payload.get("candidates",0))>0)
raise SystemExit(0 if valid else 1)
PY
}
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  if audit_cache_valid "${work}"; then
    printf '%s reusing valid fold%s inference-payload audit\n' "$(date -Ins)" "${fold}"
  else
    "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
      --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
      > "${output}/fold${fold}.audit.log" 2>&1
  fi
done

classification_values=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
values=json.load(open(sys.argv[1]))["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected three frozen thresholds")
print(" ".join(f"{float(value):.2f}" for value in values))
PY
)
segmentation=$("${python_bin}" -c \
  'import json,sys;print("{:.2f}".format(float(json.load(open(sys.argv[1]))["segmentation"]["threshold"])))' \
  "${patch_oof}")
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"

materialization_grid="${output}/materialization_grid.json"
# A terminal retry after scoring must not repeat the expensive exact-210
# shared-pass materialization. The summarizer independently verifies all case
# rows and metric means before accepting this cached summary.
if [[ ! -s "${output}/summary_by_dsc.tsv" ]]; then
"${python_bin}" - "${classification_values}" "${segmentation}" "${materialization_grid}" <<'PY'
import json,sys
from pathlib import Path
values=[float(value) for value in sys.argv[1].split()]
segmentation=float(sys.argv[2]); output=Path(sys.argv[3]); configs=[]
for classification in values:
    for mode in ("refined",):
        configs.append({
          "tag":f"stand_c{classification:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+f"_{mode}",
          "classification_threshold":classification,"mode":mode,
          "source_classification_thresholds":{},
        })
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n")
PY
materialize_grid_fold() {
  local fold="$1" work
  work="$(fold_work "${fold}")"
  "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
    --work "${work}" --base "${staging}/fold${fold}/proposals" --empty-base \
    --output-root "${output}" --configurations "${materialization_grid}" \
    --fold "${fold}" --segmentation-threshold "${segmentation}" \
    --min-seed-coverage 0 --min-refined-overlap 0 --min-volume-ratio 0 \
    --max-volume-ratio 1000000000000 > "${output}/fold${fold}.materialize_grid.log" 2>&1
}
pids=(); for fold in 0 1 2 3 4; do materialize_grid_fold "${fold}" & pids+=("$!"); done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
[[ "${status}" -eq 0 ]] || exit "${status}"

for classification in ${classification_values}; do
  for mode in refined; do
    tag="stand_c${classification/./p}_s${segmentation/./p}_${mode}"
    combined_dir="${output}/${tag}/combined"
    "${python_bin}" - "${output}/${tag}" "${combined_dir}" "${staging}/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); combined.mkdir(parents=True,exist_ok=True); seen={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in seen: raise SystemExit(f"duplicate: {source.name}")
        target=combined/source.name
        if target.is_symlink() or target.exists():
            if target.resolve(strict=True)!=source.resolve(strict=True): raise SystemExit(f"collision: {target}")
        else: target.symlink_to(source.resolve(strict=True))
        seen[source.name]=source
expected={path.name for path in labels.glob("*.nii.gz")}
if len(expected)!=210 or set(seen)!=expected: raise SystemExit("OOF coverage mismatch")
PY
    metrics="${output}/${tag}.metrics.csv"; score_log="${output}/${tag}.score.log"
    "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
      --pred "${combined_dir}" --gt "${staging}/all/labels" --csv "${metrics}" \
      --workers 4 > "${score_log}" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
      "${segmentation}" "${mode}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${output}/summary.tsv"
  done
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr; } \
  > "${output}/summary_by_dsc.tsv"
fi
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 \
  --seed 20260823 > "${output}/summarize.log" 2>&1

"${python_bin}" - "${calibration}" "${output}/proposal_selection.json" <<'PY'
import json,sys
from pathlib import Path
calibration,proposal=map(Path,sys.argv[1:])
payload=json.loads(calibration.read_text()); proposal_payload=json.loads(proposal.read_text())
payload["proposal_selection"]=proposal_payload
payload["selection_basis"]=(
  "case-held-out grouped v2 model predictions and patch-OOF-declared model thresholds; "
  "the weak-domain proposal bank/threshold was preselected using complete training-label "
  "ceiling analysis; no public or hidden data"
)
payload["calibration_caveat"] += (
  " This transfer diagnostic is not fully nested with respect to proposal-bank/threshold "
  "selection and requires independent public qualification before deployment."
)
calibration.write_text(json.dumps(payload,indent=2)+"\n")
PY

baseline_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' \
  "${multisource_calibration}")
candidate_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${calibration}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${baseline_metrics}" --candidate "${candidate_metrics}" \
  --anchor-name multisource_oof --candidate-name weak_enriched_transfer_oof \
  --output "${output}/vs_multisource_oof.robust.json" --bootstrap 20000 \
  --seed 20260823 > "${output}/vs_multisource_oof.robust.log"
date -Ins > "${complete}"
cat "${output}/summary_by_dsc.tsv"
cat "${output}/vs_multisource_oof.robust.json"

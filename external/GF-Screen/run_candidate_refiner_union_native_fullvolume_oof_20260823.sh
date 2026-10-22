#!/usr/bin/env bash
set -euo pipefail

# Generic exact-210 grouped-fold full-volume calibration on the audited
# epoch105+current-GF proposal union. Defaults run union-native checkpoints;
# the transfer wrapper supplies matched fixed-checkpoint inputs and no anchor.
# No public/hidden data or packaging path.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
exact707_priority_terminal="${root}/public_sweeps/candidate_refiner_v4_cell707_saved_20260823/SAVE_ONLY_COMPLETE.txt"
role="${UNION_OOF_ROLE:-native}"
train="${UNION_OOF_TRAIN_ROOT:-${root}/analysis/candidate_refiner_union_native_training_20260823}"
fivefold="${UNION_OOF_FIVEFOLD_ROOT:-${train}/fivefold}"
fivefold_terminal="${UNION_OOF_FIVEFOLD_TERMINAL:-${fivefold}/UNION_NATIVE_FIVEFOLD_TERMINAL.json}"
selection="${UNION_OOF_SELECTION:-${fivefold}/checkpoint_selection.json}"
patch_oof="${UNION_OOF_PATCH_OOF:-${fivefold}/five_fold_oof_summary.json}"
decoupled_terminal="${train}/decoupled_oof/TERMINAL.json"
decoupled_decision="${train}/decoupled_oof/DECISION.json"
decoupled_summary="${train}/decoupled_oof/five_fold_oof_summary.json"
index="${train}/patches_bbox24_union_v1/index.csv"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
baseline_root="${UNION_OOF_CACHE_ROOT:-${root}/analysis/candidate_refiner_v2_multisource_oof_20260823}"
baseline_complete="${UNION_OOF_CACHE_COMPLETE:-${baseline_root}/FULL_VOLUME_OOF_COMPLETE.txt}"
anchor_root="${UNION_OOF_ANCHOR_ROOT:-${root}/analysis/candidate_refiner_union_transfer_fullvolume_oof_20260823}"
anchor_terminal="${UNION_OOF_ANCHOR_TERMINAL:-${anchor_root}/UNION_TRANSFER_OOF_TERMINAL.json}"
baseline_calibration="${UNION_OOF_ANCHOR_CALIBRATION:-${anchor_root}/deployment_calibration.json}"
output="${UNION_OOF_OUTPUT:-${root}/analysis/candidate_refiner_union_native_fullvolume_oof_20260823}"
deployment_selection="${output}/deployment_checkpoint_selection.json"
calibration="${output}/deployment_calibration.json"
robust="${output}/paired_vs_transfer.robust.json"
terminal="${UNION_OOF_TERMINAL:-${output}/UNION_NATIVE_OOF_TERMINAL.json}"
run_prefix="${UNION_OOF_RUN_PREFIX:-candidate_refiner_union_native_b32_fold}"
variant_prefix="${UNION_OOF_VARIANT_PREFIX:-un}"
controller_lock="${UNION_OOF_LOCK:-${root}/runs/run_candidate_refiner_union_native_fullvolume_oof_20260823.lock}"

case "${role}" in native|transfer);; *) echo "invalid UNION_OOF_ROLE: ${role}" >&2; exit 2;; esac
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }
source_work() { printf '%s/fold%s/work_bbox24' "${baseline_root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "union ${role} full-volume OOF controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
write_failure() {
  local status=$? line="${BASH_LINENO[0]:-unknown}"
  [[ "${status}" -eq 0 ]] && return 0
  trap - EXIT
  if [[ ! -s "${terminal}" ]]; then
    "${python_bin}" - "${terminal}" "${status}" "${line}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
output=Path(sys.argv[1]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"KEEP_TRANSFER","exit_status":int(sys.argv[2]),
 "failed_line":sys.argv[3],"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  fi
  exit "${status}"
}
trap write_failure EXIT

while [[ ! -s "${fivefold_terminal}" || ! -s "${baseline_complete}" ]]; do
  printf '%s waiting for union %s five-fold and audited union cache\n' "$(date -Ins)" "${role}"
  sleep 30
done
if [[ "${role}" == native ]]; then
  while [[ ! -s "${anchor_terminal}" ]]; do
    printf '%s waiting for matched transfer full-volume calibration\n' "$(date -Ins)"
    sleep 30
  done
  # The CPU-only patch gate may qualify independently selected classification
  # and contour checkpoints. Evaluate that exact pair on full volumes rather
  # than silently falling back to the coupled segmentation checkpoint.
  if [[ -z "${UNION_OOF_SELECTION:-}" ]]; then
    while [[ ! -s "${decoupled_terminal}" ]]; do
      printf '%s waiting for union-native decoupled patch-OOF terminal\n' "$(date -Ins)"
      sleep 30
    done
    decoupled_status=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' \
      "${decoupled_decision}")
    if [[ "${decoupled_status}" == QUALIFY_DECOUPLED_FULL_VOLUME ]]; then
      "${python_bin}" - "${selection}" "${decoupled_decision}" \
        "${decoupled_summary}" "${deployment_selection}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
baseline, decision, summary, output = map(Path, sys.argv[1:])
payload = {
    "schema_version": 1,
    "created_at": datetime.datetime.now().astimezone().isoformat(),
    "selection_basis": "grouped five-fold decoupled patch OOF only; no public or hidden labels",
    "selected_strategy": "decoupled",
    "checkpoint_filename": "checkpoint_best_classification.pt",
    "prediction_filename": "validation_predictions_best_classification.npz",
    "contour_checkpoint_filename": "checkpoint_best_segmentation.pt",
    "contour_prediction_filename": "validation_predictions_best_segmentation.npz",
    "classification_eligible": True,
    "baseline_selection": str(baseline.resolve()),
    "baseline_selection_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    "decision": str(decision.resolve()),
    "decision_sha256": hashlib.sha256(decision.read_bytes()).hexdigest(),
    "patch_oof": str(summary.resolve()),
    "patch_oof_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
}
temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, output)
PY
      selection="${deployment_selection}"
      patch_oof="${decoupled_summary}"
    fi
  fi
  if [[ "${selection}" != "${deployment_selection}" ]]; then
    cp "${selection}" "${deployment_selection}"
    selection="${deployment_selection}"
  fi
fi
fivefold_status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${fivefold_terminal}")
if [[ "${fivefold_status}" != COMPLETE ]]; then
  "${python_bin}" - "${fivefold_terminal}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_TRANSFER",
 "reason":"union-native five-fold gate did not pass","source":str(source.resolve()),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
required_assets=("${selection}" "${patch_oof}" "${index}" "${manifest}" "${baseline_root}/staging/STAGING_COMPLETE.json")
[[ "${role}" == native ]] && required_assets+=("${baseline_calibration}")
for required in "${required_assets[@]}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

# Reuse only the immutable union patch manifests/NPZ inputs. Native inference
# outputs are isolated under this experiment and cannot overwrite the transfer
# baseline cache.
for fold in 0 1 2 3 4; do
  source="$(source_work "${fold}")/manifest.json"
  source_marker="$(source_work "${fold}")/PREPARE_COMPLETE.txt"
  work="$(fold_work "${fold}")"
  [[ -s "${source}" && -s "${source_marker}" ]] || {
    echo "missing prepared union OOF cache: ${source}" >&2
    exit 2
  }
  mkdir -p "${work}"
  cp "${source}" "${work}/manifest.json"
  cp "${source_marker}" "${work}/PREPARE_COMPLETE.txt"
done

"${python_bin}" - "${patch_oof}" "${selection}" "${role}" "${baseline_calibration}" "${index}" <<'PY'
import csv,json,sys
patch=json.load(open(sys.argv[1],encoding="utf-8")); selection=json.load(open(sys.argv[2],encoding="utf-8"))
role=sys.argv[3]
coverage=patch.get("coverage",{})
with open(sys.argv[5],newline="",encoding="utf-8") as stream:
    expected_patch_cases=len({row["case"] for row in csv.DictReader(stream)})
if coverage.get("cases") != expected_patch_cases or not coverage.get("every_patch_exactly_once"):
    raise SystemExit(f"union {role} patch OOF lacks exact index coverage")
if selection.get("selected_strategy") not in {"composite","segmentation","classification","decoupled"}:
    raise SystemExit(f"invalid union {role} checkpoint selection")
if role=="native":
    baseline=json.load(open(sys.argv[4],encoding="utf-8"))
    if baseline.get("cases") != 210 or not baseline.get("metrics"):
        raise SystemExit("matched transfer full-volume baseline is incomplete")
PY

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
contour_checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("contour_checkpoint_filename","NONE"))' \
  "${selection}")
# The exact-707 branch has substantially higher measured proposal headroom and
# is the only route plausibly able to produce a new candidate before the daily
# submission reset. Preserve native checkpoints but defer this lower-priority
# GPU inference until the exact branch has finished its save-or-skip terminal.
while [[ ! -s "${exact707_priority_terminal}" ]]; do
  printf '%s waiting for exact-707 priority save-or-skip terminal\n' "$(date -Ins)"
  sleep 30
done
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
gpu_is_free() {
  local used
  used=$(gpu_memory_mib "$1")
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 1024 ]]
}
wait_any_gpu() {
  while :; do
    free0=0; free1=0
    gpu_is_free 0 && free0=1
    gpu_is_free 1 && free1=1
    [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && return
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
      "$(date -Ins)" "$(gpu_memory_mib 0)" "$(gpu_memory_mib 1)"
    sleep 30
  done
}
wait_any_gpu

inference_matches() {
  local work="$1" checkpoint="$2" contour_checkpoint="$3"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" "${contour_checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker,checkpoint=map(Path,sys.argv[1:3]); contour_raw=sys.argv[3]
if not marker.is_file(): raise SystemExit(1)
digest=hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
payload=json.loads(marker.read_text(encoding="utf-8"))
contours=[] if contour_raw=="NONE" else [Path(contour_raw)]
valid=(payload.get("checkpoints")==[str(checkpoint.resolve())]
       and payload.get("checkpoint_sha256")==[digest.hexdigest()]
       and payload.get("contour_checkpoints",[])==[str(path.resolve()) for path in contours]
       and payload.get("contour_checkpoint_sha256",[])==[
           hashlib.sha256(path.read_bytes()).hexdigest() for path in contours])
raise SystemExit(0 if valid else 1)
PY
}
infer_fold() {
  local gpu="$1" fold="$2" work checkpoint contour_checkpoint contour_args=()
  work="$(fold_work "${fold}")"
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; return 2; }
  contour_checkpoint=NONE
  if [[ "${contour_checkpoint_name}" != NONE ]]; then
    contour_checkpoint="$(fold_dir "${fold}")/${contour_checkpoint_name}"
    [[ -s "${contour_checkpoint}" ]] || { echo "missing contour checkpoint: ${contour_checkpoint}" >&2; return 2; }
    contour_args+=(--contour-checkpoint "${contour_checkpoint}")
  fi
  inference_matches "${work}" "${checkpoint}" "${contour_checkpoint}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${checkpoint}" "${contour_args[@]}" \
    --batch-size 8 --workers 2 \
    > "${output}/fold${fold}.infer.log" 2>&1
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
status=0
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  infer_sequence 0 0 2 4 & infer0=$!
  infer_sequence 1 1 3 & infer1=$!
  wait "${infer0}" || status=$?
  wait "${infer1}" || status=$?
else
  if [[ "${free0}" -eq 1 ]]; then
    first_gpu=0; second_gpu=1
  else
    first_gpu=1; second_gpu=0
  fi
  # Reserve disjoint folds. Start 0/2/4 immediately on the available device;
  # launch 1/3 as soon as the occupied device is released. If the first worker
  # finishes before that release, reuse its device for 1/3 instead of idling.
  infer_sequence "${first_gpu}" 0 2 4 & first_pid=$!
  second_started=0
  while kill -0 "${first_pid}" 2>/dev/null; do
    if gpu_is_free "${second_gpu}"; then
      infer_sequence "${second_gpu}" 1 3 & second_pid=$!
      second_started=1
      break
    fi
    sleep 30
  done
  wait "${first_pid}" || status=$?
  if [[ "${second_started}" -eq 1 ]]; then
    wait "${second_pid}" || status=$?
  elif [[ "${status}" -eq 0 ]]; then
    infer_sequence "${first_gpu}" 1 3 || status=$?
  fi
fi
[[ "${status}" -eq 0 ]] || exit "${status}"

# A cache cannot be audited before its inference-complete marker exists. Audit
# every newly produced (or exactly matched reused) cache before materializing
# any full-volume predictions.
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    --allow-external-patches --require-predictions \
    > "${output}/fold${fold}.audit.log" 2>&1
done

read -r classification_center classification_values < <("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8")); values=p["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected three classifier thresholds")
print(f'{float(p["classification"]["threshold"]):.2f}', ",".join(f"{float(value):.2f}" for value in values))
PY
)
classification_values="${classification_values//,/ }"
read -r segmentation_center segmentation_values < <(PYTHONPATH="${root}" "${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
from summarize_candidate_refiner_oof import deployment_segmentation_thresholds
center=float(json.load(open(sys.argv[1],encoding="utf-8"))["segmentation"]["threshold"])
values=deployment_segmentation_thresholds(center)
print(f"{center:.2f}", ",".join(f"{value:.2f}" for value in values))
PY
)
segmentation_values="${segmentation_values//,/ }"

materialize_grid_all_folds() {
  local configurations="$1" segmentation="$2" log_tag="$3"
  local fold work base status=0 pid configuration_count group
  local pids=() groups=()
  configuration_count=$("${python_bin}" -c \
    'import json,sys;p=json.load(open(sys.argv[1]));print(len(p.get("configurations",p)))' \
    "${configurations}")
  if [[ "${configuration_count}" -le 1 ]]; then
    # Keep at most three full-volume workers after the live five-worker OOM
    # event; a one-point grid still retains the per-fold CT/probability set.
    groups=("0 1 2" "3 4")
  else
    # Nine native-size outputs can peak together on large head/neck scans.
    # Pair folds to bound memory while retaining shared within-fold resampling.
    groups=("0 1" "2 3" "4")
  fi
  for group in "${groups[@]}"; do
    pids=()
    for fold in ${group}; do
      work="$(fold_work "${fold}")"
      base="${baseline_root}/staging/fold${fold}/proposals"
      "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
        --work "${work}" --base "${base}" --empty-base --output-root "${output}" \
        --configurations "${configurations}" --fold "${fold}" \
        --segmentation-threshold "${segmentation}" --min-seed-coverage 0 \
        --min-refined-overlap 0 --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
        > "${output}/${log_tag}.fold${fold}.materialize_grid.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || return "${status}"
  done
}
score_variant() {
  local classification="$1" segmentation="$2" mode="$3" tag="$4" summary="$5"
  local combined_dir metrics score_log dsc nsd combined
  combined_dir="${output}/${tag}/combined"; mkdir -p "${combined_dir}"
  "${python_bin}" - "${output}/${tag}" "${combined_dir}" "${baseline_root}/staging/all/labels" <<'PY'
import sys
from pathlib import Path
root,combined,labels=map(Path,sys.argv[1:]); seen={}
for fold in range(5):
    for source in (root/f"fold{fold}").glob("*.nii.gz"):
        if source.name in seen: raise SystemExit(f"duplicate prediction: {source.name}")
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
    --pred "${combined_dir}" --gt "${baseline_root}/staging/all/labels" \
    --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation}" "${mode}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${summary}"
}

segmentation_sweep="${output}/segmentation_threshold_sweep.tsv"
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${segmentation_sweep}"
for segmentation in ${segmentation_values}; do
  tag="${variant_prefix}_segcal_c${classification_center/./p}_s${segmentation/./p}_refined"
  grid="${output}/${tag}.grid.json"
  "${python_bin}" - "${tag}" "${classification_center}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
tag=sys.argv[1]; threshold=float(sys.argv[2]); output=Path(sys.argv[3])
output.write_text(json.dumps({"schema_version":1,"configurations":[{
 "tag":tag,"classification_threshold":threshold,"mode":"refined",
 "source_classification_thresholds":{}}]},indent=2)+"\n",encoding="utf-8")
PY
  materialize_grid_all_folds "${grid}" "${segmentation}" "${tag}"
  score_variant "${classification_center}" "${segmentation}" refined "${tag}" "${segmentation_sweep}"
done
segmentation=$("${python_bin}" -u "${root}/select_segmentation_threshold_sweep.py" \
  --summary "${segmentation_sweep}" --patch-oof-center "${segmentation_center}" \
  --output "${output}/SEGMENTATION_THRESHOLD_SELECTION.json")

printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
main_grid="${output}/materialization_grid.json"
"${python_bin}" - "${classification_values}" "${segmentation}" "${variant_prefix}" "${main_grid}" <<'PY'
import json,sys
from pathlib import Path
values=[float(value) for value in sys.argv[1].split()]
segmentation=float(sys.argv[2]); prefix=sys.argv[3]; output=Path(sys.argv[4])
configs=[]
for classification in values:
    for mode in ("refined","seed","seed_union"):
        tag=(f"{prefix}_c{classification:.2f}_s{segmentation:.2f}_{mode}").replace(".","p")
        configs.append({"tag":tag,"classification_threshold":classification,
                        "mode":mode,"source_classification_thresholds":{}})
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n",encoding="utf-8")
PY
materialize_grid_all_folds "${main_grid}" "${segmentation}" main
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
    tag="${variant_prefix}_c${classification/./p}_s${segmentation/./p}_${mode}"
    score_variant "${classification}" "${segmentation}" "${mode}" "${tag}" "${output}/summary.tsv"
  done
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr; } > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 \
  --seed 20260823 > "${output}/summarize.log" 2>&1

if [[ "${role}" == transfer ]]; then
  "${python_bin}" - "${calibration}" "${selection}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
calibration,selection,output=map(Path,sys.argv[1:]); payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"BASELINE_READY",
 "selection_basis":"exact-210 case-held-out grouped-fold transfer full-volume OOF on union proposals",
 "calibration":str(calibration.resolve()),"selection":str(selection.resolve()),
 "calibration_payload":json.loads(calibration.read_text(encoding="utf-8")),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8"); os.replace(temporary,output)
PY
else
  anchor_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${baseline_calibration}")
  candidate_metrics=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${calibration}")
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
    --anchor "${anchor_metrics}" --candidate "${candidate_metrics}" \
    --anchor-name matched_transfer_union_oof --candidate-name union_native_oof \
    --min-dsc-gain 0.005 --min-nsd-gain 0.005 --min-win-rate 0.52 \
    --max-catastrophic-rate 0.10 --bootstrap 20000 --seed 20260823 \
    --output "${robust}" > "${output}/robust.log" 2>&1
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  "${python_bin}" - "${calibration}" "${robust}" "${decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
calibration,robust=map(Path,sys.argv[1:3]); decision=sys.argv[3]; output=Path(sys.argv[4])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"QUALIFY_FOR_FIXED_PUBLIC" if decision=="PROMOTE" else "KEEP_TRANSFER",
 "selection_basis":"exact-210 case-held-out grouped-fold full-volume OOF on identical union proposals",
 "calibration":str(calibration.resolve()),"paired_comparison":str(robust.resolve()),
 "calibration_payload":json.loads(calibration.read_text(encoding="utf-8")),
 "paired_payload":json.loads(robust.read_text(encoding="utf-8")),
 "public_read":False,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
fi
cat "${output}/summary_by_dsc.tsv"

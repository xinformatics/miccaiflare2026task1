#!/usr/bin/env bash
set -euo pipefail

# Generic one-shot public evaluation for an exact-210-calibrated union refiner.
# Defaults target union-native; the transfer wrapper supplies matched-transfer
# paths. Threshold, mode, and family are OOF-frozen. No hidden/package/upload.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
role="${UNION_PUBLIC_ROLE:-native}"
oof="${UNION_PUBLIC_OOF_ROOT:-${root}/analysis/candidate_refiner_union_native_fullvolume_oof_20260823}"
oof_terminal="${UNION_PUBLIC_OOF_TERMINAL:-${oof}/UNION_NATIVE_OOF_TERMINAL.json}"
expected_oof_decision="${UNION_PUBLIC_EXPECTED_OOF_DECISION:-QUALIFY_FOR_FIXED_PUBLIC}"
calibration="${UNION_PUBLIC_CALIBRATION:-${oof}/deployment_calibration.json}"
train="${UNION_PUBLIC_TRAIN_ROOT:-${root}/analysis/candidate_refiner_union_native_training_20260823/fivefold}"
selection="${UNION_PUBLIC_SELECTION:-${oof}/deployment_checkpoint_selection.json}"
baseline_public="${root}/public_sweeps/candidate_refiner_v2_multisource_20260823"
baseline_complete="${baseline_public}/PUBLIC_ONLY_COMPLETE.txt"
source_work="${baseline_public}/work_all_bbox24"
labels="${project}/validation/Validation-Public-Labels"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
output="${UNION_PUBLIC_OUTPUT:-${root}/public_sweeps/candidate_refiner_union_native_20260823}"
work="${output}/work_all_bbox24"
terminal="${UNION_PUBLIC_TERMINAL:-${output}/UNION_NATIVE_PUBLIC_TERMINAL.json}"
run_prefix="${UNION_PUBLIC_RUN_PREFIX:-candidate_refiner_union_native_b32_fold}"
variant_prefix="${UNION_PUBLIC_VARIANT_PREFIX:-un}"
controller_lock="${UNION_PUBLIC_LOCK:-${root}/runs/run_candidate_refiner_union_native_public_20260823.lock}"

case "${role}" in native|transfer);; *) echo "invalid UNION_PUBLIC_ROLE: ${role}" >&2; exit 2;; esac
fold_dir() { printf '%s/runs/%s%s_20260823' "${root}" "${run_prefix}" "$1"; }

mkdir -p "${output}" "${work}"
exec 9>"${controller_lock}"
flock -n 9 || { echo "union ${role} public controller already active" >&2; exit 1; }
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
 "status":"FAILED","decision":"KEEP_ANCHOR","exit_status":int(sys.argv[2]),
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

while [[ ! -s "${oof_terminal}" ]]; do
  printf '%s waiting for union %s exact-210 OOF decision\n' "$(date -Ins)" "${role}"
  sleep 30
done
oof_decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${oof_terminal}")
if [[ "${oof_decision}" != "${expected_oof_decision}" ]]; then
  "${python_bin}" - "${oof_terminal}" "${terminal}" "${role}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:3]); role=sys.argv[3]; payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_ANCHOR",
 "reason":f"union {role} model did not pass exact-210 full-volume OOF",
 "source":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
while [[ ! -s "${baseline_complete}" ]]; do
  printf '%s waiting for audited public union cache\n' "$(date -Ins)"
  sleep 30
done
for required in "${calibration}" "${selection}" "${source_work}/manifest.json" \
  "${source_work}/PREPARE_COMPLETE.txt" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
[[ $(find "${labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
[[ $(find "${base}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
cp "${source_work}/manifest.json" "${work}/manifest.json"
cp "${source_work}/PREPARE_COMPLETE.txt" "${work}/PREPARE_COMPLETE.txt"
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT.json" --workers 4 --allow-external-patches \
  > "${output}/audit_cache.log" 2>&1

checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")
contour_checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1])).get("contour_checkpoint_filename","NONE"))' \
  "${selection}")
checkpoint_args=(); checkpoint_paths=(); contour_checkpoint_args=(); contour_checkpoint_paths=()
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 2; }
  checkpoint_args+=(--checkpoint "${checkpoint}"); checkpoint_paths+=("${checkpoint}")
  if [[ "${contour_checkpoint_name}" != NONE ]]; then
    contour_checkpoint="$(fold_dir "${fold}")/${contour_checkpoint_name}"
    [[ -s "${contour_checkpoint}" ]] || { echo "missing contour checkpoint: ${contour_checkpoint}" >&2; exit 2; }
    contour_checkpoint_args+=(--contour-checkpoint "${contour_checkpoint}")
    contour_checkpoint_paths+=("${contour_checkpoint}")
  fi
done

gpu_memory_mib() { nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'; }
wait_any_gpu() {
  while :; do
    local used0 used1
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le 1024 ]]; then selected_gpu=0; return; fi
    if [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le 1024 ]]; then selected_gpu=1; return; fi
    printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
}
inference_matches() {
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" 5 \
    "${checkpoint_paths[@]}" "${contour_checkpoint_paths[@]}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); count=int(sys.argv[2])
checkpoints=list(map(Path,sys.argv[3:3+count])); contours=list(map(Path,sys.argv[3+count:]))
if not marker.is_file(): raise SystemExit(1)
def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    return digest.hexdigest()
payload=json.loads(marker.read_text(encoding="utf-8"))
valid=(payload.get("checkpoints")==[str(path.resolve()) for path in checkpoints]
       and payload.get("checkpoint_sha256")==[sha(path) for path in checkpoints]
       and payload.get("contour_checkpoints",[])==[str(path.resolve()) for path in contours]
       and payload.get("contour_checkpoint_sha256",[])==[sha(path) for path in contours])
raise SystemExit(0 if valid else 1)
PY
}
if ! inference_matches; then
  wait_any_gpu
  CUDA_VISIBLE_DEVICES="${selected_gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" "${checkpoint_args[@]}" "${contour_checkpoint_args[@]}" \
    --batch-size 8 --workers 2 \
    > "${output}/infer.log" 2>&1
fi
inference_matches
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${work}" --output "${work}/AUDIT_INFERENCE.json" --workers 4 \
  --allow-external-patches --require-predictions \
  > "${output}/audit_inference.log" 2>&1

read -r classification segmentation deployment_mode < <("${python_bin}" - "${calibration}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(f'{float(p["classification_threshold"]):.2f} {float(p["segmentation_threshold"]):.2f} {p["deployment_mode"]}')
PY
)
case "${deployment_mode}" in refined|seed|seed_union);; *) echo "invalid deployment mode" >&2; exit 2;; esac
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${output}/summary.tsv"

# Materialize the anchor and standalone deployments concurrently with the
# parity-tested shared-case grid implementation. The original CLI reloads and
# resamples every full volume once per deployment. Grid outputs are merged into
# the historical flat directories, and any partial reference outputs must be
# exactly voxel/affine identical before they are retained.
materialize_public_deployment() {
  local deployment="$1" tag prediction grid grid_root source empty_arg=()
  tag="${variant_prefix}_${deployment}_${deployment_mode}_c${classification/./p}_s${segmentation/./p}"
  prediction="${output}/${tag}"
  grid="${output}/${tag}.grid.json"
  grid_root="${output}/${tag}.grid"
  [[ "${deployment}" == standalone ]] && empty_arg+=(--empty-base)
  "${python_bin}" - "${tag}" "${classification}" "${deployment_mode}" "${grid}" <<'PY'
import json, sys
from pathlib import Path

tag, classification, mode, output = sys.argv[1], float(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
payload = {"schema_version": 1, "configurations": [{
    "tag": tag,
    "classification_threshold": classification,
    "mode": mode,
    "source_classification_thresholds": {},
}]}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
  "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
    --work "${work}" --base "${base}" --output-root "${grid_root}" \
    --configurations "${grid}" --fold 0 \
    --segmentation-threshold "${segmentation}" "${empty_arg[@]}" \
    --min-seed-coverage 0 --min-refined-overlap 0 \
    --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
    > "${prediction}.materialize.log" 2>&1
  source="${grid_root}/${tag}/fold0"
  "${python_bin}" - "${source}" "${prediction}" "${base}" <<'PY'
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

source, target, expected_root = map(Path, sys.argv[1:])
target.mkdir(parents=True, exist_ok=True)
expected = {path.name for path in expected_root.glob("*.nii.gz")}
generated = {path.name for path in source.glob("*.nii.gz")}
if len(expected) != 50 or generated != expected:
    raise SystemExit(f"public grid coverage mismatch: expected={len(expected)} generated={len(generated)}")
for name in sorted(generated):
    origin = source / name
    destination = target / name
    if destination.is_symlink() or destination.exists():
        reference = nib.load(str(destination))
        candidate = nib.load(str(origin))
        if reference.shape != candidate.shape or not np.array_equal(reference.affine, candidate.affine):
            raise SystemExit(f"public grid geometry mismatch: {name}")
        if not np.array_equal(np.asanyarray(reference.dataobj), np.asanyarray(candidate.dataobj)):
            raise SystemExit(f"public grid voxel mismatch: {name}")
    else:
        destination.symlink_to(origin.resolve(strict=True))
flat = {path.name for path in target.glob("*.nii.gz")}
if flat != expected:
    raise SystemExit(f"flat public coverage mismatch: expected={len(expected)} flat={len(flat)}")
PY
}

materialization_status=0
materialization_pids=()
for deployment in anchor standalone; do
  materialize_public_deployment "${deployment}" &
  materialization_pids+=("$!")
done
for pid in "${materialization_pids[@]}"; do
  wait "${pid}" || materialization_status=$?
done
[[ "${materialization_status}" -eq 0 ]]

for deployment in anchor standalone; do
  tag="${variant_prefix}_${deployment}_${deployment_mode}_c${classification/./p}_s${segmentation/./p}"
  prediction="${output}/${tag}"
  [[ $(find "${prediction}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]]
  metrics="${prediction}.metrics.csv"; score_log="${prediction}.score.log"; robust="${prediction}.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
    --gt "${labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" --anchor "${anchor_metrics}" \
    --candidate "${metrics}" --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  gate=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation}" "${deployment_mode}" "${deployment}" "${dsc}" "${nsd}" "${combined}" \
    "${metrics}" "${robust}" "${gate}" >> "${output}/summary.tsv"
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr; } > "${output}/summary_by_dsc.tsv"
selected=$("${python_bin}" "${root}/select_promoted_refiner_summary.py" --summary "${output}/summary.tsv")
decision=KEEP_ANCHOR; [[ -n "${selected}" ]] && decision=PROMOTE
"${python_bin}" - "${calibration}" "${output}/summary.tsv" "${decision}" "${terminal}" <<'PY'
import csv,datetime,json,os,sys
from pathlib import Path
calibration,summary=map(Path,sys.argv[1:3]); decision=sys.argv[3]; output=Path(sys.argv[4])
with summary.open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream,delimiter="\t"))
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision,"calibration":str(calibration.resolve()),
 "summary":str(summary.resolve()),
 "promoted_variants":[row["variant"] for row in rows if row.get("decision")=="PROMOTE"],
 "public_read":True,"hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"
cat "${output}/summary_by_dsc.tsv"

#!/usr/bin/env bash
set -euo pipefail

# Controlled proposal-domain transfer arm. The refiner is trained only on the
# certified epoch-105 stream, then evaluated without retraining on the
# label-blind union of epoch-105 and current-GF proposals. Full-volume OOF is
# completed before public labels are read. Hidden data are never accessed.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v2_training_20260822"
training_complete="${train}/TRAINING_COMPLETE.txt"
selection="${train}/checkpoint_selection.json"
patch_oof="${train}/five_fold_oof_summary.json"
index="${train}/patches_bbox24_min5_v5/index.csv"
manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
union_root="${root}/analysis/candidate_refiner_v2_multisource_20260823"
proposals="${union_root}/proposal_union"
union_audit="${union_root}/PROPOSAL_UNION_COMPLETE.json"
output="${root}/analysis/candidate_refiner_v2_multisource_oof_20260823"
staging="${output}/staging"
calibration="${output}/deployment_calibration.json"
primary_saved="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
focus_saved="${root}/public_sweeps/focus_candidate_refiner_saved_20260823/FOCUS_SAVE_ONLY_COMPLETE.txt"
public="${root}/public_sweeps/candidate_refiner_v2_multisource_20260823"
public_images="${project}/validation/Validation-Public-Images"
public_labels="${project}/validation/Validation-Public-Labels"
public_epoch105="${root}/public_sweeps/epoch0105/abdomen_g09_l05/predictions"
public_currentgf="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
public_union="${public}/proposal_union"
public_base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
anchor_metrics="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
public_work="${public}/work_all_bbox24"

fold_dir() { printf '%s/runs/candidate_refiner_v2_bbox24_b32_v2det_fold%s_20260823' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}" "${public}"
exec 9>"${root}/runs/run_candidate_refiner_v2_multisource_after_primary_20260823.lock"
flock -n 9 || { echo "candidate-refiner multisource controller already active" >&2; exit 1; }
[[ -s "${public}/PUBLIC_ONLY_COMPLETE.txt" ]] && exit 0
for required in "${manifest}" "${index}" "${union_audit}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
"${python_bin}" - "${union_audit}" <<'PY'
import json,sys
report=json.load(open(sys.argv[1],encoding="utf-8"))
required=("exact_file_coverage","all_masks_finite_binary","all_geometries_match_source_images","all_unions_exact_voxelwise")
if report.get("cases") != 210 or not all(report.get(key) is True for key in required) or report.get("labels_read") is not False:
    raise SystemExit("proposal union did not pass its fail-closed audit")
PY

# Safe to prepare on CPU while training continues. The union is passed as one
# binary source so prompt intensity matches the one-source training domain.
if [[ ! -s "${staging}/STAGING_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
    --index "${index}" --manifest "${manifest}" --proposals "${proposals}" \
    --output "${staging}" --folds 5 --seed 20260822 \
    > "${output}/stage.log" 2>&1
fi
prepare_fold() {
  local fold="$1" fold_stage="${staging}/fold${fold}" work
  work="$(fold_work "${fold}")"
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${fold_stage}/images" --proposal "union=${fold_stage}/proposals" \
      --base "${fold_stage}/proposals" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 \
      --bbox-margin-mm 24 --tile-stride-mm 72 \
      > "${output}/fold${fold}.prepare.log" 2>&1
  fi
  "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
    --work "${work}" --output "${work}/AUDIT.json" --workers 2 \
    > "${output}/fold${fold}.audit.log" 2>&1
}
prepare_sequence() { local fold; for fold in "$@"; do prepare_fold "${fold}"; done; }
prepare_sequence 0 2 4 & prep0=$!
prepare_sequence 1 3 & prep1=$!
status=0; wait "${prep0}" || status=$?; wait "${prep1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

# Serialize behind the primary OOF/public/save-only path so the two controlled
# experiments cannot race for GPUs or alter which result is preserved.
while [[ ! -s "${training_complete}" || ! -s "${primary_saved}" \
    || ! -s "${focus_saved}" ]]; do
  printf '%s waiting for primary and Focus save-only completion\n' "$(date -Ins)"
  sleep 30
done
for required in "${selection}" "${patch_oof}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
checkpoint_name=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' "${selection}")

inference_matches() {
  local work="$1" checkpoint="$2"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" <<'PY'
import json,sys
import hashlib
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(sys.argv[2]).resolve())]
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text())
digest=hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
valid=(payload.get("checkpoints",[]) == expected and payload.get("checkpoint_sha256",[]) == [digest])
raise SystemExit(0 if valid else 1)
PY
}

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
needs_gpu=0
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  inference_matches "$(fold_work "${fold}")" "${checkpoint}" || needs_gpu=1
done
if [[ "${needs_gpu}" -eq 1 ]]; then
  while :; do
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    if [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
        && "${used0}" -le 4096 && "${used1}" -le 4096 ]]; then break; fi
    printf '%s waiting for GPUs: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
    sleep 30
  done
else
  printf '%s all five inference caches match; bypassing GPU availability wait\n' "$(date -Ins)"
fi
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
infer_sequence 0 0 2 4 & infer0=$!
infer_sequence 1 1 3 & infer1=$!
status=0; wait "${infer0}" || status=$?; wait "${infer1}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

classification_values=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
values=json.load(open(sys.argv[1],encoding="utf-8"))["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected exactly three thresholds")
print(" ".join(f"{float(value):.2f}" for value in values))
PY
)
segmentation=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
print(f'{float(json.load(open(sys.argv[1]))["segmentation"]["threshold"]):.2f}')
PY
)
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
materialize_grid() {
  local specs=("$@") variant_args=() spec fold status=0
  for spec in "${specs[@]}"; do variant_args+=(--variant "${spec}"); done
  local group pids=()
  for group in "0 1" "2 3" "4"; do
    pids=()
    for fold in ${group}; do
      "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
        --work "$(fold_work "${fold}")" \
        --base "${staging}/fold${fold}/proposals" --output-root "${output}" \
        --fold "${fold}" --skip-existing "${variant_args[@]}" \
        > "${output}/deployment.fold${fold}.grid.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || return "${status}"
  done
}
deployment_specs=()
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
    tag="stand_c${classification/./p}_s${segmentation/./p}_${mode}"
    deployment_specs+=("${tag},${classification},${segmentation},${mode}")
  done
done
materialize_grid "${deployment_specs[@]}"
for classification in ${classification_values}; do
  for mode in refined seed seed_union; do
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
      --pred "${combined_dir}" --gt "${staging}/all/labels" --csv "${metrics}" --workers 4 \
      > "${score_log}" 2>&1
    dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" "${segmentation}" "${mode}" "${dsc}" "${nsd}" "${combined}" "${metrics}" >> "${output}/summary.tsv"
  done
done
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr; } > "${output}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${calibration}" --bootstrap-samples 5000 --seed 20260823 \
  > "${output}/summarize.log" 2>&1
date -Ins > "${output}/FULL_VOLUME_OOF_COMPLETE.txt"

# Only now materialize and audit the two-source public proposal union.
public_manifest="${public}/public_manifest.json"
"${python_bin}" - "${public_images}" "${public_manifest}" <<'PY'
import json,sys
from pathlib import Path
images=Path(sys.argv[1]); output=Path(sys.argv[2]); items=[]
for image in sorted(images.glob("*_0000.nii.gz")):
    case=image.name[:-12]
    items.append({"name":case,"prediction_name":case+".nii.gz","image":str(image.resolve()),"complete_label":False})
if len(items)!=50 or len({item["name"] for item in items})!=50: raise SystemExit("expected exact 50 public images")
output.write_text(json.dumps({"description":"unlabeled public inference geometry manifest","items":items},indent=2)+"\n")
PY
if [[ ! -s "${public}/PROPOSAL_UNION_COMPLETE.json" ]]; then
  mkdir -p "${public_union}"
  "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
    --pred-a "${public_epoch105}" --pred-b "${public_currentgf}" --fusion union \
    --min-mm3 0 --output "${public_union}" --workers 4 > "${public}/fuse_union.log" 2>&1
  "${python_bin}" -u "${root}/audit_proposal_union.py" \
    --manifest "${public_manifest}" --source-a "${public_epoch105}" \
    --source-b "${public_currentgf}" --union "${public_union}" \
    --output "${public}/PROPOSAL_UNION_COMPLETE.json" --workers 4 --allow-unlabeled \
    > "${public}/audit_union.log" 2>&1
fi

if [[ ! -s "${public_work}/PREPARE_COMPLETE.txt" ]]; then
  "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
    --images "${public_images}" --proposal "union=${public_union}" \
    --base "${public_base}" --work "${public_work}" \
    --shape 64,64,64 --spacing 1.5,1.5,1.5 \
    --min-mm3 5 --max-mm3 1000000000000 --bbox-margin-mm 24 --tile-stride-mm 72 \
    > "${public}/prepare.log" 2>&1
fi
"${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
  --work "${public_work}" --output "${public_work}/AUDIT.json" --workers 4 \
  > "${public}/audit_cache.log" 2>&1
checkpoint_args=(); checkpoint_paths=()
for fold in 0 1 2 3 4; do
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"; checkpoint_args+=(--checkpoint "${checkpoint}"); checkpoint_paths+=("${checkpoint}")
done
"${python_bin}" - "${public_work}/INFERENCE_COMPLETE.json" "${checkpoint_paths[@]}" <<'PY' || needs_inference=1
import json,sys
import hashlib
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(path).resolve()) for path in sys.argv[2:]]
if not marker.is_file(): raise SystemExit(1)
payload=json.loads(marker.read_text())
digests=[hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sys.argv[2:]]
if payload.get("checkpoints",[]) != expected or payload.get("checkpoint_sha256",[]) != digests: raise SystemExit(1)
PY
if [[ "${needs_inference:-0}" -eq 1 ]]; then
  CUDA_VISIBLE_DEVICES=0 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${public_work}" "${checkpoint_args[@]}" --batch-size 8 --workers 2 \
    > "${public}/infer.log" 2>&1
fi
classification=$("${python_bin}" -c 'import json,sys;print("{:.2f}".format(float(json.load(open(sys.argv[1]))["classification_threshold"])))' "${calibration}")
segmentation=$("${python_bin}" -c 'import json,sys;print("{:.2f}".format(float(json.load(open(sys.argv[1]))["segmentation_threshold"])))' "${calibration}")
deployment_mode=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("deployment_mode","refined"))' "${calibration}")
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tdeployment\tDSC\tNSD\tcombined\tmetrics\trobust\tdecision\n' > "${public}/summary.tsv"
anchor_tag="ms_anchor_${deployment_mode}_c${classification/./p}_s${segmentation/./p}"
standalone_tag="ms_standalone_${deployment_mode}_c${classification/./p}_s${segmentation/./p}"
"${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
  --work "${public_work}" --base "${public_base}" --output-root "${public}" \
  --flat-output \
  --variant "${anchor_tag},${classification},${segmentation},${deployment_mode},base" \
  --variant "${standalone_tag},${classification},${segmentation},${deployment_mode},empty" \
  > "${public}/public_anchor_standalone.grid.log" 2>&1
for deployment in anchor standalone; do
  if [[ "${deployment}" == anchor ]]; then tag="${anchor_tag}"; else tag="${standalone_tag}"; fi
  prediction="${public}/${tag}"
  metrics="${prediction}.metrics.csv"; score_log="${prediction}.score.log"; robust="${prediction}.robust.json"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" --pred "${prediction}" \
    --gt "${public_labels}" --csv "${metrics}" --workers 4 > "${score_log}" 2>&1
  "${python_bin}" -u "${root}/select_checkpoint_robust.py" --anchor "${anchor_metrics}" \
    --candidate "${metrics}" --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${robust}" > "${prediction}.robust.log"
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}"); nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" "${segmentation}" "${deployment_mode}" "${deployment}" "${dsc}" "${nsd}" "${combined}" "${metrics}" "${robust}" "${decision}" >> "${public}/summary.tsv"
done
{ head -1 "${public}/summary.tsv"; tail -n +2 "${public}/summary.tsv" | sort -t $'\t' -k6,6gr -k7,7gr; } > "${public}/summary_by_dsc.tsv"
date -Ins > "${public}/PUBLIC_ONLY_COMPLETE.txt"
cat "${public}/summary_by_dsc.tsv"

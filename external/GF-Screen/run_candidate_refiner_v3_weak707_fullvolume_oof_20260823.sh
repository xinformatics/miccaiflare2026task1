#!/usr/bin/env bash
set -euo pipefail

# Exact-707 grouped held-out full-volume evaluation for the headroom-qualified
# scaled refiner. Cache preparation is label-blind; labels are read only after
# every case has been inferred by its excluded fold checkpoint. No public,
# hidden, packaging, submission, or upload path exists.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse/analysis/weak_domain_biomedparse_707_selected_20260823"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
train="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
training_complete="${train}/TRAINING_COMPLETE.json"
training_skipped="${train}/TRAINING_SKIPPED.json"
selection="${train}/checkpoint_selection.json"
patch_oof="${train}/five_fold_oof_summary.json"
patches="${train}/patches_bbox24_ss_min5_v4_cell48"
index="${patches}/index.csv"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
epoch105="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050"
currentgf="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
weak_complete="${biomed}/PROPOSALS_COMPLETE.json"
margin_decision="${root}/analysis/weak707_bbox_margin_headroom_20260823/DECISION.json"
cell_scale="${root}/analysis/lung_cell707_scale_20260823"
cell_decision="${cell_scale}/DECISION.json"
cells="${cell_scale}/cells48_all707"
output="${root}/analysis/candidate_refiner_v4_cell707_fullvolume_oof_20260823"
staging="${output}/staging"
calibration="${output}/deployment_calibration.json"
standalone_calibration="${output}/deployment_calibration_standalone.json"
composition_root="${output}/anchor_currentgf"
composition_calibration="${output}/deployment_calibration_anchor_currentgf.json"
composition_robust="${output}/anchor_currentgf_vs_base.robust.json"
complete="${output}/FULL_VOLUME_OOF_COMPLETE.json"
skipped="${output}/FULL_VOLUME_OOF_SKIPPED.json"
max_idle_gpu_memory_mib="${CELL707_OOF_MAX_IDLE_GPU_MEMORY_MIB:-2048}"
[[ "${max_idle_gpu_memory_mib}" =~ ^[0-9]+$ \
    && "${max_idle_gpu_memory_mib}" -ge 1024 \
    && "${max_idle_gpu_memory_mib}" -le 4096 ]] || {
  echo "invalid idle GPU memory allowance: ${max_idle_gpu_memory_mib}" >&2
  exit 2
}

fold_dir() { printf '%s/runs/candidate_refiner_v4_cell707_b32_t05_fold%s_20260823' "${root}" "$1"; }
fold_work() { printf '%s/fold%s/work_bbox24' "${output}" "$1"; }

mkdir -p "${output}"
exec 9>"${root}/runs/run_candidate_refiner_v3_weak707_fullvolume_oof_20260823.lock"
flock -n 9 || { echo "weak707 full-volume OOF already active" >&2; exit 1; }
[[ -s "${complete}" || -s "${skipped}" ]] && exit 0

# The training controller can explicitly decline the experiment when proposal
# headroom is insufficient. Do not wait indefinitely in that case.
while [[ ! -s "${training_complete}" && ! -s "${training_skipped}" \
    && ! -s "${patches}/COMPLETE.txt" ]]; do
  printf '%s waiting for weak707 patch cache or terminal training decision\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${training_skipped}" ]]; then
  "${python_bin}" - "${training_skipped}" "${skipped}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={
  "created_at":datetime.now().astimezone().isoformat(),
  "decision":"SKIP_FULL_VOLUME_OOF",
  "reason":"scaled training did not pass its proposal-headroom prerequisite",
  "training_decision":json.loads(source.read_text(encoding="utf-8")),
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY
  exit 0
fi

# As soon as the training patch index exists, prewarm the deployment caches on
# CPU. This does not inspect the labels linked into staging.
for required in "${manifest}" "${index}" "${weak_complete}" "${margin_decision}" \
  "${cell_decision}" "${cells}/PROPOSALS_COMPLETE.json"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
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
cell_stage_args=()
for colour in 0 1 2 3 4 5 6 7; do
  cell_stage_args+=(--proposal "lung_cell${colour}=${cells}/colour${colour}")
done
if [[ ! -s "${staging}/STAGING_COMPLETE.json" ]]; then
  "${python_bin}" -u "${root}/stage_candidate_refiner_oof_cases.py" \
    --index "${index}" --manifest "${manifest}" \
    --proposal "epoch105=${epoch105}" --proposal "currentgf=${currentgf}" \
    --proposal "autopet_weak=${autopet}" \
    --proposal "lndb_weak=${lndb}" "${cell_stage_args[@]}" --output "${staging}" \
    --folds 5 --seed 20260823 > "${output}/stage.log" 2>&1
fi

prepare_fold() {
  local fold="$1" fold_stage="${staging}/fold${fold}" work
  work="$(fold_work "${fold}")"
  if [[ ! -s "${work}/PREPARE_COMPLETE.txt" ]]; then
    local cell_fold_args=() colour
    for colour in 0 1 2 3 4 5 6 7; do
      cell_fold_args+=(--proposal "lung_cell${colour}=${fold_stage}/proposals/lung_cell${colour}")
    done
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${fold_stage}/images" \
      --proposal "epoch105=${fold_stage}/proposals/epoch105" \
      --proposal "currentgf=${fold_stage}/proposals/currentgf" \
      --proposal "autopet_weak=${fold_stage}/proposals/autopet_weak" \
      --proposal "lndb_weak=${fold_stage}/proposals/lndb_weak" \
      "${cell_fold_args[@]}" \
      --component-policy source_separate \
      --base "${fold_stage}/proposals/epoch105" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 \
      --min-mm3 5 --max-mm3 1000000000000 \
      --bbox-margin-mm "${bbox_margin}" --tile-stride-mm 72 --resume \
      > "${output}/fold${fold}.prepare.log" 2>&1
  fi
}
prepare_sequence() { local fold; for fold in "$@"; do prepare_fold "${fold}"; done; }
# Every fold writes to an independent work tree.  Run one resumable CPU
# builder per fold so cache creation completes alongside GPU training instead
# of serializing three folds behind fold 0.  Five builders remain bounded well
# below the audited host RAM headroom and do not inspect labels.
prepare_pids=()
for prepare_fold_index in 0 1 2 3 4; do
  prepare_sequence "${prepare_fold_index}" &
  prepare_pids+=("$!")
done
status=0
for prepare_pid in "${prepare_pids[@]}"; do
  wait "${prepare_pid}" || status=$?
done
[[ "${status}" -eq 0 ]] || exit "${status}"

while [[ ! -s "${training_complete}" && ! -s "${training_skipped}" ]]; do
  printf '%s deployment caches ready; waiting for five-fold training\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${training_skipped}" ]]; then
  printf '%s training skipped after cache preparation\n' "$(date -Ins)" > "${skipped}"
  exit 0
fi
for required in "${selection}" "${patch_oof}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
checkpoint_name=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["checkpoint_filename"])' \
  "${selection}")

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
  free0=0; free1=0
  [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le "${max_idle_gpu_memory_mib}" ]] && free0=1
  [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le "${max_idle_gpu_memory_mib}" ]] && free1=1
  [[ "${free0}" -eq 1 || "${free1}" -eq 1 ]] && break
  printf '%s waiting for any GPU: gpu0=%s MiB gpu1=%s MiB\n' \
    "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done
# The paired fold trainers normally release the two GPUs within a few minutes
# of one another.  Give the second device a bounded grace period so one early
# release cannot accidentally serialize all five full-volume folds.  If the
# second device belongs to another workload, inference still proceeds on the
# first device after the deadline.
if [[ $((free0 + free1)) -eq 1 ]]; then
  second_gpu_deadline=$((SECONDS + 300))
  while [[ "${SECONDS}" -lt "${second_gpu_deadline}" ]]; do
    printf '%s one GPU free; waiting up to five minutes for dual-GPU OOF dispatch\n' \
      "$(date -Ins)"
    sleep 30
    used0=$(gpu_memory_mib 0); used1=$(gpu_memory_mib 1)
    [[ "${used0}" =~ ^[0-9]+$ && "${used0}" -le "${max_idle_gpu_memory_mib}" ]] && free0=1 || free0=0
    [[ "${used1}" =~ ^[0-9]+$ && "${used1}" -le "${max_idle_gpu_memory_mib}" ]] && free1=1 || free1=0
    [[ $((free0 + free1)) -eq 2 ]] && break
  done
fi

inference_matches() {
  local work="$1" checkpoint="$2"
  "${python_bin}" - "${work}/INFERENCE_COMPLETE.json" "${checkpoint}" <<'PY'
import hashlib,json,sys
from pathlib import Path
marker=Path(sys.argv[1]); expected=[str(Path(sys.argv[2]).resolve())]
if not marker.is_file(): raise SystemExit(1)
checkpoint=Path(sys.argv[2]); digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
payload=json.loads(marker.read_text())
valid=(payload.get("checkpoints",[])==expected
       and payload.get("checkpoint_sha256",[])==[digest]
       and payload.get("contour_checkpoints",[])==[]
       and payload.get("contour_checkpoint_sha256",[])==[]
       and payload.get("prediction_cache_compressed") is False)
raise SystemExit(0 if valid else 1)
PY
}
infer_fold() {
  local gpu="$1" fold="$2" work checkpoint
  work="$(fold_work "${fold}")"
  checkpoint="$(fold_dir "${fold}")/${checkpoint_name}"
  [[ -s "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; return 2; }
  inference_matches "${work}" "${checkpoint}" && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${checkpoint}" --batch-size 32 --workers 2 \
    --uncompressed-predictions --resume \
    > "${output}/fold${fold}.infer.log" 2>&1
}
infer_sequence() { local gpu="$1"; shift; local fold; for fold in "$@"; do infer_fold "${gpu}" "${fold}"; done; }
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

audit_matches() {
  local work="$1"
  "${python_bin}" - "${work}/AUDIT.json" "${work}/manifest.json" \
    "${work}/INFERENCE_COMPLETE.json" <<'PY'
import json,sys
from pathlib import Path
audit_path,manifest_path,inference_path=map(Path,sys.argv[1:])
if not all(path.is_file() for path in (audit_path,manifest_path,inference_path)):
    raise SystemExit(1)
try:
    audit=json.loads(audit_path.read_text(encoding="utf-8"))
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
except (OSError,ValueError):
    raise SystemExit(1)
checks=(
    int(audit.get("cases",-1))==len(manifest.get("case_records",[])),
    int(audit.get("candidates",-1))==len(manifest.get("records",[])),
    audit.get("exact_file_coverage") is True,
    audit.get("all_prompts_nonempty") is True,
    audit.get("all_values_finite_and_normalized") is True,
    audit.get("predictions_required") is True,
    audit.get("prediction_exact_file_coverage") is True,
    audit.get("prediction_values_finite_and_normalized") is True,
    audit.get("inference_plan_bound") is True,
)
raise SystemExit(0 if all(checks) else 1)
PY
}
for fold in 0 1 2 3 4; do
  work="$(fold_work "${fold}")"
  if ! audit_matches "${work}"; then
    "${python_bin}" -u "${root}/audit_candidate_refiner_inference_cache.py" \
      --work "${work}" --output "${work}/AUDIT.json" --workers 16 \
      --require-predictions --require-inference-plan \
      > "${output}/fold${fold}.audit.log" 2>&1
  fi
done

classification_values=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
values=json.load(open(sys.argv[1],encoding="utf-8"))["classification"]["deployment_thresholds"]
if len(values)!=3 or len(set(map(float,values)))!=3: raise SystemExit("expected exactly three thresholds")
print(" ".join(f"{float(value):.2f}" for value in values))
PY
)
classification_center=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
print(f'{float(json.load(open(sys.argv[1]))["classification"]["threshold"]):.2f}')
PY
)
source_threshold_json=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
reports=payload.get("source_classification",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"} | {f"lung_cell{i}" for i in range(8)}
if set(reports)!=expected: raise SystemExit(f"source calibration mismatch: {sorted(reports)}")
thresholds={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0 < value < 1 for value in thresholds.values()):
    raise SystemExit("invalid source-specific threshold")
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
source_segmentation_threshold_json=$("${python_bin}" - "${patch_oof}" <<'PY'
import json,sys
payload=json.load(open(sys.argv[1],encoding="utf-8"))
reports=payload.get("source_segmentation",{})
expected={"epoch105","currentgf","autopet_weak","lndb_weak"} | {f"lung_cell{i}" for i in range(8)}
if set(reports)!=expected: raise SystemExit(f"source contour calibration mismatch: {sorted(reports)}")
thresholds={name:round(float(reports[name]["threshold"]),2) for name in sorted(reports)}
if any(not 0 < value < 1 for value in thresholds.values()): raise SystemExit("invalid source contour threshold")
if len({thresholds[f"lung_cell{i}"] for i in range(8)})!=1: raise SystemExit("pooled lung-cell contour thresholds diverged")
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
source_recall_threshold_json=$("${python_bin}" - "${source_threshold_json}" <<'PY'
import json,sys
thresholds=json.loads(sys.argv[1])
cells=[f"lung_cell{i}" for i in range(8)]
values={float(thresholds[name]) for name in cells}
if len(values)!=1:
    raise SystemExit(f"pooled lung-cell thresholds diverged: {sorted(values)}")
lowered=max(0.05,round(next(iter(values))-0.10,2))
for name in cells: thresholds[name]=lowered
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
source_high_recall_threshold_json=$("${python_bin}" - "${source_threshold_json}" <<'PY'
import json,sys
thresholds=json.loads(sys.argv[1])
cells=[f"lung_cell{i}" for i in range(8)]
values={float(thresholds[name]) for name in cells}
if len(values)!=1:
    raise SystemExit(f"pooled lung-cell thresholds diverged: {sorted(values)}")
lowered=max(0.01,round(next(iter(values))-0.20,2))
for name in cells: thresholds[name]=lowered
print(json.dumps(thresholds,sort_keys=True,separators=(",",":")))
PY
)
mapfile -t source_threshold_entries < <("${python_bin}" - "${source_threshold_json}" <<'PY'
import json,sys
for name,value in sorted(json.loads(sys.argv[1]).items()): print(f"{name}={float(value):.2f}")
PY
)
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
  local configurations="$1" segmentation_threshold="$2" log_tag="$3"
  local deployment="${4:-standalone}" destination_root="${5:-${output}}"
  local fold fold_group pid status=0 work base reuse_log
  local -a empty_arg=() validator_args=()
  case "${deployment}" in
    standalone) empty_arg+=(--empty-base);;
    anchor_currentgf) ;;
    *) echo "invalid OOF deployment: ${deployment}" >&2; return 2;;
  esac
  # Five concurrent materializers triggered systemd-oomd despite nominal host
  # headroom because the user service slice was under sustained memory
  # pressure.  Bound each wave to three folds; this still avoids five serial
  # NVMe passes while leaving margin for the rest of the pipeline.
  for fold_group in "0 1 2" "3 4"; do
    pids=()
    for fold in ${fold_group}; do
      work="$(fold_work "${fold}")"
      if [[ "${deployment}" == anchor_currentgf ]]; then
        base="${staging}/fold${fold}/proposals/currentgf"
      else
        base="${staging}/fold${fold}/proposals/epoch105"
      fi
      reuse_log="${output}/${log_tag}.fold${fold}.reuse.log"
      validator_args=(
        --configurations "${configurations}" --output-root "${destination_root}"
        --fold "${fold}" --expected-dir "${staging}/fold${fold}/labels"
        --base "${base}" --segmentation-threshold "${segmentation_threshold}"
        --min-seed-coverage 0 --min-refined-overlap 0
        --min-volume-ratio 0 --max-volume-ratio 1000000000000
      )
      [[ "${deployment}" == standalone ]] && validator_args+=(--empty-base)
      if "${python_bin}" -u "${root}/validate_candidate_refiner_grid_fold.py" \
          "${validator_args[@]}" > "${reuse_log}" 2>&1; then
        printf '%s reusing validated %s fold %s materialization\n' \
          "$(date -Ins)" "${log_tag}" "${fold}"
        continue
      fi
      "${python_bin}" -u "${root}/materialize_candidate_refiner_grid.py" \
        --work "${work}" --base "${base}" "${empty_arg[@]}" \
        --output-root "${destination_root}" \
        --configurations "${configurations}" --fold "${fold}" \
        --segmentation-threshold "${segmentation_threshold}" --min-seed-coverage 0 \
        --min-refined-overlap 0 --min-volume-ratio 0 --max-volume-ratio 1000000000000 \
        > "${output}/${log_tag}.fold${fold}.materialize_grid.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || status=$?; done
    [[ "${status}" -eq 0 ]] || return "${status}"
  done
}

score_variant() {
  local tag="$1" classification="$2" mode="$3" source_json="$4"
  local source_segmentation_json="$5" source_mode_json="$6"
  local geometry_json="$7" segmentation_threshold="$8" summary="$9"
  local deployment_root="${10:-${output}}"
  local combined_dir metrics score_log dsc nsd combined
  combined_dir="${deployment_root}/${tag}/combined"
  "${python_bin}" - "${deployment_root}/${tag}" "${combined_dir}" "${staging}/all/labels" <<'PY'
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
if len(expected)!=707 or set(seen)!=expected: raise SystemExit("OOF coverage mismatch")
PY
  metrics="${deployment_root}/${tag}.metrics.csv"
  score_log="${deployment_root}/${tag}.score.log"
  "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
    --pred "${combined_dir}" --gt "${staging}/all/labels" \
    --csv "${metrics}" --workers 6 > "${score_log}" 2>&1
  dsc=$(awk '/DSC  mean/ {print $3}' "${score_log}")
  nsd=$(awk '/NSD  mean/ {print $3}' "${score_log}")
  combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN {printf "%.6f",(d+n)/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${tag}" "${classification}" \
    "${segmentation_threshold}" "${mode}" "${source_json}" "${source_segmentation_json}" "${source_mode_json}" "${geometry_json}" "${dsc}" "${nsd}" \
    "${combined}" "${metrics}" >> "${summary}"
}

# Patch Dice only centers the predeclared threshold grid. Calibrate that same
# frozen grid independently for standalone and current-GF-anchor composition;
# neither deployment inherits the other one's operating point.
write_segmentation_grid() {
  local values="$1" grid="$2"
  "${python_bin}" - "${values}" "${classification_center}" "${grid}" <<'PY'
import json,sys
from pathlib import Path
values=[float(value) for value in sys.argv[1].split()]
classification=float(sys.argv[2]); output=Path(sys.argv[3])
configs=[]
for segmentation in values:
 tag=(f"weak707_segcal_c{classification:.2f}".replace(".","p")+
      f"_s{segmentation:.2f}".replace(".","p")+"_refined")
 configs.append({"tag":tag,"classification_threshold":classification,
  "segmentation_threshold":segmentation,"mode":"refined",
  "source_classification_thresholds":{},"source_segmentation_thresholds":{}})
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n",encoding="utf-8")
PY
}
run_segmentation_sweep() {
  local deployment="$1" destination_root="$2" log_prefix="$3" selection_output="$4"
  local sweep="${destination_root}/segmentation_threshold_sweep.tsv"
  local segmentation_candidate tag grid
  mkdir -p "${destination_root}"
  printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tsource_deployment_modes\tgeometry_gates\tDSC\tNSD\tcombined\tmetrics\n' > "${sweep}"
  grid="${destination_root}/segmentation_threshold_sweep.grid.json"
  write_segmentation_grid "${segmentation_values}" "${grid}"
  materialize_grid_all_folds "${grid}" "${segmentation_center}" \
    "${log_prefix}_segmentation_threshold_sweep" "${deployment}" "${destination_root}"
  for segmentation_candidate in ${segmentation_values}; do
    tag="weak707_segcal_c${classification_center/./p}_s${segmentation_candidate/./p}_refined"
    score_variant "${tag}" "${classification_center}" refined '{}' '{}' '{}' '{}' \
      "${segmentation_candidate}" "${sweep}" "${destination_root}"
  done
  "${python_bin}" -u "${root}/select_segmentation_threshold_sweep.py" \
    --summary "${sweep}" --patch-oof-center "${segmentation_center}" \
    --expected-cases 707 --output "${selection_output}"
}

standalone_segmentation_value="${output}/SEGMENTATION_THRESHOLD_SELECTION.value.txt"
composition_segmentation_value="${composition_root}/SEGMENTATION_THRESHOLD_SELECTION.value.txt"
# Redirections are opened before the function body runs, so the composition
# directory must exist before redirecting the selector's stdout into it.  Do
# not rely on run_segmentation_sweep's internal mkdir for this path.
mkdir -p "${composition_root}"
# Each sweep already uses up to five independent fold workers.  Run the two
# deployment families sequentially so the acceleration cannot create ten
# simultaneous full-volume materializers.
run_segmentation_sweep standalone "${output}" standalone \
  "${output}/SEGMENTATION_THRESHOLD_SELECTION.json" \
  > "${standalone_segmentation_value}"
run_segmentation_sweep anchor_currentgf "${composition_root}" composition \
  "${composition_root}/SEGMENTATION_THRESHOLD_SELECTION.json" \
  > "${composition_segmentation_value}"
segmentation=$(tail -n 1 "${standalone_segmentation_value}")
composition_segmentation=$(tail -n 1 "${composition_segmentation_value}")
[[ "${segmentation}" =~ ^0\.[0-9][0-9]$ \
    && "${composition_segmentation}" =~ ^0\.[0-9][0-9]$ ]] || {
  echo "invalid selected segmentation thresholds: ${segmentation}, ${composition_segmentation}" >&2
  exit 2
}

write_materialization_grid() {
  local segmentation_value="$1" grid_output="$2"
  "${python_bin}" - "${classification_values}" "${classification_center}" \
    "${segmentation_value}" "${source_threshold_json}" \
    "${source_recall_threshold_json}" "${source_high_recall_threshold_json}" \
    "${source_segmentation_threshold_json}" "${grid_output}" <<'PY'
import json,sys
from pathlib import Path
values=[float(value) for value in sys.argv[1].split()]
center=float(sys.argv[2]); segmentation=float(sys.argv[3])
source_thresholds=json.loads(sys.argv[4])
source_recall_thresholds=json.loads(sys.argv[5])
source_high_recall_thresholds=json.loads(sys.argv[6])
source_segmentation_thresholds=json.loads(sys.argv[7]); output=Path(sys.argv[8])
configs=[]
for classification in values:
    configs.append({
      "tag":f"weak707_c{classification:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+"_refined",
      "classification_threshold":classification,"mode":"refined",
      "source_classification_thresholds":{},"source_segmentation_thresholds":{},
    })
for prefix,thresholds in (("src",source_thresholds),("srclo",source_recall_thresholds),
                          ("srclo2",source_high_recall_thresholds)):
    configs.append({
      "tag":f"weak707_{prefix}_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+"_refined",
      "classification_threshold":center,"mode":"refined",
      "source_classification_thresholds":thresholds,"source_segmentation_thresholds":{},
    })
configs.append({
  "tag":f"weak707_srcseg_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p")+"_refined",
  "classification_threshold":center,"mode":"refined",
  "source_classification_thresholds":source_thresholds,
  "source_segmentation_thresholds":source_segmentation_thresholds,
})
configs.extend([
 {
  "tag":f"weak707_seedgf_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p"),
  "classification_threshold":center,"mode":"refined",
  "source_classification_thresholds":source_thresholds,
  "source_segmentation_thresholds":source_segmentation_thresholds,
  "source_deployment_modes":{"epoch105":"seed","currentgf":"seed"},
 },
 {
  "tag":f"weak707_seedcropgf_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p"),
  "classification_threshold":center,"mode":"refined",
  "source_classification_thresholds":source_thresholds,
  "source_segmentation_thresholds":source_segmentation_thresholds,
  "source_deployment_modes":{"epoch105":"seed_crop","currentgf":"seed_crop"},
 },
])
for suffix,gates in (
 ("geomlo",{"min_seed_coverage":0.05,"min_refined_overlap":0.05,
            "min_volume_ratio":0.10,"max_volume_ratio":10.0}),
 ("geommid",{"min_seed_coverage":0.10,"min_refined_overlap":0.10,
             "min_volume_ratio":0.25,"max_volume_ratio":4.0}),
 ("geomhi",{"min_seed_coverage":0.25,"min_refined_overlap":0.25,
            "min_volume_ratio":0.25,"max_volume_ratio":4.0}),
):
    configs.append({
      "tag":f"weak707_{suffix}_c{center:.2f}".replace(".","p")+f"_s{segmentation:.2f}".replace(".","p"),
      "classification_threshold":center,"mode":"refined",
      "source_classification_thresholds":source_thresholds,
      "source_segmentation_thresholds":source_segmentation_thresholds,
      "geometry_gates":gates,
    })
output.write_text(json.dumps({"schema_version":1,"configurations":configs},indent=2)+"\n")
PY
}
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tsource_deployment_modes\tgeometry_gates\tDSC\tNSD\tcombined\tmetrics\n' > "${output}/summary.tsv"
materialization_grid="${output}/materialization_grid.json"
composition_materialization_grid="${composition_root}/materialization_grid.json"
write_materialization_grid "${segmentation}" "${materialization_grid}"
write_materialization_grid "${composition_segmentation}" "${composition_materialization_grid}"
score_main_grid() {
  local summary_path="$1" deployment_root="$2" selected_segmentation="$3"
  local classification tag mode_variant source_mode_json geometry_variant geometry_json
  for classification in ${classification_values}; do
    tag="weak707_c${classification/./p}_s${selected_segmentation/./p}_refined"
    score_variant "${tag}" "${classification}" refined '{}' '{}' '{}' '{}' \
      "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  done
  tag="weak707_src_c${classification_center/./p}_s${selected_segmentation/./p}_refined"
  score_variant "${tag}" "${classification_center}" refined "${source_threshold_json}" \
    '{}' '{}' '{}' "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  tag="weak707_srclo_c${classification_center/./p}_s${selected_segmentation/./p}_refined"
  score_variant "${tag}" "${classification_center}" refined "${source_recall_threshold_json}" \
    '{}' '{}' '{}' "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  tag="weak707_srclo2_c${classification_center/./p}_s${selected_segmentation/./p}_refined"
  score_variant "${tag}" "${classification_center}" refined "${source_high_recall_threshold_json}" \
    '{}' '{}' '{}' "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  tag="weak707_srcseg_c${classification_center/./p}_s${selected_segmentation/./p}_refined"
  score_variant "${tag}" "${classification_center}" refined "${source_threshold_json}" \
    "${source_segmentation_threshold_json}" '{}' '{}' "${selected_segmentation}" \
    "${summary_path}" "${deployment_root}"
  for mode_variant in seedgf seedcropgf; do
    tag="weak707_${mode_variant}_c${classification_center/./p}_s${selected_segmentation/./p}"
    if [[ "${mode_variant}" == seedgf ]]; then
      source_mode_json='{"currentgf":"seed","epoch105":"seed"}'
    else
      source_mode_json='{"currentgf":"seed_crop","epoch105":"seed_crop"}'
    fi
    score_variant "${tag}" "${classification_center}" refined "${source_threshold_json}" \
      "${source_segmentation_threshold_json}" "${source_mode_json}" '{}' \
      "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  done
  for geometry_variant in geomlo geommid geomhi; do
    tag="weak707_${geometry_variant}_c${classification_center/./p}_s${selected_segmentation/./p}"
    case "${geometry_variant}" in
      geomlo) geometry_json='{"max_volume_ratio":10.0,"min_refined_overlap":0.05,"min_seed_coverage":0.05,"min_volume_ratio":0.1}' ;;
      geommid) geometry_json='{"max_volume_ratio":4.0,"min_refined_overlap":0.1,"min_seed_coverage":0.1,"min_volume_ratio":0.25}' ;;
      geomhi) geometry_json='{"max_volume_ratio":4.0,"min_refined_overlap":0.25,"min_seed_coverage":0.25,"min_volume_ratio":0.25}' ;;
    esac
    score_variant "${tag}" "${classification_center}" refined "${source_threshold_json}" \
      "${source_segmentation_threshold_json}" '{}' "${geometry_json}" \
      "${selected_segmentation}" "${summary_path}" "${deployment_root}"
  done
}

# Calibrate both the candidate as a standalone segmentation and its actual
# incremental deployment role on top of the frozen, label-blind current-GF
# base. The latter mirrors the public champion composition without using any
# public data; only the added fold-held-out refiner predictions vary.
materialize_grid_all_folds "${materialization_grid}" "${segmentation}" main \
  standalone "${output}"
printf 'variant\tclassification_threshold\tsegmentation_threshold\tdeployment_mode\tsource_classification_thresholds\tsource_segmentation_thresholds\tsource_deployment_modes\tgeometry_gates\tDSC\tNSD\tcombined\tmetrics\n' \
  > "${composition_root}/summary.tsv"
materialize_grid_all_folds "${composition_materialization_grid}" \
  "${composition_segmentation}" composition_main anchor_currentgf "${composition_root}"
score_pids=(); score_status=0
score_main_grid "${output}/summary.tsv" "${output}" "${segmentation}" \
  & score_pids+=("$!")
score_main_grid "${composition_root}/summary.tsv" "${composition_root}" \
  "${composition_segmentation}" & score_pids+=("$!")
for pid in "${score_pids[@]}"; do wait "${pid}" || score_status=$?; done
[[ "${score_status}" -eq 0 ]] || exit "${score_status}"
{ head -1 "${output}/summary.tsv"; tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k9,9gr -k10,10gr; } \
  > "${output}/summary_by_dsc.tsv"
{ head -1 "${composition_root}/summary.tsv"; tail -n +2 "${composition_root}/summary.tsv" | sort -t $'\t' -k9,9gr -k10,10gr; } \
  > "${composition_root}/summary_by_dsc.tsv"
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${output}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${standalone_calibration}" \
  --bootstrap-samples 5000 --seed 20260823 > "${output}/summarize_standalone.log" 2>&1
"${python_bin}" -u "${root}/summarize_candidate_refiner_fullvolume_oof.py" \
  --summary "${composition_root}/summary_by_dsc.tsv" --checkpoint-selection "${selection}" \
  --manifest "${manifest}" --output "${composition_calibration}" \
  --bootstrap-samples 5000 --seed 20260824 > "${composition_root}/summarize.log" 2>&1

anchor_metrics="${composition_root}/currentgf_anchor.metrics.csv"
"${python_bin}" -u "${root}/evaluate_public_parallel.py" \
  --pred "${staging}/all/proposals/currentgf" --gt "${staging}/all/labels" \
  --csv "${anchor_metrics}" --workers 6 > "${composition_root}/currentgf_anchor.score.log" 2>&1
composition_metrics=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["metrics"])' "${composition_calibration}")
"${python_bin}" -u "${root}/select_checkpoint_robust.py" \
  --anchor "${anchor_metrics}" --candidate "${composition_metrics}" \
  --anchor-name currentgf_fixed_base --candidate-name refiner_anchor_currentgf \
  --min-dsc-gain 0.005 --min-nsd-gain 0.005 --min-win-rate 0.52 \
  --max-catastrophic-rate 0.10 --bootstrap 20000 --seed 20260824 \
  --output "${composition_robust}" > "${composition_root}/robust.log" 2>&1
"${python_bin}" - "${standalone_calibration}" "${composition_calibration}" \
  "${composition_robust}" "${calibration}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
standalone,composition,gate,output=map(Path,sys.argv[1:])
robust=json.loads(gate.read_text(encoding="utf-8"))
selected=composition if robust.get("decision")=="PROMOTE" else standalone
payload=json.loads(selected.read_text(encoding="utf-8"))
payload["calibration_deployment"]=(
    "anchor_currentgf" if selected==composition else "standalone"
)
payload["standalone_calibration"]=str(standalone.resolve())
payload["composition_calibration"]=str(composition.resolve())
payload["composition_gate"]=str(gate.resolve())
payload["composition_gate_payload"]=robust
payload["selection_basis"] += (
    "; final deployment family selected by a paired fixed-currentgf-base "
    "incremental DSC/NSD gate"
)
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY

"${python_bin}" - "${calibration}" "${staging}/STAGING_COMPLETE.json" "${margin_decision}" "${bbox_margin}" "${complete}" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
calibration_path,staging_path,margin_path=map(Path,sys.argv[1:4]); margin=int(sys.argv[4]); output=Path(sys.argv[5])
calibration=json.loads(calibration_path.read_text(encoding="utf-8"))
staging=json.loads(staging_path.read_text(encoding="utf-8"))
if staging.get("cases")!=707 or staging.get("proposal_layout")!="named_sources" or not staging.get("exact_proposal_coverage"):
    raise SystemExit("named-source OOF staging audit failed")
payload={
  "schema_version":1,
  "created_at":datetime.now().astimezone().isoformat(),
  "selection_basis":"exact-707 case-held-out grouped five-fold full-volume OOF",
  "cases":707,
  "component_policy":"source_separate",
  "bbox_margin_mm":margin,
  "bbox_margin_decision":str(margin_path.resolve()),
  "Lesion_DSC":calibration["Lesion_DSC"],
  "Lesion_NSD":calibration["Lesion_NSD"],
  "classification_threshold":calibration["classification_threshold"],
  "source_classification_thresholds":calibration.get("source_classification_thresholds",{}),
  "segmentation_threshold":calibration["segmentation_threshold"],
  "source_segmentation_thresholds":calibration.get("source_segmentation_thresholds",{}),
  "source_deployment_modes":calibration.get("source_deployment_modes",{}),
  "geometry_gates":calibration.get("geometry_gates",{}),
  "calibration_deployment":calibration.get("calibration_deployment","standalone"),
  "composition_gate":calibration.get("composition_gate"),
  "composition_gate_payload":calibration.get("composition_gate_payload"),
  "deployment_mode":calibration.get("deployment_mode","refined"),
  "calibration":str(calibration_path.resolve()),
  "public_read":False,
  "hidden_read":False,
  "submission_created":False,
}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
print(json.dumps(payload,indent=2))
PY
cat "${output}/summary_by_dsc.tsv"

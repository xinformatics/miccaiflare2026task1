#!/usr/bin/env bash
set -euo pipefail

# Label-blind scale-up of the selected epoch-105 proposal stream. It reuses
# the exact 210 pilot files by hard link and generates only the 497 new cases.
# No labels, patch targets, training, public selection, or upload occur here.
# Submission preservation and the bounded weak-domain proposal audits have GPU
# priority; this larger 497-case generation starts only after their save/gate
# markers exist.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
pilot_complete="${root}/analysis/candidate_refiner_v2_training_20260822/TRAINING_COMPLETE.txt"
oof="${root}/analysis/candidate_refiner_v2_training_20260822/five_fold_oof_summary.json"
save_complete="${root}/public_sweeps/candidate_refiner_v2_bbox24_saved_20260823/SAVE_ONLY_COMPLETE.txt"
multisource_save_complete="${root}/public_sweeps/candidate_refiner_v2_multisource_saved_20260823/SAVE_ONLY_COMPLETE.txt"
weak_extension_complete="${root}/../BiomedParse/analysis/weak_domain_biomedparse_recall_extension_20260823/EXTENSION_COMPLETE.txt"
weak_transfer_gpu_complete="${root}/analysis/candidate_refiner_v2_weak_transfer_oof_20260823/GPU_PHASE_COMPLETE.json"
focus_pilot_decision="${root}/analysis/focus_candidate_refiner_pilot_20260823/PILOT_DECISION.json"
focus_fivefold_terminal="${root}/analysis/focus_candidate_refiner_fivefold_20260823/FOCUS_FIVEFOLD_TERMINAL.json"
focus_save_terminal="${root}/public_sweeps/focus_candidate_refiner_saved_20260823/FOCUS_SAVE_ONLY_COMPLETE.txt"
checkpoint="${root}/runs/gf_screen_full_unfreeze_20260811_110459/checkpoint_epoch_0106.pt"
pilot_manifest="${root}/analysis/fpr_training_mining_20260822/manifest.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
images="${root}/analysis/fpr_training_mining_707_20260822/staged_images"
pilot_proposals="${root}/analysis/candidate_refiner_v2_training_20260822/epoch105_abd_g090_l050"
output="${root}/analysis/candidate_refiner_v2_training_707_20260823/epoch105_abd_g090_l050"
marker="${output}/PROPOSALS_COMPLETE.json"
decision_root="${root}/analysis/candidate_refiner_v2_training_707_20260823"
expected=707

mkdir -p "${output}" "${decision_root}"
exec 9>"${root}/runs/run_epoch105_training_proposals_707_after_pilot_20260823.lock"
flock -n 9 || { echo "707-case proposal controller already active" >&2; exit 1; }
[[ -s "${marker}" ]] && exit 0
for required in "${oof}" "${checkpoint}" "${pilot_manifest}" "${manifest}"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done
for directory in "${images}" "${pilot_proposals}"; do
  [[ -d "${directory}" ]] || { echo "missing directory: ${directory}" >&2; exit 2; }
done

recommendation=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["scale_recommendation"])' "${oof}")
if [[ "${recommendation}" != PROMISING_FOR_707_SCALE ]]; then
  printf '%s %s\n' "$(date -Ins)" "${recommendation}" > "${decision_root}/NOT_PROMISING_FOR_707_SCALE.txt"
  exit 0
fi

"${python_bin}" - "${pilot_manifest}" "${manifest}" "${pilot_proposals}" "${output}" <<'PY'
import json,os,sys
from pathlib import Path
pilot_manifest,manifest,pilot_dir,output=map(Path,sys.argv[1:])
pilot={str(item["prediction_name"]) for item in json.loads(pilot_manifest.read_text())["items"]}
large={str(item["prediction_name"]) for item in json.loads(manifest.read_text())["items"]}
if len(pilot)!=210 or len(large)!=707 or not pilot < large:
    raise SystemExit(f"manifest relation mismatch pilot={len(pilot)} large={len(large)}")
for name in sorted(pilot):
    source=pilot_dir/name; target=output/name
    if not source.is_file(): raise FileNotFoundError(source)
    if target.exists():
        if not target.is_file(): raise RuntimeError(f"non-file target: {target}")
        if not os.path.samefile(source,target):
            raise RuntimeError(f"pilot reuse is not the expected hard link: {target}")
        continue
    os.link(source,target)
PY

# Validate and reuse the 210 pilot cases without occupying a GPU. Only the 497
# new cases remain behind the bounded save/transfer/Focus priority gates.
while [[ ! -s "${pilot_complete}" || ! -s "${save_complete}" \
    || ! -s "${multisource_save_complete}" || ! -s "${weak_extension_complete}" \
    || ! -s "${weak_transfer_gpu_complete}" \
    || ! -s "${focus_pilot_decision}" \
    || ! -s "${focus_fivefold_terminal}" \
    || ! -s "${focus_save_terminal}" ]]; do
  printf '%s reused 210 pilot cases; waiting for save-only stages, bounded transfer gates, and Focus save-only gate\n' "$(date -Ins)"
  sleep 30
done

coverage_complete=0
if "${python_bin}" - "${manifest}" "${output}" <<'PY'
import json,sys
from pathlib import Path
manifest,output=map(Path,sys.argv[1:])
expected={str(item["prediction_name"]) for item in json.loads(manifest.read_text())["items"]}
observed={path.name for path in output.glob("*.nii.gz")}
raise SystemExit(0 if len(expected)==707 and observed==expected else 1)
PY
then
  coverage_complete=1
  printf '%s reusing exact-707 manifest-matched proposal cache; skipping GPU scan\n' "$(date -Ins)"
fi

gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
if [[ "${coverage_complete}" -eq 0 ]]; then
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

run_shard() {
  local gpu="$1" shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.90 --lesion_threshold 0.50 \
    --sw_batch_size 2 --workers 2 --num_shards 2 --shard_id "${shard}" \
    --skip_existing > "${output}/shard${shard}.log" 2>&1
}
if [[ "${free0}" -eq 1 && "${free1}" -eq 1 ]]; then
  run_shard 0 0 & pid0=$!; run_shard 1 1 & pid1=$!
  status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
elif [[ "${free0}" -eq 1 ]]; then
  run_shard 0 0; run_shard 0 1
else
  run_shard 1 0; run_shard 1 1
fi
fi

"${python_bin}" - "${manifest}" "${output}" "${checkpoint}" "${marker}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
manifest,output,checkpoint,marker=map(Path,sys.argv[1:])
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
expected={str(item["prediction_name"]) for item in items}
observed={path.name for path in output.glob("*.nii.gz")}
if len(items)!=707 or len(expected)!=707 or observed!=expected:
    raise SystemExit(f"coverage mismatch items={len(items)} names={len(expected)} observed={len(observed)} missing={sorted(expected-observed)[:3]} extra={sorted(observed-expected)[:3]}")
digest=hashlib.sha256()
with checkpoint.open("rb") as stream:
    for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
payload={
    "created_at":datetime.now().astimezone().isoformat(),"cases":707,
    "reused_pilot_cases":210,"generated_additional_cases":497,
    "labels_read":False,"checkpoint":str(checkpoint.resolve()),
    "checkpoint_sha256":digest.hexdigest(),"output":str(output.resolve()),
    "window":[-175,250],"glance_threshold":0.90,"lesion_threshold":0.50,
}
marker.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
print(json.dumps(payload,indent=2))
PY

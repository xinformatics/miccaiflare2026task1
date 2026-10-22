#!/usr/bin/env bash
set -euo pipefail

# Continue the local-only dual-window Focus experiment after its fold-0 gate.
# No full-volume public/hidden inference, packaging, or upload occurs here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
training="${root}/analysis/candidate_refiner_v4_cell707_training_20260823"
index="${training}/patches_bbox24_ss_min5_v4_cell48/index.csv"
b32_complete="${training}/TRAINING_COMPLETE.json"
b32_summary="${training}/five_fold_oof_summary.json"
b32_source_root="${training}/decoupled_oof"
b32_source_summary="${b32_source_root}/five_fold_oof_summary.json"
b32_source_decision="${b32_source_root}/DECISION.json"
pilot="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/DECISION.json"
mixed_pilot="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/MIXED_DECISION.json"
mixed_early_pilot="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/MIXED_EARLY_DECISION.json"
pilot_skipped="${root}/analysis/focus_dualwindow_cell707_pilot_20260823/SKIPPED.json"
pretrained="/mnt/nvme3-data/share/wiseFocusA015_ep150_fLR3e-4_gLR3e-4_bs8x1_wu5_to300/checkpoint_epoch_0240.pt"
work="${root}/analysis/focus_dualwindow_cell707_fivefold_20260823"
summary="${work}/focus_patch_oof.json"
epoch_routes="${work}/focus_epoch_routes.json"
epoch_route_decision="${work}/focus_epoch_routes_decision.json"
mixed_route_decision="${work}/focus_b32_source_hybrid_decision.json"
decision="${work}/DECISION.json"
terminal="${work}/FIVEFOLD_TERMINAL.json"

fold_dir() {
  if [[ "$1" -eq 0 ]]; then
    printf '%s/runs/focus_dualwindow_cell707_release10_fold0_20260823' "${root}"
  else
    printf '%s/runs/focus_dualwindow_cell707_release10_fold%s_20260823' "${root}" "$1"
  fi
}

mkdir -p "${work}"
exec 9>"${root}/runs/run_focus_dualwindow_weak707_fivefold_20260823.lock"
flock -n 9 || { echo "dual-window weak707 five-fold controller already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0
while [[ ! -s "${pilot_skipped}" && ( ! -s "${pilot}" || ! -s "${mixed_pilot}" ) ]]; do
  printf '%s waiting for global and mixed dual-window fold-0 decisions\n' "$(date -Ins)"
  sleep 30
done
if [[ -s "${pilot_skipped}" ]]; then
  "${python_bin}" - "${pilot_skipped}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:])
payload={"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_B32",
 "reason":"dual-window pilot prerequisites did not qualify",
 "pilot":str(source.resolve()),"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
  exit 0
fi
pilot_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${pilot}")
mixed_pilot_decision=$("${python_bin}" -c \
  'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${mixed_pilot}")
mixed_early_pilot_decision=KEEP_B32
if [[ -s "${mixed_early_pilot}" ]]; then
  mixed_early_pilot_decision=$("${python_bin}" - \
    "${mixed_early_pilot}" "$(fold_dir 0)/checkpoint_epoch_0001.pt" \
    "$(fold_dir 0)/validation_predictions_epoch_0001.npz" <<'PY'
import hashlib,json,sys
from pathlib import Path
decision,checkpoint,prediction=map(Path,sys.argv[1:])
payload=json.loads(decision.read_text(encoding="utf-8"))
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
if payload.get("decision")!="PROMOTE_MIXED_TO_FIVEFOLD": raise SystemExit("early mixed decision did not promote")
if set(payload.get("eligible_focus_groups",[]))!={"weak_sources","lung_cells"}: raise SystemExit("early mixed groups mismatch")
if payload.get("fixed_b32_groups")!=["mature"]: raise SystemExit("early mature route mismatch")
if Path(payload.get("focus_checkpoint","")).resolve()!=checkpoint.resolve(): raise SystemExit("early checkpoint path mismatch")
if Path(payload.get("focus_prediction","")).resolve()!=prediction.resolve(): raise SystemExit("early prediction path mismatch")
if payload.get("focus_checkpoint_sha256")!=sha(checkpoint): raise SystemExit("early checkpoint hash mismatch")
if payload.get("focus_prediction_sha256")!=sha(prediction): raise SystemExit("early prediction hash mismatch")
if payload.get("public_read") or payload.get("hidden_read") or payload.get("submission_created"): raise SystemExit("early decision is not local-only")
print(payload["decision"])
PY
  )
fi
if [[ "${pilot_decision}" != PROMOTE_TO_FIVEFOLD && \
      "${mixed_pilot_decision}" != PROMOTE_MIXED_TO_FIVEFOLD && \
      "${mixed_early_pilot_decision}" != PROMOTE_MIXED_TO_FIVEFOLD ]]; then
  "${python_bin}" - "${pilot}" "${mixed_pilot}" "${mixed_early_pilot}" \
    "${terminal}" <<'PY'
import datetime,json,sys
from pathlib import Path
source,mixed,early,output=map(Path,sys.argv[1:])
payload={
 "created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"SKIPPED","decision":"KEEP_B32",
 "reason":"no global, final-source, or early-source Focus fold-0 gate passed",
 "pilot":str(source.resolve()),"mixed_pilot":str(mixed.resolve()),
 "mixed_early_pilot":str(early.resolve()) if early.is_file() else None,
 "public_read":False,"hidden_read":False,
 "submission_created":False,
}
output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
PY
  exit 0
fi
for required in "${index}" "${pretrained}" "$(fold_dir 0)/checkpoint_best.pt"; do
  [[ -s "${required}" ]] || { echo "missing required asset: ${required}" >&2; exit 2; }
done

gpu="${FOCUS_DUALWINDOW_FIVEFOLD_GPU:-1}"
gpu_memory_mib() {
  nvidia-smi --id="$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}
while :; do
  used=$(gpu_memory_mib "${gpu}")
  [[ "${used}" =~ ^[0-9]+$ && "${used}" -le 4096 ]] && break
  printf '%s waiting for GPU %s: %s MiB\n' "$(date -Ins)" "${gpu}" "${used}"
  sleep 30
done
for fold in 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  mkdir -p "${output}"
  if [[ ! -s "${output}/COMPLETE.txt" || ! -s "${output}/checkpoint_best.pt" ]]; then
    export MPLCONFIGDIR="${output}/matplotlib"
    export TMPDIR="/tmp/gf_focus_dualwindow_weak707_fold${fold}_20260823"
    mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
      "${root}/train_focus_candidate_refiner.py" \
      --index "${index}" --pretrained "${pretrained}" --output "${output}" \
      --folds 5 --fold "${fold}" --epochs 10 --batch-size 8 --workers 2 \
      --lr 3e-5 --weight-decay 1e-5 --loss-recipe release_dice_ce \
      --sampling-temperature 0.5 --samples-per-epoch 2048 \
      --segmentation-threshold-max 0.95 \
      --freeze-encoder --input-mode soft_lung_seed --seed 20260823 \
      --wandb-project pancancer-ct \
      --wandb-name "focus_dualwindow_cell707_release10_fold${fold}_20260823" \
      --resume > "${output}/train.log" 2>&1
  fi
done

args=()
for fold in 0 1 2 3 4; do
  output="$(fold_dir "${fold}")"
  args+=(--checkpoint "${output}/checkpoint_best.pt")
  args+=(--prediction "${output}/validation_predictions_best.npz")
done
"${python_bin}" -u "${root}/summarize_focus_candidate_refiner_oof.py" \
  --index "${index}" "${args[@]}" --folds 5 --seed 20260823 \
  --output "${summary}" > "${work}/summarize.log" 2>&1

# Preserve a diagnostic source-routed epoch policy before any public data can
# be read.  This does not alter the primary Focus gate; it supplies a strictly
# held-out candidate for a later paired full-volume comparison if the observed
# patch gain is large enough to justify routed inference.
route_args=()
for fold in 0 1 2 3 4; do
  route_args+=(--run "$(fold_dir "${fold}")")
done
"${python_bin}" -u "${root}/summarize_focus_epoch_routes.py" \
  --index "${index}" "${route_args[@]}" --folds 5 --seed 20260823 \
  --expected-epochs 11 --output "${epoch_routes}" \
  > "${work}/summarize_epoch_routes.log" 2>&1
"${python_bin}" -u "${root}/select_focus_epoch_routes.py" \
  --baseline "${summary}" --candidate "${epoch_routes}" \
  --output "${epoch_route_decision}" \
  > "${work}/select_epoch_routes.log" 2>&1

while [[ ! -s "${b32_complete}" ]]; do
  printf '%s waiting for matched B32 five-fold OOF summary\n' "$(date -Ins)"
  sleep 30
done
[[ -s "${b32_summary}" ]] || { echo "missing B32 summary" >&2; exit 2; }
while [[ ! -s "${b32_source_summary}" || ! -s "${b32_source_decision}" ]]; do
  printf '%s waiting for source-pooled B32 contour summary\n' "$(date -Ins)"
  sleep 30
done
"${python_bin}" - "${b32_summary}" "${b32_source_summary}" \
  "${b32_source_decision}" <<'PY'
import json,math,sys
from pathlib import Path
baseline,source_summary,decision_path=map(Path,sys.argv[1:])
b=json.load(baseline.open()); s=json.load(source_summary.open())
d=json.load(decision_path.open())
if d.get("decision")!="KEEP_COUPLED":
 raise SystemExit("source-pooled B32 summary was not a contour-preserving decoupled audit")
if Path(d.get("baseline","")).resolve()!=baseline.resolve() or Path(d.get("candidate","")).resolve()!=source_summary.resolve():
 raise SystemExit("decoupled B32 decision is not bound to the requested summaries")
if b.get("index_sha256")!=s.get("index_sha256"):
 raise SystemExit("source-pooled B32 index mismatch")
if not math.isclose(float(b["segmentation"]["dice"]),float(s["segmentation"]["dice"]),abs_tol=1e-12):
 raise SystemExit("source-pooled B32 contours differ from the primary")
if set(s.get("source_segmentation_groups",{}))!={"mature","weak_sources","lung_cells"}:
 raise SystemExit("source-pooled B32 contour groups are incomplete")
PY
"${python_bin}" -u "${root}/select_focus_dualwindow_weak707_fivefold.py" \
  --focus-summary "${summary}" --b32-summary "${b32_summary}" \
  --output "${decision}" > "${work}/select.log" 2>&1
"${python_bin}" -u "${root}/select_focus_b32_source_hybrid.py" \
  --focus-summary "${summary}" --b32-summary "${b32_source_summary}" \
  --output "${mixed_route_decision}" > "${work}/select_mixed_route.log" 2>&1
"${python_bin}" - "${decision}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
source,output=map(Path,sys.argv[1:]); decision=json.loads(source.read_text())
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":decision["decision"],
 "fivefold_decision":str(source.resolve()),"public_read":False,
 "hidden_read":False,"submission_created":False,
}
temporary=output.with_name(f".{output.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,output)
PY
cat "${decision}"

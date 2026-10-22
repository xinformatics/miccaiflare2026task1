#!/usr/bin/env bash
set -euo pipefail

# Training-only proposal inference for a future precision gate. Uses the
# locally selected source-stratified STU endpoint on 707 documentation-verified
# complete-label images, but inference never reads those labels. Binary masks
# only: no public/hidden input, probability cache, packaging, or upload.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
gf_root="${project_root}/external/GF-Screen"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
predictor="${root}/predict_stunet_safe.py"
manifest="${gf_root}/analysis/fpr_training_mining_707_20260822/manifest.json"
images="${gf_root}/analysis/fpr_training_mining_707_20260822/staged_images"
public_root="${root}/public_sweeps/stunetb_sourcestrat_e40_e80_20260826"
public_terminal="${public_root}/TERMINAL.json"
model="${project_root}/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBoxE160SourceStratified__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
output="${root}/analysis/source_stunet_verified707_proposals_20260827"
predictions="${output}/predictions"
terminal="${output}/PROPOSALS_COMPLETE.json"

mkdir -p "${output}" "${predictions}"
exec 9>"${output}/controller.lock"
flock -n 9 || { echo "source STU verified-707 proposal miner already active" >&2; exit 1; }
if [[ -s "${terminal}" ]]; then
  status=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["status"])' "${terminal}")
  case "${status}" in
    COMPLETE) exit 0 ;;
    FAILED)
      mv --no-clobber "${terminal}" \
        "${terminal%.json}.failed_retry_$(date +%Y%m%dT%H%M%S%N).json"
      ;;
    *) echo "unexpected verified-707 proposal terminal: ${status}" >&2; exit 2 ;;
  esac
fi
failure_line=unknown
finalize() {
  local status=$? line="${failure_line}"
  trap - EXIT
  if [[ "${status}" -ne 0 && ! -s "${terminal}" ]]; then
    "${python_bin}" - "${line}" "${status}" "${terminal}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
line,status,output=sys.argv[1],int(sys.argv[2]),Path(sys.argv[3])
p={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"FAILED","decision":"RETRY_MASK_ONLY_INFERENCE",
 "reason":f"failed at line {line}, status {status}","training_images_read":True,
 "training_labels_read":False,"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp")
t.write_text(json.dumps(p,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
  fi
  exit "${status}"
}
trap 'failure_line=${LINENO}' ERR
trap finalize EXIT

while [[ ! -s "${public_terminal}" ]]; do
  printf '%s waiting for source-stratified fixed-public selection\n' "$(date -Ins)"
  sleep 30
done
read -r epoch checkpoint checkpoint_sha < <("${python_bin}" - "${public_terminal}" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=json.load(open(sys.argv[1],encoding="utf-8")); s=p.get("selected") or {}
if (p.get("status")!="COMPLETE" or p.get("predeclared_epochs")!=[40,80]
    or p.get("hidden_read") or p.get("submission_created") or p.get("uploaded")):
 raise SystemExit("invalid source public terminal")
epoch=int(s.get("epoch",-1)); checkpoint=Path(s.get("checkpoint",{}).get("path",""))
expected=s.get("checkpoint",{}).get("sha256")
if (epoch not in (40,80) or s.get("configuration") not in
    {"stunet_standalone","gf_anchor_confirm_either_p70"}
    or not checkpoint.is_file()
    or hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=expected):
 raise SystemExit("invalid selected source checkpoint")
print(epoch,checkpoint,expected)
PY
)
checkpoint_name="$(basename "${checkpoint}")"
[[ "${checkpoint}" == "${model}/fold_all/checkpoint_inference_epoch_$(printf '%04d' "${epoch}").pth" ]]

"${python_bin}" - "${manifest}" "${images}" "${predictor}" <<'PY'
import json,sys
from pathlib import Path
manifest,images,predictor=map(Path,sys.argv[1:]); p=json.load(open(manifest,encoding="utf-8")); items=p.get("items") or []
names={str(item.get("prediction_name")) for item in items}
staged={path.name.removesuffix("_0000.nii.gz")+".nii.gz" for path in images.glob("*_0000.nii.gz")}
if (len(items)!=707 or len(names)!=707 or any(item.get("complete_label") is not True for item in items)
    or staged!=names or not predictor.is_file()):
 raise SystemExit("invalid verified-707 manifest/stage/predictor contract")
for item in items:
 image=Path(item["image"]); label=Path(item["label"])
 if "validation" in {part.casefold() for part in image.resolve().parts}:
  raise SystemExit("validation image leaked into training manifest")
 if not image.is_file() or not label.is_file(): raise SystemExit("missing training asset")
PY

while :; do
  used0=$(nvidia-smi --id=0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  used1=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')
  [[ "${used0}" =~ ^[0-9]+$ && "${used1}" =~ ^[0-9]+$ \
     && "${used0}" -le 1024 && "${used1}" -le 1024 ]] && break
  printf '%s waiting for both GPUs: gpu0=%s MiB gpu1=%s MiB\n' "$(date -Ins)" "${used0}" "${used1}"
  sleep 30
done

run_shard() {
  local gpu="$1" shard="$2" shard_terminal="${output}/SHARD_${shard}.json"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${predictor}" \
    --images "${images}" --output "${predictions}" --model-dir "${model}" \
    --checkpoint "${checkpoint_name}" --gpu 0 --part-id "${shard}" --num-parts 2 \
    --workers 1 --sequential --dataset-role training_complete \
    --terminal "${shard_terminal}" >"${output}/shard_${shard}.log" 2>&1
}
run_shard 0 0 & pid0=$!
run_shard 1 1 & pid1=$!
status=0; wait "${pid0}" || status=$?; wait "${pid1}" || status=$?
[[ "${status}" -eq 0 ]]

"${python_bin}" - "${manifest}" "${predictions}" "${public_terminal}" \
  "${output}/SHARD_0.json" "${output}/SHARD_1.json" "${checkpoint}" \
  "${checkpoint_sha}" "${predictor}" "${terminal}" <<'PY'
import datetime,hashlib,json,os,sys
from pathlib import Path
manifest,predictions,public,shard0,shard1,checkpoint=map(Path,sys.argv[1:7]); expected=sys.argv[7]; predictor=Path(sys.argv[8]); output=Path(sys.argv[9])
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest(); items=json.load(open(manifest,encoding="utf-8"))["items"]
names={str(item["prediction_name"]) for item in items}; observed={p.name for p in predictions.glob("*.nii.gz")}
if len(names)!=707 or observed!=names or sha(checkpoint)!=expected: raise SystemExit("proposal coverage/checkpoint mismatch")
records=[]
for shard in (shard0,shard1):
 p=json.load(open(shard,encoding="utf-8")); records.append({"path":str(shard.resolve()),"sha256":sha(shard)})
 if (p.get("status")!="COMPLETE" or p.get("dataset_role")!="training_complete"
     or not p.get("training_images_read") or p.get("public_images_read")
     or p.get("public_labels_read") or p.get("hidden_read") or p.get("hidden_labels_read")
     or p.get("save_probabilities") or p.get("submission_created") or p.get("uploaded")
     or p.get("checkpoint",{}).get("sha256")!=expected):
  raise SystemExit("invalid training-only inference shard")
payload={"schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","decision":"TRAIN_STU_PROPOSAL_PRECISION_GATE",
 "cases":707,"proposal_threshold":0.5,"probabilities_saved":False,
 "predictions":str(predictions.resolve()),"manifest":{"path":str(manifest.resolve()),"sha256":sha(manifest)},
 "source_public":{"path":str(public.resolve()),"sha256":sha(public)},
 "checkpoint":{"path":str(checkpoint.resolve()),"sha256":expected},
 "predictor":{"path":str(predictor.resolve()),"sha256":sha(predictor)},"shards":records,
 "training_images_read":True,"training_labels_read":False,"public_read":False,"hidden_read":False,
 "submission_created":False,"uploaded":False}
t=output.with_name(f".{output.name}.{os.getpid()}.tmp");t.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8");os.replace(t,output)
PY
trap - EXIT

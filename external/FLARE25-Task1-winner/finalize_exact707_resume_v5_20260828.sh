#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/nvme2-data/shashank/MICCAI26"
root="${project_root}/external/FLARE25-Task1-winner"
output="${root}/exact707_primary_audit_20260828"
predictions="${output}/predictions"
terminal="${output}/INFERENCE_TERMINAL.json"
manifest="${project_root}/external/GF-Screen/analysis/fpr_training_mining_707_20260822/manifest.json"
checkpoint="${root}/nnUNet_results/nnUNet_results/Dataset523_train_all/nnUNetTrainer_Epoch5000_Lr1e3__nnUNetPlans__3d_fullres_S4D2W32/fold_all/checkpoint_final.pth"
container="hias25-exact707-resume-v5-20260828"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"

while docker inspect "${container}" >/dev/null 2>&1 &&
      [[ "$(docker inspect -f '{{.State.Running}}' "${container}")" == true ]]; do
  sleep 30
done

exit_code="$(docker inspect -f '{{.State.ExitCode}}' "${container}")"
oom="$(docker inspect -f '{{.State.OOMKilled}}' "${container}")"
count="$(find "${predictions}" -maxdepth 1 -name '*.nii.gz' | wc -l)"
[[ "${exit_code}" == 0 && "${oom}" == false && "${count}" -eq 707 ]] || {
  echo "resume failed: exit=${exit_code}, oom=${oom}, outputs=${count}" >&2
  exit 1
}

"${python_bin}" - "${terminal}" "${manifest}" "${checkpoint}" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path

terminal, manifest, checkpoint = map(Path, sys.argv[1:4])
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
now = datetime.datetime.now().astimezone().isoformat()
payload = {
    "schema_version": 1,
    "created_at": now,
    "status": "COMPLETE",
    "cases": 707,
    "finished_at": now,
    "model": "official FLARE25 Task-1 winner",
    "manifest": {"path": str(manifest.resolve()), "sha256": digest(manifest)},
    "checkpoint": {"path": str(checkpoint.resolve()), "sha256": digest(checkpoint)},
    "execution": {"physical_gpu": 1, "npp": 1, "nps": 1, "tta": True,
                  "resume_container": "hias25-exact707-resume-v5-20260828"},
    "decision_role": "PRIMARY_LESION_EXPERT_DIAGNOSTIC_ONLY",
    "labels_read": False,
    "public_read": False,
    "hidden_read": False,
    "submission_created": False,
    "uploaded": False,
    "optimizer_started": False,
}
temporary = terminal.with_name(f".{terminal.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, terminal)
PY

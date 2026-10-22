#!/usr/bin/env bash
set -euo pipefail

project_root="/mnt/nvme2-data/shashank/MICCAI26"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
preprocess="${project_root}/nnunet/resume_preprocess.py"
runner="${project_root}/nnunet/run_flare25_multiteacher_distill.py"
checkpoint="${project_root}/external/FLARE25-Task1-winner/nnUNet_results/nnUNet_results/Dataset523_train_all/nnUNetTrainer_Epoch5000_Lr1e3__nnUNetPlans__3d_fullres_S4D2W32/fold_all/checkpoint_final.pth"
dataset="${project_root}/nnUNet_preprocessed/Dataset504_FLARE25MultiTeacher"
cache="${dataset}/nnUNetPlans_3d_fullres_S4D2W32"
build_marker="${dataset}/DATASET_BUILD_COMPLETE.json"
preprocess_marker="${dataset}/PREPROCESSING_COMPLETE.json"
work="${project_root}/nnunet/runs/flare25_multiteacher_stage10_20260830"
terminal="${work}/TERMINAL.json"
log="${work}/controller.log"

mkdir -p "${work}"
exec 9>"${work}/controller.lock"
flock -n 9 || { echo "FLARE25 multi-teacher controller already active" >&2; exit 1; }
exec > >(tee -a "${log}") 2>&1

export nnUNet_raw="${project_root}/nnUNet_raw"
export nnUNet_preprocessed="${project_root}/nnUNet_preprocessed"
export nnUNet_results="${project_root}/nnUNet_results"
export PYTHONPATH="${project_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${work}/matplotlib"
export WANDB_PROJECT="pancancer-ct"
export WANDB_MODE="${WANDB_MODE:-online}"
export nnUNet_compile="false"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2 PYTHONUNBUFFERED=1
export PANCANCER_AUGMENTATION_SEED_OFFSET=2026083000
mkdir -p "${MPLCONFIGDIR}"

write_terminal() {
  local status="$1" stage="$2" detail="$3"
  "${python_bin}" - "${terminal}" "${status}" "${stage}" "${detail}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
path=Path(sys.argv[1])
payload={
 "schema_version":1,"created_at":datetime.datetime.now().astimezone().isoformat(),
 "status":sys.argv[2],"stage":sys.argv[3],"detail":sys.argv[4],
 "public_used_for_training":False,"hidden_used_for_training":False,
 "submission_created":False,"uploaded":False,
}
temporary=path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,path)
PY
}
trap 'code=$?; write_terminal FAILED controller "exit ${code} at line ${LINENO}"; exit ${code}' ERR

while [[ ! -s "${build_marker}" ]]; do
  printf '%s waiting for Dataset504 target build\n' "$(date -Ins)"
  sleep 30
done

read -r pseudo total < <("${python_bin}" - "${build_marker}" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
if (p.get("status")!="COMPLETE" or p.get("target_dataset")!="Dataset504_FLARE25MultiTeacher"
    or p.get("labeled_cases")!=17575 or p.get("public_read") or p.get("hidden_read")):
    raise SystemExit("invalid Dataset504 build marker")
print(p["pseudo_cases"],p["total_cases"])
PY
)
[[ "${pseudo}" -ge 500 && "${total}" -eq $((17575 + pseudo)) ]]

if [[ ! -s "${preprocess_marker}" ]]; then
  write_terminal RUNNING preprocessing "preprocessing ${pseudo} new pseudo cases with 4 workers"
  "${python_bin}" -u "${preprocess}" -d 504 -c 3d_fullres_ResEncS6 \
    -plans_name nnUNetPlans -np 4 >"${work}/preprocess.log" 2>&1
  "${python_bin}" - "${cache}" "${pseudo}" "${total}" "${preprocess_marker}" <<'PY'
import datetime,json,os,sys
from pathlib import Path
cache=Path(sys.argv[1]); pseudo=int(sys.argv[2]); total=int(sys.argv[3]); output=Path(sys.argv[4])
data={p.name.removesuffix('.b2nd') for p in cache.glob('*.b2nd') if not p.name.endswith('_seg.b2nd')}
seg={p.name.removesuffix('_seg.b2nd') for p in cache.glob('*_seg.b2nd')}
props={p.name.removesuffix('.pkl') for p in cache.glob('*.pkl')}
if len(data)!=total or data!=seg or data!=props:
    raise SystemExit('Dataset504 preprocessed triplet inventory mismatch')
if len({name for name in data if name.startswith('Pseudo-')})!=pseudo:
    raise SystemExit('Dataset504 pseudo cache inventory mismatch')
payload={
 'schema_version':1,'created_at':datetime.datetime.now().astimezone().isoformat(),
 'status':'COMPLETE','dataset':'Dataset504_FLARE25MultiTeacher','workers':4,
 'labeled_reused_cases':17575,'pseudo_preprocessed_cases':pseudo,'total_cases':total,
 'public_read':False,'hidden_read':False,'submission_created':False,'uploaded':False,
}
temporary=output.with_name(f'.{output.name}.{os.getpid()}.tmp')
temporary.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8');os.replace(temporary,output)
PY
fi

expected_sha="c3f136cf916ea135025dcd43742e8e42ce6fcb6c9aab7802522bd8fb3e3b6941"
observed_sha="$(sha256sum "${checkpoint}" | cut -d' ' -f1)"
[[ "${observed_sha}" == "${expected_sha}" ]]

write_terminal RUNNING gpu_smoke "two-GPU global-batch-8 backward preflight"
"${python_bin}" -u "${runner}" Dataset504_FLARE25MultiTeacher \
  3d_fullres_FLARE25MTSmoke all \
  -tr nnUNetTrainerFLARE25MultiTeacherSmoke -p nnUNetPlans \
  -pretrained_weights "${checkpoint}" -num_gpus 2 \
  >"${work}/smoke.log" 2>&1

write_terminal RUNNING training "official FLARE25 student through fixed epoch 10"
export FLARE25_MT_STAGE_END_EPOCH=10
"${python_bin}" -u "${runner}" Dataset504_FLARE25MultiTeacher \
  3d_fullres_FLARE25MT all \
  -tr nnUNetTrainerFLARE25MultiTeacherDistill -p nnUNetPlans \
  -pretrained_weights "${checkpoint}" -num_gpus 2 \
  2>&1 | tee "${work}/training.log"

model="${project_root}/nnUNet_results/Dataset504_FLARE25MultiTeacher/nnUNetTrainerFLARE25MultiTeacherDistill__nnUNetPlans__3d_fullres_FLARE25MT/fold_all"
for name in checkpoint_epoch_0010.pth checkpoint_final.pth; do
  [[ -s "${model}/${name}" ]] || { echo "missing stage-10 checkpoint: ${name}" >&2; exit 5; }
done
write_terminal COMPLETE epoch10 "stage 10 complete; fixed-public evaluation required before resume"

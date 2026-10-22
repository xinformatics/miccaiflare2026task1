#!/usr/bin/env bash
set -euo pipefail

# Speculatively infer completed folds with the currently leading composite
# checkpoint family. The authoritative five-fold selector may choose another
# family; its controller verifies exact checkpoint paths and will overwrite
# mismatched inference. No materialization, labels, public/hidden data, or
# submission output is touched here.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
output="${root}/analysis/candidate_refiner_v2_fullvolume_oof_20260823"
terminal="${output}/COMPOSITE_INFERENCE_PREWARM.json"

exec 9>"${root}/runs/prewarm_candidate_refiner_v2_oof_composite_infer_20260823.lock"
flock -n 9 || { echo "OOF composite inference prewarm already active" >&2; exit 1; }
[[ -s "${terminal}" ]] && exit 0

records="${output}/.composite_inference_prewarm_records.tsv"
: > "${records}"
for fold in 0 1 2 3; do
  work="${output}/fold${fold}/work_bbox24"
  checkpoint="${root}/runs/candidate_refiner_v2_bbox24_b32_v2det_fold${fold}_20260823/checkpoint_best.pt"
  for required in "${work}/PREPARE_COMPLETE.txt" "${work}/AUDIT.json" "${checkpoint}"; do
    [[ -s "${required}" ]] || { echo "missing prewarm asset: ${required}" >&2; exit 2; }
  done
  exec 8>"${work}/INFERENCE_PREWARM.lock"
  flock 8
  CUDA_VISIBLE_DEVICES=1 "${python_bin}" -u "${root}/apply_candidate_refiner.py" infer \
    --work "${work}" --checkpoint "${checkpoint}" --batch-size 8 --workers 2 \
    > "${output}/fold${fold}.composite_prewarm.infer.log" 2>&1
  printf '%s\t%s\t%s\n' "${fold}" "${checkpoint}" "${work}/INFERENCE_COMPLETE.json" >> "${records}"
done

"${python_bin}" - "${records}" "${terminal}" <<'PY'
import hashlib,json,sys
from datetime import datetime
from pathlib import Path
records,output=map(Path,sys.argv[1:]); folds=[]
for line in records.read_text(encoding="utf-8").splitlines():
    fold,checkpoint,marker=line.split("\t")
    checkpoint=Path(checkpoint); marker=Path(marker)
    payload=json.loads(marker.read_text(encoding="utf-8"))
    expected=[str(checkpoint.resolve())]
    if payload.get("checkpoints")!=expected:
        raise SystemExit(f"inference checkpoint mismatch for fold {fold}")
    digest=hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): digest.update(block)
    folds.append({"fold":int(fold),"checkpoint":expected[0],
                  "checkpoint_sha256":digest.hexdigest(),"marker":str(marker.resolve())})
if [row["fold"] for row in folds]!=[0,1,2,3]: raise SystemExit("fold coverage mismatch")
payload={"schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","strategy":"composite","folds":folds,
 "authoritative_selection_pending":True,"public_read":False,"hidden_read":False,
 "submission_created":False}
temporary=output.with_name(f".{output.name}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(output)
PY

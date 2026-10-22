#!/usr/bin/env bash
set -euo pipefail

root=/mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen
dataset="${root}/analysis/subtractive_gate_diag_20260823/gate_dataset_707.csv"
dataset_marker="${root}/analysis/subtractive_gate_diag_20260823/GATE_DATASET_707_COMPLETE.json"
legacy_builder_log="${root}/analysis/subtractive_gate_diag_20260823/gate707.log"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
predictions="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
output="${root}/analysis/subtractive_gate_cv_707_20260823"
terminal="${output}/COMPLETE.txt"

mkdir -p "${output}"
if [[ -s "${terminal}" && -s "${output}/report.json" ]]; then
  exit 0
fi

# The upstream builder writes the CSV only after all 707 cases have been
# processed.  Also wait for its process to exit so a partially flushed table
# can never enter cross-validation.
while pgrep -f '[b]uild_gate_dataset_707.py' >/dev/null 2>&1; do
  printf '%s waiting for complete 707-case subtractive-gate table\n' "$(date -Ins)"
  sleep 60
done
[[ -s "${dataset}" ]] || { echo "missing completed gate dataset: ${dataset}" >&2; exit 2; }
# The first builder process began before atomic marker support was added. Its
# final JSON is emitted only after all 707 Pool results have been consumed and
# the CSV handle has closed. Convert that terminal evidence once, checking CSV
# row/case counts, so the completed work is not discarded.
if [[ ! -s "${dataset_marker}" ]]; then
  /home/m348911/miniconda3/envs/radoncAI/bin/python - "${dataset}" \
    "${legacy_builder_log}" "${manifest}" "${predictions}" "${dataset_marker}" <<'PY'
import csv,hashlib,json,os,sys
from datetime import datetime
from pathlib import Path
dataset,log,manifest,predictions,marker=map(Path,sys.argv[1:])
text=log.read_text(encoding="utf-8")
start=text.rfind("{")
if start < 0: raise SystemExit("legacy builder has no terminal JSON")
terminal=json.loads(text[start:])
with dataset.open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
items=json.loads(manifest.read_text(encoding="utf-8"))["items"]
checks={
 "manifest_cases":len(items)==707 and len({str(item["name"]) for item in items})==707,
 "rows":len(rows)==int(terminal.get("rows",-1)),
 "cases":len({row["case"] for row in rows})==int(terminal.get("cases",-1)),
 "tp":sum(int(row["is_tp"]) for row in rows)==int(terminal.get("tp",-1)),
}
failed=[name for name,passed in checks.items() if not passed]
if failed: raise SystemExit(f"legacy builder terminal validation failed: {failed}")
payload={
 "schema_version":1,"created_at":datetime.now().astimezone().isoformat(),
 "status":"COMPLETE","items_processed":707,"rows":len(rows),
 "cases_with_components":len({row["case"] for row in rows}),
 "tp":sum(int(row["is_tp"]) for row in rows),
 "dataset":str(dataset.resolve()),
 "dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest(),
 "manifest":str(manifest.resolve()),
 "manifest_sha256":hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "predictions":str(predictions.resolve()),"min_mm3":1000.0,
 "completion_proof":"legacy terminal JSON emitted after full 707-item pool loop and closed CSV",
 "labels_read":True,"public_or_hidden_data_read":False,
}
temporary=marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
os.replace(temporary,marker)
PY
fi
[[ -s "${dataset_marker}" ]] || { echo "missing gate dataset completion marker: ${dataset_marker}" >&2; exit 2; }
[[ -s "${manifest}" ]] || { echo "missing manifest: ${manifest}" >&2; exit 2; }
[[ -d "${predictions}" ]] || { echo "missing frozen predictions: ${predictions}" >&2; exit 2; }

/home/m348911/miniconda3/envs/radoncAI/bin/python - "${dataset}" "${dataset_marker}" \
  "${manifest}" "${predictions}" <<'PY'
import hashlib,json,sys
from pathlib import Path
dataset,marker,manifest,predictions=map(Path,sys.argv[1:])
payload=json.loads(marker.read_text(encoding="utf-8"))
checks={
 "status":payload.get("status")=="COMPLETE",
 "items":payload.get("items_processed")==707,
 "dataset":payload.get("dataset")==str(dataset.resolve()),
 "dataset_sha256":payload.get("dataset_sha256")==hashlib.sha256(dataset.read_bytes()).hexdigest(),
 "manifest":payload.get("manifest")==str(manifest.resolve()),
 "manifest_sha256":payload.get("manifest_sha256")==hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "predictions":payload.get("predictions")==str(predictions.resolve()),
 "min_mm3":float(payload.get("min_mm3",-1))==1000.0,
 "public_blind":payload.get("public_or_hidden_data_read") is False,
}
failed=[name for name,passed in checks.items() if not passed]
if failed: raise SystemExit(f"invalid gate dataset completion marker: {failed}")
PY

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${output}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
exec nice -n 19 /home/m348911/miniconda3/envs/radoncAI/bin/python -u \
  "${root}/analyze_subtractive_component_gate_cv.py" \
  --dataset "${dataset}" \
  --manifest "${manifest}" \
  --predictions "${predictions}" \
  --output "${output}" \
  --expected-cases 707 --min-mm3 1000 --seed 20260823 --bootstrap 5000

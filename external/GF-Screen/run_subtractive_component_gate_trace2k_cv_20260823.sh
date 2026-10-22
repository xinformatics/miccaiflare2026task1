#!/usr/bin/env bash
set -euo pipefail

# Deployment-scope qualification: the champion's separately guided additions
# are all <=2,000 mm3, so only larger GF components may be deleted.
root=/mnt/nvme2-data/shashank/MICCAI26/external/GF-Screen
dataset="${root}/analysis/subtractive_gate_diag_20260823/gate_dataset_707.csv"
marker="${root}/analysis/subtractive_gate_diag_20260823/GATE_DATASET_707_COMPLETE.json"
manifest="${root}/analysis/fpr_training_mining_707_20260822/manifest.json"
predictions="${root}/analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
output="${root}/analysis/subtractive_gate_trace2k_cv_707_20260823"
terminal="${output}/COMPLETE.txt"
anchor_cache="${root}/analysis/subtractive_gate_risk01_cv_707_20260823/official_metrics.tsv"

mkdir -p "${output}"
if [[ -s "${terminal}" && -s "${output}/report.json" ]]; then exit 0; fi
[[ -s "${dataset}" && -s "${marker}" && -s "${manifest}" && -s "${anchor_cache}" ]] || {
  echo "missing completed 707-case gate inputs" >&2; exit 2;
}
[[ -d "${predictions}" ]] || { echo "missing frozen GF predictions" >&2; exit 2; }

/home/m348911/miniconda3/envs/radoncAI/bin/python - \
  "${dataset}" "${marker}" "${manifest}" "${predictions}" <<'PY'
import hashlib,json,sys
from pathlib import Path
dataset,marker,manifest,predictions=map(Path,sys.argv[1:])
p=json.loads(marker.read_text(encoding="utf-8"))
checks={
 "complete":p.get("status")=="COMPLETE" and p.get("items_processed")==707,
 "dataset":p.get("dataset")==str(dataset.resolve()),
 "dataset_sha":p.get("dataset_sha256")==hashlib.sha256(dataset.read_bytes()).hexdigest(),
 "manifest":p.get("manifest")==str(manifest.resolve()),
 "manifest_sha":p.get("manifest_sha256")==hashlib.sha256(manifest.read_bytes()).hexdigest(),
 "predictions":p.get("predictions")==str(predictions.resolve()),
 "min_mm3":float(p.get("min_mm3",-1))==1000.0,
 "blind":p.get("public_or_hidden_data_read") is False,
}
failed=[key for key,value in checks.items() if not value]
if failed: raise SystemExit(f"invalid completed gate inputs: {failed}")
PY

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${output}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"
exec nice -n 19 /home/m348911/miniconda3/envs/radoncAI/bin/python -u \
  "${root}/analyze_subtractive_component_gate_cv.py" \
  --dataset "${dataset}" --manifest "${manifest}" \
  --predictions "${predictions}" --output "${output}" \
  --expected-cases 707 --min-mm3 1000 --seed 20260823 --bootstrap 5000 \
  --inner-catastrophic-cap 0.01 --minimum-deletion-mm3-exclusive 2000 \
  --anchor-metrics-cache "${anchor_cache}" --workers 3

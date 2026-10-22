#!/usr/bin/env bash
set -euo pipefail

# CPU-only, resumable deterministic-cache stage for the balanced positive-only
# Train-Unlabel manifest. It never reads public/hidden images or labels.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
cache_root="/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
pseudo_root="${root}/public_sweeps/gf_train_unlabel_cbg_direct_20260825"
pseudo_terminal="${pseudo_root}/PSEUDO_LABELS_COMPLETE.json"
manifest="${pseudo_root}/balanced/manifest.json"
audit="${pseudo_root}/balanced/audit.json"
contract="${pseudo_root}/PSEUDO_CACHE_DATASET_CONTRACT.json"
complete="${pseudo_root}/PSEUDO_CACHE_COMPLETE.json"
lock="${root}/runs/run_preprocess_positive_unlabeled_cache_20260825.lock"

mkdir -p "${root}/runs"
mkdir -p "${cache_root}/tmp/pseudo_positive_cache"
export TMPDIR="${cache_root}/tmp/pseudo_positive_cache"
exec 9>"${lock}"
flock -n 9 || { echo "positive-unlabeled cache builder already active" >&2; exit 1; }

while [[ ! -s "${pseudo_terminal}" ]]; do
  printf '%s waiting for audited Train-Unlabel pseudo labels before cache build\n' "$(date -Ins)"
  sleep 30
done
for file in "${manifest}" "${audit}" "${cache_root}/PREPROCESSING_COMPLETE.json"; do
  [[ -s "${file}" ]] || { echo "missing pseudo-cache prerequisite: ${file}" >&2; exit 2; }
done

if [[ -s "${complete}" ]]; then
  [[ -s "${contract}" ]] || { echo "missing completed pseudo-cache contract" >&2; exit 2; }
  "${python_bin}" "${root}/preprocess_positive_unlabeled_cache.py" \
    --cache-dir "${cache_root}" --manifest "${manifest}" \
    --dataset-contract "${contract}" --output "${complete}" --verify-only
  exit 0
fi

"${python_bin}" "${root}/validate_positive_unlabeled_manifest.py" \
  --pseudo-terminal "${pseudo_terminal}" --manifest "${manifest}" \
  --pseudo-audit "${audit}" --output "${contract}" \
  > "${pseudo_root}/pseudo_cache_contract.log"

"${python_bin}" -u "${root}/preprocess_positive_unlabeled_cache.py" \
  --cache-dir "${cache_root}" --manifest "${manifest}" \
  --dataset-contract "${contract}" --output "${complete}" \
  --workers 2 --minimum-free-gib 100

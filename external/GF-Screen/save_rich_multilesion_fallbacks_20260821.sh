#!/usr/bin/env bash
set -euo pipefail

# Save-only watcher. It preserves at most one near-anchor candidate per arm
# because the current hidden champion itself was a local KEEP_ANCHOR result.
# This never uploads to Codabench and never broadens the public selection grid.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
audit="${root}/public_sweeps/gf_rich16_full_dual_deployed_20260821"
marker="${audit}/PUBLIC_ONLY_COMPLETE.txt"
anchor="${root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"
selected="${audit}/near_anchor_fallbacks.tsv"
saved="${audit}/saved_fallbacks.tsv"

exec 9>"${root}/runs/save_rich_multilesion_fallbacks_20260821.lock"
if ! flock -n 9; then
  echo "rich multi-lesion fallback watcher is already active" >&2
  exit 1
fi
while [[ ! -s "${marker}" ]]; do
  sleep 30
done
# The main controller holds this lock through strict PROMOTE materialization.
# Acquire it before fallback inference so two save paths cannot contend for the
# same GPUs. This also guarantees the strict queue is always handled first.
exec 8>"${root}/runs/run_dual_rich_multilesion_full_label_20260821.lock"
flock 8
"${python_bin}" "${root}/select_near_anchor_fallbacks.py" \
  --summary "${audit}/summary.tsv" --anchor "${anchor}" \
  --output "${selected}" --count 2

printf 'rank\ttag\tvariant\tcheckpoint\tfocus\tDSC\tNSD\tmetrics\trobust\tsubmission\n' > "${saved}"
rank=0
while IFS=$'\t' read -r variant checkpoint_tag checkpoint focus dsc nsd \
    combined metrics robust anchor_dsc anchor_nsd dsc_drop nsd_drop catastrophic; do
  [[ "${variant}" == "variant" ]] && continue
  rank=$((rank + 1))
  arm=std
  [[ "${variant}" == *comp050* ]] && arm=c50
  epoch_tag="${checkpoint_tag#checkpoint_epoch_}"
  [[ "${checkpoint_tag}" == "model" ]] && epoch_tag=best
  focus_tag="${focus/./}"
  tag="gf_r16${arm}_e${epoch_tag}_l${focus_tag}"
  submission="${project}/submissions/${tag}.zip"
  if [[ ! -s "${submission}" ]]; then
    CHECKPOINT="${checkpoint}" TAG="${tag}" METRICS="${metrics}" \
      ROBUST_JSON="${robust}" GLANCE_THRESHOLD=0.50 LESION_THRESHOLD="${focus}" \
      ALLOW_FALLBACK=1 bash "${root}/materialize_deployed_candidate_submission_20260821.sh"
  fi
  unzip -t "${submission}" > "${submission}.zip_test.log"
  [[ $(unzip -Z1 "${submission}" | sed '/^$/d' | wc -l) -eq 100 ]]
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${rank}" "${tag}" "${variant}" "${checkpoint}" "${focus}" "${dsc}" \
    "${nsd}" "${metrics}" "${robust}" "${submission}" >> "${saved}"
done < "${selected}"

if [[ "${rank}" -eq 0 ]]; then
  printf '%s\n' 'No candidate met the bounded near-anchor fallback gate; no exploratory ZIP saved.' \
    > "${audit}/NO_NEAR_ANCHOR_FALLBACK.txt"
fi

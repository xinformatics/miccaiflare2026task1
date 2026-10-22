#!/usr/bin/env bash
set -euo pipefail

# Recover only the aggregate table/marker after the already-complete evaluator
# parent saw its source file change while running. No inference, postprocessing,
# or metric computation is repeated.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output="${root}/public_sweeps/gf_rich04_08_full_comp050_deployed_20260821"
arm_a="gf_rich04_full_comp050_ep280_lr1e5_3ep_20260821"
arm_b="gf_rich08_full_comp050_ep280_lr1e5_3ep_20260821"

for arm in "${arm_a}" "${arm_b}"; do
  summary="${output}/${arm}/summary.tsv"
  [[ $(wc -l < "${summary}") -eq 7 ]] || {
    echo "expected header plus six complete rows: ${summary}" >&2; exit 2;
  }
  for epoch in 281 282 283; do
    for threshold in 50 60; do
      metrics="${output}/${arm}/checkpoint_epoch_0${epoch}/l0p${threshold}.metrics.csv"
      [[ $(wc -l < "${metrics}") -eq 51 ]] || {
        echo "expected header plus 50 metric rows: ${metrics}" >&2; exit 2;
      }
    done
  done
done

head -1 "${output}/${arm_a}/summary.tsv" > "${output}/summary.tsv"
tail -n +2 "${output}/${arm_a}/summary.tsv" >> "${output}/summary.tsv"
tail -n +2 "${output}/${arm_b}/summary.tsv" >> "${output}/summary.tsv"
{
  head -1 "${output}/summary.tsv"
  tail -n +2 "${output}/summary.tsv" | sort -t $'\t' -k5,5gr -k6,6gr
} > "${output}/summary_by_dsc.tsv"
cat "${output}/summary_by_dsc.tsv"
date -Ins > "${output}/PUBLIC_ONLY_COMPLETE.txt"

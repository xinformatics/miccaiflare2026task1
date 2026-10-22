#!/usr/bin/env bash
set -euo pipefail

# Label-free CPU cache preparation for the isotropic candidate refiner. This
# can overlap upstream GPU inference and is safe to resume by completion marker.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
images="${project}/validation/Validation-Public-Images"
proposals="${root}/predictions/gf_a055_bbox_pu_best_ep280/public_raw"
base="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
output="${root}/public_sweeps/candidate_refiner_v1_20260822"
disconnected="${output}/work_iso15"
connected="${output}/work_connected_iso15"

for directory in "${images}" "${proposals}" "${base}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 NIfTI files: ${directory}" >&2; exit 2;
  }
done

prepare_cache() {
  local work="$1" selection="$2" log="$3"
  mkdir -p "${work}"
  (
    flock -x 8
    [[ -s "${work}/PREPARE_COMPLETE.txt" ]] && exit 0
    "${python_bin}" -u "${root}/apply_candidate_refiner.py" prepare \
      --images "${images}" --proposal "gf_l050=${proposals}" \
      --base "${base}" "${selection}" --work "${work}" \
      --shape 64,64,64 --spacing 1.5,1.5,1.5 --min-mm3 5 --max-mm3 20000 \
      > "${log}" 2>&1
  ) 8>"${work}.prepare.lock"
}

prepare_cache "${disconnected}" --only-disconnected "${output}/prepare_public_iso15.log"
prepare_cache "${connected}" --only-connected "${output}/prepare_public_connected_iso15.log"

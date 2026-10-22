#!/usr/bin/env bash
set -euo pipefail

# Public-only Focus flip-TTA audit on the immutable champion checkpoint.
#
# Why this exists: the 2026-08-20 TTA runner
# (run_dual_focus_tta_after_safe_negatives.sh) was implemented, tested, and
# queued, but never executed -- both of its logs are zero bytes. It was lost
# when the downstream watcher cascade was stopped after an OOM. TTA is the only
# remaining cheap lever that is a *model-level* change rather than a knob tuned
# on the 50 public cases, so it is the only one expected to transfer to hidden
# at ~1x instead of the measured 0.20x post-processing rate.
#
# Scope deliberately bounded, differing from the 08-20 script:
#   * public 50 only -- NO hidden inference, NO ZIP, NO packaging, NO upload
#   * computes its own no-TTA baseline in the same run, so the comparison is
#     head-to-head at the identical pipeline stage instead of against a stale
#     metrics file from an older anchor
#   * stops at the mm1000 filter stage; prompt confirmation and the
#     disconnected-component step are left to the existing controllers if this
#     clears
#   * waits for a genuinely free GPU (codex's 1536 MiB convention) so it cannot
#     collide with active training
#
# Runtime guard: the champion measures 8.37 s/case mean, 18.92 s max. Any
# variant whose measured public maximum exceeds 55 s is ineligible under the
# 60 s challenge limit and is reported as such.

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${script_root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="${PYTHON_BIN:-/home/m348911/miniconda3/envs/radoncAI/bin/python}"
checkpoint="${CHECKPOINT:-${script_root}/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt}"
public_images="${PUBLIC_IMAGES:-${project}/validation/Validation-Public-Images}"
public_labels="${PUBLIC_LABELS:-${project}/validation/Validation-Public-Labels}"
root="${OUTPUT_ROOT:-${script_root}/public_sweeps/gf_champion_focus_tta_20260824}"
thresholds="${THRESHOLDS:-0.50}"
min_mm3="${MIN_MM3:-1000}"
idle_mib="${IDLE_MIB:-1536}"
gpu_poll_seconds="${GPU_POLL_SECONDS:-10}"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
guided_candidate="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"
anchor_metrics="${script_root}/public_sweeps/disconnected_guided_support_20260821/metrics/new_min0_max2000_dist0_sf0.25_sm100.metrics.csv"

[[ "${thresholds}" == 0.50 && "${min_mm3}" == 1000 ]] || {
  echo "the champion-reproduction TTA audit requires threshold 0.50 and min-mm3 1000" >&2
  exit 2
}

[[ -f "${checkpoint}" ]] || { echo "missing checkpoint: ${checkpoint}" >&2; exit 1; }
[[ -d "${public_images}" ]] || { echo "missing images: ${public_images}" >&2; exit 1; }
[[ $(find "${public_labels}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
  echo "expected 50 public labels" >&2; exit 1; }
for directory in "${prompt_support}" "${guided_candidate}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public support masks: ${directory}" >&2; exit 1; }
done
for required in "${prompt_config}" "${anchor_metrics}"; do
  [[ -s "${required}" ]] || { echo "missing deployment asset: ${required}" >&2; exit 1; }
done
mkdir -p "${root}"

# Block until a GPU is below the idle threshold, then claim it.
wait_for_free_gpu() {
  [[ "${gpu_poll_seconds}" =~ ^[0-9]+$ && "${gpu_poll_seconds}" -ge 5 ]] || {
    echo "GPU_POLL_SECONDS must be an integer >=5" >&2
    return 2
  }
  while true; do
    while IFS=, read -r index used; do
      used="${used// /}"
      if [[ "${used}" -lt "${idle_mib}" ]]; then echo "${index}"; return 0; fi
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
    sleep "${gpu_poll_seconds}"
  done
}

infer_variant() {
  local gpu="$1" tag="$2" axes="$3"
  local output="${root}/${tag}"
  mkdir -p "${output}"
  echo "[$(date -Is)] ${tag} (axes='${axes}') on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u val_GF_FLARE.py \
    --test_data_path "${public_images}" --save_prediction_path "${output}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.50 --lesion_thresholds "${thresholds}" \
    --focus_tta_axes "${axes}" --focus_batch_size 2 --workers 2 \
    > "${output}.inference.log" 2>&1
}

# baseline first, so a head-to-head pair exists even if TTA is interrupted
gpu="$(wait_for_free_gpu)"
infer_variant "${gpu}" notta ""
mapfile -t free_gpus < <(
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v limit="${idle_mib}" '{gsub(/ /,"",$2); if ($2 < limit) print $1}'
)
if [[ "${#free_gpus[@]}" -ge 2 ]]; then
  infer_variant "${free_gpus[0]}" tta_x x & pid_x=$!
  infer_variant "${free_gpus[1]}" tta_xy xy & pid_xy=$!
  status=0; wait "${pid_x}" || status=$?; wait "${pid_xy}" || status=$?
  [[ "${status}" -eq 0 ]] || exit "${status}"
else
  gpu="$(wait_for_free_gpu)"; infer_variant "${gpu}" tta_x x
  gpu="$(wait_for_free_gpu)"; infer_variant "${gpu}" tta_xy xy
fi

grid_args=()
printf 'variant\tflip_axes\tmean_seconds\tmax_seconds\truntime_eligible\n' > "${root}/runtime.tsv"
for specification in "notta:" "tta_x:x" "tta_xy:xy"; do
  tag="${specification%%:*}"; axes="${specification#*:}"
  source_dir="${root}/${tag}/lesion_${thresholds}"
  [[ $(find "${source_dir}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "incomplete predictions: ${source_dir}" >&2; exit 1; }
  read -r mean_seconds max_seconds <<< "$(awk '/^Duration:/ {s+=$2; n++; if($2>m)m=$2} END {printf "%.3f %.3f", (n?s/n:0), m}' "${root}/${tag}.inference.log")"
  eligible=yes
  awk -v m="${max_seconds}" 'BEGIN {exit !(m > 55.0)}' && eligible=no
  printf '%s\t%s\t%s\t%s\t%s\n' "${tag}" "${axes:-none}" "${mean_seconds}" "${max_seconds}" "${eligible}" \
    >> "${root}/runtime.tsv"
  state="${root}/scored/${tag}_mm${min_mm3}"
  filtered="${state}/filtered"
  prompted="${state}/prompted"
  deployed="${state}/deployed_new2k_s25"
  mkdir -p "${filtered}" "${prompted}" "${deployed}"
  "${python_bin}" -u fuse_and_filter_masks.py \
    --pred-a "${source_dir}" --fusion a --min-mm3 "${min_mm3}" \
    --output "${filtered}" > "${state}/filter.log" 2>&1
  "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
    --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
    --output "${prompted}" > "${state}/prompted.log" 2>&1
  "${python_bin}" -u "${script_root}/screen_disconnected_guided_additions.py" materialize \
    --base "${prompted}" --candidate "${guided_candidate}" \
    --support "${prompt_support}" --output "${deployed}" \
    --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
    --min-support-fraction 0.25 --min-support-mm3 100 \
    --support-dilation-mm inf --workers 4 > "${state}/deployed.log" 2>&1
  [[ $(find "${deployed}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "incomplete deployed predictions: ${deployed}" >&2; exit 1; }
  grid_args+=(--candidate "${tag}_mm${min_mm3}=${deployed}")
done

"${python_bin}" -u evaluate_public_grid.py --gt "${public_labels}" \
  --output-dir "${root}/scored" --workers 8 "${grid_args[@]}" \
  > "${root}/grid.score.log" 2>&1

champion_anchor_match=no
if "${python_bin}" -u compare_focus_deployment_metrics.py \
  --anchor "${anchor_metrics}" \
  --candidate "${root}/scored/notta_mm${min_mm3}.metrics.csv" \
  --output "${root}/notta_deployment_equivalence.json" \
  > "${root}/notta_deployment_equivalence.log" 2>&1; then
  champion_anchor_match=yes
fi

# Paired robustness of each TTA variant against the run's own no-TTA baseline.
for tag in tta_x tta_xy; do
  "${python_bin}" -u select_checkpoint_robust.py \
    --anchor "${root}/scored/notta_mm${min_mm3}.metrics.csv" \
    --candidate "${root}/scored/${tag}_mm${min_mm3}.metrics.csv" \
    --anchor-name notta --candidate-name "${tag}" \
    --output "${root}/robust_${tag}.json" > "${root}/robust_${tag}.log" 2>&1 || true
  "${python_bin}" -u select_checkpoint_robust.py \
    --anchor "${anchor_metrics}" \
    --candidate "${root}/scored/${tag}_mm${min_mm3}.metrics.csv" \
    --anchor-name gf_champ_new2k_s25 --candidate-name "${tag}" \
    --output "${root}/robust_${tag}_vs_champion.json" \
    > "${root}/robust_${tag}_vs_champion.log" 2>&1 || true
done

printf 'variant\tDSC\tNSD\tcombined\truntime_eligible\tvs_notta\tvs_champion\tchampion_anchor_match\n' > "${root}/summary.tsv"
for tag in notta tta_x tta_xy; do
  m="${root}/scored/${tag}_mm${min_mm3}.metrics.csv"
  dsc=$(awk -F, 'NR>1 {s+=$2; n++} END {printf "%.6f", s/n}' "${m}")
  nsd=$(awk -F, 'NR>1 {s+=$3; n++} END {printf "%.6f", s/n}' "${m}")
  runtime_eligible=$(awk -F'\t' -v t="${tag}" '$1==t {print $5}' "${root}/runtime.tsv")
  notta_decision=ANCHOR
  champion_decision=ANCHOR
  if [[ "${tag}" != notta ]]; then
    notta_decision=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' \
      "${root}/robust_${tag}.json")
    champion_decision=$("${python_bin}" -c \
      'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' \
      "${root}/robust_${tag}_vs_champion.json")
  fi
  awk -v t="${tag}" -v d="${dsc}" -v n="${nsd}" -v r="${runtime_eligible}" \
    -v a="${notta_decision}" -v c="${champion_decision}" \
    -v m="${champion_anchor_match}" \
    'BEGIN {printf "%s\t%s\t%s\t%.6f\t%s\t%s\t%s\t%s\n", t, d, n, (d+n)/2, r, a, c, m}' \
    >> "${root}/summary.tsv"
done
echo "=== runtime ==="; cat "${root}/runtime.tsv"
echo "=== final deployed public scores (champion anchor is 0.590780 / 0.498244) ==="
cat "${root}/summary.tsv"
echo
echo "No hidden inference, archive, or upload was performed by this script."

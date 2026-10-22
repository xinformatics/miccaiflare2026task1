#!/usr/bin/env bash
set -euo pipefail

# Current-best, public-only Focus flip-TTA audit. The CBG epoch-283 Glance
# checkpoint is fixed, gf_conf is the immutable deployed anchor, and hidden
# inference/packaging is forbidden here. A separate saver may consume only a
# publicly qualified row.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project="$(cd "${root}/../.." && pwd)"
biomed="${project}/external/BiomedParse"
python_bin="/home/m348911/miniconda3/envs/radoncAI/bin/python"
checkpoint="${root}/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt"
images="${project}/validation/Validation-Public-Images"
labels="${project}/validation/Validation-Public-Labels"
output="${root}/public_sweeps/gf_cbg_focus_tta_current_20260823"
champion="${root}/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
gfconf="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/predictions/cbg_confirm_distance_mm_0_rescue_min_mm3_inf"
anchor="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/exact0.metrics.csv"
confirm_config="${root}/public_sweeps/champion_cbg41_component_confirmation_20260822/configs/cbg_confirm_distance_mm_0_rescue_min_mm3_inf.json"
prompt_support="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/supports/union_e050"
prompt_config="${biomed}/public_sweeps/combined_prompt_confirmation_20260820/screen/configs/prompt_or_e050_confirm_distance_mm_0_rescue_min_mm3_inf.json"
whole="${biomed}/public_sweeps/automatic_text_proposals_20260820/whole_body_lesion_e050_nifti"
tumor="${biomed}/public_sweeps/automatic_text_proposals_20260820/tumor_e050_nifti"
guided="${biomed}/public_sweeps/automatic_text_proposals_20260820/prompted_ranking/predictions/guided_whole_body_lesion_e050_v500_l0.50_mm500"

mkdir -p "${output}/robust" "${output}/candidates"
exec 9>"${output}.lock"
flock -n 9 || { echo "current CBG Focus-TTA audit already active" >&2; exit 1; }
[[ -s "${output}/PUBLIC_ONLY_TERMINAL.json" ]] && exit 0
for file in "${checkpoint}" "${anchor}" "${confirm_config}" "${prompt_config}"; do
  [[ -s "${file}" ]] || { echo "missing TTA asset: ${file}" >&2; exit 2; }
done
for directory in "${images}" "${labels}" "${champion}" "${gfconf}" \
    "${prompt_support}" "${whole}" "${tumor}" "${guided}"; do
  [[ $(find "${directory}" -maxdepth 1 -name '*.nii.gz' | wc -l) -eq 50 ]] || {
    echo "expected 50 public masks: ${directory}" >&2; exit 2;
  }
done

infer_variant() {
  local gpu="$1" tag="$2" axes="$3" prediction="${output}/${tag}"
  mkdir -p "${prediction}"
  [[ $(find "${prediction}/lesion_0.45" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 50 \
     && $(find "${prediction}/lesion_0.60" -maxdepth 1 -name '*.nii.gz' 2>/dev/null | wc -l) -eq 50 ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u "${root}/val_GF_FLARE.py" \
    --test_data_path "${images}" --save_prediction_path "${prediction}" \
    --trained_pth "${checkpoint}" --a_min -175 --a_max 250 \
    --glance_threshold 0.20 --lesion_thresholds 0.45,0.50,0.55,0.60 \
    --focus_tta_axes "${axes}" --focus_batch_size 2 --workers 2 \
    > "${output}/${tag}.inference.log" 2>&1
}
infer_variant 0 tta_x x & pid_x=$!
infer_variant 1 tta_xy xy & pid_xy=$!
status=0; wait "${pid_x}" || status=$?; wait "${pid_xy}" || status=$?
[[ "${status}" -eq 0 ]] || exit "${status}"

printf 'variant\tflip_axes\tmean_seconds\tmax_seconds\truntime_eligible\n' > "${output}/runtime.tsv"
for spec in tta_x:x tta_xy:xy; do
  tag="${spec%%:*}"; axes="${spec#*:}"
  read -r mean_seconds max_seconds count < <(awk '/^Duration:/ {s+=$2;n++;if($2>m)m=$2} END{printf "%.6f %.6f %d\n",(n?s/n:999),(n?m:999),n}' "${output}/${tag}.inference.log")
  eligible=no
  [[ "${count}" -eq 50 ]] && awk -v m="${max_seconds}" 'BEGIN{exit !(m<=55)}' && eligible=yes
  printf '%s\t%s\t%s\t%s\t%s\n' "${tag}" "${axes}" "${mean_seconds}" "${max_seconds}" "${eligible}" >> "${output}/runtime.tsv"
done

printf 'family\ttag\tvariant\tlesion_threshold\tconfig\tDSC\tNSD\tcombined\tmean_seconds\tmax_seconds\truntime_eligible\tdecision\tpredictions\tmetrics\trobust\n' \
  > "${output}/candidates.tsv"

# Role 1: use TTA CBG support to confirm complete components from the immutable
# champion. This is the exact mechanism that produced gf_conf.
for spec in tta_x:x tta_xy:xy; do
  tag="${spec%%:*}"; axes="${spec#*:}"
  runtime=$(awk -F '\t' -v t="${tag}" '$1==t{print $3,$4,$5}' "${output}/runtime.tsv")
  read -r mean_seconds max_seconds eligible <<< "${runtime}"
  for threshold in 0.45 0.50 0.55 0.60; do
    threshold_tag="${threshold/./p}"
    work="${output}/candidates/confirm_${tag}_l${threshold_tag}"
    filtered="${work}/filtered"; prompted="${work}/prompted"; deployed="${work}/deployed"; prediction="${work}/prediction"
    mkdir -p "${filtered}" "${prompted}" "${deployed}" "${prediction}"
    "${python_bin}" -u "${root}/fuse_and_filter_masks.py" \
      --pred-a "${output}/${tag}/lesion_${threshold}" --fusion a --min-mm3 1000 \
      --workers 4 --output "${filtered}" > "${work}/filter.log" 2>&1
    "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
      --base "${filtered}" --support "${prompt_support}" --config "${prompt_config}" \
      --output "${prompted}" > "${work}/prompt.log" 2>&1
    "${python_bin}" -u "${root}/screen_disconnected_guided_additions.py" materialize \
      --base "${prompted}" --candidate "${guided}" --support "${prompt_support}" \
      --output "${deployed}" --min-mm3 0 --max-mm3 2000 --min-distance-mm 0 \
      --min-support-fraction 0.25 --min-support-mm3 100 \
      --support-dilation-mm inf --workers 4 > "${work}/guided.log" 2>&1
    "${python_bin}" -u "${biomed}/biomedparse_fusion.py" materialize \
      --base "${champion}" --support "${deployed}" --config "${confirm_config}" \
      --output "${prediction}" > "${work}/confirm.log" 2>&1
    metrics="${work}/metrics.csv"; robust="${output}/robust/confirm_${tag}_l${threshold_tag}.json"
    "${python_bin}" -u "${root}/evaluate_public_parallel.py" \
      --pred "${prediction}" --gt "${labels}" --csv "${metrics}" --workers 4 \
      > "${work}/score.log" 2>&1
    "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
      --anchor "${anchor}" --candidate "${metrics}" --anchor-name gf_conf_cbg41_e283 \
      --candidate-name "confirm_${tag}_l${threshold_tag}" --output "${robust}" \
      > "${work}/robust.log"
    read -r dsc nsd < <("${python_bin}" - "${metrics}" <<'PY'
import csv,sys,numpy as np
r=list(csv.DictReader(open(sys.argv[1])));print(np.mean([float(x['Lesion_DSC']) for x in r]),np.mean([float(x['Lesion_NSD']) for x in r]))
PY
    )
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.9f",(d+n)/2}')
    decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
    printf 'confirmation\tconfirm_%s_l%s\t%s\t%s\t-\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${tag}" "${threshold_tag}" "${tag}" "${threshold}" "${dsc}" "${nsd}" "${combined}" \
      "${mean_seconds}" "${max_seconds}" "${eligible}" "${decision}" "${prediction}" "${metrics}" "${robust}" \
      >> "${output}/candidates.tsv"
  done
done

# Role 2: retain gf_conf unchanged and add only seeded, persistent, independently
# prompt-supported components. This directly targets missed-lesion DSC.
eligible_x=$(awk -F '\t' '$1=="tta_x"{print $5}' "${output}/runtime.tsv")
eligible_xy=$(awk -F '\t' '$1=="tta_xy"{print $5}' "${output}/runtime.tsv")
if [[ "${eligible_x}" == yes && "${eligible_xy}" == yes ]]; then
  addition="${output}/additions"
  mkdir -p "${addition}/robust"
  "${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" screen \
    --base "${gfconf}" --candidate "${output}/tta_x/lesion_0.45" \
    --seed "${output}/tta_x/lesion_0.60" --persistence "${output}/tta_xy/lesion_0.50" \
    --whole "${whole}" --tumor "${tumor}" --gt "${labels}" \
    --output "${addition}" --workers 4 > "${addition}/screen.log" 2>&1
  for config in union25_persist_2k_full vote2_seed_2k_full union25_persist_5k_full; do
    prediction="${addition}/predictions/${config}"; mkdir -p "${prediction}"
    "${python_bin}" -u "${root}/screen_component_hysteresis_consensus.py" materialize \
      --base "${gfconf}" --candidate "${output}/tta_x/lesion_0.45" \
      --seed "${output}/tta_x/lesion_0.60" --persistence "${output}/tta_xy/lesion_0.50" \
      --whole "${whole}" --tumor "${tumor}" --config "${config}" \
      --output "${prediction}" --workers 4 > "${addition}/${config}.materialize.log" 2>&1
    metrics="${addition}/metrics/${config}.metrics.csv"; robust="${addition}/robust/${config}.json"
    "${python_bin}" -u "${root}/select_checkpoint_robust.py" \
      --anchor "${anchor}" --candidate "${metrics}" --anchor-name gf_conf_cbg41_e283 \
      --candidate-name "tta_add_${config}" --output "${robust}" > "${addition}/robust/${config}.log"
    read -r dsc nsd < <("${python_bin}" - "${metrics}" <<'PY'
import csv,sys,numpy as np
r=list(csv.DictReader(open(sys.argv[1])));print(np.mean([float(x['Lesion_DSC']) for x in r]),np.mean([float(x['Lesion_NSD']) for x in r]))
PY
    )
    combined=$(awk -v d="${dsc}" -v n="${nsd}" 'BEGIN{printf "%.9f",(d+n)/2}')
    decision=$("${python_bin}" -c 'import json,sys;print(json.load(open(sys.argv[1]))["decision"])' "${robust}")
    mean_seconds=$(awk -F '\t' 'NR>1{s+=$3}END{print s/(NR-1)}' "${output}/runtime.tsv")
    max_seconds=$(awk -F '\t' 'NR>1&&$4>m{m=$4}END{print m}' "${output}/runtime.tsv")
    printf 'addition\tadd_%s\ttta_x+tta_xy\t0.45/0.60+0.50\t%s\t%s\t%s\t%s\t%s\t%s\tyes\t%s\t%s\t%s\t%s\n' \
      "${config}" "${config}" "${dsc}" "${nsd}" "${combined}" "${mean_seconds}" "${max_seconds}" \
      "${decision}" "${prediction}" "${metrics}" "${robust}" >> "${output}/candidates.tsv"
  done
fi

{
  head -1 "${output}/candidates.tsv"
  tail -n +2 "${output}/candidates.tsv" | awk -F '\t' '$11=="yes"' | \
    sort -t $'\t' -k12,12r -k8,8gr -k6,6gr
} > "${output}/candidates_ranked.tsv"
"${python_bin}" - "${output}/candidates_ranked.tsv" "${output}/PUBLIC_ONLY_TERMINAL.json" <<'PY'
import csv,datetime,json,os,sys
from pathlib import Path
table,output=map(Path,sys.argv[1:]); rows=list(csv.DictReader(table.open(),delimiter='\t'))
promoted=[r for r in rows if r['decision']=='PROMOTE']
payload={'schema_version':1,'created_at':datetime.datetime.now().astimezone().isoformat(),
 'status':'COMPLETE','decision':'PROMOTE_PUBLIC_ONLY' if promoted else 'KEEP_GFCONF',
 'selected':promoted[0] if promoted else (rows[0] if rows else None),
 'runtime_safe_candidates':len(rows),'promoted_candidates':len(promoted),
 'public_read':True,'hidden_read':False,'submission_created':False,'uploaded':False}
tmp=output.with_name('.'+output.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2)+'\n');os.replace(tmp,output)
PY
cat "${output}/candidates_ranked.tsv"

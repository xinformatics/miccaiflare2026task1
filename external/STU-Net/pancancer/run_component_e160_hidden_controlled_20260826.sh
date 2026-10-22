#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
saver="${root}/save_component_e160_p70_now_20260826.sh"
dual_launcher="${root}/launch_stunetb_dual_after_e100_20260826.sh"
e200_watcher="miccai-stunetb-component-e200-public-20260826.service"
base_unit="miccai-stunetb-base-gpu0-post-e100-20260826.service"
output="${root}/public_sweeps/stunetb_component_e160_p70_saved_20260826"

mkdir -p "${output}"
exec 9>"${output}/training_control.lock"
flock -n 9 || { echo "component epoch-160 hidden controller already active" >&2; exit 1; }

base_paused=false
rearm_e200=false
resume_services() {
  if [[ "${base_paused}" == true ]]; then
    /usr/bin/bash "${dual_launcher}" || true
    base_paused=false
  fi
  if [[ "${rearm_e200}" == true ]] && \
     ! systemctl --user is-active --quiet "${e200_watcher}"; then
    systemd-run --user --unit="${e200_watcher%.service}" --collect \
      --property=MemoryHigh=56G --property=MemoryMax=72G \
      /usr/bin/bash "${root}/run_component_e200_public_20260826.sh" || true
    rearm_e200=false
  fi
}
trap resume_services EXIT

if systemctl --user is-active --quiet "${e200_watcher}"; then
  systemctl --user stop "${e200_watcher}"
  rearm_e200=true
fi
if systemctl --user is-active --quiet "${base_unit}"; then
  systemctl --user stop "${base_unit}"
fi
base_paused=true
while systemctl --user is-active --quiet "${base_unit}"; do sleep 2; done

/usr/bin/bash "${saver}"
resume_services
trap - EXIT

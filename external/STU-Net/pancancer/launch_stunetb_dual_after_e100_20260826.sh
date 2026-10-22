#!/usr/bin/env bash
set -euo pipefail

# Launch matched single-GPU continuations from the exact epoch-100 optimizer
# state. GPU 0 preserves the original global-box recipe; GPU 1 changes only the
# trusted-region construction to a union of per-component boxes.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
root="${project_root}/external/STU-Net/pancancer"
base_script="${root}/continue_stunetb_base_gpu0_after_e100_20260826.sh"
component_script="${root}/continue_stunetb_component_gpu1_after_e100_20260826.sh"
equivalence_script="${root}/validate_direct_compact_e100_real_case_20260826.sh"

for script in "${base_script}" "${component_script}"; do
  [[ -x "${script}" ]] || { echo "missing executable continuation: ${script}" >&2; exit 2; }
done

# Spend one public case once to prove the faster probability exporter is
# bit-identical before either branch starts. A failed equivalence check is
# fail-closed for exporter activation but must not strand model training; all
# later consumers fall back to the legacy exporter unless COMPLETE is proven.
if ! /usr/bin/bash "${equivalence_script}"; then
  printf '%s direct-compact equivalence did not pass; continuing training with legacy export available\n' "$(date -Ins)" >&2
fi

if ! systemctl --user is-active --quiet miccai-stunetb-base-gpu0-post-e100-20260826.service; then
  systemd-run --user --unit=miccai-stunetb-base-gpu0-post-e100-20260826 --collect \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
    /usr/bin/bash "${base_script}"
fi
if ! systemctl --user is-active --quiet miccai-stunetb-component-gpu1-post-e100-20260826.service; then
  systemd-run --user --unit=miccai-stunetb-component-gpu1-post-e100-20260826 --collect \
    --property=Restart=on-failure --property=RestartSec=60 \
    --property=Nice=0 --property=CPUWeight=100 --property=IOWeight=100 \
    /usr/bin/bash "${component_script}"
fi

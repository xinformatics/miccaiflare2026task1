#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gf="$(cd "${root}/../GF-Screen" && pwd)"

# The LR evaluation queue holds this lock from launch until all new GF
# checkpoints have been scored.  Waiting here guarantees that organ-prompt
# inference cannot contend for either GPU.
exec 9>"${gf}/runs/lr_escape_eval_queue_20260821.lock"
flock 9
bash "${root}/run_organ_specific_rescue_public_20260821.sh"

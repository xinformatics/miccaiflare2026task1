#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec 9>"${root}/runs/run_unlabeled_consensus_queue_20260822.lock"
flock -n 9 || { echo "unlabeled consensus queue already active" >&2; exit 1; }
bash "${root}/evaluate_unlabeled_glance_consensus_public_20260822.sh"
bash "${root}/save_unlabeled_consensus_candidate_20260822.sh"

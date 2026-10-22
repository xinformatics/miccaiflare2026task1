#!/usr/bin/env bash
set -euo pipefail

# Compatibility entry point. The active higher-headroom schedule runs the
# controlled Focus loss gate before 707 scale-out. Keeping this wrapper avoids
# a second writer if an older launch command is reused.
compat_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${compat_root}/run_focus_candidate_refiner_pilot_before_707_20260823.sh"

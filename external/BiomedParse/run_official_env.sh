#!/usr/bin/env bash
set -euo pipefail

# Run the author-provided BiomedParse v2 Python 3.10 environment extracted
# from the official CVPR challenge image without loading that image into the
# space-constrained system Docker daemon.
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
site="${root}/container_python/usr/local/lib/python3.10/dist-packages"
workspace="${root}/container_workspace"

[[ -x /usr/bin/python3.10 ]] || {
  echo "System Python 3.10 is required" >&2
  exit 1
}
[[ -d "${site}/torch" && -f "${workspace}/model_weights/biomedparse_3D_AllData_MultiView_edge.ckpt" ]] || {
  echo "The extracted official BiomedParse runtime/checkpoint is incomplete" >&2
  exit 1
}

library_path="${site}/torch/lib"
for directory in "${site}"/nvidia/*/lib; do
  [[ -d "${directory}" ]] && library_path="${library_path}:${directory}"
done

export PYTHONPATH="${workspace}:${site}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${library_path}:/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "${workspace}"
exec /usr/bin/python3.10 "$@"

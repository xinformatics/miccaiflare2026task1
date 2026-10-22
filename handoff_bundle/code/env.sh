# Source this before any nnU-Net command:  source code/env.sh
export PROJ=/media/shashank/NewVolume/PancancerCTSeg

# Generated data lives on the 8TB HDD (/dev/sda, 6.0TB free). The preprocessed
# cache projects to ~226GB at 2.4x1.6x1.6mm (measured 13.2 MB/case), and the NVMe
# holding the raw corpus has only ~126GB free -- it would have run dry mid-run.
# nnUNet_raw stays on the NVMe because it is only symlinks into the raw download.
export HDD=/mnt/yadav/PancancerCTSeg
export nnUNet_raw="$PROJ/nnUNet_raw"
export nnUNet_preprocessed="$HDD/nnUNet_preprocessed"
export nnUNet_results="$HDD/nnUNet_results"
export PREDS="$HDD/preds"

export PY="$PROJ/.venv/bin/python"
# Measured on this box: 667 -> 537 ms/iteration (1.24x). Over a 2000-epoch run
# that is ~18 hours saved. Costs ~22s of warmup at startup.
export nnUNet_compile=1
export nnUNet_n_proc_DA=12
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export PATH="$PROJ/.venv/bin:$PATH"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$PREDS"

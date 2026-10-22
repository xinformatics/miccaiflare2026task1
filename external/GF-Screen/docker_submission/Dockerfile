FROM mrigroupopbg/mri-python3:latest

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN python -m pip install --no-cache-dir --upgrade "pip<25" setuptools wheel && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu116 \
      torch==1.13.1+cu116 && \
    python -m pip install --no-cache-dir \
      numpy==1.24.4 \
      scipy==1.10.1 \
      nibabel==5.2.1 \
      SimpleITK==2.3.1 \
      monai==1.3.0

RUN python -m pip install --no-cache-dir einops==0.6.1

WORKDIR /workspace
COPY models /workspace/models
COPY utils/__init__.py utils/utils.py /workspace/utils/
COPY docker_submission/inference.py /workspace/inference.py
COPY docker_submission/predict.sh /workspace/predict.sh
COPY docker_submission/model.pt /workspace/model.pt

RUN chmod 755 /workspace/predict.sh /workspace/inference.py && \
    python - <<'PY'
import torch
from types import SimpleNamespace
from models.GF_Screen_RL import GFScreen_RL
args = SimpleNamespace(
    glance_backbone="resnet18", in_channels=1, out_channels=2,
    feature_size=48, use_checkpoint=False, roi_x=96, roi_y=96, roi_z=64,
    batch_size=1, sw_batch_size=1, focus_batch_size=1, glance_threshold=0.5,
)
checkpoint = torch.load("/workspace/model.pt", map_location="cpu")
model = GFScreen_RL(args)
model.load_state_dict(checkpoint["state_dict"], strict=True)
print("Checkpoint architecture verified:", sum(p.numel() for p in model.parameters()))
PY

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    MPLCONFIGDIR=/tmp/matplotlib

CMD ["/bin/bash"]

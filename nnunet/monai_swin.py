"""MONAI SwinUNETR adapter for nnU-Net plans and official SSL weights."""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import SwinUNETR


class SwinUNETRForNNUNet(SwinUNETR):
    """Expose nnU-Net's constructor names and MONAI SSL encoder loading."""

    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        deep_supervision: bool = False,
        **kwargs,
    ) -> None:
        if deep_supervision:
            raise ValueError("SwinUNETRForNNUNet does not implement deep supervision")
        super().__init__(
            in_channels=input_channels,
            out_channels=num_classes,
            **kwargs,
        )

    def load_pretrained_encoder(self, checkpoint_path: str | Path) -> int:
        """Load MONAI's official 5,000-epoch self-supervised CT encoder.

        The released checkpoint uses ``swinViT.*`` and ``mlp.linear*`` names,
        while MONAI 1.6's ``load_from`` compatibility loader expects
        ``module.*`` and ``mlp.fc*``. Only names are translated; tensors are
        passed unchanged and MONAI performs shape-checked copies.
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        source = checkpoint.get("state_dict", checkpoint)
        converted = {}
        for name, tensor in source.items():
            if name.startswith("swinViT."):
                name = "module." + name[len("swinViT.") :]
            name = name.replace(".mlp.linear1.", ".mlp.fc1.")
            name = name.replace(".mlp.linear2.", ".mlp.fc2.")
            converted[name] = tensor
        self.load_from({"state_dict": converted})
        return len(converted)

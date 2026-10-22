"""Conditional 0.20/0.80 Tversky branch with the audited source sampler."""

from __future__ import annotations

import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

from STUNetTrainerPanCancer import PartialBoxTverskyCELoss, STUNetTrainerPanCancer
from STUNetTrainerPanCancerSourceBalanced import (
    STUNetTrainerPanCancerComponentBoxE160SourceStratified,
)


class STUNetTrainerPanCancerComponentBoxE160SourceAggressive(
    STUNetTrainerPanCancerComponentBoxE160SourceStratified
):
    """Stronger FN pressure; launch only after the public diagnostic qualifies."""

    def _build_loss(self):
        loss = PartialBoxTverskyCELoss(alpha=0.20, beta=0.80)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = np.asarray([1.0 / (2**index) for index in range(len(scales))])
            weights[-1] = 1e-6 if self.is_ddp else 0.0
            weights /= weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_start(self):
        # Do not attach this conditional branch to either completed control.
        STUNetTrainerPanCancer.on_train_start(self)
        if self.local_rank == 0:
            self._initialize_wandb(
                name="stunetb_componentbox_e160_source_a020b080_80ep_20260826",
                run_id="stunetbcompe160sourcea020b08020260826",
                partial_label_policy="ignore_outside_union_of_component_bboxes",
                loss_name="trusted_component_box_CE_plus_Tversky_a020_b080",
                extra_config={
                    "parent_epoch": 160,
                    "parent_checkpoint_sha256": self.parent_checkpoint_sha256,
                    "optimizer_restart": True,
                    "optimizer": "SGD_Nesterov_momentum_0.99",
                    "lr_schedule": "polynomial_80_epochs",
                    "case_sampling": "size_preserving_bounded_source_mixture",
                    "source_fraction": self.source_fraction,
                    "raw_per_case_ratio_cap": self.raw_ratio_cap,
                    "patch_center_sampling": (
                        "75% foreground-voxel uniform plus 25% annotated-component uniform"
                    ),
                    "component_center_fraction": self.component_center_fraction,
                    "expected_training_cases": self.expected_case_count,
                    "expected_sources": self.expected_source_count,
                    "expected_identifier_sha256": self.expected_identifier_sha256,
                    "expected_size_bin_counts": self.expected_size_bin_counts,
                    "launch_gate": "QUALIFY_TVERSKY_A020_B080_FOLLOWUP",
                },
            )

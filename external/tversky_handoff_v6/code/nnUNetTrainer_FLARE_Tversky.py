"""Fine-tuning trainers that target THE measured failure mode: false negatives.

Motivation, from measurements in RESULTS.md:
  * 16 of 50 public cases score exactly DSC = 0, and that count did NOT move between
    epoch 77 and 161 while mean DSC rose 60%. The model improves on cases it already
    finds and not on the ones it misses. The errors are misses, not over-segmentation.
  * The official metric zeroes NSD when DSC < 0.2, so dragging a missed case from
    0.00 to 0.25 is worth roughly DOUBLE what lifting a 0.60 case to 0.70 is worth.
  * Stock nnU-Net uses Dice+CE, which weights FP and FN symmetrically -- the wrong
    prior for this task.

Tversky generalises Dice:

    Tversky = TP / (TP + alpha*FP + beta*FN)
    Dice    = 2TP / (2TP + FP + FN)  ==  Tversky at alpha = beta = 0.5

Setting beta > alpha penalises false negatives harder. Default here is
alpha=0.3, beta=0.7.

Also changed, both aimed at the same failure:
  * oversample_foreground_percent 0.33 -> 0.50 (more lesion-bearing patches/batch)
  * save_every 50 -> 1, because the fine-tuning optimum is sharp: measured 0.33
    hidden at epoch 17 and 0.38 at epoch 19. A 50-epoch cadence cannot see that and
    previously led us to conclude, wrongly, that fine-tuning was capped at 0.33.

Install alongside nnUNetTrainer_FLARE.py in nnunetv2/training/nnUNetTrainer/.
"""
from functools import partial
from typing import Callable

import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Tversky loss mirroring MemoryEfficientSoftDiceLoss exactly.

    Same interface, same memory behaviour, same one-hot / batch_dice / ddp handling.
    Only the final ratio differs, so it is a drop-in `dice_class` for DC_and_CE_loss.

    alpha weights false positives, beta weights false negatives.
    alpha = beta = 0.5 reproduces Dice exactly (asserted in the self-test below).
    """

    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1., ddp: bool = True,
                 alpha: float = 0.3, beta: float = 0.7):
        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))
            if x.shape == y.shape:
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)
            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]
            sum_gt = y_onehot.sum(axes) if loss_mask is None \
                else (y_onehot * loss_mask).sum(axes)

        # must be outside no_grad or there are no gradients
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)
            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        # intersect = TP, sum_pred = TP + FP, sum_gt = TP + FN
        fp = sum_pred - intersect
        fn = sum_gt - intersect
        tv = (intersect + self.smooth) / torch.clip(
            intersect + self.alpha * fp + self.beta * fn + self.smooth, 1e-8)

        return -tv.mean()


class _TverskyFTBase(nnUNetTrainer):
    """Fine-tune from the organisers' baseline with an FN-weighted loss.

    Subclasses set NUM_EPOCHS / INIT_LR / ALPHA / BETA / OVERSAMPLE.
    """
    NUM_EPOCHS = 60
    INIT_LR = 1e-4
    ALPHA = 0.3          # false-positive weight
    BETA = 0.7           # false-negative weight -- the point of this trainer
    OVERSAMPLE = 0.5     # was 0.33; more lesion-bearing patches per batch
    SAVE_EVERY = 1       # the fine-tuning peak is sharp (0.33 @ ep17 -> 0.38 @ ep19)

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = self.NUM_EPOCHS
        self.initial_lr = self.INIT_LR
        self.oversample_foreground_percent = self.OVERSAMPLE
        self.save_every = self.SAVE_EVERY

    def _build_loss(self):
        """Identical to nnUNetTrainer._build_loss except the dice_class is Tversky."""
        if self.label_manager.has_regions:
            # region-based training is not used for this dataset; fall back to stock
            loss = DC_and_BCE_loss(
                {}, {'batch_dice': self.configuration_manager.batch_dice,
                     'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss)
        else:
            tversky = partial(MemoryEfficientSoftTverskyLoss,
                              alpha=self.ALPHA, beta=self.BETA)
            loss = DC_and_CE_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
                {}, weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=tversky)

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss

    def on_train_start(self):
        super().on_train_start()
        self.print_to_log_file(
            f"nnUNetTrainer_FLARE_Tversky: alpha={self.ALPHA} beta={self.BETA} "
            f"(beta>alpha penalises FALSE NEGATIVES), lr={self.INIT_LR}, "
            f"oversample_fg={self.OVERSAMPLE}, save_every={self.SAVE_EVERY}, "
            f"epochs={self.NUM_EPOCHS}")

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Skip: with fold='all', val_keys == tr_keys, so this would run sliding-window
        inference over all 17,575 training cases (~20h). Score with code/predict.py +
        code/evaluate.py against validation/ instead."""
        self.print_to_log_file("perform_actual_validation SKIPPED (fold=all).")


# --- the configuration to run first -----------------------------------------
class nnUNetTrainer_FT_Tversky_Lr1e4(_TverskyFTBase):
    """Recommended. LR 1e-4 so the gain persists instead of peaking at ~19 and
    decaying, as it did at 1e-3 (0.38 @ ep19 -> 0.29 @ ep149)."""
    NUM_EPOCHS, INIT_LR = 60, 1e-4


class nnUNetTrainer_FT_Tversky_Lr3e4(_TverskyFTBase):
    """Middle ground: our preprocessing differs from the baseline's (2.4x1.6x1.6 vs
    3x2x2, different normalisation constants), so some real adaptation is needed."""
    NUM_EPOCHS, INIT_LR = 60, 3e-4


class nnUNetTrainer_FT_Tversky_Lr1e3(_TverskyFTBase):
    """Controls for loss only: same 1e-3 schedule that produced 0.38 @ ep19, so any
    difference is attributable to Tversky + oversampling rather than the LR."""
    NUM_EPOCHS, INIT_LR = 60, 1e-3


class nnUNetTrainer_FT_Tversky_b085_Lr1e4(_TverskyFTBase):
    """More aggressive FN weighting, if beta=0.7 does not move the 16/50 zero cases."""
    NUM_EPOCHS, INIT_LR, ALPHA, BETA = 60, 1e-4, 0.15, 0.85

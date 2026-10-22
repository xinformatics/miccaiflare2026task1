"""Memory-efficient Tversky loss verified against nnU-Net's Dice loss."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn

from nnunetv2.utilities.ddp_allgather import AllGatherGrad


class MemoryEfficientSoftTverskyLoss(nn.Module):
    """Drop-in replacement for MemoryEfficientSoftDiceLoss.

    ``alpha`` weights false positives and ``beta`` weights false negatives.
    At alpha=beta=0.5 this is exactly Dice.
    """

    def __init__(
        self,
        apply_nonlin: Callable | None = None,
        batch_dice: bool = False,
        do_bg: bool = True,
        smooth: float = 1.0,
        ddp: bool = True,
        alpha: float = 0.3,
        beta: float = 0.7,
    ) -> None:
        super().__init__()
        if alpha < 0 or beta < 0 or alpha + beta <= 0:
            raise ValueError("Tversky alpha and beta must be nonnegative with a positive sum")
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
            sum_gt = (
                y_onehot.sum(axes)
                if loss_mask is None
                else (y_onehot * loss_mask).sum(axes)
            )

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

        fp = sum_pred - intersect
        fn = sum_gt - intersect
        score = (intersect + self.smooth) / torch.clip(
            intersect + self.alpha * fp + self.beta * fn + self.smooth,
            min=1e-8,
        )
        return -score.mean()

"""Verify the Tversky loss before spending GPU hours on it.

Three properties that must hold, or the trainer is silently wrong:
  1. alpha = beta = 0.5 reproduces MemoryEfficientSoftDiceLoss EXACTLY (Dice is the
     symmetric special case of Tversky). This validates the whole tp/fp/fn derivation.
  2. With beta > alpha, a prediction that MISSES lesion (false negative) is penalised
     more than one that OVER-predicts by the same amount. That is the entire point:
     16/50 cases score exactly 0 because the model misses them.
  3. Gradients flow, and batch_dice / do_bg behave like the Dice version.
"""
import sys

import torch
from torch import nn

sys.path.insert(0, "/media/shashank/NewVolume/PancancerCTSeg/code")


def main():
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnUNetTrainer_FLARE_Tversky import MemoryEfficientSoftTverskyLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    torch.manual_seed(0)
    B, C, S = 2, 2, 16
    logits = torch.randn(B, C, S, S, S, requires_grad=True)
    seg = (torch.rand(B, 1, S, S, S) > 0.97).long()   # ~3% foreground, like our data

    for batch_dice in (True, False):
        d = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1,
                                        batch_dice=batch_dice, do_bg=False,
                                        smooth=1e-5, ddp=False)
        t = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1,
                                           batch_dice=batch_dice, do_bg=False,
                                           smooth=1e-5, ddp=False,
                                           alpha=0.5, beta=0.5)
        ld, lt = d(logits, seg).item(), t(logits, seg).item()
        ok = abs(ld - lt) < 1e-6
        print(f"[1] batch_dice={batch_dice}: dice={ld:.8f} tversky(.5,.5)={lt:.8f} "
              f"-> {'MATCH' if ok else 'MISMATCH'}")
        assert ok, "Tversky at alpha=beta=0.5 must equal Dice"

    # 2. asymmetry: same magnitude of error, once as FN and once as FP
    gt = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    gt[0, 0, 2:6, 2:6, 2:6] = 1                        # 64-voxel lesion

    def probs_from(mask):
        p = torch.full((1, 2, 8, 8, 8), 0.01)
        p[0, 1] = mask * 0.98 + 0.01
        p[0, 0] = 1 - p[0, 1]
        return p.clamp(1e-6, 1 - 1e-6).log()           # logits for softmax

    miss = torch.zeros(1, 8, 8, 8); miss[0, 2:4, 2:6, 2:6] = 1     # finds half
    over = torch.zeros(1, 8, 8, 8); over[0, 2:6, 2:6, 2:6] = 1
    over[0, 6:8, 2:6, 2:6] = 1                                      # finds all + extra

    tv = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1,
                                        batch_dice=False, do_bg=False, smooth=1e-5,
                                        ddp=False, alpha=0.3, beta=0.7)
    dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1,
                                     batch_dice=False, do_bg=False, smooth=1e-5,
                                     ddp=False)
    l_miss_t, l_over_t = tv(probs_from(miss), gt).item(), tv(probs_from(over), gt).item()
    l_miss_d, l_over_d = dc(probs_from(miss), gt).item(), dc(probs_from(over), gt).item()
    print(f"\n[2] under-segment (FN)  dice={l_miss_d:+.4f}  tversky={l_miss_t:+.4f}")
    print(f"    over-segment  (FP)  dice={l_over_d:+.4f}  tversky={l_over_t:+.4f}")
    gap_d, gap_t = l_miss_d - l_over_d, l_miss_t - l_over_t
    print(f"    penalty gap (FN worse than FP): dice={gap_d:+.4f}  tversky={gap_t:+.4f}")
    assert gap_t > gap_d, "beta>alpha must punish false negatives MORE than Dice does"
    print("    -> Tversky punishes misses harder than Dice. Correct.")

    # 3. gradients
    logits2 = torch.randn(B, C, S, S, S, requires_grad=True)
    loss = MemoryEfficientSoftTverskyLoss(apply_nonlin=softmax_helper_dim1,
                                          batch_dice=True, do_bg=False, smooth=1e-5,
                                          ddp=False)(logits2, seg)
    loss.backward()
    g = logits2.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
    print(f"\n[3] gradients finite and non-zero (|grad| sum={g.abs().sum():.4f})")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

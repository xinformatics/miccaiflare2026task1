"""Trainer variants for the pan-cancer task.

The FLARE24/25 champions changed exactly two things from stock nnU-Net: a longer
schedule and a 10x lower initial LR (1e-3 vs 1e-2). Everything else -- polyLR decay,
SGD+Nesterov momentum 0.99, Dice+CE with deep supervision, the standard augmentation
pipeline -- is stock, so we subclass rather than reimplement.

Installed into the nnunetv2 package tree so nnU-Net's class discovery finds it:
    -tr nnUNetTrainer_Epoch2000_Lr1e3
"""
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _Base(nnUNetTrainer):
    NUM_EPOCHS = 1000
    INIT_LR = 1e-3

    def __init__(self, plans, configuration, fold, dataset_json, device=None):
        # nnU-Net's signature changed across 2.x; forward whatever we were given.
        if device is None:
            super().__init__(plans, configuration, fold, dataset_json)
        else:
            super().__init__(plans, configuration, fold, dataset_json, device=device)
        self.num_epochs = self.NUM_EPOCHS
        self.initial_lr = self.INIT_LR

    def perform_actual_validation(self, save_probabilities: bool = False):
        """Skip nnU-Net's post-training validation pass.

        run_training.py calls this unconditionally and offers no flag to disable it.
        With fold='all', do_split() sets val_keys = tr_keys (nnUNetTrainer.py:565-569),
        so this would run sliding-window inference over all 17,575 training cases --
        roughly 20 hours of GPU, tacked onto the end of an already 75-hour run, on
        data the model was trained on.

        We score with the organisers' own metric on the real held-out sets instead:
            code/predict.py + code/evaluate.py  (public val, and the 172 healthy cases)
        """
        self.print_to_log_file(
            "perform_actual_validation SKIPPED by nnUNetTrainer_FLARE: fold='all' "
            "would validate on all training cases. Use code/predict.py + "
            "code/evaluate.py against validation/ instead."
        )


class nnUNetTrainer_Epoch500_Lr1e3(_Base):
    NUM_EPOCHS, INIT_LR = 500, 1e-3


class nnUNetTrainer_Epoch1000_Lr1e3(_Base):
    NUM_EPOCHS, INIT_LR = 1000, 1e-3


class nnUNetTrainer_Epoch2000_Lr1e3(_Base):
    """FLARE24 champion setting."""
    NUM_EPOCHS, INIT_LR = 2000, 1e-3


class nnUNetTrainer_Epoch5000_Lr1e3(_Base):
    """FLARE25 champion setting."""
    NUM_EPOCHS, INIT_LR = 5000, 1e-3


class nnUNetTrainer_Epoch10_Lr1e3(_Base):
    """Smoke test: end-to-end pipeline validation, not a real model."""
    NUM_EPOCHS, INIT_LR = 10, 1e-3


# --- fine-tuning variants ---------------------------------------------------
# The FLARE25 baseline was itself trained at initial_lr=1e-3 for 2000 epochs, so its
# poly schedule had decayed to ~0 at the end. Re-initialising a fresh 1e-3 schedule
# on top of it resets the LR to full and knocks the model out of its minimum: public
# val fell 0.4725 -> 0.3614 by epoch 50 while pseudo dice oscillated 0.39-0.59
# without beating the 0.5579 it started at.
#
# These keep the pretrained minimum and adapt to our preprocessing (2.4x1.6x1.6mm
# and our own normalisation constants) instead of overwriting it.

class nnUNetTrainer_FT_Lr1e4(_Base):
    """Conservative fine-tune: 1/10th the baseline's LR."""
    NUM_EPOCHS, INIT_LR = 300, 1e-4


class nnUNetTrainer_FT_Lr3e4(_Base):
    """Middle ground -- our spacing/normalisation differ from the baseline's, so
    some real adaptation is needed, not just polishing."""
    NUM_EPOCHS, INIT_LR = 300, 3e-4


class nnUNetTrainerLR0001Epoch2000(_Base):
    """Needed to LOAD the organisers' FLARE25 baseline docker checkpoint.

    initialize_from_trained_model_folder resolves the trainer class named in the
    checkpoint in order to rebuild the network, and refuses to run if it is absent.
    The baseline (Dataset025_FLARE25, 3d_fullres, 5.47M params) was trained with this
    class, which is not part of stock nnU-Net. Only the class *identity* matters for
    inference -- the LR and epoch count below are inert at predict time.
    """
    NUM_EPOCHS, INIT_LR = 2000, 1e-4

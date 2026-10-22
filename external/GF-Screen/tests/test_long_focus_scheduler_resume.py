import copy
import unittest

import torch

from utils.optimizers import LinearWarmupCosineAnnealingLR


class LongFocusSchedulerResumeTest(unittest.TestCase):
    steps_per_epoch = 1024
    total_steps = 30 * steps_per_epoch
    warmup_steps = 3 * steps_per_epoch
    peak_lr = 3e-5

    def make(self, max_steps: int):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=self.peak_lr)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_steps,
            max_epochs=max_steps,
        )
        return parameter, optimizer, scheduler

    @staticmethod
    def step(parameter, optimizer, scheduler):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        scheduler.step()

    def test_exact_resume_matches_uninterrupted_schedule(self):
        parameter, optimizer, scheduler = self.make(self.total_steps)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        interrupt_step = 10 * self.steps_per_epoch
        for _ in range(interrupt_step):
            self.step(parameter, optimizer, scheduler)

        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        saved_scheduler = copy.deepcopy(scheduler.state_dict())
        saved_lr = scheduler.get_last_lr()[0]

        resumed_parameter, resumed_optimizer, resumed_scheduler = self.make(
            self.total_steps - interrupt_step
        )
        resumed_optimizer.load_state_dict(saved_optimizer)
        resumed_scheduler.load_state_dict(saved_scheduler)
        for group, lr in zip(
            resumed_optimizer.param_groups, resumed_scheduler.get_last_lr()
        ):
            group["lr"] = lr

        self.assertEqual(resumed_scheduler.max_epochs, self.total_steps)
        self.assertAlmostEqual(resumed_optimizer.param_groups[0]["lr"], saved_lr)
        checkpoints = {15 * self.steps_per_epoch, self.total_steps}
        for step in range(interrupt_step + 1, self.total_steps + 1):
            self.step(parameter, optimizer, scheduler)
            self.step(resumed_parameter, resumed_optimizer, resumed_scheduler)
            if step in checkpoints:
                self.assertAlmostEqual(
                    resumed_optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[0]["lr"],
                    places=14,
                )

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0, places=14)


if __name__ == "__main__":
    unittest.main()

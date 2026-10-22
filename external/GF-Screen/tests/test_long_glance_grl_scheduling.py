import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from monai import data

from models.GF_Screen_RL import GFScreen_RL, masked_group_relative_advantages
from utils.data_utils import Sampler


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (ROOT / "run_long_glance_grl_after_focus_20260825.sh").read_text(encoding="utf-8")
TRAJECTORY = (ROOT / "evaluate_long_glance_grl_trajectory_public_20260825.sh").read_text(encoding="utf-8")
SELECTOR = (ROOT / "select_long_glance_grl_trajectory.py").read_text(encoding="utf-8")
CHILD = (ROOT / "evaluate_long_focus_pu_champion_public_20260825.sh").read_text(encoding="utf-8")
MODEL = (ROOT / "models/GF_Screen_RL.py").read_text(encoding="utf-8")


class _ZeroOldPolicy:
    def get_actions(self, crops):
        return torch.zeros((crops.shape[0], 2), dtype=crops.dtype)


def _distributed_grl_worker(rank, rendezvous, output_dir):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        model = GFScreen_RL.__new__(GFScreen_RL)
        torch.nn.Module.__init__(model)
        model.args = SimpleNamespace(
            glance_teacher_labels=True,
            global_group_advantages=True,
            glance_pseudo_positive_threshold=0.98,
            glance_pseudo_negative_threshold=0.02,
            glance_pseudo_positive_min_voxels=64,
        )
        actions = torch.zeros((8, 2), requires_grad=True)
        crops = torch.zeros((8, 1, 2, 2, 2))
        labels = torch.ones((8, 1, 2, 2, 2))
        if rank == 0:
            labels[1:] = 0.0
        segmentation = torch.zeros((8, 2, 2, 2, 2))
        segmentation[:, 1 if rank == 1 else 0] = 1.0
        loss = model.get_RL_loss(
            actions,
            _ZeroOldPolicy(),
            crops,
            segmentation,
            labels,
        )
        Path(output_dir, f"rank{rank}.json").write_text(
            json.dumps({"loss": float(loss.detach())}), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


class LongGlanceGRLSchedulingTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("RUN_DISTRIBUTED_GLOO_TEST") == "1",
        "requires a host namespace that permits local Gloo sockets",
    )
    def test_real_two_rank_teacher_safe_all_gather_uses_global_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            rendezvous = Path(directory) / "rendezvous"
            mp.spawn(
                _distributed_grl_worker,
                args=(str(rendezvous), directory),
                nprocs=2,
                join=True,
            )
            losses = [
                json.loads(
                    (Path(directory) / f"rank{rank}.json").read_text(encoding="utf-8")
                )["loss"]
                for rank in range(2)
            ]
        self.assertGreater(losses[0], 0.5)
        self.assertLess(losses[1], -0.5)
        self.assertAlmostEqual(losses[0], -losses[1], places=5)

    def test_runtime_batch_and_sampler_contract_is_global_n16(self):
        class TwoCropDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 16

            def __getitem__(self, index):
                return [
                    {
                        "image": torch.full((1, 2, 2, 2), float(index)),
                        "label": torch.zeros((1, 2, 2, 2)),
                    }
                    for _ in range(2)
                ]

        loader = data.DataLoader(TwoCropDataset(), batch_size=4, num_workers=0)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["image"].shape), (8, 1, 2, 2, 2))

        dataset = list(range(20_000))
        rank0 = Sampler(
            dataset, num_replicas=2, rank=0, num_samples_total=8192
        )
        rank1 = Sampler(
            dataset, num_replicas=2, rank=1, num_samples_total=8192
        )
        self.assertEqual(len(rank0), 4096)
        self.assertEqual(len(rank1), 4096)
        left, right = list(rank0), list(rank1)
        self.assertEqual(len(left) + len(right), 8192)
        self.assertFalse(set(left) & set(right))

        rewards = torch.tensor([0.0] * 8 + [1.0] * 8)
        advantages = masked_group_relative_advantages(
            rewards, torch.ones(16, dtype=torch.bool)
        )
        self.assertTrue(torch.all(advantages[:8] < 0))
        self.assertTrue(torch.all(advantages[8:] > 0))

    def test_is_one_long_paper_aligned_grl_run(self):
        for token in (
            "--distributed",
            "--freeze_focus",
            "--global_group_advantages",
            "--glance_classification_weight 0.1",
            "--glance_rl_weight 1.0",
            "--samples_per_epoch 8192",
            "--batch_size 4",
            "--sw_batch_size 2",
            "max_epochs=$((source_epoch + 30))",
        ):
            self.assertIn(token, CONTROLLER)
        self.assertNotIn("train_arm", CONTROLLER)
        self.assertIn("torch.distributed.all_gather", MODEL)
        self.assertIn("stored_epoch+1", CONTROLLER)
        self.assertIn("source checkpoint filename/stored epoch mismatch", CONTROLLER)

    def test_uses_all_partial_and_positive_only_contracts(self):
        self.assertIn("all_partial_primary_lesion_only", CONTROLLER)
        self.assertIn("--pseudo_abdomen_json", CONTROLLER)
        self.assertIn("jsons/local/no_proxy_healthy.json", CONTROLLER)
        self.assertIn("--glance_pseudo_negative_threshold 0.02", CONTROLLER)
        self.assertNotIn("--complete_label_cohorts_json", CONTROLLER)
        self.assertNotIn("verified7", CONTROLLER)
        self.assertNotIn("707", CONTROLLER)

    def test_waits_for_focus_prediction_consumers_before_gpu_use(self):
        self.assertIn("focus_encoder_notta_saved_20260825/TERMINAL.json", CONTROLLER)
        self.assertIn("gf_focus_pu_long30_trajectory_saved_20260825/TERMINAL.json", CONTROLLER)
        self.assertLess(CONTROLLER.index('for marker in'), CONTROLLER.index('CUDA_VISIBLE_DEVICES=0,1'))

    def test_failed_upstream_terminal_is_not_consumed_as_final(self):
        self.assertIn('COMPLETE|SKIPPED) break', CONTROLLER)
        self.assertIn('waiting for replacement of failed upstream terminal', CONTROLLER)
        self.assertIn("trap 'failure_line=${LINENO}' ERR", CONTROLLER)

    def test_own_failed_terminal_is_preserved_for_exact_resume(self):
        self.assertIn('COMPLETE|SKIPPED) trap - EXIT; exit 0', CONTROLLER)
        self.assertIn('TERMINAL.failed_retry_', CONTROLLER)
        self.assertIn('preserved failed long Glance terminal for exact resume', CONTROLLER)
        self.assertIn('>> "${run_dir}/train.log" 2>&1', CONTROLLER)

    def test_public_trajectory_retries_failed_terminal_only(self):
        self.assertIn('COMPLETE) exit 0', TRAJECTORY)
        self.assertIn('FAILED)', TRAJECTORY)
        self.assertIn('TERMINAL.failed_retry_', TRAJECTORY)
        self.assertNotIn('[[ -s "${terminal}" ]] && exit 0', TRAJECTORY)

    def test_fixed_public_grid_is_predeclared_and_save_free(self):
        self.assertIn("thresholds=(0.10 0.20 0.35)", TRAJECTORY)
        self.assertIn("OFFSETS = (10, 20, 30)", SELECTOR)
        self.assertIn("source_epoch + offset for offset in OFFSETS", SELECTOR)
        self.assertIn("LONG_FOCUS_GLANCE_THRESHOLD", TRAJECTORY)
        self.assertIn("LONG_FOCUS_GLANCE_THRESHOLD", CHILD)
        lowered = (CONTROLLER + TRAJECTORY + SELECTOR).lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("package_codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertIn('"submission_created":false', lowered)

    def test_gate_matches_operating_instructions(self):
        self.assertIn("0.6154316379500749", SELECTOR)
        self.assertIn("0.5104225107769019", SELECTOR)
        self.assertIn('row["gate"] == "PROMOTE"', SELECTOR)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import sys
import types
import unittest
from contextlib import redirect_stderr
from io import StringIO

from fault_tolerant_wandb import (
    safe_wandb_define_metrics,
    safe_wandb_finish,
    safe_wandb_init,
    safe_wandb_log,
    stable_wandb_run_id,
)


class BrokenRun:
    def __init__(self) -> None:
        self.log_calls = 0
        self.finish_calls = 0

    def log(self, record, step):
        del record, step
        self.log_calls += 1
        raise ConnectionResetError("connection lost")

    def finish(self):
        self.finish_calls += 1
        raise ConnectionResetError("connection lost")


class RecordingRun:
    def __init__(self) -> None:
        self.metrics = []
        self.logs = []

    def define_metric(self, name, **kwargs):
        self.metrics.append((name, kwargs))

    def log(self, record, step, **kwargs):
        self.logs.append((record, step, kwargs))


class FaultTolerantWandbTests(unittest.TestCase):
    def test_log_failure_is_nonfatal_and_disables_future_logging(self):
        run = BrokenRun()
        stderr = StringIO()
        with redirect_stderr(stderr):
            run = safe_wandb_log(run, {"loss": 1.0}, step=1)
        self.assertIsNone(run)
        self.assertIn("continuing training without W&B", stderr.getvalue())

    def test_finish_failure_is_nonfatal(self):
        run = BrokenRun()
        stderr = StringIO()
        with redirect_stderr(stderr):
            safe_wandb_finish(run)
        self.assertEqual(run.finish_calls, 1)
        self.assertIn("W&B finish failed", stderr.getvalue())

    def test_metric_definition_and_commit_options_are_preserved(self):
        run = RecordingRun()
        self.assertIs(
            safe_wandb_define_metrics(
                run, [("epoch", {}), ("train/*", {"step_metric": "epoch"})]
            ),
            run,
        )
        self.assertIs(
            safe_wandb_log(run, {"loss": 1.0}, step=3, commit=False), run
        )
        self.assertEqual(run.metrics[1], ("train/*", {"step_metric": "epoch"}))
        self.assertEqual(run.logs[0][1:], (3, {"commit": False}))

    def test_stable_id_is_resume_safe_and_run_specific(self):
        first = stable_wandb_run_id("/tmp/run-a", "experiment")
        self.assertEqual(first, stable_wandb_run_id("/tmp/run-a", "experiment"))
        self.assertNotEqual(first, stable_wandb_run_id("/tmp/run-b", "experiment"))
        self.assertEqual(len(first), 16)

    def test_init_failure_is_nonfatal(self):
        module = types.SimpleNamespace(
            init=lambda **kwargs: (_ for _ in ()).throw(
                ConnectionResetError("offline")
            )
        )
        previous = sys.modules.get("wandb")
        sys.modules["wandb"] = module
        stderr = StringIO()
        try:
            with redirect_stderr(stderr):
                self.assertIsNone(safe_wandb_init(project="test"))
        finally:
            if previous is None:
                sys.modules.pop("wandb", None)
            else:
                sys.modules["wandb"] = previous
        self.assertIn("W&B initialization failed", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

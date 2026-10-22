"""Stream nnU-Net training metrics to Weights & Biases.

Deliberately a LOG TAILER, not a trainer subclass. Patching nnUNetTrainer to call
wandb would mean restarting an in-flight 32-hour run and would put third-party
network I/O inside the training loop, where a wandb outage or auth hiccup could kill
the job. This process can crash, be restarted, or be killed with zero effect on
training.

Scrapes nnU-Net's own training_log_*.txt, which emits per epoch:
    Epoch <n> / Current learning rate / train_loss / val_loss / Pseudo dice / Epoch time
and joins in the real public-validation scores from code/val_curve.csv (written by
val_watcher.py), which are the numbers that actually matter -- pseudo dice is
measured on training data when fold=all and reads optimistically.

Resumable: uses a fixed run id and replays the whole log on start, so restarts
backfill rather than duplicate or lose history.

  python code/wandb_export.py --project pancancer-ct --name resenc-s6
"""
import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")

RE_EPOCH = re.compile(r":\s*Epoch (\d+)\s*$")
RE_LR = re.compile(r"Current learning rate:\s*([0-9.eE+-]+)")
RE_TRAIN = re.compile(r"train_loss\s*([-0-9.eE+]+)")
RE_VAL = re.compile(r"val_loss\s*([-0-9.eE+]+)")
RE_DICE = re.compile(r"Pseudo dice\s*\[([^\]]*)\]")
RE_TIME = re.compile(r"Epoch time:\s*([0-9.]+)")


def parse_log(path):
    """Yield one dict per completed epoch, in order."""
    cur = {}
    epoch = None
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_EPOCH.search(line)
            if m:
                epoch = int(m.group(1))
                cur = {"epoch": epoch}
                continue
            if epoch is None:
                continue
            if (m := RE_LR.search(line)):
                cur["lr"] = float(m.group(1))
            elif (m := RE_TRAIN.search(line)):
                cur["train_loss"] = float(m.group(1))
            elif (m := RE_VAL.search(line)):
                cur["val_loss"] = float(m.group(1))
            elif (m := RE_DICE.search(line)):
                vals = [float(x) for x in m.group(1).replace(",", " ").split() if x]
                if vals:
                    cur["pseudo_dice"] = sum(vals) / len(vals)
            elif (m := RE_TIME.search(line)):
                cur["epoch_time_s"] = float(m.group(1))
                yield dict(cur)          # Epoch time is emitted last -> epoch complete


def read_val_curve(path):
    """{epoch: {public_dsc, public_nsd, healthy_clean_frac}} from val_watcher.py."""
    out = {}
    if not Path(path).is_file():
        return out
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                e = int(r["epoch"])
            except (ValueError, KeyError, TypeError):
                continue
            d = {}
            for src, dst in (("dsc", "public_dsc"), ("nsd", "public_nsd")):
                if r.get(src):
                    d[dst] = float(r[src])
            if r.get("healthy_clean") and r.get("healthy_n"):
                try:
                    d["healthy_clean_frac"] = int(r["healthy_clean"]) / int(r["healthy_n"])
                except ValueError:
                    pass
            if d:
                out[e] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="pancancer-ct")
    ap.add_argument("--name", default="resenc-s6")
    ap.add_argument("--run-id", default=None, help="fixed id so restarts resume")
    ap.add_argument("--logdir", default=None, help="fold_all dir with training_log_*.txt")
    ap.add_argument("--val-curve", default=str(PROJ / "code/val_curve.csv"))
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--config", default=None, help="config name for wandb metadata")
    a = ap.parse_args()

    import wandb

    logdir = Path(a.logdir) if a.logdir else None
    if logdir is None:
        sys.exit("--logdir is required (the fold_all directory)")

    run_id = a.run_id or a.name
    wandb.init(project=a.project, name=a.name, id=run_id, resume="allow",
               config={"config": a.config or "", "logdir": str(logdir)})
    # epoch is our x-axis, not wandb's internal step counter
    wandb.define_metric("epoch")
    for k in ("train_loss", "val_loss", "pseudo_dice", "lr", "epoch_time_s",
              "public_dsc", "public_nsd", "healthy_clean_frac"):
        wandb.define_metric(k, step_metric="epoch")

    print(f"wandb run: {wandb.run.url}", flush=True)
    sent = set()
    while True:
        logs = sorted(logdir.glob("training_log_*.txt"))
        vc = read_val_curve(a.val_curve)
        rows = {}
        for lg in logs:                      # later logs win: restarts overwrite
            for r in parse_log(lg):
                rows[r["epoch"]] = r
        for e in sorted(rows):
            payload = dict(rows[e])
            payload.update(vc.get(e, {}))
            key = (e, tuple(sorted(payload.items())))
            if key in sent:
                continue
            sent.add(key)
            wandb.log(payload)
        # public-val scores can land for epochs already sent; re-emit those
        for e, d in vc.items():
            key = (e, tuple(sorted({**rows.get(e, {"epoch": e}), **d}.items())))
            if key not in sent:
                sent.add(key)
                wandb.log({**rows.get(e, {"epoch": e}), **d})
        time.sleep(a.poll)


if __name__ == "__main__":
    main()

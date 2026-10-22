"""Track public-validation DSC/NSD across a training run, without slowing it down.

Validating every epoch would cost more than the training itself: sliding-window
inference over the 50 public cases takes ~215s against a ~150s epoch, turning a
3.5-day run into 8.4 days. nnU-Net checkpoints every 50 epochs anyway
(nnUNetTrainer.save_every = 50), so this watches for new checkpoints and scores
those -- ~3% overhead for 40 points across a 2000-epoch run.

Runs as a SEPARATE process from training. It snapshots (copies) the checkpoint
before reading it, because the trainer may overwrite checkpoint_latest.pth mid-read.
It shares the GPU with training, so each evaluation is somewhat slower than a
standalone run; that is accounted for in the overhead figure above.

Scores with the organisers' metric (code/evaluate.py -> SurfaceDice.py), so the
numbers are directly comparable to the leaderboard.

NOTE ON RULES: Information.md:30 forbids *training* on the public validation set.
This only measures it. Nothing here feeds back into the optimiser; use the curve to
monitor progress, and be aware that picking a final checkpoint by this number is a
mild form of selection on it.

  python code/val_watcher.py -tr nnUNetTrainer_Epoch2000_Lr1e3
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")


def checkpoint_epoch(path):
    """Epoch index stored in an nnU-Net checkpoint."""
    import torch
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return int(ck.get("current_epoch", -1))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", default="501")
    ap.add_argument("-c", "--config", default="3d_fullres_S4D2W32")
    ap.add_argument("-tr", "--trainer", default="nnUNetTrainer_Epoch2000_Lr1e3")
    ap.add_argument("-p", "--plans", default="nnUNetPlans")
    ap.add_argument("--fold", default="all")
    ap.add_argument("--poll", type=int, default=180)
    ap.add_argument("--healthy-every", type=int, default=4,
                    help="also score the 172 healthy cases every Nth evaluation "
                         "(0 disables). They cost ~740s, hence not every time.")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    results = Path(os.environ["nnUNet_results"])
    preds = Path(os.environ.get("PREDS", "/mnt/yadav/PancancerCTSeg/preds"))
    model_dir = results / f"Dataset{int(a.dataset):03d}_PanCancer" / \
        f"{a.trainer}__{a.plans}__{a.config}"
    ckpt = model_dir / f"fold_{a.fold}" / "checkpoint_latest.pth"
    snap = preds / "val_watch_snapshot"
    out_csv = Path(a.csv) if a.csv else PROJ / "code/val_curve.csv"

    if not out_csv.exists():
        with open(out_csv, "w", newline="") as f:
            csv.writer(f).writerow(
                ["epoch", "dsc", "nsd", "healthy_clean", "healthy_n", "timestamp"])

    print(f"watching  {ckpt}")
    print(f"curve     {out_csv}")
    print(f"poll      {a.poll}s\n", flush=True)

    seen_mtime, n_eval = 0.0, 0
    while True:
        if not ckpt.exists():
            time.sleep(a.poll)
            continue
        m = ckpt.stat().st_mtime
        if m <= seen_mtime:
            time.sleep(a.poll)
            continue
        time.sleep(20)                      # let the write settle
        seen_mtime = ckpt.stat().st_mtime

        # snapshot so training can overwrite the live checkpoint freely
        (snap / f"fold_{a.fold}").mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(ckpt, snap / f"fold_{a.fold}" / "checkpoint_latest.pth")
            for j in ("plans.json", "dataset.json"):
                shutil.copy2(model_dir / j, snap / j)
        except Exception as e:
            print(f"snapshot failed: {e}", flush=True)
            time.sleep(a.poll)
            continue

        ep = checkpoint_epoch(snap / f"fold_{a.fold}" / "checkpoint_latest.pth")
        n_eval += 1
        print(f"\n=== epoch {ep}: scoring public val ===", flush=True)
        t0 = time.time()

        pdir = preds / "val_watch"
        r = subprocess.run(
            [sys.executable, str(PROJ / "code/predict.py"),
             "-i", str(PROJ / "validation/Validation-Public-Images"),
             "-o", str(pdir), "--model-dir", str(snap), "-f", a.fold,
             "--checkpoint", "checkpoint_latest.pth"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print("predict failed:\n" + r.stdout[-1500:] + r.stderr[-1500:], flush=True)
            continue

        e = subprocess.run(
            [sys.executable, str(PROJ / "code/evaluate.py"), "--pred", str(pdir),
             "--csv", str(preds / f"metrics_epoch{ep}.csv")],
            capture_output=True, text=True)
        dsc = nsd = float("nan")
        for line in e.stdout.splitlines():
            if line.startswith("DSC  mean"):
                dsc = float(line.split()[2])
            if line.startswith("NSD  mean"):
                nsd = float(line.split()[2])

        clean = hn = ""
        if a.healthy_every and n_eval % a.healthy_every == 0:
            print(f"=== epoch {ep}: scoring 172 healthy cases ===", flush=True)
            hdir = preds / "val_watch_healthy"
            rh = subprocess.run(
                [sys.executable, str(PROJ / "code/predict.py"),
                 "-i", str(PROJ / "validation/HealthyImages-noLesion"),
                 "-o", str(hdir), "--model-dir", str(snap), "-f", a.fold,
                 "--checkpoint", "checkpoint_latest.pth"],
                capture_output=True, text=True)
            if rh.returncode == 0:
                eh = subprocess.run(
                    [sys.executable, str(PROJ / "code/evaluate.py"), "--pred",
                     str(hdir), "--healthy",
                     "--csv", str(preds / f"metrics_healthy_epoch{ep}.csv")],
                    capture_output=True, text=True)
                for line in eh.stdout.splitlines():
                    if "clean (zero FP)" in line:
                        frac = line.split(":")[1].split("->")[0].strip()
                        clean, hn = frac.split("/")

        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([ep, round(dsc, 4), round(nsd, 4), clean, hn,
                                    time.strftime("%F %T")])
        print(f"epoch {ep}: DSC {dsc:.4f}  NSD {nsd:.4f}"
              + (f"  healthy clean {clean}/{hn}" if clean else "")
              + f"   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()

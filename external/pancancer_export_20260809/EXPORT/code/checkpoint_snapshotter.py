"""Preserve a copy of every per-epoch checkpoint during a fine-tune.

`save_every = 1` makes nnU-Net write a checkpoint each epoch, but it always writes to
the SAME file (`checkpoint_latest.pth`), overwriting the previous one. The whole point
of the dense cadence is to find the peak — measured 0.33 at ep17 and 0.38 at ep19 —
which is impossible if only the newest epoch survives.

This watches the file and copies each new epoch to `epochs/checkpoint_ep0019.pth`.
At ~44MB for the 5.5M model, 60 epochs is ~2.6GB.

Runs as a separate process: if it dies, training is unaffected.
"""
import argparse
import os
import shutil
import time
from pathlib import Path


def epoch_of(path):
    import torch
    try:
        return int(torch.load(path, map_location="cpu",
                              weights_only=False).get("current_epoch", -1))
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="the fold_all directory")
    ap.add_argument("--out", default=None, help="default <model-dir>/epochs")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--max-epoch", type=int, default=10**6,
                    help="stop once this epoch has been captured")
    a = ap.parse_args()

    os.environ["nnUNet_compile"] = "0"
    src = Path(a.model_dir) / "checkpoint_latest.pth"
    out = Path(a.out) if a.out else Path(a.model_dir) / "epochs"
    out.mkdir(parents=True, exist_ok=True)
    print(f"watching {src}\nsaving   {out}", flush=True)

    seen_mtime, captured = 0.0, set()
    while True:
        if src.is_file():
            m = src.stat().st_mtime
            if m > seen_mtime:
                time.sleep(6)                      # let the write settle
                try:
                    m = src.stat().st_mtime
                    tmp = out / ".partial.pth"
                    shutil.copy2(src, tmp)         # copy first, then identify
                    ep = epoch_of(tmp)
                    if ep >= 0 and ep not in captured:
                        dst = out / f"checkpoint_ep{ep:04d}.pth"
                        tmp.replace(dst)
                        captured.add(ep)
                        seen_mtime = m
                        print(f"captured epoch {ep} -> {dst.name} "
                              f"({len(captured)} kept)", flush=True)
                        if ep >= a.max_epoch:
                            print("reached max epoch, exiting", flush=True)
                            return
                    else:
                        tmp.unlink(missing_ok=True)
                        seen_mtime = m
                except Exception as e:
                    print(f"snapshot failed: {type(e).__name__}: {e}", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()

"""Which (architecture, patch, spacing) buys the most accuracy inside the 4GB budget?

Efficiency is half the ranking weight, and GPU memory is scored as
max(0, peak - 4096MB). Both published top teams overshoot it:
    #16 leaderboard leader : DSC 59.0 / NSD 52.0 @ 14044 MB, 23-31 s
    #9  "winner" paper     : DSC 53.5 / NSD 43.8 @  6904 MB, ~18 s
So the play is to maximise accuracy subject to staying under 4096 MB, where we
score a perfect efficiency rank and they do not.

This sweeps candidate configs and reports peak inference VRAM + estimated GPU time
on the largest real validation geometries.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa

# (label, n_stages, features, patch (z,y,x), spacing (z,y,x))
CANDIDATES = [
    ("S4W32 @3.0/2.0/2.0 p96",   4, [32, 64, 128, 256],            [96, 128, 128],  [3.0, 2.0, 2.0]),
    ("S4W32 @2.4/1.6/1.6 p128",  4, [32, 64, 128, 256],            [128, 160, 160], [2.4, 1.6, 1.6]),
    ("S5W32 @2.4/1.6/1.6 p128",  5, [32, 64, 128, 256, 320],       [128, 160, 160], [2.4, 1.6, 1.6]),
    ("S6W32 @2.4/1.6/1.6 p128",  6, [32, 64, 128, 256, 320, 320],  [128, 160, 160], [2.4, 1.6, 1.6]),
    ("S5W32 @2.0/1.5/1.5 p128",  5, [32, 64, 128, 256, 320],       [128, 160, 160], [2.0, 1.5, 1.5]),
]

# the geometries that stress the budget hardest, from the real val set
CASES = [
    ("val_0083 512x512x361", (512, 512, 361), (0.75, 0.75, 1.25)),
    ("val_0099 512x512x334", (512, 512, 334), (0.879, 0.879, 1.5)),
    ("860mm z-FOV",          (512, 512, 688), (0.98, 0.98, 1.25)),
]


def build(n_stages, feats):
    ak = {
        "n_stages": n_stages, "features_per_stage": feats,
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * n_stages,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        "n_conv_per_stage": [2] * n_stages,
        "n_conv_per_stage_decoder": [2] * (n_stages - 1),
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
    }
    return get_network_from_plans(
        "dynamic_network_architectures.architectures.unet.PlainConvUNet", ak,
        ["conv_op", "norm_op", "dropout_op", "nonlin"], 1, 2, True)


def n_tiles(shape, patch, overlap=0.5):
    tot = 1
    for s, p in zip(shape, patch):
        tot *= 1 if s <= p else int(np.ceil((s - p) / (p * (1 - overlap)))) + 1
    return tot


def main():
    dev = torch.device("cuda")
    print(f"{'config':28s} {'params':>8s} {'trainVRAM':>10s} "
          f"{'infVRAM':>9s} {'maxTime':>8s}  verdict")
    print("-" * 82)

    for label, ns, feats, patch, spacing in CANDIDATES:
        net = build(ns, feats).to(dev)
        n_par = sum(p.numel() for p in net.parameters()) / 1e6

        # training peak
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda")
        try:
            x = torch.randn(2, 1, *patch, device=dev)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(x)
                    loss = sum(t.float().mean() for t in o) if isinstance(o, (list, tuple)) \
                        else o.float().mean()
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            train_vram = torch.cuda.max_memory_allocated() / 2**20
            del x, o, loss
        except torch.cuda.OutOfMemoryError:
            train_vram = float("inf")
        del opt, scaler
        net.eval(); torch.cuda.empty_cache()

        # inference peak + time over the worst geometry
        worst_vram, worst_time = 0.0, 0.0
        for _, shp_xyz, sp_xyz in CASES:
            shp = np.array([shp_xyz[2], shp_xyz[1], shp_xyz[0]], float)
            sp = np.array([sp_xyz[2], sp_xyz[1], sp_xyz[0]], float)
            new_shape = np.maximum(np.round(shp * sp / np.array(spacing)).astype(int), patch)
            tiles = n_tiles(new_shape, patch)

            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                logits = torch.zeros((2, *new_shape), dtype=torch.half, device=dev)
                g = torch.ones(patch, dtype=torch.half, device=dev)
                pt = torch.randn(1, 1, *patch, device=dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                sample = min(tiles, 20)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    for _ in range(sample):
                        o = net(pt)
                        o = o[0] if isinstance(o, (list, tuple)) else o
                        logits[:, :patch[0], :patch[1], :patch[2]] += o[0].half() * g
                torch.cuda.synchronize()
                est = (time.perf_counter() - t0) / sample * tiles
                worst_vram = max(worst_vram, torch.cuda.max_memory_allocated() / 2**20)
                worst_time = max(worst_time, est)
                del logits, g, pt, o
            except torch.cuda.OutOfMemoryError:
                worst_vram, worst_time = float("inf"), float("inf")
            torch.cuda.empty_cache()

        ok = worst_vram <= 4096 and worst_time <= 60 and train_vram <= 15000
        print(f"{label:28s} {n_par:7.2f}M {train_vram:9.0f}M "
              f"{worst_vram:8.0f}M {worst_time:7.1f}s  {'OK' if ok else 'OVER'}")
        del net
        torch.cuda.empty_cache()

    print("\nbudget: inference peak <=4096 MB, <=60 s/case; train must fit 16380 MB")
    print("inference VRAM here counts the net + logit map only; predict.py adds the")
    print("resampled input and the output volume, so treat these as a lower bound.")


if __name__ == "__main__":
    main()

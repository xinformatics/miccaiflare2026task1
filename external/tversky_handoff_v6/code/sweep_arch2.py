"""Is the 5.5M PlainConvUNet the right capacity, given how much budget is unused?

Measured reality of the current config: 2468 MB of the 4096 MB VRAM allowance and
12.5 s of the 60 s per-case allowance. The hidden leaderboard now tops out at 0.71
DSC while this architecture's lineage (FLARE24/25 baseline) sits at 0.33. Capacity
is the obvious thing we are not spending.

Sweeps deeper and residual-encoder variants at the same spacing/patch, so the only
variable is the network. Inference VRAM here counts net + logit map only; the
measured full-pipeline peak ran 1.42x that (2468 measured vs 1737 swept), so the
'est real' column applies that factor before comparing to 4096 MB.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa

PATCH = [128, 160, 160]
SPACING = [2.4, 1.6, 1.6]
MEASURED_FACTOR = 1.42          # full-pipeline peak / swept peak, from real runs
PLAIN = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
RESENC = "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"

# (label, class, n_stages, features, n_conv_per_stage(enc), strides_tail)
CANDIDATES = [
    ("current: Plain S4 W32",  PLAIN,  4, [32, 64, 128, 256],                 [2, 2, 2, 2],          None),
    ("Plain S5 W32",           PLAIN,  5, [32, 64, 128, 256, 320],            [2] * 5,               None),
    ("Plain S6 W32",           PLAIN,  6, [32, 64, 128, 256, 320, 320],       [2] * 6,               None),
    ("ResEnc S5 W32",          RESENC, 5, [32, 64, 128, 256, 320],            [1, 3, 4, 6, 6],       None),
    ("ResEnc S6 W32",          RESENC, 6, [32, 64, 128, 256, 320, 320],       [1, 3, 4, 6, 6, 6],    None),
    ("ResEnc S6 W48",          RESENC, 6, [48, 96, 192, 384, 480, 480],       [1, 3, 4, 6, 6, 6],    None),
]

CASES = [((512, 512, 361), (0.75, 0.75, 1.25)),
         ((512, 512, 688), (0.98, 0.98, 1.25))]


def build(cls, n_stages, feats, nconv):
    ak = {
        "n_stages": n_stages, "features_per_stage": feats,
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * n_stages,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        "n_conv_per_stage" if cls == PLAIN else "n_blocks_per_stage": nconv,
        "n_conv_per_stage_decoder": [2] * (n_stages - 1),
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
    }
    return get_network_from_plans(cls, ak, ["conv_op", "norm_op", "dropout_op", "nonlin"],
                                  1, 2, True), ak


def n_tiles(shape, patch, overlap=0.5):
    t = 1
    for s, p in zip(shape, patch):
        t *= 1 if s <= p else int(np.ceil((s - p) / (p * (1 - overlap)))) + 1
    return t


def main():
    dev = torch.device("cuda")
    print(f"{'config':24s} {'params':>9s} {'trainVRAM':>10s} {'infVRAM':>9s} "
          f"{'est real':>9s} {'maxTime':>8s}  verdict")
    print("-" * 88)
    for label, cls, ns, feats, nconv, _ in CANDIDATES:
        try:
            net, ak = build(cls, ns, feats, nconv)
            net = net.to(dev)
        except Exception as e:
            print(f"{label:24s} BUILD FAILED: {str(e)[:40]}")
            continue
        npar = sum(p.numel() for p in net.parameters()) / 1e6

        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        net.train()
        opt = torch.optim.SGD(net.parameters(), lr=1e-3)
        sc = torch.amp.GradScaler("cuda")
        try:
            x = torch.randn(2, 1, *PATCH, device=dev)
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    o = net(x)
                    loss = sum(t.float().mean() for t in o) if isinstance(o, (list, tuple)) else o.float().mean()
                sc.scale(loss).backward(); sc.step(opt); sc.update()
            torch.cuda.synchronize()
            tv = torch.cuda.max_memory_allocated() / 2**20
            del x, o, loss
        except torch.cuda.OutOfMemoryError:
            tv = float("inf")
        del opt, sc
        net.eval(); torch.cuda.empty_cache()

        wv, wt = 0.0, 0.0
        for shp_xyz, sp_xyz in CASES:
            shp = np.array([shp_xyz[2], shp_xyz[1], shp_xyz[0]], float)
            sp = np.array([sp_xyz[2], sp_xyz[1], sp_xyz[0]], float)
            new = np.maximum(np.round(shp * sp / np.array(SPACING)).astype(int), PATCH)
            tiles = n_tiles(new, PATCH)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                lg = torch.zeros((2, *new), dtype=torch.half, device=dev)
                pt = torch.randn(1, 1, *PATCH, device=dev)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                k = min(tiles, 12)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    for _ in range(k):
                        o = net(pt); o = o[0] if isinstance(o, (list, tuple)) else o
                        lg[:, :PATCH[0], :PATCH[1], :PATCH[2]] += o[0].half()
                torch.cuda.synchronize()
                wt = max(wt, (time.perf_counter() - t0) / k * tiles)
                wv = max(wv, torch.cuda.max_memory_allocated() / 2**20)
                del lg, pt, o
            except torch.cuda.OutOfMemoryError:
                wv, wt = float("inf"), float("inf")
            torch.cuda.empty_cache()

        est = wv * MEASURED_FACTOR
        ok = est <= 4096 and wt * 2 <= 60 and tv <= 15000   # x2 for the slower eval GPU
        print(f"{label:24s} {npar:8.2f}M {tv:9.0f}M {wv:8.0f}M {est:8.0f}M "
              f"{wt:7.1f}s  {'OK' if ok else 'OVER'}")
        del net; torch.cuda.empty_cache()

    print("\nbudget: est real inference peak <=4096 MB; time doubled for the")
    print("Quadro RTX 5000 eval GPU and compared to the 60 s cliff.")
    print("Training must fit 16380 MB (shared with nothing else).")


if __name__ == "__main__":
    main()

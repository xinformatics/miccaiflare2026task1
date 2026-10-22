"""Training cost of the bigger architectures: does it fit 16GB, and how slow?

Inference already showed ResEnc S6 W32 (107M) fits the 4096MB serving budget. The
open question is the TRAINING side -- VRAM against this 16380MB card, and seconds
per iteration, which sets whether a run is 12h or 60h.

Measures the real thing: fp16 autocast, SGD+nesterov momentum 0.99, deep supervision
losses summed, at the actual patch size and batch size. Falls back to smaller batch
sizes when a config OOMs, so we learn what WOULD work rather than just that it fails.
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa

PATCH = [128, 160, 160]
PLAIN = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
RESENC = "dynamic_network_architectures.architectures.residual_unet.ResidualEncoderUNet"
ITERS_PER_EPOCH = 250

CANDIDATES = [
    ("current: Plain S4 W32", PLAIN, 4, [32, 64, 128, 256], [2, 2, 2, 2]),
    ("Plain S6 W32", PLAIN, 6, [32, 64, 128, 256, 320, 320], [2] * 6),
    ("ResEnc S5 W32", RESENC, 5, [32, 64, 128, 256, 320], [1, 3, 4, 6, 6]),
    ("ResEnc S6 W32", RESENC, 6, [32, 64, 128, 256, 320, 320], [1, 3, 4, 6, 6, 6]),
]


def build(cls, n_stages, feats, nconv):
    key = "n_conv_per_stage" if cls == PLAIN else "n_blocks_per_stage"
    ak = {
        "n_stages": n_stages, "features_per_stage": feats,
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * n_stages,
        "strides": [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1),
        key: nconv,
        "n_conv_per_stage_decoder": [2] * (n_stages - 1),
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None, "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU", "nonlin_kwargs": {"inplace": True},
    }
    return get_network_from_plans(cls, ak, ["conv_op", "norm_op", "dropout_op", "nonlin"],
                                  1, 2, True)


def try_batch(net, bs, dev, n=8):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(bs, 1, *PATCH, device=dev)
    net.train()
    try:
        for _ in range(3):                      # warmup
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x)
                loss = sum(t.float().mean() for t in o) if isinstance(o, (list, tuple)) else o.float().mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = net(x)
                loss = sum(t.float().mean() for t in o) if isinstance(o, (list, tuple)) else o.float().mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n
        peak = torch.cuda.max_memory_allocated() / 2**20
        return dt, peak
    except torch.cuda.OutOfMemoryError:
        return None, None
    finally:
        del x, opt, scaler
        torch.cuda.empty_cache()


def main():
    dev = torch.device("cuda")
    free = torch.cuda.mem_get_info()[0] / 2**20
    print(f"free VRAM at start: {free:.0f} MB of {torch.cuda.mem_get_info()[1]/2**20:.0f} MB\n")
    print(f"{'config':24s} {'params':>9s} {'bs':>3s} {'VRAM':>8s} {'s/iter':>8s} "
          f"{'s/epoch':>8s} {'300ep':>8s} {'500ep':>8s}")
    print("-" * 86)
    for label, cls, ns, feats, nconv in CANDIDATES:
        net = build(cls, ns, feats, nconv).to(dev)
        npar = sum(p.numel() for p in net.parameters()) / 1e6
        got = False
        for bs in (2, 1):
            dt, peak = try_batch(net, bs, dev)
            if dt is not None:
                ep = dt * ITERS_PER_EPOCH
                note = "" if bs == 2 else "  <- needed bs=1"
                print(f"{label:24s} {npar:8.2f}M {bs:3d} {peak:7.0f}M {dt:7.3f}s "
                      f"{ep:7.0f}s {ep*300/3600:7.1f}h {ep*500/3600:7.1f}h{note}")
                got = True
                break
        if not got:
            print(f"{label:24s} {npar:8.2f}M   -  OOM even at batch size 1")
        del net
        torch.cuda.empty_cache()
    print("\ns/epoch = 250 iterations (nnU-Net's fixed epoch). Add ~10% for dataloading")
    print("off the HDD and ~11% if the val watcher runs concurrently.")


if __name__ == "__main__":
    main()

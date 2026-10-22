"""Does the S4D2W32 net fit the challenge budget? Measure before training, not after.

Budget (Information.md): peak GPU memory <=4096MB earns a perfect efficiency score,
and any case taking >60s scores DSC=0. Ranking weights efficiency as half of the
four metrics, so this is not a footnote.

Simulates nnU-Net sliding-window inference (fp16, no mirroring TTA) over the largest
real validation geometries, resampled to the target 2x2x3mm grid.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from arch import ARCHITECTURE, BATCH_SIZE, PATCH_SIZE, SPACING  # noqa: E402

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans  # noqa: E402


def make_net():
    return get_network_from_plans(
        ARCHITECTURE["network_class_name"], ARCHITECTURE["arch_kwargs"],
        ARCHITECTURE["_kw_requires_import"], 1, 2, True)


def n_tiles(shape, patch, overlap=0.5):
    """nnU-Net's sliding-window tile count."""
    tot = 1
    per = []
    for s, p in zip(shape, patch):
        if s <= p:
            k = 1
        else:
            step = p * (1 - overlap)
            k = int(np.ceil((s - p) / step)) + 1
        per.append(k)
        tot *= k
    return tot, per


def main():
    dev = torch.device("cuda")
    net = make_net().to(dev).eval()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"S4D2W32: {n_par/1e6:.2f}M params  patch={PATCH_SIZE} spacing={SPACING}")
    print(f"(target from FLARE24 champion README: 5.6M)\n")

    # --- training-step memory ---
    torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.SGD(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    x = torch.randn(BATCH_SIZE, 1, *PATCH_SIZE, device=dev)
    net.train()
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
            loss = sum(o.float().mean() for o in out) if isinstance(out, (list, tuple)) \
                else out.float().mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    torch.cuda.synchronize()
    print(f"TRAIN peak VRAM (bs={BATCH_SIZE}, fp16): "
          f"{torch.cuda.max_memory_allocated()/2**20:.0f} MB  "
          f"[have 16380 MB]")

    del opt, scaler, x, out, loss
    net.eval()
    torch.cuda.empty_cache()

    # --- inference on the largest real val geometries ---
    print("\n--- sliding-window inference @ 2x2x3mm, fp16, no mirroring ---")
    cases = [
        ("FLARE23Ts_0083 (512x512x361 @.75/.75/1.25)", (512, 512, 361), (0.75, 0.75, 1.25)),
        ("FLARE23Ts_0099 (512x512x334 @.88/.88/1.5)", (512, 512, 334), (0.879, 0.879, 1.5)),
        ("860mm z-FOV worst case", (512, 512, 688), (0.98, 0.98, 1.25)),
        ("KiTS23-like (512x512x581 @.71/.71/.5)", (512, 512, 581), (0.71, 0.71, 0.5)),
    ]
    for name, shp_xyz, sp_xyz in cases:
        # to nnU-Net (z,y,x) order, then resample to target spacing
        shp = np.array([shp_xyz[2], shp_xyz[1], shp_xyz[0]], dtype=float)
        sp = np.array([sp_xyz[2], sp_xyz[1], sp_xyz[0]], dtype=float)
        new_shape = np.round(shp * sp / np.array(SPACING)).astype(int)
        new_shape = np.maximum(new_shape, PATCH_SIZE)
        tiles, per = n_tiles(new_shape, PATCH_SIZE)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # nnU-Net keeps the full logit map on GPU: n_classes x resampled volume, fp16
        logits = torch.zeros((2, *new_shape), dtype=torch.half, device=dev)
        gaussian = torch.ones(PATCH_SIZE, dtype=torch.half, device=dev)
        patch = torch.randn(1, 1, *PATCH_SIZE, device=dev, dtype=torch.float32)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(min(tiles, 40)):        # cap the timing sample
                o = net(patch)
                o = o[0] if isinstance(o, (list, tuple)) else o
                logits[:, :PATCH_SIZE[0], :PATCH_SIZE[1], :PATCH_SIZE[2]] += \
                    (o[0].half() * gaussian)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sampled = min(tiles, 40)
        per_tile = dt / sampled
        est = per_tile * tiles
        peak = torch.cuda.max_memory_allocated() / 2**20
        flag = "OK " if peak <= 4096 and est <= 60 else "!! "
        print(f"{flag}{name}")
        print(f"    resampled {new_shape.tolist()}  tiles={tiles} {per}")
        print(f"    peak VRAM {peak:7.0f} MB   est GPU time {est:5.1f}s "
              f"({per_tile*1000:.0f} ms/tile)")
        del logits, gaussian, patch
        torch.cuda.empty_cache()

    print("\nBudget: <=4096 MB VRAM for perfect efficiency score; >60s/case = DSC 0.")
    print("Note: est GPU time excludes disk I/O, resampling and NIfTI writing.")


if __name__ == "__main__":
    main()

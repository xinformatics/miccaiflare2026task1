"""Label champion components TP/FP on the 707 complete-label internal cases.

Reproduces the deployed component stage (>=1000 mm3 filter) on the champion's
raw output, then records for every surviving component a TP/FP label, the exact
macro-Dice utility of deleting it, and only inference-visible features. Labels
are used for supervision/scoring, never as features. No public or hidden case is
touched. CPU-only.
"""
import csv, hashlib, json, os, sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = Path("/mnt/nvme2-data/shashank/MICCAI26")
GF   = ROOT / "external/GF-Screen"
PRED = GF / "analysis/candidate_refiner_v3_weak707_training_20260823/currentgf_g050_l050"
MAN  = GF / "analysis/fpr_training_mining_707_20260822/manifest.json"
OUT  = GF / "analysis/subtractive_gate_diag_20260823"
MIN_MM3 = 1000.0          # deployed champion component floor
DET_OVERLAP = 0.10        # detection criterion used by the champion audit

def one(item):
    name, img_p, lab_p, cohort = item["name"], item["image"], item["label"], item["cohort"]
    pp = PRED / item["prediction_name"]
    if not pp.exists(): return []
    pn = nib.load(str(pp))
    pred = np.asarray(pn.dataobj) > 0
    gt = np.asarray(nib.load(lab_p).dataobj) > 0
    if pred.shape != gt.shape: return []
    img = nib.load(img_p).dataobj
    vox = float(abs(np.linalg.det(pn.affine[:3, :3])))

    lab, nl = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    if nl == 0: return []
    sizes = ndimage.sum(pred, lab, index=np.arange(1, nl + 1))
    keep = {k for k in range(1, nl + 1) if sizes[k - 1] * vox >= MIN_MM3}
    if not keep: return []
    deployed = np.isin(lab, list(keep))
    glab, ngl = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    gt_sizes = ndimage.sum(gt, glab, index=np.arange(1, ngl + 1)) if ngl else np.array([])

    ps, gs = int(deployed.sum()), int(gt.sum())
    inter0 = int((deployed & gt).sum())
    base = 0.0 if ps + gs == 0 else 2 * inter0 / (ps + gs)
    objs = ndimage.find_objects(lab)
    zdim = pred.shape[2] if pred.ndim == 3 else 1
    rows = []
    vols = sorted((sizes[k - 1] for k in keep), reverse=True)
    for k in sorted(keep):
        m = lab == k
        n = int(sizes[k - 1]); i = int((m & gt).sum())
        det = 0
        if i and ngl:
            for gi in np.unique(glab[m]):
                if gi and int((m & (glab == gi)).sum()) >= DET_OVERLAP * gt_sizes[gi - 1]:
                    det += 1
        sl = objs[k - 1]
        ext = [ (s.stop - s.start) for s in sl ]
        # local context shell (inference-visible)
        pad = 6
        ex = tuple(slice(max(0, s.start - pad), min(d, s.stop + pad)) for s, d in zip(sl, pred.shape))
        sub_m = m[ex]
        sub_i = np.asarray(img[ex]).astype(np.float32)
        shell = ndimage.binary_dilation(sub_m, iterations=3) & ~sub_m
        iv = sub_i[sub_m]
        sv = sub_i[shell] if shell.any() else np.array([np.nan], np.float32)
        cz = float(ndimage.center_of_mass(m)[2]) / max(1, zdim - 1)
        d_removed = 0.0 if (ps - n) + gs == 0 else 2 * (inter0 - i) / ((ps - n) + gs)
        rows.append(dict(
            case=name, cohort=cohort, comp=k, is_tp=int(i > 0), gt_comps_detected=det,
            overlap_frac=i / n, delta_dice_if_removed=d_removed - base, base_dice=base,
            # --- inference-visible features only, below ---
            mm3=n * vox, log_mm3=float(np.log10(n * vox)),
            hu_mean=float(iv.mean()), hu_std=float(iv.std()),
            hu_p10=float(np.percentile(iv, 10)), hu_p50=float(np.median(iv)),
            hu_p90=float(np.percentile(iv, 90)),
            shell_hu_p50=float(np.nanmedian(sv)),
            contrast=float(np.median(iv) - np.nanmedian(sv)),
            ext_x=ext[0], ext_y=ext[1], ext_z=ext[2],
            elongation=float(max(ext) / max(1, min(ext))),
            fill_frac=float(n / max(1, ext[0] * ext[1] * ext[2])),
            surface_ratio=float(ndimage.binary_dilation(sub_m).sum() - sub_m.sum()) / max(1, n),
            z_centroid_norm=cz,
            case_n_comps=len(keep), case_pred_mm3=ps * vox,
            vol_rank=vols.index(sizes[k - 1]) + 1,
            vol_frac_of_case=float(n / max(1, ps)),
        ))
    return rows

if __name__ == "__main__":
    items = json.load(open(MAN))["items"]
    os.nice(19)
    with Pool(3) as pool:
        allrows = []
        for j, rs in enumerate(pool.imap_unordered(one, items, chunksize=1), 1):
            allrows.extend(rs)
            if j % 20 == 0: print(f"[{j}/{len(items)}] rows={len(allrows)}", flush=True)
    keys = sorted({k for r in allrows for k in r})
    dataset = OUT / "gate_dataset_707.csv"
    temporary = dataset.with_name(f".{dataset.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(allrows)
        f.flush(); os.fsync(f.fileno())
    os.replace(temporary, dataset)
    payload = dict(
        schema_version=1,
        created_at=datetime.now().astimezone().isoformat(),
        status="COMPLETE",
        items_processed=len(items),
        rows=len(allrows),
        cases_with_components=len({r["case"] for r in allrows}),
        tp=sum(r["is_tp"] for r in allrows),
        dataset=str(dataset.resolve()),
        dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
        manifest=str(MAN.resolve()),
        manifest_sha256=hashlib.sha256(MAN.read_bytes()).hexdigest(),
        predictions=str(PRED.resolve()),
        min_mm3=MIN_MM3,
        labels_read=True,
        public_or_hidden_data_read=False,
    )
    marker = OUT / "GATE_DATASET_707_COMPLETE.json"
    marker_temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    marker_temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(marker_temporary, marker)
    print(json.dumps(payload, indent=2))

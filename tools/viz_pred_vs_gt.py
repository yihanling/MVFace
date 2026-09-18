import _init_paths  # noqa: F401
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvface.checkpoint import load_model
from mvface.output_dir import OutputDir
from mvface.data.facescape_dataset import (
    MultiViewFaceScape, denormalize_rgb, discover_subject_folders,
    subject_train_val_split)
from mvface.units import MM_PER_METRE


def project_np(world: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(68,3) world -> (68,2) pixels through P (3,4)."""
    ph = np.concatenate([world, np.ones((world.shape[0], 1))], axis=1)
    uvw = ph @ P.T
    return uvw[:, :2] / uvw[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best.pth / last.pth")
    ap.add_argument("--root", default=None, help="override dataset root (default: from ckpt args)")
    ap.add_argument("--out", default=None,
                    help="output PNG (default: <run>/figures/pred_vs_gt.png)")
    ap.add_argument("--n", type=int, default=5, help="#validation subjects to show")
    ap.add_argument("--seed", type=int, default=None,
                    help="which val subjects to sample (default: random)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, ckpt, t = load_model(args.ckpt, args.device)
    run = OutputDir.from_checkpoint(args.ckpt)
    root = args.root or run.data_root() or t["root"]
    print(f"ckpt: {args.ckpt}\n  trained {t['epochs']} ep, "
          f"val_mpjpe(best)={ckpt.get('val_mpjpe', float('nan')):.3f} mm"
          f"  depth={'OFF' if t.get('no_depth') else 'ON'}")

    # The held-out subjects the model never saw. Prefer the split frozen at
    # train time; only re-derive it for runs that recorded none.
    split = run.read_split()
    if split is not None:
        val_ids, how = split[1], "recorded"
    else:
        subs = discover_subject_folders(root)
        if t.get("limit"):
            subs = subs[: t["limit"]]
        _, val_ids = subject_train_val_split(subs, t["val_frac"], t["seed"])
        how = f"re-derived (val_frac={t['val_frac']}, seed={t['seed']})"
    ds = MultiViewFaceScape(root, val_ids)
    print(f"  val subjects: {len(val_ids)}  [{how}]")

    rng = random.Random(args.seed if args.seed is not None else random.randrange(1_000_000))
    pick = rng.sample(range(len(val_ids)), min(args.n, len(val_ids)))
    print(f"  showing: {[val_ids[i] for i in pick]}")

    ncol = None
    rows = []
    for idx in pick:
        s = ds[idx]
        rgbd = s["rgbd"].unsqueeze(0).to(args.device)              # (1,N,4,H,W)
        proj = s["proj"].unsqueeze(0).to(args.device)              # (1,N,3,4)
        hw = (rgbd.shape[-2], rgbd.shape[-1])
        with torch.no_grad():
            preds_3d, _ = model(rgbd, proj, hw)
        pred3d = preds_3d[-1][0].cpu().numpy()                     # (68,3) world, metres
        gt3d = s["landmarks_3d"].numpy()                           # (68,3) world, metres
        mpjpe = float(np.linalg.norm(pred3d - gt3d, axis=1).mean()) * MM_PER_METRE
        rows.append({
            "id": val_ids[idx], "mpjpe": mpjpe,
            # de-normalize: the stored pixels are ImageNet-normalized, not [0,1]
            "rgb": denormalize_rgb(s["rgbd"][:, :3]).permute(0, 2, 3, 1).numpy(),  # (N,H,W,3)
            "P": s["proj"].numpy(),                                # (N,3,4)
            "gt2d": s["landmarks_2d"].numpy(),                     # (N,68,2) stored GT pixels
            "pred3d": pred3d,
        })
        ncol = rows[-1]["rgb"].shape[0]
        print(f"    {val_ids[idx]:>10s}  MPJPE {mpjpe:6.2f} mm")

    mean_mpjpe = np.mean([r["mpjpe"] for r in rows])
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.7 * ncol, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, row in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.imshow(row["rgb"][c]); ax.axis("off")
            gt = row["gt2d"][c]
            pr = project_np(row["pred3d"], row["P"][c])
            # residual lines (faint) then the two point sets
            for k in range(gt.shape[0]):
                ax.plot([gt[k, 0], pr[k, 0]], [gt[k, 1], pr[k, 1]],
                        c="yellow", lw=0.4, alpha=0.6)
            ax.scatter(gt[:, 0], gt[:, 1], s=6, c="lime", edgecolors="none", label="GT")
            ax.scatter(pr[:, 0], pr[:, 1], s=6, c="red", edgecolors="none", label="pred")
            if c == 0:
                ax.set_title(f"{row['id']}  v{c}\nMPJPE {row['mpjpe']:.2f} mm",
                             fontsize=8, loc="left")
            else:
                ax.set_title(f"v{c}", fontsize=8, loc="left")
    handles = [plt.Line2D([], [], marker="o", ls="", c="lime", label="ground truth"),
               plt.Line2D([], [], marker="o", ls="", c="red", label="prediction (3D reprojected)"),
               plt.Line2D([], [], c="yellow", label="residual")]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle(f"pred vs GT landmarks — held-out val — mean MPJPE {mean_mpjpe:.2f} mm "
                 f"(over {len(rows)} subjects)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(args.out) if args.out else run.figure("pred_vs_gt.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=115)
    print(f"\nmean MPJPE over shown subjects: {mean_mpjpe:.2f} mm")
    print(f"panel -> {out}")


if __name__ == "__main__":
    main()

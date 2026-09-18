"""
Example:
    .venv/bin/python tools/train.py \
        --root data/facescape/virtual_camera_data --epochs 40 --bs 2 --lr 1e-4
"""

import _init_paths  # noqa: F401
import argparse
import csv
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _init_paths import REPO_ROOT
from mvface.checkpoint import save_checkpoint
from mvface.data.facescape_dataset import (
    MultiViewFaceScape, _base_identity, discover_subject_folders,
    limit_to_subjects, subject_train_val_split)
from mvface.losses import decoder_losses, mpjpe_mm
from mvface.model import MultiViewLandmark3D
from mvface.output_dir import OutputDir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO_ROOT / "data/facescape/virtual_camera_data"))
    p.add_argument("--assets", default=str(REPO_ROOT / "src/mvface/assets"))
    p.add_argument("--out", default=str(REPO_ROOT / "output/early_fusion"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1e-3,
                   help="max grad-norm for clipping; guards against NaN blow-ups")
    p.add_argument("--lambda-2d", type=float, default=1e-4,
                   help="weight on the 2D reprojection loss term")
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0,
                   help="cap #subjects/identities (0=all) for quick smoke tests; "
                        "keeps ALL expressions and variants of those subjects")
    p.add_argument("--no-depth", action="store_true", help="RGB-only ablation arm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true",
                   help="overwrite a non-empty run directory without asking")
    p.add_argument("--resume", action="store_true",
                   help="continue an existing run from checkpoints/last.pth")
    return p.parse_args()


def move(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, lambda_2d):
    """Return (val_loss, val_mpjpe_mm) averaged over the loader (per-sample)."""
    model.eval()
    loss_sum, err_sum, n = 0.0, 0.0, 0
    for batch in loader:
        batch = move(batch, device)
        hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
        preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
        b = batch["rgbd"].shape[0]
        losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                batch["landmarks_2d"], batch["vis"], lambda_2d=lambda_2d)
        loss_sum += float(losses["total"]) * b
        err_sum += float(mpjpe_mm(preds_3d[-1], batch["landmarks_3d"])) * b
        n += b
    n = max(n, 1)
    return loss_sum / n, err_sum / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    # OutputDir owns the layout. An explicit flag wins; otherwise an occupied
    # directory prompts (or refuses, when not on a TTY).
    on_exists = "force" if args.force else ("resume" if args.resume else "ask")
    run = OutputDir(args.out).create(on_exists=on_exists)
    resuming = run.action == "resumed"

    # Append when resuming so the existing history is not truncated.
    mode = "a" if resuming else "w"
    logf = open(run.train_log, mode)

    def logprint(msg):
        print(msg)
        logf.write(msg + "\n"); logf.flush()

    csvf = open(run.train_csv, mode, newline="")
    writer = csv.writer(csvf)
    if not resuming:
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_mpjpe",
                         "skipped", "sec"])
        csvf.flush()

    if resuming:
        # The frozen split is authoritative -- re-deriving it could silently move
        # subjects between train and val mid-run if --limit/--val-frac changed.
        split = run.read_split()
        if split is None:
            raise SystemExit(f"cannot resume {run.root}: no split.csv")
        train_ids, val_ids = split
        logprint(f"resuming {run.root}  train {len(train_ids)}  val {len(val_ids)}")
    else:
        subs = discover_subject_folders(args.root)
        if args.limit:
            subs = limit_to_subjects(subs, args.limit)
        train_ids, val_ids = subject_train_val_split(subs, args.val_frac, args.seed)
        n_subj = len({_base_identity(s) for s in subs})

        if not train_ids or not val_ids:
            side = "train" if not train_ids else "val"
            raise SystemExit(
                f"empty {side} split: {n_subj} subject(s) / {len(subs)} items at "
                f"--val-frac {args.val_frac} gave train {len(train_ids)}, "
                f"val {len(val_ids)}.\n"
                f"An identity-disjoint split needs at least two subjects. "
                f"Raise --limit (it caps subjects, not items) or lower --val-frac.")
        logprint(f"items: {len(subs)}  subjects: {n_subj}  train {len(train_ids)}  "
                 f"val {len(val_ids)}  depth={'OFF' if args.no_depth else 'ON'}  "
                 f"-> {run.root}")

        # Frozen subject train/val split (one row per subject) + the run's identity
        # record: args, git sha, torch/CUDA versions, resolved data root.
        run.write_split(train_ids, val_ids)
        run.write_config(vars(args),
                         data_root=str(Path(args.root).resolve()),
                         n_subjects=len(subs),
                         n_train=len(train_ids),
                         n_val=len(val_ids))


    train_ds = MultiViewFaceScape(args.root, train_ids)
    val_ds = MultiViewFaceScape(args.root, val_ids)
    train_ld = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=args.workers, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=args.bs, shuffle=False,
                        num_workers=args.workers)

    model = MultiViewLandmark3D(args.assets, num_layers=args.num_layers,
                                use_depth=not args.no_depth,
                                img_size=args.img_size).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    start_epoch, best = 0, float("inf")
    if resuming:
        ck = torch.load(run.checkpoint("last"), map_location=args.device,
                        weights_only=False)
        model.load_state_dict(ck["model"])
        # Adam moments and the cosine schedule's position: without these the run
        # restarts its optimizer state and its learning-rate curve.
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        else:
            logprint("  warning: no scheduler state in last.pth -- lr curve restarts")
        start_epoch = ck.get("epoch", 0)
        best = ck.get("best", ck.get("val_mpjpe", float("inf")))
        prev_epochs = (run.read_config() or {}).get("args", {}).get("epochs")
        if prev_epochs and prev_epochs != args.epochs:
            logprint(f"  warning: --epochs {args.epochs} differs from the original "
                     f"{prev_epochs}; the cosine schedule shape changes")
        logprint(f"  from epoch {start_epoch + 1}/{args.epochs}, best so far {best:.2f} mm")
        if start_epoch >= args.epochs:
            logprint(f"nothing to do: {start_epoch} epochs already complete")
            csvf.close(); logf.close()
            return

    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0, running, skipped = time.time(), 0.0, 0
        for it, batch in enumerate(train_ld):
            batch = move(batch, args.device)
            hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
            preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
            losses = decoder_losses(preds_3d, preds_2d, batch["landmarks_3d"],
                                    batch["landmarks_2d"], batch["vis"],
                                    lambda_2d=args.lambda_2d)
            loss = losses["total"]
            # Skip a batch whose loss is already non-finite (do not backward NaN).
            if not torch.isfinite(loss):
                skipped += 1
                continue
            opt.zero_grad()
            loss.backward()
            # if the grads themselves are non-finite (e.g. an SVD-backward NaN), skip the step so the optimizer never writes NaN into the weights.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isfinite(gnorm):
                opt.step()
                running += loss.item()
            else:
                skipped += 1
        lr = opt.param_groups[0]["lr"]
        sched.step()

        train_loss = running / max(len(train_ld) - skipped, 1)
        val_loss, val_mpjpe = evaluate(model, val_ld, args.device, args.lambda_2d)
        sec = time.time() - t0
        skip_note = f"  skipped {skipped}" if skipped else ""
        logprint(f"epoch {epoch:3d}  train_loss {train_loss:8.3f}  "
                 f"val_loss {val_loss:8.3f}  val_MPJPE {val_mpjpe:7.2f} mm  "
                 f"({sec:.0f}s){skip_note}")
        writer.writerow([epoch, f"{lr:.3e}", f"{train_loss:.6f}",
                         f"{val_loss:.6f}", f"{val_mpjpe:.6f}", skipped,
                         f"{sec:.1f}"])
        csvf.flush()

        ckpt = {"model": model.state_dict(), "epoch": epoch, "val_mpjpe": val_mpjpe,
                "val_loss": val_loss, "args": vars(args)}
        if val_mpjpe < best:
            best = val_mpjpe
            save_checkpoint(ckpt, run.checkpoint("best"))   # lean: for eval / deploy
        # last.pth additionally carries what a resume needs.
        save_checkpoint({**ckpt, "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "best": best}, run.checkpoint("last"))
    logprint(f"done. best val MPJPE {best:.2f} mm  ->  {run.checkpoint('best')}")
    csvf.close(); logf.close()


if __name__ == "__main__":
    main()

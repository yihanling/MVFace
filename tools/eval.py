"""Evaluate a model checkpoint

Runs in up to three phases:

  1. accuracy  stream every held-out val subject -> MPJPE, P-MPJPE, loss, optional per-subject / per-joint breakdowns.
  2. timing    one fixed batch pinned on the device, warmed up and synchronised, timed over many iterations -> latency and Hz.
  3. report    write everything to <ckpt_dir>/eval.json.

Examples:
    .venv/bin/python tools/eval.py --ckpt output/messy_rgbd_v2/best.pth
    .venv/bin/python tools/eval.py --ckpt output/messy_rgbd_v2/best.pth --per-joint
    .venv/bin/python tools/eval.py --ckpt output/messy_rgbd_v2/best.pth --no-bench
    # timing only -- works on a checkpoint with no run directory:
    .venv/bin/python tools/eval.py --ckpt old_run/best.pth --bench-only --root <data>
"""

import _init_paths  # noqa: F401
import argparse
import csv
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from mvface.checkpoint import load_model
from mvface.data.facescape_dataset import MultiViewFaceScape, discover_subject_folders
from mvface.head_pose import landmark_group
from mvface.losses import decoder_losses, p_mpjpe, per_joint_pjpe
from mvface.output_dir import OutputDir
from mvface.units import MM_PER_METRE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to best.pth / last.pth")
    p.add_argument("--root", default=None,
                   help="override the dataset root recorded in the run config")
    p.add_argument("--assets", default=None,
                   help="decoder assets dir (default: the training-time --assets)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap #val subjects (0=all); ignored if --subjects given")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="explicit subject ids to score (must be a subset of val_ids)")
    p.add_argument("--per-subject", action="store_true",
                   help="print each subject's MPJPE, worst first")
    p.add_argument("--per-joint", action="store_true",
                   help="print the 10 worst landmarks by mean error")
    p.add_argument("--bs", type=int, default=2, help="accuracy-phase batch size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = p.add_argument_group("timing")
    g.add_argument("--no-bench", action="store_true",
                   help="skip the timing phase")
    g.add_argument("--bench-only", action="store_true",
                   help="skip the accuracy phase (and its need for a recorded split)")
    g.add_argument("--bench-bs", type=int, default=1,
                   help="batch size to TIME at; 1 = single-frame-set latency, "
                        "which is what a real-time rig actually sees")
    g.add_argument("--iters", type=int, default=100, help="timed iterations")
    g.add_argument("--warmup", type=int, default=15, help="untimed warmup iterations")

    g2 = p.add_argument_group("report")
    g2.add_argument("--out-dir", default=None,
                    help="write the report here instead of <run>/eval/")
    g2.add_argument("--no-report", action="store_true", help="do not write a report")
    return p.parse_args()


def build_val_ids(split, args):
    val_ids = split["val_ids"]
    if args.subjects:
        val_set = set(val_ids)
        bad = [s for s in args.subjects if s not in val_set]
        if bad:
            raise SystemExit(f"not in the val split (would leak): {bad}")
        return list(args.subjects)
    return val_ids[: args.limit] if args.limit else val_ids


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def run_accuracy(model, loader, val_ids, lambda_2d, device) -> dict:
    """Stream the val set once. Returns a dict of metrics (mm where applicable).

    shuffle=False + drop_last=False -> the loader yields subjects in val_ids
    order, so a running counter maps per-sample errors back to subject ids.
    """
    per_subject, joint_err = [], []
    loss_sum, err_sum, palign_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
            preds_3d, preds_2d = model(batch["rgbd"], batch["proj"], hw)
            b = batch["rgbd"].shape[0]
            gt = batch["landmarks_3d"]

            losses = decoder_losses(preds_3d, preds_2d, gt, batch["landmarks_2d"],
                                    batch["vis"], lambda_2d=lambda_2d)
            loss_sum += float(losses["total"]) * b
            palign_sum += p_mpjpe(preds_3d[-1], gt) * b
            joint_err.append(per_joint_pjpe(preds_3d[-1], gt) * b)

            # per-sample MPJPE in mm: L2 per joint, mean over the 68 joints -> (b,)
            err = (preds_3d[-1] - gt).norm(dim=-1).mean(dim=-1) * MM_PER_METRE
            for e in err.tolist():
                per_subject.append((val_ids[n], e))
                err_sum += e
                n += 1

    n = max(n, 1)
    return {
        "n_val_subjects": n,
        "mpjpe_mm": err_sum / n,
        "p_mpjpe_mm": palign_sum / n,
        "loss": loss_sum / n,
        "per_joint_mm": (torch.stack(joint_err).sum(0) / n).tolist(),
        "per_subject_mm": per_subject,
    }


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def bench_subjects(root, val_ids, n) -> list[str]:
    if val_ids:
        return val_ids[:n]
    found = discover_subject_folders(root)
    if len(found) < n:
        raise SystemExit(f"only {len(found)} subjects under {root}, need {n}")
    return found[:n]


def run_timing(model, root, subjects, bench_bs, iters, warmup, device, workers) -> dict:
    ds = MultiViewFaceScape(str(root), subjects)
    loader = DataLoader(ds, batch_size=bench_bs, shuffle=False, num_workers=workers)
    it = iter(loader)
    batch = {k: v.to(device) for k, v in next(it).items()}

    # Release the loader's worker PROCESSES before timing. They sit idle but still
    # compete for CPU, and a CUDA kernel launch is CPU work -- with workers alive
    # the launch queue runs dry and the GPU idles between kernels, costing ~20%
    # (measured 56 Hz with 4 workers alive vs 68 Hz released, same batch).
    del it, loader, ds
    gc.collect()

    hw = (batch["rgbd"].shape[-2], batch["rgbd"].shape[-1])
    B, N, _, H, W = batch["rgbd"].shape

    cuda = device.startswith("cuda")

    def run():
        with torch.no_grad():
            model(batch["rgbd"], batch["proj"], hw)

    for _ in range(warmup):
        run()
    if cuda:
        torch.cuda.synchronize()     # queued != done; block until the GPU catches up

    t0 = time.perf_counter()
    for _ in range(iters):
        run()
    if cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    return {
        "precision": "fp32",
        "bench_loader_released": True,   # workers torn down before timing
        "bench_bs": B,
        "bench_views": N,
        "bench_input_hw": [H, W],
        "iters": iters,
        "warmup": warmup,
        "latency_ms": dt / iters * 1e3,
        "hz": iters / dt,
        "device_name": torch.cuda.get_device_name(0) if cuda else "cpu",
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(run: OutputDir, name: str, summary: dict,
                 per_subject: list | None, per_joint: list | None,
                 out_dir=None) -> Path:
    """Scalars -> eval/<name>.json; the two detail tables -> sibling CSVs.

    Split by shape: the summary is what you read every time and stays small
    enough to `cat`; the per-subject and per-joint tables are things you sort
    and filter, which is CSV's job.
    """
    if out_dir:
        d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
        js, ps, pj = (d / f"{name}.json", d / f"{name}_per_subject.csv",
                      d / f"{name}_per_joint.csv")
    else:
        js, ps, pj = (run.eval_json(name), run.eval_per_subject(name),
                      run.eval_per_joint(name))
        js.parent.mkdir(parents=True, exist_ok=True)

    js.write_text(json.dumps(summary, indent=2))

    if per_subject is not None:
        with open(ps, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["subject_id", "mpjpe_mm"])
            for sid, e in sorted(per_subject, key=lambda x: -x[1]):
                w.writerow([sid, f"{e:.4f}"])
    if per_joint is not None:
        with open(pj, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["joint_idx", "group", "mpjpe_mm"])
            for k, e in enumerate(per_joint):
                w.writerow([k, landmark_group(k), f"{e:.4f}"])
    return js


def main():
    args = parse_args()
    if args.bench_only and args.no_bench:
        raise SystemExit("--bench-only and --no-bench are mutually exclusive")

    model, ckpt, a = load_model(args.ckpt, args.device, args.assets)
    run = OutputDir.from_checkpoint(args.ckpt)
    name = Path(args.ckpt).stem                     # best.pth -> "best"
    split = run.read_split()                        # None on runs with no split

    print(f"ckpt {args.ckpt}  (epoch {ckpt.get('epoch','?')}, "
          f"train-time val_MPJPE {ckpt.get('val_mpjpe', float('nan')):.3f} mm)")
    print(f"run  {run.root}")

    summary = {
        "ckpt": str(args.ckpt),
        "evaluated": datetime.now().isoformat(timespec="seconds"),
        "epoch": ckpt.get("epoch"),
        "train_val_mpjpe_mm": ckpt.get("val_mpjpe"),
    }
    per_subject = per_joint = None
    val_ids = []
    root = args.root or run.data_root() or a["root"]

    # ---- phase 1: accuracy -------------------------------------------------
    if not args.bench_only:
        if split is None:
            raise SystemExit(
                f"no split recorded in {run.root} -- cannot score held-out "
                "subjects. Use --bench-only to time this checkpoint anyway.")
        val_ids = build_val_ids({"val_ids": split[1]}, args)
        print(f"scoring {len(val_ids)} val subjects  root={root}  "
              f"depth={'OFF' if a['no_depth'] else 'ON'}")

        loader = DataLoader(MultiViewFaceScape(root, val_ids), batch_size=args.bs,
                            shuffle=False, num_workers=args.workers)
        acc = run_accuracy(model, loader, val_ids, a["lambda_2d"], args.device)

        print(f"\nMPJPE    {acc['mpjpe_mm']:7.3f} mm   "
              f"(mean over {acc['n_val_subjects']} subjects)")
        print(f"P-MPJPE  {acc['p_mpjpe_mm']:7.3f} mm   (Procrustes-aligned; shape only)")
        print(f"loss     {acc['loss']:7.3f}      (deep-supervised total, for parity)")

        per_joint = acc["per_joint_mm"]
        per_subject = acc["per_subject_mm"]

        if args.per_joint:
            j = torch.tensor(per_joint)
            print("\n10 worst landmarks:")
            for k in torch.argsort(j, descending=True)[:10].tolist():
                print(f"  lm {k:2d}  {landmark_group(k):<12s} {float(j[k]):7.3f} mm")

        if args.per_subject:
            print("\nper-subject MPJPE (worst first):")
            for sid, e in sorted(per_subject, key=lambda x: -x[1]):
                print(f"  {sid:>10s}  {e:7.3f} mm")

        summary.update({k: v for k, v in acc.items()
                        if k not in ("per_subject_mm", "per_joint_mm")})
        summary["val_root"] = str(root)

    # ---- phase 2: timing ---------------------------------------------------
    if not args.no_bench:
        if not Path(root).exists():
            raise SystemExit(f"data root not found: {root} (pass --root)")
        subs = bench_subjects(root, val_ids, args.bench_bs)
        t = run_timing(model, root, subs, args.bench_bs, args.iters,
                       args.warmup, args.device, args.workers)
        print(f"\ntiming   {t['device_name']}  fp32  "
              f"bs={t['bench_bs']} x {t['bench_views']} views "
              f"@ {t['bench_input_hw'][0]}x{t['bench_input_hw'][1]}  "
              f"({t['iters']} iters, {t['warmup']} warmup)")
        print(f"latency  {t['latency_ms']:7.2f} ms / forward")
        print(f"rate     {t['hz']:7.2f} Hz  (frame-sets / s)")
        summary.update(t)

    # ---- phase 3: report ---------------------------------------------------
    if args.no_report:
        return
    # A narrowed eval is not the run's canonical result -- writing it to
    # eval/<name>.json would silently replace a full-val record with a subset.
    partial = bool(args.limit or args.subjects)
    if partial and not args.out_dir:
        print("\nnarrowed eval (--limit/--subjects): not written. "
              "Pass --out-dir to save it somewhere explicit.")
        return
    out = write_report(run, name, summary, per_subject, per_joint, args.out_dir)
    print(f"\nreport -> {out}"
          + (f"\n         {out.with_name(name + '_per_subject.csv').name}, "
             f"{out.with_name(name + '_per_joint.csv').name}"
             if per_subject is not None else ""))


if __name__ == "__main__":
    main()

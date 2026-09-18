"""Load a trained model checkpoint, shared functions between multiple files
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from mvface.model import MultiViewLandmark3D


def load_checkpoint(path, device: str = "cpu") -> tuple[dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "args" not in ckpt:
        raise SystemExit(
            f"{path} has no 'args' -- cannot rebuild the model it belongs to. "
            "Only checkpoints written by tools/train.py are supported.")
    return ckpt, ckpt["args"]


def build_model(train_args: dict, device: str = "cpu",
                assets: str | None = None,
                weights: dict | None = None) -> MultiViewLandmark3D:
    model = MultiViewLandmark3D(
        assets or train_args["assets"],
        num_layers=train_args["num_layers"],
        use_depth=not train_args["no_depth"],
        img_size=train_args["img_size"],
    )
    if weights is not None:
        model.load_state_dict(weights)
    return model.to(device).eval()


def load_model(path, device: str = "cpu", assets: str | None = None):
    """Checkpoint path -> ready-to-run model. Returns (model, ckpt, train_args)."""
    ckpt, train_args = load_checkpoint(path, device)
    model = build_model(train_args, device, assets, weights=ckpt["model"])
    return model, ckpt, train_args


def save_checkpoint(obj: dict, path) -> Path:
    """Write a checkpoint atomically -- a crash mid-write leaves the old file intact.

    last.pth is ~425 MB (weights + Adam moments) and is the ONLY resume point a
    run has. torch.save takes a second or two, and a process death inside that
    window would leave a truncated file, making the whole run unrecoverable.
    Writing to a sibling temp file and renaming avoids that: os.replace is atomic
    on POSIX, so the path is always either the previous checkpoint or the new one.

    The temp file is deliberately a sibling, since os.replace is only atomic
    within one filesystem.

    This protects against process death (crash, OOM, Ctrl-C), not against power
    loss -- surviving that would need an fsync of both file and directory, which
    would cost a full flush of ~425 MB every epoch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:                 # includes KeyboardInterrupt
        tmp.unlink(missing_ok=True)
        raise
    return path

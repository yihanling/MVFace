from __future__ import annotations

import torch
import torch.nn.functional as F

from mvface.units import MM_PER_METRE

# Landmarks are in metres (IOD ~0.105 m = ~105 mm physically).
# Multiply by MM_PER_METRE to report all metrics in mm.


def masked_l1_2d(pred: torch.Tensor, gt: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    """visibility-masked 2D loss"""
    m = vis.unsqueeze(-1)
    diff = (pred - gt).abs() * m
    return diff.sum() / (m.sum() * 2 + 1e-6)


def decoder_losses(preds_3d, preds_2d, gt_3d, gt_2d, vis, lambda_2d: float = 1e-4):
    """Returns a dict with the total and each component (for logging)."""
    loss_3d = preds_3d[0].new_zeros(())
    loss_2d = preds_3d[0].new_zeros(())
    for p3, p2 in zip(preds_3d, preds_2d):
        loss_3d = loss_3d + F.l1_loss(p3, gt_3d)
        loss_2d = loss_2d + masked_l1_2d(p2, gt_2d, vis)
    total = loss_3d + lambda_2d * loss_2d
    return {"total": total, "loss_3d": loss_3d, "loss_2d": loss_2d}


# ---------------------------------------------------------------------------
# Evaluation metrics — all return honest mm via MM_PER_METRE.
# vis: optional (B, J) mask; None averages over all landmarks (matches how
# nme_interocular is computed in train.py, so the two metrics are comparable).
# ---------------------------------------------------------------------------
def _masked_mean(err: torch.Tensor, vis: torch.Tensor | None) -> torch.Tensor:
    if vis is None:
        return err.mean()
    vis = vis.to(err.dtype)
    return (err * vis).sum() / vis.sum().clamp(min=1.0)


@torch.no_grad()
def mpjpe_mm(pred_3d: torch.Tensor, gt_3d: torch.Tensor,
             vis: torch.Tensor | None = None,
             scale: float = MM_PER_METRE) -> float:
    """Mean per-joint position error in mm."""
    err = (pred_3d - gt_3d).norm(dim=-1)              # (B, J)
    return float(_masked_mean(err, vis) * scale)


@torch.no_grad()
def per_joint_pjpe(pred_3d: torch.Tensor, gt_3d: torch.Tensor,
                   vis: torch.Tensor | None = None,
                   scale: float = MM_PER_METRE) -> torch.Tensor:
    """Per-landmark position error in mm averaged over the batch -> (J,)."""
    err = (pred_3d - gt_3d).norm(dim=-1)              # (B, J)
    if vis is None:
        return err.mean(0) * scale
    vis = vis.to(err.dtype)
    return (err * vis).sum(0) / vis.sum(0).clamp(min=1.0) * scale


@torch.no_grad()
def p_mpjpe(pred_3d: torch.Tensor, gt_3d: torch.Tensor,
            vis: torch.Tensor | None = None,
            scale: float = MM_PER_METRE) -> float:
    """Procrustes-aligned MPJPE in mm. Removes per-sample rigid+scale transform
    before measuring error — isolates shape accuracy from global misalignment.
    A large gap vs mpjpe_mm means error is mostly pose/scale, not landmark shape."""
    mu_p = pred_3d.mean(dim=1, keepdim=True)
    mu_g = gt_3d.mean(dim=1, keepdim=True)
    Xp = pred_3d - mu_p                               # (B, J, 3)
    Xg = gt_3d - mu_g

    C = Xg.transpose(1, 2) @ Xp                       # (B, 3, 3)
    U, S, Vh = torch.linalg.svd(C)
    det = torch.linalg.det(U @ Vh)                    # (B,)
    D = torch.eye(3, dtype=pred_3d.dtype, device=pred_3d.device)
    D = D.expand(pred_3d.shape[0], 3, 3).clone()
    D[:, 2, 2] = torch.sign(det)
    R = U @ D @ Vh                                     # (B, 3, 3)

    var_p = (Xp ** 2).sum(dim=(1, 2))                 # (B,)
    s = (S[:, 0] + S[:, 1] + torch.sign(det) * S[:, 2]) / var_p.clamp(min=1e-9)
    pred_aligned = s[:, None, None] * (Xp @ R.transpose(1, 2)) + mu_g

    err = (pred_aligned - gt_3d).norm(dim=-1)         # (B, J)
    return float(_masked_mean(err, vis) * scale)

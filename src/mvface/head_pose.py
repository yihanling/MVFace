"""6-DoF head pose from 3D facial landmarks, for OCT eye targeting.

Pipeline:
  1. eye_targets(landmarks)         -> eye centres in world/metric coords (what
                                        the OCT aims at). Straight from landmarks,
                                        no model fit.
  2. estimate_head_pose(landmarks)  -> rigid transform (R, t, s) aligning a
                                        canonical face to the observation. R is
                                        head orientation, t its reference point.
                                        Closed-form Kabsch/Umeyama, no training.

Conventions
  - Landmarks are iBUG-68 order, in metric world coords (metres in this project).
  - Pose is fit on a RIGID subset (eye corners + nose) so it is invariant to
    expression (mouth/jaw/brow motion does not move the head pose).
  - The canonical reference frame defines what "zero rotation" means; Euler
    angles are reported in that frame, so verify the reference's axis convention
    once (see head_axes / the note in rotation_to_euler).
"""
from __future__ import annotations

import torch


# iBUG-68 landmark groups
RIGHT_EYE = list(range(36, 42))
LEFT_EYE  = list(range(42, 48))

# The standard iBUG-68 partition, as (name, first, last-inclusive). Used to label
# per-joint error so a result reads "the jaw contour is worst" rather than
# "landmarks 0, 3 and 16 are worst".
LANDMARK_GROUPS = [
    ("jaw",         0, 16),
    ("right_brow", 17, 21),
    ("left_brow",  22, 26),
    ("nose_bridge", 27, 30),
    ("nose_base",  31, 35),
    ("right_eye",  36, 41),
    ("left_eye",   42, 47),
    ("mouth_outer", 48, 59),
    ("mouth_inner", 60, 67),
]


def landmark_group(idx: int) -> str:
    """iBUG-68 index -> group name."""
    for name, lo, hi in LANDMARK_GROUPS:
        if lo <= idx <= hi:
            return name
    return "unknown"
# rigid, (near) expression-invariant subset for pose fitting:
#   eye corners (36,39,42,45) + nose bridge/tip (27,28,29,30,33)
RIGID_IDX = [36, 39, 42, 45, 27, 28, 29, 30, 33]


def eye_targets(landmarks_3d: torch.Tensor) -> dict:
    """Eye centres and interocular midpoint in the landmarks' own frame.

    Args:
        landmarks_3d: (..., 68, 3) metric world coords.
    Returns:
        dict with 'right' (...,3), 'left' (...,3), 'mid' (...,3),
        and 'iod' (...,) the interocular distance.
    """
    right = landmarks_3d[..., RIGHT_EYE, :].mean(dim=-2)
    left  = landmarks_3d[..., LEFT_EYE, :].mean(dim=-2)
    mid   = 0.5 * (right + left)
    iod   = (left - right).norm(dim=-1)
    return {"right": right, "left": left, "mid": mid, "iod": iod}


def kabsch(src: torch.Tensor, dst: torch.Tensor,
           weights: torch.Tensor | None = None,
           allow_scale: bool = True):
    """Rigid/similarity fit:  find s, R, t  s.t.  s * R @ src + t ~= dst.

    Batched. src/dst are (B, N, 3); weights (B, N) optional. Returns
    R (B,3,3), t (B,3), s (B,). With allow_scale=False, s is fixed to 1.
    """
    if src.dim() == 2:
        src = src[None]; dst = dst[None]; squeeze = True
    else:
        squeeze = False
    B, N, _ = src.shape
    if weights is None:
        weights = src.new_ones(B, N)
    w = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)   # (B,N)

    mu_s = (w[..., None] * src).sum(dim=1)                # (B,3) weighted centroid
    mu_d = (w[..., None] * dst).sum(dim=1)
    Sc = src - mu_s[:, None]
    Dc = dst - mu_d[:, None]

    # weighted cross-covariance  H = Sc^T diag(w) Dc
    H = torch.einsum('bn,bni,bnj->bij', w, Sc, Dc)        # (B,3,3)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(1, 2)))   # (B,) reflection fix
    D = torch.eye(3, device=src.device, dtype=src.dtype).expand(B, 3, 3).clone()
    D[:, 2, 2] = d
    R = V @ D @ U.transpose(1, 2)                         # (B,3,3): src -> dst

    if allow_scale:
        var_s = (w * (Sc ** 2).sum(-1)).sum(dim=1)        # (B,) weighted var of src
        s = (S.sum(dim=1) * torch.where(d < 0, -1.0, 1.0).clamp(min=-1) if False
             else (S[:, 0] + S[:, 1] + d * S[:, 2])) / var_s.clamp(min=1e-9)
    else:
        s = src.new_ones(B)

    t = mu_d - s[:, None] * torch.einsum('bij,bj->bi', R, mu_s)
    if squeeze:
        return R[0], t[0], s[0]
    return R, t, s


def estimate_head_pose(landmarks_3d: torch.Tensor,
                       ref_landmarks: torch.Tensor,
                       rigid_idx: list[int] | None = None,
                       weights: torch.Tensor | None = None,
                       allow_scale: bool = True) -> dict:
    """6-DoF head pose by rigid alignment of a canonical face to the observation.

    Args:
        landmarks_3d:  (..., 68, 3) observed landmarks (metric world coords).
        ref_landmarks: (68, 3) canonical face (e.g. assets/mean_face_68.npy).
        rigid_idx:     landmark indices used for the fit; default RIGID_IDX
                       (eye corners + nose), which is expression-invariant.
        weights:       optional (..., len(rigid_idx)) per-landmark weights.
        allow_scale:   solve a uniform scale (True) or lock to the reference
                       size (False, true rigid 6-DoF).

    Returns:
        dict: R (...,3,3), t (...,3), s (...,), rmsd (...,) fit residual in the
        landmarks' units, and 'eyes' (from eye_targets on the full landmarks).
        R, t map canonical -> world:  world = s * R @ canonical + t.
    """
    if rigid_idx is None:
        rigid_idx = RIGID_IDX
    idx = torch.as_tensor(rigid_idx, device=landmarks_3d.device)

    obs = landmarks_3d.index_select(-2, idx)             # (...,K,3)
    ref = ref_landmarks.index_select(0, idx)             # (K,3)
    lead = obs.shape[:-2]
    obs_f = obs.reshape(-1, len(rigid_idx), 3)
    ref_f = ref[None].expand(obs_f.shape[0], -1, -1)
    w_f = weights.reshape(-1, len(rigid_idx)) if weights is not None else None

    R, t, s = kabsch(ref_f, obs_f, weights=w_f, allow_scale=allow_scale)

    # fit residual (RMSD) on the rigid points
    fit = s[:, None, None] * torch.einsum('bij,bnj->bni', R, ref_f) + t[:, None]
    rmsd = (fit - obs_f).norm(dim=-1).mean(dim=1)         # (prod(lead),)

    R = R.reshape((*lead, 3, 3))
    t = t.reshape((*lead, 3))
    s = s.reshape(lead)
    rmsd = rmsd.reshape(lead)
    return {"R": R, "t": t, "s": s, "rmsd": rmsd,
            "eyes": eye_targets(landmarks_3d)}


def rotation_to_euler(R: torch.Tensor, order: str = "xyz") -> torch.Tensor:
    """Euler angles (radians) from R, default intrinsic XYZ = pitch, yaw, roll.

    NOTE: the physical meaning (which way is 'yaw') depends on the canonical
    reference's axis convention. With a canonical face of +x=right, +y=up,
    +z=out-of-face:  x-rot = pitch (nod), y-rot = yaw (shake), z-rot = roll (tilt).
    Verify against your mean_face_68 orientation before trusting the labels.
    Returns (...,3) in the given order.
    """
    if order != "xyz":
        raise NotImplementedError("only intrinsic xyz implemented")
    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = sy < 1e-6
    x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    y = torch.atan2(-R[..., 2, 0], sy)
    z = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    # gimbal-lock fallback
    x_s = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    z_s = torch.zeros_like(z)
    x = torch.where(singular, x_s, x)
    z = torch.where(singular, z_s, z)
    return torch.stack([x, y, z], dim=-1)


def head_axes(R: torch.Tensor, ref_forward=(0.0, 0.0, 1.0)) -> torch.Tensor:
    """Head forward axis in world coords = R @ canonical_forward.

    This is the approximate facial-normal / OCT approach direction. Set
    ref_forward to the canonical face's out-of-face axis (default +z).
    Returns (...,3) unit vector.
    """
    f = R.new_tensor(ref_forward)
    fwd = torch.einsum('...ij,j->...i', R, f)
    return fwd / fwd.norm(dim=-1, keepdim=True).clamp(min=1e-9)

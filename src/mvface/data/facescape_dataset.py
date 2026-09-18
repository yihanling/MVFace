from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes
from torch.utils.data import Dataset

from mvface.units import MM_PER_METRE

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def denormalize_rgb(rgb_chw: torch.Tensor) -> torch.Tensor:
    """Undo the ImageNet normalization -> displayable RGB in [0,1].

    Args:
        rgb_chw: (..., 3, H, W) normalized RGB, e.g. `sample["rgbd"][:, :3]`.
    Returns:
        same shape, clamped to [0, 1], safe to hand to imshow.
    """
    mean = torch.as_tensor(IMAGENET_MEAN, device=rgb_chw.device).view(3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=rgb_chw.device).view(3, 1, 1)
    return (rgb_chw * std + mean).clamp(0, 1)


def _base_identity(name: str) -> str:
    """Identity of a subject folder. Seperate from subject variants and
    expressions: `<id>`, `<id>_<variant>` or `<id>_<expression>_<variant>`.
    """
    return name.split("_")[0]


def limit_to_subjects(folder_names, n_subjects: int):
    """A cap on subjects identities, not on item folders. 
    """
    if n_subjects <= 0:
        return list(folder_names)
    kept, seen = [], set()
    for name in folder_names:
        base = _base_identity(name)
        if base not in seen:
            if len(seen) >= n_subjects:
                continue                  # identity past the cap: drop its items
            seen.add(base)
        kept.append(name)
    return kept


def subject_train_val_split(subject_ids, val_frac: float = 0.2, seed: int = 0):
    """Partition subject ids into (train, val) with no subjects in both. Randomized by `seed`
    """
    bases = sorted({_base_identity(x) for x in subject_ids})
    random.Random(seed).shuffle(bases)
    n_val = max(1, int(round(len(bases) * val_frac)))
    val_bases = set(bases[:n_val])
    ids = sorted(subject_ids)
    train = [x for x in ids if _base_identity(x) not in val_bases]
    val = [x for x in ids if _base_identity(x) in val_bases]
    return train, val


def discover_subject_folders(root):
    """Discover all valid subject folders
    
    Returns:
        list[str]: List of valid folder names, ordered by
            (subject_id, expression_id, variant_id).
    """
    root = Path(root)
    def ok(n: str) -> bool:
        """Folder is valid if its name is '<subject_id>', '<subject_id>_<variant_id>'
        or '<subject_id>_<expression_id>_<variant_id>' -- every field all-digits.
        """
        parts = n.split("_")
        return len(parts) in (1, 2, 3) and all(p.isdigit() for p in parts)
    def key(n: str):
        """sort key by subject_id, then expression_id, then variant_id
        """
        parts = [int(p) for p in n.split("_")]
        if len(parts) == 1:
            return (parts[0], -1, -1)
        if len(parts) == 2:
            return (parts[0], -1, parts[1])
        return (parts[0], parts[1], parts[2])
    return sorted((d.name for d in root.iterdir() if d.is_dir() and ok(d.name)), key=key)


def _project_np(pts_world: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Project 3D points world -> (68,2) 2D pixels through P.
    
    Args:
        pts_world (68,3): 68 landmarks in 3D, world frame
        P (3,4): projection matrix
    
    Returns:
        (68, 2) pixel coordinates
    """
    ph = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1))], axis=1)
    uvw = ph @ P.T
    return uvw[:, :2] / uvw[:, 2:3]


class MultiViewFaceScape(Dataset):
    """PyTorch dataset representing one face seen from all N views at once.

    One item is a dict of five tensors:

    ```
    rgbd          (N, 4, H, W)  RGB in [0,1] + normalized depth as the 4th channel
    proj          (N, 3, 4)     projection matrix P = K @ [R|t] per view (t in metres)
    landmarks_3d  (68, 3)       GT in the WORLD frame, metres (shared across views)
    landmarks_2d  (N, 68, 2)    GT pixel landmarks per view
    vis           (N, 68)       per-view visibility (geometric occlusion test)
    ```
    """
    def __init__(
        self,
        root: str | Path,
        subject_ids: list[str],
        depth_scale: float = 200.0,
        vis_tol: float = 10.0,
        transform=None,
        # augmentor=None,
        # aug_deterministic: bool = False,
        # aug_seed: int = 0,
    ):
        """
        depth_scale: used to normalize depth channel to better match the rgb channels
        
        vis_tol: how far behind the surface a point can still be count as "visible"
        
        transform: easy hook for future modifications/tests, ignore for iteration 1
        """
        
        self.root = Path(root)
        self.subject_ids = list(subject_ids)
        self.depth_scale = depth_scale
        self.vis_tol = vis_tol
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subject_ids)

    def _read_view(self, vd: Path) -> dict:
        """load one camera's raw file information
        
        Args:
            vd: view directory that contains meta.json, rgb.png, depth.npy, landmarks_3d.npy, landmarks_2d.npy
        """
        meta = json.loads((vd / "meta.json").read_text())
        K = np.asarray(meta["K"], dtype=np.float64)
        rgb = np.asarray(Image.open(vd / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0
        depth = np.load(vd / "depth.npy").astype(np.float32)

        R = np.asarray(meta["R"], dtype=np.float64)
        t = np.asarray(meta["t"], dtype=np.float64).reshape(3)
        lm_cam = np.load(vd / "landmarks_3d.npy").astype(np.float64)
        uv = np.load(vd / "landmarks_2d.npy").astype(np.float64)
        return {"K": K, "R": R, "t": t, "rgb": rgb, "depth": depth,
                "lm_cam": lm_cam, "uv": uv}

    def __getitem__(self, i: int) -> dict:
        subj_dir = self.root / self.subject_ids[i]
        view_dirs = sorted((d for d in subj_dir.iterdir() if d.is_dir()),
                           key=lambda d: int(d.name))
        raws = [self._read_view(vd) for vd in view_dirs]
        
        # Establish 3D landmarks in world coordinate by inverting x_cam = R @ x_world + t
        r0 = raws[0]
        lm_world = (r0["lm_cam"] - r0["t"]) @ r0["R"]

        # per-view loop
        rgbd, proj, lm2d, vis = [], [], [], []
        for j, r in enumerate(raws):
            K, R, t = r["K"], r["R"], r["t"]
            rgb, depth, lm_cam, uv = r["rgb"], r["depth"], r["lm_cam"], r["uv"]

            face = depth > 0 # face without holes
            filled = binary_fill_holes(face) # face with holes
            holes = filled & ~face # eyes and possibly mouth if open
            med = np.median(depth[face])
            depth = np.where(holes, med, depth)

            # normalize depth
            depth_n = (depth - med) / self.depth_scale

            # ImageNet-normalize RGB
            rgb_n = (rgb - IMAGENET_MEAN) / IMAGENET_STD

            # rgb is (H,W,3); transpose to (3, H, W) first and append depth_n as channel 4.
            x = np.concatenate([rgb_n.transpose(2, 0, 1), depth_n[None]], axis=0).astype(np.float32)

            # Build projection matrix P = K @ [R|t] in raw (mm) units for the consistency
            # checks below, since lm_world/lm_cam/t are still in mm at this point.
            P_mm = K @ np.hstack([R, t[:, None]])

            # assert camera -> world inverse still correct, verify R, t
            assert np.allclose((lm_cam - t) @ R, lm_world, atol=1e-3)
            # end-to-end assert world lanmarks project onto the correct pixel with P, verify P
            assert np.allclose(_project_np(lm_world, P_mm), uv, atol=1.0)

            # Output P uses t scaled to metres to match the scaled landmarks_3d below;
            # the perspective divide cancels the shared scale factor so pixels are unaffected.
            P = K @ np.hstack([R, (t / MM_PER_METRE)[:, None]])

            # Visibility
            H, W = depth.shape
            in_img = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            # uv in integer
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
            
            at_or_front = lm_cam[:, 2] <= depth[vi, ui] + self.vis_tol
            on_hole = holes[vi, ui]
            visible = (in_img & (on_hole | at_or_front)).astype(np.float32)
            
            rgbd.append(x)
            proj.append(P)
            lm2d.append(uv)
            vis.append(visible)

        sample = {
            "rgbd": torch.from_numpy(np.stack(rgbd)).float(),
            "proj": torch.from_numpy(np.stack(proj)).float(),
            "landmarks_3d": torch.from_numpy(lm_world / MM_PER_METRE).float(),  # mm -> m
            "landmarks_2d": torch.from_numpy(np.stack(lm2d)).float(),
            "vis": torch.from_numpy(np.stack(vis)).float(),
        }
        # ignore this for now
        if self.transform is not None:
            sample = self.transform(sample, i)
            
        return sample

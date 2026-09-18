"""
FaceScape multi-view dataloader that outputs the clean dict interface expected by
the colleague's model (backbone.py / decoder.py / model.py), while reusing the
proven coordinate / depth / crop logic from the original facescape_multiview.py.

Per-sample output (default-collated into a leading batch dim B):

    rgbd          (N, 4, Ht, Wt)  RGB (ImageNet-normalized) + normalized depth 4th channel
    depth_raw     (N, Ht, Wt)     RAW metric depth in mm (0 = background/invalid) for late fusion
    proj          (N, 3, 4)       P' = K' @ [R | t], maps centered-world mm -> resized pixels
    landmarks_3d  (68, 3)         GT landmarks, centered-world TU-scale mm (shared across views)
    landmarks_2d  (N, 68, 2)      GT pixel landmarks in resized space, per view
    vis           (N, 68)         per-view visibility (in-bounds & in-front & not occluded)

Fixed image size is IMAGE_SIZE = (Ht, Wt); pass this as `image_hw` to model.forward.

KEY CONVENTIONS (these are the ones that have bitten us before):
  * Landmarks are TU-scale mm, per-capture centered (subtract face centroid).
    Cameras are re-centered to match via global_center so projection stays valid.
  * proj is a TRUE pinhole P' = K' @ [R | t]. We adjust K for crop+resize instead
    of composing an affine, so P'[2] stays exactly the camera z-axis. This is what
    makes the late-fusion depth constraint (p3 . Xh = z_cam) correct.
  * depth_raw is eye-space z (pyrender returns z along the optical axis, NOT radial)
    in mm, resized with NEAREST so values are not blended. 0 means no surface.
  * The backbone's 4th channel is (depth - median_face) / depth_scale; late fusion
    uses depth_raw, NOT this normalized channel.
"""

from __future__ import annotations

import os
import sys as _sys
_RF_ROOT = '/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface'
if _RF_ROOT not in _sys.path:
    _sys.path.insert(0, _RF_ROOT)
import json
import logging
import random

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

from mvface.units import MM_PER_METRE

logger = logging.getLogger(__name__)

NUM_LANDMARKS      = 68
NUM_VIEWS_SAMPLE   = 5
SPLIT_AT_SUBJ      = 300
ROOT_LANDMARK      = 30
IMAGE_SIZE         = (256, 256)   # (Ht, Wt) — pass as image_hw to model.forward
DEPTH_SCALE        = 200.0        # for the normalized backbone channel only
VIS_TOL_MM         = 10.0         # depth occlusion tolerance in mm

INVALID_CAM_INDICES = {45, 46, 49, 50, 51, 52, 57}

LM_INDICES = [
    5696, 23350, 5702,  4651,  4650, 20322, 21351,  5013,  1681,  1692,
   11486, 10439, 1338,  1339,  2369, 13524,  2363, 24759,  3549, 24702,
   24687, 24632,14837, 14899, 14914,   237, 14968,  6053,  6041,  1870,
    1855,  4728, 4870,  1807,  1551,  1419,  3434,  3414,  3447,  3457,
    3309,  3373, 3179,   151,   127,   143,  3236,    47, 21018,  4985,
    4898,  6571, 1575,  1663,  1599,  1899, 12138,  5231, 21978,  5101,
   21067, 21239,11378, 11369, 11553, 12048,  5212, 21892,
]


# ── TU landmark loading (unchanged logic) ─────────────────────────────────────

def _load_tu_landmarks_world(obj_path, scale, Rt):
    verts = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
    verts = np.array(verts, dtype=np.float32)
    tu_lm = verts[LM_INDICES]
    Rt = np.array(Rt, dtype=np.float64)
    lm_world = (Rt[:3, :3].T @ (tu_lm.astype(np.float64) - Rt[:3, 3]).T).T / scale
    return lm_world.astype(np.float32)  # (68,3) meters


def _tu_landmarks_scaled(obj_path, scale, Rt):
    lm_m = _load_tu_landmarks_world(obj_path, scale, Rt)
    return (lm_m * scale).astype(np.float32)  # (68,3) TU-scale mm


# ── camera parsing (unchanged logic) ──────────────────────────────────────────

def _parse_params_json(params_path, scale_factor, global_center=None):
    with open(params_path, 'r') as f:
        raw = json.load(f)

    view_indices = set()
    for key in raw:
        if key.endswith('_K'):
            try:
                view_indices.add(int(key[:-2]))
            except ValueError:
                pass

    cameras = {}
    for i in sorted(view_indices):
        if i in INVALID_CAM_INDICES:
            continue
        if not raw.get(f'{i}_valid', False):
            continue

        K    = np.array(raw[f'{i}_K'],  dtype=np.float64)[:3, :3]
        Rt   = np.array(raw[f'{i}_Rt'], dtype=np.float64)
        R    = Rt[:, :3]
        t    = Rt[:, 3] * scale_factor
        if global_center is not None:
            t = t + R @ global_center.astype(np.float64)
        dist = np.array(raw[f'{i}_distortion'], dtype=np.float64)
        W    = int(raw[f'{i}_width'])
        H    = int(raw[f'{i}_height'])

        cameras[i] = {
            'R': R, 't': t, 't_raw': Rt[:, 3].copy(), 'K': K, 'distCoef': dist, 'width': W, 'height': H,
        }
    return cameras


# ── depth rendering (returns eye-space z in mm) ───────────────

def _render_depth(obj_path, cam, scale, Rt_tu):
    """
    Render eye-space z-depth (mm) at the camera's native resolution.

    pyrender returns depth as z along the optical axis (NOT radial distance),
    which is exactly what the late-fusion constraint needs. 0 = no surface hit.
    """
    try:
        os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
        import pyrender, trimesh
        mesh  = trimesh.load(obj_path, force="mesh")
        V = np.array(mesh.vertices)
        R_tu = np.array(Rt_tu)[:3,:3]; t_tu = np.array(Rt_tu)[:3,3]
        mesh.vertices = (R_tu.T @ (V - t_tu).T).T / scale
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(mesh))

        R = cam['R']
        t = cam["t_raw"].reshape(3)
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = R.T
        cam_pose[:3,  3] = -R.T @ t
        cam_pose[:, 1] *= -1
        cam_pose[:, 2] *= -1

        camera = pyrender.IntrinsicsCamera(
            fx=float(cam['K'][0, 0]), fy=float(cam['K'][1, 1]),
            cx=float(cam['K'][0, 2]), cy=float(cam['K'][1, 2]),
            znear=0.01, zfar=100.0)
        scene.add(camera, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(cam['width'], cam['height'])
        _, depth = renderer.render(scene)
        renderer.delete()
        return (depth * MM_PER_METRE).astype(np.float32)  # meters -> mm
    except Exception as e:
        logger.warning(f'Depth render failed: {e}')
        return np.zeros((cam['height'], cam['width']), dtype=np.float32)


# ── intrinsics adjustment for crop + resize ───────────────────────────────────

def _adjust_K(K, x1, y1, sx, sy):
    """Return K' for a crop at (x1,y1) followed by a resize with scales (sx,sy).

    Keeps K' a proper upper-triangular pinhole intrinsic, so P' = K' @ [R|t]
    still has P'[2] == camera z-axis (required for the depth constraint).
    """
    Kp = K.copy()
    Kp[0, 2] -= x1
    Kp[1, 2] -= y1
    Kp[0, 0] *= sx; Kp[0, 2] *= sx
    Kp[1, 1] *= sy; Kp[1, 2] *= sy
    return Kp


class FaceScapeMultiView(Dataset):
    """Outputs the colleague's model interface; reuses our proven data logic."""

    def __init__(self, cfg, image_set, is_train, transform=None,
                 render_depth=True, depth_cache_dir=None):
        super().__init__()
        self.cfg            = cfg
        self.is_train       = is_train
        self.render_depth    = render_depth
        self.depth_cache_dir = depth_cache_dir
        self.num_views      = getattr(cfg.DATASET, 'NUM_VIEWS', NUM_VIEWS_SAMPLE)
        self.use_retinaface = getattr(cfg.DATASET, 'USE_RETINAFACE', False)
        self.image_size     = tuple(getattr(cfg.NETWORK, 'IMAGE_SIZE', IMAGE_SIZE))  # (Ht, Wt)
        self.depth_scale    = getattr(cfg.DATASET, 'DEPTH_SCALE', DEPTH_SCALE)

        self.data_root  = os.path.join(cfg.DATASET.ROOT, 'multi_view_data')
        self.image_root = os.path.join(self.data_root, 'image')
        self.tu_root    = os.path.join(self.data_root, 'tu')

        with open(os.path.join(self.tu_root, 'Rt_scale_dict.json')) as f:
            self.rt_scale_dict = json.load(f)

        self.detector = None
        if self.use_retinaface:
            self.detector = self._init_retinaface(cfg)

        self.db = self._build_db(is_train)
        split = 'train' if is_train else 'val'
        logger.info(
            f'FaceScapeMultiView {split}: {len(self.db)} captures, '
            f'retinaface={self.use_retinaface}, views={self.num_views}, '
            f'image_size={self.image_size}'
        )

    # ── RetinaFace (unchanged) ────────────────────────────────────────────────

    def _init_retinaface(self, cfg):
        try:
            import sys
            retinaface_root = getattr(cfg.DATASET, 'RETINAFACE_ROOT',
                                      '/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface')
            checkpoint = getattr(cfg.DATASET, 'RETINAFACE_CHECKPOINT',
                                 '/nfs/turbo/coe-igmr-pub/seoin/trained_weights/Resnet50_Final.pth')
            sys.path.insert(0, retinaface_root)
            from models.retinaface import RetinaFace
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location('retinaface_config',
                '/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface/data/config.py')
            _rcfg = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_rcfg)
            cfg_re50 = _rcfg.cfg_re50
            net = RetinaFace(cfg=cfg_re50, phase='test')
            net.load_state_dict(torch.load(checkpoint, map_location='cpu'))
            net = net.float()  # force fp32; checkpoint loaded some params as fp64
            net.eval()
            logger.info('RetinaFace loaded')
            return net
        except Exception as e:
            logger.warning(f'RetinaFace init failed: {e}. Using full frame.')
            return None

    def _retinaface_bbox(self, img, cam):
        """Return (x1, y1, x2, y2) crop box in pixels, or full frame on failure."""
        try:
            from layers.functions.prior_box import PriorBox
            from utils.box_utils import decode
            from utils.nms.py_cpu_nms import py_cpu_nms
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location('retinaface_config',
                '/nfs/turbo/coe-igmr-pub/seoin/Pytorch_Retinaface/data/config.py')
            _rcfg = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_rcfg)
            cfg_re50 = _rcfg.cfg_re50

            device = next(self.detector.parameters()).device
            h, w = img.shape[:2]
            scale_img = torch.Tensor([w, h, w, h]).to(device)
            img_t = (np.float32(img) - (104, 117, 123)).astype(np.float32)
            img_t = torch.from_numpy(img_t.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                loc, conf, _ = self.detector(img_t)
            priorbox = PriorBox(cfg_re50, image_size=(h, w))
            priors   = priorbox.forward().to(device)
            boxes    = decode(loc.squeeze(0), priors, cfg_re50['variance'])
            boxes    = (boxes * scale_img).cpu().numpy()
            scores   = conf.squeeze(0).cpu().numpy()[:, 1]
            keep     = py_cpu_nms(np.hstack([boxes, scores[:, None]]).astype(np.float32), 0.4)
            boxes, scores = boxes[keep], scores[keep]
            if len(boxes) == 0:
                return 0, 0, w, h
            x1, y1, x2, y2 = boxes[np.argmax(scores)].astype(int)
            pad_x = int((x2 - x1) * 0.2); pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x); y2 = min(h, y2 + pad_y)
            if x2 <= x1 or y2 <= y1:
                return 0, 0, w, h
            return x1, y1, x2, y2
        except Exception as e:
            logger.warning(f'RetinaFace detect failed: {e}')
            h, w = img.shape[:2]
            return 0, 0, w, h

    # ── db build ────────────────────────────────────────────

    def _build_db(self, is_train):
        db = []
        subject_dirs = sorted(
            int(d) for d in os.listdir(self.image_root)
            if os.path.isdir(os.path.join(self.image_root, d)) and d.isdigit()
        )
        for subj in subject_dirs:
            if is_train     and subj >= SPLIT_AT_SUBJ: continue
            if not is_train and subj <  SPLIT_AT_SUBJ: continue

            subj_str = str(subj)
            if subj_str not in self.rt_scale_dict:
                continue
            subj_image_dir = os.path.join(self.image_root, subj_str)
            subj_tu_dir    = os.path.join(self.tu_root, subj_str, 'models_reg')
            if not os.path.isdir(subj_tu_dir):
                continue

            for expr in sorted(os.listdir(subj_image_dir)):
                expr_image_dir = os.path.join(subj_image_dir, expr)
                if not os.path.isdir(expr_image_dir):
                    continue
                expr_idx = expr.split('_')[0]
                if expr_idx not in self.rt_scale_dict[subj_str]:
                    continue
                params_path = os.path.join(expr_image_dir, 'params.json')
                obj_path    = os.path.join(subj_tu_dir, f'{expr}.obj')
                if not os.path.isfile(params_path) or not os.path.isfile(obj_path):
                    continue

                entry = self.rt_scale_dict[subj_str][expr_idx]
                try:
                    lm_check  = _load_tu_landmarks_world(obj_path, entry[0], entry[1])
                    if np.linalg.norm(lm_check[45] - lm_check[36]) > 1.0:
                        continue  # bad registration (IOD > 1m)
                except Exception:
                    continue

                view_indices = sorted(
                    int(os.path.splitext(f)[0])
                    for f in os.listdir(expr_image_dir) if f.endswith('.jpg')
                )
                if len(view_indices) < self.num_views:
                    continue

                db.append({
                    'subject': subj, 'expression': expr,
                    'image_dir': expr_image_dir, 'params_path': params_path,
                    'obj_path': obj_path, 'all_views': view_indices,
                    'scale': entry[0], 'Rt': entry[1],
                })
        return db

    def __len__(self):
        return len(self.db)

    # ── per-view processing ───────────────────────────────────────────────────

    def _process_view(self, rec, cam, img_path, lm_world_centered, view_id=None):
        """Return (rgbd_chw, depth_raw_resized, P_prime, lm2d, vis) for one view."""
        Ht, Wt = self.image_size

        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f'Cannot read: {img_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.undistort(img, cam['K'].astype(np.float32), cam['distCoef'].astype(np.float32))

        # render depth at native resolution (before crop/resize)
        # acquire depth: cached npy > rendered > zeros (RGB-only fast path)
        if self.depth_cache_dir is not None and view_id is not None:
            # prerender_depth.py layout: {cache}/{subject}/{expression}/{view}.npy
            # stored as float16 in mm at original resolution.
            cpath = os.path.join(self.depth_cache_dir, str(rec['subject']),
                                 str(rec['expression']), f'{view_id}.npy')
            if os.path.isfile(cpath):
                depth_full = np.load(cpath).astype(np.float32)  # mm
            else:
                depth_full = _render_depth(rec["obj_path"], cam, rec["scale"], rec["Rt"])
        elif self.render_depth:
            depth_full = _render_depth(rec["obj_path"], cam, rec["scale"], rec["Rt"])  # (H0, W0) mm
        else:
            depth_full = np.zeros((cam['height'], cam['width']), dtype=np.float32)

        # crop box
        if self.use_retinaface and self.detector is not None:
            x1, y1, x2, y2 = self._retinaface_bbox(img, cam)
        else:
            x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]

        img_crop   = img[y1:y2, x1:x2]
        depth_crop = depth_full[y1:y2, x1:x2]
        crop_h, crop_w = img_crop.shape[:2]

        sx = Wt / float(crop_w)
        sy = Ht / float(crop_h)

        img_resized   = cv2.resize(img_crop, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        depth_resized = cv2.resize(depth_crop, (Wt, Ht), interpolation=cv2.INTER_NEAREST)

        # adjusted intrinsics + true pinhole projection matrix
        Kp = _adjust_K(cam['K'], x1, y1, sx, sy)
        t_scaled = cam['t'].reshape(3, 1) / MM_PER_METRE
        Rt = np.hstack([cam['R'], t_scaled])
        P_prime = (Kp @ Rt).astype(np.float32)  # (3,4) centered-world (m) -> resized px

        # RGB: ImageNet normalize, CHW
        rgb = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb  = (rgb - mean) / std
        rgb_chw = rgb.transpose(2, 0, 1)

        # normalized depth channel for the backbone (median over face region)
        face = depth_resized > 0
        if face.any():
            med = np.median(depth_resized[face])
        else:
            med = 0.0
        depth_norm = np.where(face, (depth_resized - med) / self.depth_scale, 0.0).astype(np.float32)

        rgbd_chw = np.concatenate([rgb_chw, depth_norm[None]], axis=0).astype(np.float32)

        # GT 2D landmarks in resized pixel space + camera-space z (for occlusion test)
        ph      = np.concatenate([lm_world_centered, np.ones((NUM_LANDMARKS, 1))], axis=1)
        uvw     = ph @ P_prime.T
        z_cam   = uvw[:, 2]
        lm2d    = (uvw[:, :2] / uvw[:, 2:3]).astype(np.float32)

        # visibility: in-front & in-bounds & not occluded by rendered surface
        u = lm2d[:, 0]; v = lm2d[:, 1]
        in_front  = z_cam > 0
        in_bounds = (u >= 0) & (u < Wt) & (v >= 0) & (v < Ht)
        ui = np.clip(np.round(u).astype(int), 0, Wt - 1)
        vi = np.clip(np.round(v).astype(int), 0, Ht - 1)
        surf = depth_resized[vi, ui] / MM_PER_METRE  # mm -> m to match z_cam
        # visible if no surface recorded there (background/hole) OR landmark is at/in front of surface
        not_occluded = (surf <= 0) | (z_cam <= surf + VIS_TOL_MM / MM_PER_METRE)
        vis = (in_front & in_bounds & not_occluded).astype(np.float32)

        return rgbd_chw, (depth_resized / MM_PER_METRE).astype(np.float32), P_prime, lm2d, vis

    def __getitem__(self, idx):
        rec = self.db[idx]

        joints_3d   = _tu_landmarks_scaled(rec['obj_path'], rec['scale'], rec['Rt'])
        face_center = joints_3d.mean(axis=0)
        joints_3d   = joints_3d - face_center  # per-capture centering
        joints_3d   = joints_3d / MM_PER_METRE  # mm -> m (keeps proj well-conditioned)

        all_cameras = _parse_params_json(rec['params_path'], scale_factor=rec['scale'],
                                         global_center=face_center)
        valid_views = [v for v in rec['all_views'] if v in all_cameras]
        if len(valid_views) < self.num_views:
            valid_views = (valid_views * ((self.num_views // max(1, len(valid_views))) + 1))[:self.num_views]

        if self.is_train:
            chosen = random.sample(valid_views, self.num_views)
        else:
            chosen = valid_views[:self.num_views]

        rgbd, depth_raw, proj, lm2d, vis = [], [], [], [], []
        for v in chosen:
            cam      = all_cameras[v]
            img_path = os.path.join(rec['image_dir'], f'{v}.jpg')
            r_chw, d_raw, P, l2d, vs = self._process_view(rec, cam, img_path, joints_3d, view_id=v)
            rgbd.append(r_chw); depth_raw.append(d_raw); proj.append(P)
            lm2d.append(l2d);   vis.append(vs)

        return {
            'rgbd':         torch.from_numpy(np.stack(rgbd)).float(),        # (N,4,Ht,Wt)
            'depth_raw':    torch.from_numpy(np.stack(depth_raw)).float(),   # (N,Ht,Wt) mm (raw, NOT divided by MM_PER_METRE)
            'proj':         torch.from_numpy(np.stack(proj)).float(),        # (N,3,4)
            'landmarks_3d': torch.from_numpy(joints_3d).float(),             # (68,3) m  (divided by MM_PER_METRE)
            'landmarks_2d': torch.from_numpy(np.stack(lm2d)).float(),        # (N,68,2)
            'vis':          torch.from_numpy(np.stack(vis)).float(),         # (N,68)
        }

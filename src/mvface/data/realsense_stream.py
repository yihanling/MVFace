"""Live 3x RealSense D435i streaming loader for MVFace inference.

Yields the tensors model.forward expects, with preprocessing IDENTICAL to
data/facescape_multiview.py so a FaceScape-trained model stays in-distribution:

    rgbd       (V, 4, 256, 256)  RGB (ImageNet-normalized) + normalized-depth channel
    depth_raw  (V, 256, 256)     metric depth in METRES (for late-fusion depth rows)
    proj       (V, 3, 4)         K' @ [R|t], centered-world metres -> resized pixels

Frame conventions match training exactly:
  * RGB: BGR->RGB, undistort(K,dist), crop, resize LINEAR, /255, ImageNet norm, CHW.
  * depth: RealSense raw * depth_scale = METRES. depth_raw is metres; the backbone
    4th channel is (depth_mm - median_face_mm) / DEPTH_SCALE, NEAREST-resized.
  * proj: _adjust_K for crop+resize (keeps p3 == camera z-axis), t from rig_calib.
  * world_center: training centers each face at the origin. At inference we can't
    center per-face, so we shift the world origin to a fixed expected head location
    (metres, board frame) so predictions land in the same range the model trained on.
    Camera-space z is invariant to this shift, so depth_raw stays consistent.

Sync: hardware genlock (one master inter_cam_sync_mode=1, others=2 through the 4-pin header). 
Software timestamp-matching is applied regardless as a safety net.

RealSense I/O (pyrealsense2) can only run with hardware attached
process_view and the proj math are unit-tested
Verify the streaming loop on-device
"""
from __future__ import annotations

import json
import time

import numpy as np
import cv2
import torch

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from mvface.units import MM_PER_METRE

IMAGE_SIZE  = (256, 256)     # (Ht, Wt) — match training
DEPTH_SCALE = 200.0          # backbone 4th-channel normalization (mm), from training
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _adjust_K(K, x1, y1, sx, sy):
    """from facescape_multiview: K' for crop at (x1,y1) then resize (sx,sy)."""
    Kp = K.copy()
    Kp[0, 2] -= x1
    Kp[1, 2] -= y1
    Kp[0, 0] *= sx; Kp[0, 2] *= sx
    Kp[1, 1] *= sy; Kp[1, 2] *= sy
    return Kp


def process_view(color_bgr, depth_units, depth_scale, K, dist, R, t,
                 image_size=IMAGE_SIZE, world_center=None, crop=None):
    """Preprocess one camera's frame into (rgbd_chw, depth_m_resized, P_prime).

    Args:
        color_bgr:   (H, W, 3) uint8 BGR from RealSense color stream.
        depth_units: (H, W) uint16 raw depth, aligned to color.
        depth_scale: metres per raw unit (device.first_depth_sensor().get_depth_scale()).
        K:           (3, 3) color intrinsics.
        dist:        (5,) color distortion coeffs.
        R, t:        (3,3),(3,) world->camera extrinsics from rig_calib (metres).
        world_center:(3,) expected head location in board frame (metres) or None.
        crop:        (x1,y1,x2,y2) face crop, or None for full frame.

    Returns:
        rgbd_chw (4,Ht,Wt) float32, depth_m (Ht,Wt) float32 metres, P (3,4) float32.
    """
    Ht, Wt = image_size
    K = np.asarray(K, np.float64); dist = np.asarray(dist, np.float64)
    R = np.asarray(R, np.float64); t = np.asarray(t, np.float64).reshape(3, 1)

    img = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.undistort(img, K.astype(np.float32), dist.astype(np.float32))

    depth_m_full  = depth_units.astype(np.float32) * float(depth_scale)   # metres
    depth_mm_full = depth_m_full * MM_PER_METRE                           # mm

    if crop is None:
        x1, y1, x2, y2 = 0, 0, img.shape[1], img.shape[0]
    else:
        x1, y1, x2, y2 = crop

    img_c  = img[y1:y2, x1:x2]
    dmm_c  = depth_mm_full[y1:y2, x1:x2]
    dm_c   = depth_m_full[y1:y2, x1:x2]
    ch, cw = img_c.shape[:2]
    sx, sy = Wt / float(cw), Ht / float(ch)

    img_r = cv2.resize(img_c, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
    dmm_r = cv2.resize(dmm_c, (Wt, Ht), interpolation=cv2.INTER_NEAREST)
    dm_r  = cv2.resize(dm_c,  (Wt, Ht), interpolation=cv2.INTER_NEAREST)

    # proj: adjusted K @ [R|t], with optional world-origin shift (camera z invariant)
    Kp = _adjust_K(K, x1, y1, sx, sy)
    tt = t.copy()
    if world_center is not None:
        tt = tt + R @ np.asarray(world_center, np.float64).reshape(3, 1)
    P = (Kp @ np.hstack([R, tt])).astype(np.float32)

    # RGB: ImageNet-normalized CHW
    rgb = img_r.astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    rgb_chw = rgb.transpose(2, 0, 1)

    # backbone 4th channel: (depth_mm - median_face_mm) / DEPTH_SCALE, 0 for holes
    face = dmm_r > 0
    med  = np.median(dmm_r[face]) if face.any() else 0.0
    depth_norm = np.where(face, (dmm_r - med) / DEPTH_SCALE, 0.0).astype(np.float32)

    rgbd_chw = np.concatenate([rgb_chw, depth_norm[None]], axis=0).astype(np.float32)
    return rgbd_chw, dm_r.astype(np.float32), P


# ---------------------------------------------------------------------------
# Live streaming rig (needs hardware)
# ---------------------------------------------------------------------------
class RealSenseRig:
    def __init__(self, rig_calib_path, world_center=None,
                 width=1280, height=720, fps=30, sync_tol_ms=8.0,
                 hw_sync=True, detector=None):
        if rs is None:
            raise RuntimeError("pyrealsense2 not installed")
        calib = json.load(open(rig_calib_path))
        assert calib.get("units") == "m", "rig_calib must be in metres"
        self.order = calib["view_order"]
        self.cams  = calib["cameras"]
        self.world_center = None if world_center is None else np.asarray(world_center, np.float64)
        self.sync_tol_ms = sync_tol_ms
        self.detector = detector          # optional RetinaFace-style bbox provider
        self.image_size = IMAGE_SIZE

        self.pipes, self.aligners, self.depth_scales = {}, {}, {}
        for i, serial in enumerate(self.order):
            pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16,  fps)
            profile = pipe.start(cfg)
            dsensor = profile.get_device().first_depth_sensor()
            if hw_sync and dsensor.supports(rs.option.inter_cam_sync_mode):
                # first camera = master (1), rest = slaves (2). Requires sync cable.
                dsensor.set_option(rs.option.inter_cam_sync_mode, 1 if i == 0 else 2)
            self.depth_scales[serial] = dsensor.get_depth_scale()
            self.pipes[serial]    = pipe
            self.aligners[serial] = rs.align(rs.stream.color)  # depth -> color
        time.sleep(1.0)  # let auto-exposure settle

    def _grab(self, serial):
        frames = self.pipes[serial].wait_for_frames()
        frames = self.aligners[serial].process(frames)
        c = frames.get_color_frame(); d = frames.get_depth_frame()
        if not c or not d:
            return None
        return (np.asanyarray(c.get_data()),
                np.asanyarray(d.get_data()),
                c.get_timestamp())

    def read(self):
        """Grab one synchronized set across all cameras -> model ready dict or None."""
        grabs = {s: self._grab(s) for s in self.order}
        if any(v is None for v in grabs.values()):
            return None
        ts = [grabs[s][2] for s in self.order]
        if max(ts) - min(ts) > self.sync_tol_ms:
            return None  # out of sync this cycle; caller retries

        rgbd, depth_raw, proj = [], [], []
        for serial in self.order:
            color, depth_u, _ = grabs[serial]
            c = self.cams[serial]
            crop = self.detector(color) if self.detector is not None else None
            r, dm, P = process_view(
                color, depth_u, self.depth_scales[serial],
                np.array(c["K"]), np.array(c.get("dist", np.zeros(5))),
                np.array(c["R"]), np.array(c["t"]),
                image_size=self.image_size, world_center=self.world_center, crop=crop)
            rgbd.append(r); depth_raw.append(dm); proj.append(P)

        return {
            "rgbd":      torch.from_numpy(np.stack(rgbd)).float(),        # (V,4,H,W)
            "depth_raw": torch.from_numpy(np.stack(depth_raw)).float(),   # (V,H,W) metres
            "proj":      torch.from_numpy(np.stack(proj)).float(),        # (V,3,4)
            "image_hw":  self.image_size,
        }

    def stream(self, max_retries=5):
        """Generator of synchronized frame dicts, skips out-of-sync cycles."""
        while True:
            out, tries = None, 0
            while out is None and tries < max_retries:
                out = self.read(); tries += 1
            if out is None:
                continue
            yield out

    def close(self):
        for p in self.pipes.values():
            p.stop()

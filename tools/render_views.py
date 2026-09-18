"""Render FaceScape TU meshes through virtual cameras (RGB + depth + landmarks)."""

import _init_paths  # noqa: F401
import argparse
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import cv2
import trimesh
import OpenGL
import pyrender          # run with PYOPENGL_PLATFORM=egl for headless GPU
from PIL import Image
from tqdm import tqdm

if tuple(int(x) for x in OpenGL.__version__.split(".")[:3]) < (3, 1, 9):
    raise ImportError(
        f"PyOpenGL {OpenGL.__version__} crashes when rendering textured meshes. "
        'Run `bash setup_render.sh`, or `pip install --upgrade "PyOpenGL>=3.1.9"`.'
    )

from _init_paths import REPO_ROOT
from mvface.data.augment import AugConfig, MultiViewAugmentor
from mvface.expressions import parse_expression_spec, stem

ASSETS_DIR = REPO_ROOT / "src/mvface/assets"

# Files every finished view directory must contain; used for the resume check.
VIEW_FILES = ("rgb.png", "depth.npy", "landmarks_2d.npy", "landmarks_3d.npy", "meta.json")


@dataclass
class Camera:
    id: str
    W: int
    H: int
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.t = np.asarray(self.t, dtype=float)


@dataclass
class Light:
    intensity: float
    ambient: float
    direction: np.ndarray

    def __post_init__(self):
        self.direction = np.asarray(self.direction, dtype=float)


def look_at_cv(eye, target, up=(0.0, 1.0, 0.0)):   # +Y up in the TU frame
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)            # +Z: forward, toward the target
    x = np.cross(z, up)
    x /= np.linalg.norm(x)            # +X: right  (cross(z,up), so world-up stays up)
    y = np.cross(z, x)                # +Y: down  (right-handed: y = z x x)
    R_c2w = np.stack([x, y, z], axis=1)   # columns are the camera axes in world coords
    R = R_c2w.T                       # world->camera rotation
    t = -R @ eye                      # world->camera translation
    return R, t


def intrinsics_from_fov(fov_deg, W, H):
    fx = fy = (W / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1.0]])


def light_pose_from_direction(direction):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    z = -d                                       # local +Z is opposite the travel dir
    up = np.array([0.0, 1.0, 0.0])
    if abs(z @ up) > 0.99:                        # light almost vertical -> new up
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, :3] = np.stack([x, y, z], axis=1)   # columns = light axes in world
    return pose


def rotmat_to_quat(R):
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0                 # s = 4*w
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0   # s = 4*x
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0   # s = 4*y
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0   # s = 4*z
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def item_seed(base_seed, id, exp_id, k):
    """Deterministic per-item seed.
        For interuption or need to resume
    """
    return (int(base_seed) * 1_000_003 + int(id) * 10_007
            + int(exp_id) * 101 + int(k)) % (2 ** 32)


class ViewRenderer:
    def __init__(self,
                 data_root="data/facescape",
                 out_root="data/facescape/virtual_camera_expr",
                 assets_dir=ASSETS_DIR):
        self.data_root = Path(data_root)
        self.out_root = Path(out_root)
        # (68,) vertex indices into the TU mesh, shipped as package data rather
        # than read from the old repo's third_party/ checkout.
        self.lm_idx = np.load(Path(assets_dir) / "landmark_indices.npz")["v10"]

    def find_id_range_folder(self, id):
        n = int(id)
        for d in self.data_root.iterdir():
            parts = d.name.split("_")          # a bucket is exactly <digits>_<digits>
            if not d.is_dir() or len(parts) != 2 or not all(p.isdigit() for p in parts):
                continue                       # skips _selftest, virtual_camera_data, ...
            lo, hi = int(parts[0]), int(parts[1])
            if lo <= n <= hi:
                return d.name

        raise FileNotFoundError(f"no folder contains id {id} under {self.data_root}")

    def mesh_path(self, id, exp_stem):
        return self.data_root / self.find_id_range_folder(id) / id / f"{exp_stem}.obj"

    def load_mesh(self, id, exp="1_neutral"):
        obj = self.mesh_path(id, exp)
        mesh = trimesh.load(obj, process=False)   # process=False keeps vertex order
        if isinstance(mesh, trimesh.Scene):       # rare, but guard it
            mesh = mesh.dump(concatenate=True)

        mat = mesh.visual.material
        if hasattr(mat, "baseColorFactor"):
            mat.baseColorFactor = np.array([255, 255, 255, 255], dtype=np.uint8)
        if hasattr(mat, "diffuse"):
            mat.diffuse = np.array([255, 255, 255, 255], dtype=np.uint8)

        raw = np.array([ln.split()[1:4] for ln in obj.read_text().splitlines()
                        if ln.startswith("v ")], dtype=float)   # (26317, 3), file order
        mesh.metadata["lm_world"] = raw[self.lm_idx]            # (68, 3)
        return mesh

    def orient_head(self, mesh, roll=0.0, pitch=0.0, yaw=0.0):
        pivot = mesh.vertices.mean(axis=0)     # head origin = centroid, in world coords
        if roll == 0 and pitch == 0 and yaw == 0:
            mesh.metadata["head_R"] = np.eye(3)   # head->world rotation (identity at 0,0,0)
            mesh.metadata["head_t"] = pivot       # head origin in world frame
            return mesh
        a_pitch = np.radians(-pitch)   # +pitch = up               (nose toward +Y)
        a_yaw = np.radians(-yaw)       # +yaw   = subject's right   (nose toward -X)
        a_roll = np.radians(roll)      # +roll  = subject's right   (top toward -X)
        cx, sx = np.cos(a_pitch), np.sin(a_pitch)
        cy, sy = np.cos(a_yaw), np.sin(a_yaw)
        cz, sz = np.cos(a_roll), np.sin(a_roll)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])   # pitch about X (lateral)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])   # yaw   about Y (up)
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])   # roll  about Z (forward)
        R = Ry @ Rx @ Rz                       # yaw-pitch-roll intrinsic order
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pivot - R @ pivot           # rotate about `pivot`, not the origin
        mesh.apply_transform(T)                # mutates in place -> reload per orientation
        lm = mesh.metadata["lm_world"]         # carry the landmarks through the same transform
        mesh.metadata["lm_world"] = (R @ lm.T).T + T[:3, 3]
        mesh.metadata["head_R"] = R            # head->world rotation
        mesh.metadata["head_t"] = pivot        # head origin (centroid) in world
        return mesh

    # ---- helper: OpenCV (R,t) -> pyrender's OpenGL pose -----------------
    def _gl_pose(self, R, t):
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        return c2w @ np.diag([1.0, -1.0, -1.0, 1.0])

    def render(self, mesh, cam, light: Light = None):
        scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[light.ambient if light else 0.5]*3)
        scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        pcam = pyrender.IntrinsicsCamera(fx=cam.K[0, 0], fy=cam.K[1, 1],
                                         cx=cam.K[0, 2], cy=cam.K[1, 2],
                                         znear=1.0, zfar=5000.0)
        pose = self._gl_pose(cam.R, cam.t)
        scene.add(pcam, pose=pose)

        if light is None:
            scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
        else:
            scene.add(pyrender.DirectionalLight(intensity=light.intensity), pose=light_pose_from_direction(light.direction))

        renderer = pyrender.OffscreenRenderer(cam.W, cam.H)
        color, depth = renderer.render(scene)     # (H,W,3) uint8, (H,W) float32
        renderer.delete()
        return color, depth

    def project_landmarks(self, mesh, cam):
        pts = mesh.metadata["lm_world"]   # (68,3) carried from the raw 'v' order
        x_cam = (cam.R@pts.T).T + cam.t
        fx, fy = cam.K[0,0], cam.K[1,1]
        cx, cy = cam.K[0,2], cam.K[1,2]
        x, y, z = x_cam.T
        u = fx*x/z + cx
        v = fy*y/z + cy
        return np.stack([u, v], axis=1)

    def backproject(self, depth, K, color=None):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        H, W = depth.shape
        vv, uu = np.mgrid[0:H, 0:W]          # vv = row index (v), uu = col index (u)
        mask = depth > 0                     # hit pixels only (0 = miss)
        u = uu[mask].astype(float)           # (N,)
        v = vv[mask].astype(float)           # (N,)
        z = depth[mask]                      # (N,) camera-space distance
        x = (u-cx)/fx*z
        y = (v-cy)/fy*z
        pts = np.stack([x,y,z], axis = 1)
        colors = color[mask] if color is not None else None   # (N,3) RGB, aligned to pts
        return pts, colors

    def draw_landmarks(self, img, landmarks):
        new_img = img.copy()
        for landmark in landmarks:
            cv2.circle(img=new_img, center=(int(landmark[0]), int(landmark[1])), radius=2, color=(255,0,0), thickness=-1)
        return new_img

    def make_panel(self, images):
        pil = [Image.fromarray(im) for im in images]
        h = max(im.height for im in pil)
        panel = Image.new("RGB", (sum(im.width for im in pil), h), (0, 0, 0))
        x = 0
        for im in pil:
            panel.paste(im, (x, 0))
            x += im.width
        return panel

    def save_panel(self, path, images):
        panel = self.make_panel(images=images)
        panel.save(path)

    def save_grid(self, path, grid):
        strips = [self.make_panel(panel) for panel in grid]

        W = max(s.width for s in strips)
        H = sum(s.height for s in strips)

        grid = Image.new("RGB", (W,H), (0,0,0))

        y=0

        for strip in strips:
            grid.paste(strip, (0, y))
            y+=strip.height

        grid.save(path)

    # -- resume ----------------------------------------------------------
    def item_complete(self, subj_out, n_views) -> bool:
        """True if this item already has all n_views fully written.
        """
        item = self.out_root / subj_out
        if not item.is_dir():
            return False
        for i in range(n_views):
            vd = item / str(i)
            if not all((vd / f).is_file() for f in VIEW_FILES):
                return False
        return True

    def run(self, ids, expressions, cameras=None, orientation=(0.0, 0.0, 0.0),
            lighting=False, rand_pose=False, rand_ring=False, variants=1,
            augmentor=None, debug_artifacts=False, views=5, seed=0, force=False):
        """Render every (id, expression, variant) combination.
        """
        total = len(ids) * len(expressions) * variants
        bar = tqdm(total=total, desc="rendering", unit="item")
        n_missing = n_skipped = n_done = 0

        for id in ids:
            for exp_id in expressions:
                exp_stem = stem(exp_id)
                # Some FaceScape subjects ship textures but no .obj geometry for
                # a given expression -- skip instead of crashing the whole run.
                try:
                    obj = self.mesh_path(id, exp_stem)
                except FileNotFoundError as e:
                    print(f"  warning: {e}")
                    n_missing += variants
                    bar.update(variants)
                    continue
                if not obj.is_file():
                    print(f"  warning: no mesh for id {id} {exp_stem}, skipping: {obj}")
                    n_missing += variants
                    bar.update(variants)
                    continue

                for k in range(variants):
                    subj_out = f"{id}_{exp_id}_{k}"
                    bar.set_postfix_str(subj_out)

                    n_views = len(load_camera(cameras)) if cameras else views
                    if not force and self.item_complete(subj_out, n_views):
                        n_skipped += 1
                        bar.update(1)
                        continue

                    # Same item name -> same random draws, across interruptions.
                    s = item_seed(seed, id, exp_id, k)
                    np.random.seed(s)
                    rng = np.random.default_rng(s)

                    roll, pitch, yaw = random_orientation() if rand_pose else orientation
                    mesh = self.load_mesh(id=id, exp=exp_stem)
                    self.orient_head(mesh, roll, pitch, yaw)
                    lm_w = mesh.metadata["lm_world"]    # landmark in world frame
                    if cameras:
                        cams = load_camera(cameras)
                    elif rand_ring:
                        cams = random_ring(mesh, n=views)   # resampled per variant
                    else:
                        cams = default_ring(mesh, n=views)
                    overlays = []                       # per-cam overlays for the panel

                    for i, cam in enumerate(cams):
                        cam.id = str(i)                 # output ids are ALWAYS 0,1,2,...
                        color, depth = self.render(mesh, cam)
                        # Bake domain randomization (bg composite + photometric)
                        if augmentor is not None:
                            rgb01 = augmentor.apply(color.astype(np.float32) / 255.0,
                                                    depth > 0, rng)
                            color = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
                        landmarks = self.project_landmarks(mesh, cam)
                        lm_cam = (cam.R @ lm_w.T).T + cam.t     # landmark in camera frame
                        R_ch   = cam.R @ mesh.metadata["head_R"]    # head orientation in camera frame
                        t_ch   = cam.R @ mesh.metadata["head_t"] + cam.t    # head position in camera frame
                        quat   = rotmat_to_quat(R_ch)   # R_ch in quaternion

                        out_dir = self.out_root / subj_out / cam.id
                        out_dir.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(color).save(out_dir / "rgb.png")
                        np.save(out_dir / "depth.npy", depth)
                        meta = {
                            "id": cam.id, "W": cam.W, "H": cam.H,
                            "K": cam.K.tolist(), "R": cam.R.tolist(), "t": cam.t.tolist(),
                            "head_quat": quat.tolist(), "head_t_cam": t_ch.tolist(),
                            "orientation_deg": [roll, pitch, yaw],
                            "expression_id": int(exp_id), "expression": exp_stem,
                            "variant": int(k), "units": "facescape_world",
                        }
                        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                        np.save(out_dir / "landmarks_2d.npy", landmarks)   # (68,2) uv
                        np.save(out_dir / "landmarks_3d.npy", lm_cam)       # (68,3) camera frame

                        # Debug artifacts (landmark overlay, point clouds) -- off by
                        # default; they dominate disk on a full multi-expression bake.
                        if debug_artifacts:
                            overlay = self.draw_landmarks(color, landmarks)
                            Image.fromarray(overlay).save(out_dir / "rgb_landmark_overlay.png")
                            overlays.append(overlay)
                            pts, cols = self.backproject(depth=depth, K=cam.K, color=color)
                            flip = np.array([1.0, -1.0, -1.0])
                            pts_gl = pts * flip
                            trimesh.PointCloud(vertices=pts_gl, colors=cols).export(out_dir / "rgbd.ply")
                            lm_dots = np.tile([255, 0, 0], (len(lm_cam), 1))
                            all_pts = np.vstack([pts_gl, lm_cam * flip])
                            all_colors = np.vstack([cols, lm_dots])
                            trimesh.PointCloud(vertices=all_pts, colors=all_colors).export(out_dir / "rgbd_landmarks_overlay.ply")

                    if debug_artifacts:
                        # all camera overlays side by side, under the item folder
                        self.save_panel(self.out_root / subj_out / "panel.png", overlays)
                        if lighting:
                            grid = []
                            for kk in range(1, 5):
                                row = []
                                light = generate_random_light()
                                for i, cam in enumerate(cams):
                                    cam.id = str(i)
                                    color, _ = self.render(mesh, cam, light=light)
                                    out_dir = self.out_root / subj_out / cam.id
                                    Image.fromarray(color).save(out_dir / f"rgb_{kk}.png")
                                    row.append(color)
                                grid.append(row)
                            self.save_grid(path=self.out_root / subj_out / "lighting_panel.png", grid=grid)

                    n_done += 1
                    bar.update(1)

        bar.close()
        print(f"rendered {n_done} items, skipped {n_skipped} already complete, "
              f"{n_missing} missing meshes  ->  {self.out_root}")


def load_camera(path):
    entries = json.loads(Path(path).read_text())
    return [
        Camera(id=e["id"], W=e["W"], H=e["H"], K=e["K"], R=e["R"], t=e["t"])
        for e in entries
    ]


def default_ring(mesh, n=5, radius=380.0, fov_deg=40.0, W=512, H=512,
                 az_min_deg=-50.0, az_max_deg=50.0):
    """n look-at cameras spanned evenly in azimuth across [az_min_deg, az_max_deg]
    around the head centroid, centered on the front (az=0). A front-biased arc, not
    a full circle -- the back of the head carries no useful info. Defaults: 5 cams
    from -50deg to +50deg. Returns Camera objects (same type as load_camera)."""
    centroid = mesh.vertices.mean(axis=0)
    K = intrinsics_from_fov(fov_deg, W, H)
    azimuths = np.radians(np.linspace(az_min_deg, az_max_deg, n))   # az=0 is the front
    cams = []
    for i, az in enumerate(azimuths):
        eye = centroid + radius * np.array([np.sin(az), 0.0, np.cos(az)])
        R, t = look_at_cv(eye, centroid)       # +Y up; given primitive
        cams.append(Camera(id=f"cam{i:02d}", W=W, H=H, K=K, R=R, t=t))
    return cams


def random_ring(mesh, n=5, W=512, H=512):
    centroid = mesh.vertices.mean(axis=0)
    radius = np.random.uniform(340.0, 420.0)
    fov_deg = np.random.uniform(35.0, 45.0)
    K = intrinsics_from_fov(fov_deg, W, H)
    # azimuth: random arc center + half-width, plus small per-camera jitter.
    az_center = np.random.uniform(-15.0, 15.0)
    az_half = np.random.uniform(35.0, 60.0)
    az_deg = np.linspace(az_center - az_half, az_center + az_half, n) \
        + np.random.uniform(-5.0, 5.0, size=n)
    # elevation: random ring tilt + per-camera jitter (vertical viewpoint diversity).
    el_deg = np.random.uniform(-12.0, 12.0) + np.random.uniform(-5.0, 5.0, size=n)
    azr, elr = np.radians(az_deg), np.radians(el_deg)
    cams = []
    for i, (az, el) in enumerate(zip(azr, elr)):
        direction = np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
        eye = centroid + radius * direction
        R, t = look_at_cv(eye, centroid)       # +Y up; given primitive
        cams.append(Camera(id=f"cam{i:02d}", W=W, H=H, K=K, R=R, t=t))
    return cams


def random_orientation(pitch_max=25.0, yaw_max=35.0):
    pitch = np.random.uniform(-pitch_max, pitch_max)
    yaw   = np.random.uniform(-yaw_max, yaw_max)
    return (0.0, pitch, yaw)


def generate_random_light() -> Light:
    x = np.random.uniform(-1.0, 1.0)    # left/right: full swing to either cheek
    y = np.random.uniform(-0.75, 0.5)    # up/down: gentler, avoids harsh top/bottom light
    z = np.random.uniform(-1.0, 0.3)   # ALWAYS negative -> light comes from the front

    intensity = np.random.uniform(5.0, 10.0)
    ambient = np.random.uniform(0.2, 0.5)
    return Light(intensity=intensity, ambient=ambient, direction=(x,y,z))


def parse_id_spec(spec: str) -> list[str]:
    """Parse --id-range: '801', '801-847', or '1,5,9-12'."""
    out: set[int] = set()
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(chunk))
    return [str(i) for i in sorted(out)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--id-range", required=True, help='e.g. "801-805", "801" or "1,5,9-12"')
    parser.add_argument("--expressions", default="all",
                        help="'all' (default), '18', '1,18,20' or '1-5'")
    parser.add_argument("--cameras", help="path to cameras.json")
    parser.add_argument("--views", type=int, default=5,
                        help="cameras per item when not using --cameras")
    parser.add_argument("--orientation", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--lighting", action="store_true",
                        help="render a random-lighting grid (debug artifacts only)")
    parser.add_argument("--rand-pose", action="store_true",
                        help="randomize head pitch+yaw per variant (overrides --orientation)")
    parser.add_argument("--rand-ring", action="store_true",
                        help="randomize the camera ring geometry per variant")
    parser.add_argument("--variants", type=int, default=1,
                        help="baked augmentation variants per (id, expression); each is a "
                             "fresh pose+ring+RGB-aug draw written as <id>_<exp>_<k>")
    parser.add_argument("--bg-dir", default=str(REPO_ROOT / "data/backgrounds/indoor/Images"),
                        help="background image pool for baked bg compositing")
    parser.add_argument("--bg-prob", type=float, default=0.0,
                        help="per-view prob of baking a random background (0=off)")
    parser.add_argument("--photometric", action="store_true",
                        help="bake HRNet's photometric ISP jitter into the saved RGB")
    parser.add_argument("--debug-artifacts", action="store_true",
                        help="also write landmark overlays, .ply point clouds and panels; "
                             "OFF by default because they dominate a full bake's disk")
    parser.add_argument("--seed", type=int, default=0,
                        help="base seed; per-item draws are derived from (seed, id, exp, variant)")
    parser.add_argument("--force", action="store_true",
                        help="re-render items that are already complete on disk")
    parser.add_argument("--data-root", default=str(REPO_ROOT / "data/facescape"),
                        help="root holding the TU meshes")
    parser.add_argument("--assets", default=str(ASSETS_DIR),
                        help="package assets dir (landmark_indices.npz)")
    parser.add_argument("--out-root", default=str(REPO_ROOT / "data/facescape/virtual_camera_expr"),
                        help="output dir; use a NEW dir to keep an existing set")
    args = parser.parse_args()

    ids = parse_id_spec(args.id_range)
    expressions = parse_expression_spec(args.expressions)

    # Build the shared bg+photometric augmentor if baking is requested.
    aug_cfg = AugConfig(bg_dir=args.bg_dir, bg_prob=args.bg_prob,
                        photometric=args.photometric)
    augmentor = MultiViewAugmentor(aug_cfg) if aug_cfg.enabled else None

    render = ViewRenderer(data_root=args.data_root, out_root=args.out_root,
                          assets_dir=args.assets)
    render.run(ids, expressions, cameras=args.cameras, orientation=args.orientation,
               lighting=args.lighting, rand_pose=args.rand_pose, rand_ring=args.rand_ring,
               variants=args.variants, augmentor=augmentor,
               debug_artifacts=args.debug_artifacts, views=args.views,
               seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()

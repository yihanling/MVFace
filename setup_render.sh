#!/usr/bin/env bash
# Install the FaceScape data-generation stack in one command:
#
#     bash setup_render.sh                 # into ./.venv
#     bash setup_render.sh /path/to/venv   # into another venv
#
# Two pip passes are unavoidable. pyrender hard-pins PyOpenGL==3.1.0, which
# dies in glGenTextures (ctypes.ArgumentError) on Python 3.14 as soon as a
# *textured* mesh is rendered -- untextured geometry is fine, so the failure
# only shows up on real FaceScape meshes. Asking pip for pyrender and a newer
# PyOpenGL together makes it downgrade pyrender to 0.1.18 instead, so PyOpenGL
# has to be upgraded afterwards, on its own.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"

[ -x "$PY" ] || { echo "no python at $PY (pass a venv path as \$1)" >&2; exit 1; }

echo "== installing mvface[render] into $VENV"
"$PY" -m pip install -e "$REPO_ROOT[render]"

echo "== upgrading PyOpenGL past pyrender's stale pin (conflict warning is expected)"
"$PY" -m pip install --upgrade "PyOpenGL>=3.1.9"

echo "== verifying"
PYOPENGL_PLATFORM=egl "$PY" - <<'PYCODE'
import sys
import OpenGL, pyrender, trimesh, cv2, numpy as np

print(f"PyOpenGL {OpenGL.__version__}  pyrender {pyrender.__version__} "
      f" trimesh {trimesh.__version__}  cv2 {cv2.__version__}")

gl = tuple(int(x) for x in OpenGL.__version__.split(".")[:3])
if gl < (3, 1, 9):
    sys.exit(f"PyOpenGL {OpenGL.__version__} is too old; textured renders will crash")
if tuple(int(x) for x in pyrender.__version__.split(".")) < (0, 1, 45):
    sys.exit(f"pyrender {pyrender.__version__} got downgraded; expected >= 0.1.45")

# Textured offscreen render -- the case PyOpenGL 3.1.0 fails on. Kept small and
# self-contained so it needs no FaceScape data.
mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
mesh.visual = trimesh.visual.TextureVisuals(
    uv=np.zeros((len(mesh.vertices), 2)),
    image=__import__("PIL.Image", fromlist=["Image"]).new("RGB", (8, 8), (200, 120, 90)),
)
scene = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
scene.add(pyrender.Mesh.from_trimesh(mesh))
pose = np.eye(4)
pose[2, 3] = 4.0
scene.add(pyrender.IntrinsicsCamera(fx=100, fy=100, cx=32, cy=32,
                                    znear=0.1, zfar=100.0), pose=pose)
scene.add(pyrender.DirectionalLight(intensity=3.0), pose=pose)
r = pyrender.OffscreenRenderer(64, 64)
color, depth = r.render(scene)
r.delete()
if not (depth > 0).any():
    sys.exit("EGL render produced no pixels")
print(f"headless textured render OK ({int((depth > 0).sum())} px)")
PYCODE

echo "== done. Run renders with PYOPENGL_PLATFORM=egl"

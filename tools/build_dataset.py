"""Build the full FaceScape render set one zip bucket at a time."""

import _init_paths  # noqa: F401
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import timedelta
from pathlib import Path

from _init_paths import REPO_ROOT
from mvface.expressions import parse_expression_spec

TOOLS = REPO_ROOT / "tools"
STATE_NAME = ".build_state.json"

MB_PER_ITEM = 7.5
MB_PER_ID_MESH = 115.0


def human(mb: float) -> str:
    return f"{mb/1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


class Bucket:
    def __init__(self, zip_path: Path):
        self.zip_path = zip_path
        with zipfile.ZipFile(zip_path) as zf:
            ids = sorted({n.split("/")[0] for n in zf.namelist()
                          if n.split("/")[0].isdigit()}, key=int)
        if not ids:
            raise ValueError(f"{zip_path.name}: no subject ids inside")
        self.ids = ids
        self.name = f"{ids[0]}_{ids[-1]}"

    def chunks(self, jobs: int) -> list[list[str]]:
        """Split the ids into `jobs` contiguous, near-equal ranges.
        """
        n = min(jobs, len(self.ids))
        size, extra = divmod(len(self.ids), n)
        out, i = [], 0
        for k in range(n):
            take = size + (1 if k < extra else 0)
            out.append(self.ids[i:i + take])
            i += take
        return out


class Baker:
    def __init__(self, args):
        self.a = args
        self.out_root = Path(args.out_root)
        self.data_root = Path(args.data_root)
        self.state_path = self.out_root / STATE_NAME
        self.n_exp = len(parse_expression_spec(args.expressions))

    # -- state -----------------------------------------------------------
    def load_state(self) -> dict:
        if self.state_path.is_file():
            return json.loads(self.state_path.read_text())
        return {"config": self.config(), "done": []}

    def config(self) -> dict:
        a = self.a
        return {"expressions": a.expressions, "variants": a.variants,
                "bg_prob": a.bg_prob, "seed": a.seed, "views": a.views}

    def save_state(self, state: dict) -> None:
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2))

    def check_config(self, state: dict) -> None:
        """Refuse to mix two recipes in one output dir -- the mismatch would be
        invisible later, and half the set would carry different augmentation."""
        old, new = state.get("config", {}), self.config()
        if old and old != new:
            diff = {k: (old.get(k), new.get(k)) for k in set(old) | set(new)
                    if old.get(k) != new.get(k)}
            sys.exit(f"{self.out_root} was built with a different recipe: {diff}\n"
                     "Use a fresh --out-root, or match the recorded settings.")

    # -- logging ---------------------------------------------------------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.out_root.mkdir(parents=True, exist_ok=True)
        with open(self.out_root / "build.log", "a") as f:
            f.write(line + "\n")

    # -- disk ------------------------------------------------------------
    def guard_disk(self, bucket: Bucket) -> None:
        need = (len(bucket.ids) * self.n_exp * self.a.variants * MB_PER_ITEM
                + len(bucket.ids) * MB_PER_ID_MESH)
        free = shutil.disk_usage(self.out_root).free / 1e6
        if free < need + self.a.disk_margin * 1024:
            sys.exit(f"stopping before {bucket.name}: needs ~{human(need)} plus a "
                     f"{self.a.disk_margin} GB margin, only {human(free)} free")

    # -- steps -----------------------------------------------------------
    def extract(self, bucket: Bucket) -> None:
        cmd = [sys.executable, str(TOOLS / "extract_facescape.py"), str(bucket.zip_path),
               "--out-root", str(self.data_root), "--expressions", self.a.expressions]
        self.run(cmd, f"extract {bucket.name}")

    def render_cmd(self, ids: list[str]) -> list[str]:
        a = self.a
        cmd = [sys.executable, str(TOOLS / "render_views.py"),
               "--id-range", f"{ids[0]}-{ids[-1]}",
               "--expressions", a.expressions,
               "--variants", str(a.variants),
               "--views", str(a.views),
               "--bg-prob", str(a.bg_prob),
               "--bg-dir", a.bg_dir,
               "--seed", str(a.seed),
               "--data-root", str(self.data_root),
               "--out-root", str(self.out_root)]
        for flag, on in (("--rand-pose", a.rand_pose), ("--rand-ring", a.rand_ring),
                         ("--photometric", a.photometric), ("--force", a.force)):
            if on:
                cmd.append(flag)
        return cmd

    def render(self, bucket: Bucket) -> None:
        """Run `--jobs` renderers over disjoint id ranges, each logging to its
        own file so the progress bars do not interleave."""
        chunks = bucket.chunks(self.a.jobs)
        env = {**os.environ, "PYOPENGL_PLATFORM": "egl"}
        logs = self.out_root / "logs"
        logs.mkdir(parents=True, exist_ok=True)

        procs = []
        for k, ids in enumerate(chunks):
            path = logs / f"{bucket.name}.job{k}.log"
            fh = open(path, "w")
            self.log(f"  render {ids[0]}-{ids[-1]} ({len(ids)} ids) -> {path.name}")
            procs.append((subprocess.Popen(self.render_cmd(ids), env=env,
                                           stdout=fh, stderr=subprocess.STDOUT), fh, path))

        failed = []
        for p, fh, path in procs:
            rc = p.wait()
            fh.close()
            if rc != 0:
                failed.append((path, rc))
        if failed:
            for path, rc in failed:
                self.log(f"  FAILED (exit {rc}): tail of {path}")
                self.log("    " + "    ".join(path.read_text().splitlines(True)[-8:]))
            sys.exit(f"{bucket.name}: {len(failed)}/{len(procs)} render jobs failed; "
                     "meshes kept for a retry")

    def drop_meshes(self, bucket: Bucket) -> None:
        d = self.data_root / bucket.name
        if self.a.keep_meshes or not d.is_dir():
            return
        size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 1e6
        shutil.rmtree(d)
        self.log(f"  dropped meshes {d} ({human(size)})")

    def run(self, cmd: list[str], what: str) -> None:
        r = subprocess.run(cmd, env={**os.environ, "PYOPENGL_PLATFORM": "egl"})
        if r.returncode != 0:
            sys.exit(f"{what} failed (exit {r.returncode})")

    # -- driver ----------------------------------------------------------
    def go(self, buckets: list[Bucket]) -> None:
        state = self.load_state()
        self.check_config(state)
        todo = [b for b in buckets if b.name not in state["done"]]

        items = sum(len(b.ids) for b in todo) * self.n_exp * self.a.variants
        self.log(f"{len(todo)}/{len(buckets)} buckets to go, {items} items, "
                 f"~{human(items * MB_PER_ITEM)} of renders, --jobs {self.a.jobs}")
        if state["done"]:
            self.log(f"already done: {', '.join(state['done'])}")
        if self.a.dry_run:
            for b in todo:
                chunks = b.chunks(self.a.jobs)
                self.log(f"  {b.name}: {len(b.ids)} ids -> "
                         + ", ".join(f"{c[0]}-{c[-1]}" for c in chunks))
            return

        self.save_state(state)
        t_all = time.time()
        for i, b in enumerate(todo, 1):
            t0 = time.time()
            self.log(f"=== [{i}/{len(todo)}] bucket {b.name} ({len(b.ids)} ids)")
            self.guard_disk(b)
            self.extract(b)
            self.render(b)
            self.drop_meshes(b)
            state["done"].append(b.name)
            self.save_state(state)
            self.log(f"  bucket done in {timedelta(seconds=int(time.time() - t0))}; "
                     f"elapsed {timedelta(seconds=int(time.time() - t_all))}")
        self.log(f"build complete in {timedelta(seconds=int(time.time() - t_all))} "
                 f"-> {self.out_root}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zips", default=None,
                   help="glob of trainset zips (default: all under --data-root)")
    p.add_argument("--jobs", type=int, default=4,
                   help="renderer processes per bucket, over disjoint id ranges")
    p.add_argument("--expressions", default="all")
    p.add_argument("--variants", type=int, default=2)
    p.add_argument("--views", type=int, default=5)
    p.add_argument("--bg-prob", type=float, default=0.9,
                   help="matches the iteration-2 messy set (measured 88%% of views)")
    p.add_argument("--bg-dir", default=str(REPO_ROOT / "data/backgrounds/indoor/Images"))
    p.add_argument("--seed", type=int, default=0)
    # On by default -- this is the messy recipe. --no-rand-pose etc. turn them off.
    p.add_argument("--rand-pose", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rand-ring", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--photometric", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--force", action="store_true",
                   help="re-render items already complete on disk")
    p.add_argument("--keep-meshes", action="store_true",
                   help="do not delete each bucket's meshes after rendering")
    p.add_argument("--disk-margin", type=float, default=25.0,
                   help="GB of free space to keep in hand; stops before a bucket "
                        "that would eat into it")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--data-root", default=str(REPO_ROOT / "data/facescape"))
    p.add_argument("--out-root", default=str(REPO_ROOT / "data/facescape/virtual_camera_expr"))
    args = p.parse_args()

    root = Path(args.data_root)
    zips = (sorted(Path().glob(args.zips)) if args.zips
            else sorted(root.glob("facescape_trainset_*.zip")))
    if not zips:
        sys.exit(f"no trainset zips found under {root}")

    Baker(args).go([Bucket(z) for z in zips])


if __name__ == "__main__":
    main()

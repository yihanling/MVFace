"""Extract FaceScape TU-model assets (.obj/.jpg/.obj.mtl) from the trainset zips."""

import _init_paths  # noqa: F401
import argparse
import zipfile
from pathlib import Path

from _init_paths import REPO_ROOT
from mvface.expressions import parse_expression_spec, stem

try:                                       # progress bar is a nicety, not a dep
    from tqdm import tqdm
except ImportError:                        # extraction is otherwise pure stdlib
    def tqdm(it, **kw):
        return it


class Extract:
    """Pulls the three per-expression files for a set of ids out of one zip."""

    def __init__(self, zip_path, out_root="data/facescape"):
        self.zip_path = Path(zip_path)
        self.out_root = Path(out_root)

    # -- layout ----------------------------------------------------------
    def discover_ids(self, zf: zipfile.ZipFile) -> list[str]:
        ids = {name.split("/")[0] for name in zf.namelist() if name.split("/")[0].isdigit()}
        return sorted(ids, key=int)

    def bucket_name(self, ids: list[str]) -> str:
        """Folder holding this zip's ids, e.g. '1_100' -- the renderer parses
        this back into an inclusive id range."""
        return f"{ids[0]}_{ids[-1]}"

    def member_names(self, id: str, exp_stem: str) -> list[str]:
        """The three zip entries making up one textured expression mesh."""
        return [f"{id}/models_reg/{exp_stem}.obj",
                f"{id}/models_reg/{exp_stem}.jpg",
                f"{id}/models_reg/{exp_stem}.obj.mtl"]

    # -- extraction ------------------------------------------------------
    def extract_one(self, zf, bucket, id, exp_stem, force=False) -> tuple[int, int]:
        """Extract one (id, expression). Returns (written, missing) counts."""
        written = missing = 0
        for name in self.member_names(id, exp_stem):
            out = self.out_root / bucket / id / Path(name).name
            if out.is_file() and out.stat().st_size > 0 and not force:
                continue                   # already extracted -> resumable
            try:
                data = zf.read(name)
            except KeyError:
                # Some subjects (e.g. 832) ship textures but no geometry.
                print(f"  warning: missing in zip, skipped: {name}")
                missing += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            written += 1
        return written, missing

    def run(self, expressions, ids=None, force=False, delete_zip=False,
            all_expressions=False) -> str:
        with zipfile.ZipFile(self.zip_path) as zf:
            present = self.discover_ids(zf)
            bucket = self.bucket_name(present)
            targets = [i for i in present if ids is None or int(i) in ids]

            if ids is not None and not targets:
                print(f"{self.zip_path.name}: no requested ids in this zip, skipping")
                return bucket

            written = missing = 0
            bar = tqdm(targets, desc=f"extract {bucket}", unit="id")
            for id in bar:
                for exp_id in expressions:
                    w, m = self.extract_one(zf, bucket, id, stem(exp_id), force=force)
                    written += w
                    missing += m

        print(f"{self.zip_path.name}: {len(targets)} ids x {len(expressions)} expressions "
              f"-> {written} files written, {missing} missing, into {self.out_root / bucket}")

        # Only safe to drop the source once EVERY expression is out of it; a
        # partial set would mean re-downloading to finish the job later.
        if delete_zip:
            if not all_expressions:
                print("  refusing --delete-zip: not all expressions were requested")
            elif ids is not None:
                print("  refusing --delete-zip: --ids was given, so this zip is not exhausted")
            elif missing:
                print(f"  refusing --delete-zip: {missing} members were missing")
            else:
                self.zip_path.unlink()
                print(f"  deleted {self.zip_path}")
        return bucket


def parse_ids(spec: str | None) -> set[int] | None:
    """Parse --ids: '805', '801-847', '1,5,9-12', or None for every id."""
    if not spec:
        return None
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
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zips", nargs="+", help="paths to FaceScape trainset zips")
    p.add_argument("--out-root", default=str(REPO_ROOT / "data/facescape"),
                   help="root that will hold the <lo>_<hi>/<id>/ mesh buckets")
    p.add_argument("--expressions", default="all",
                   help="'all' (default), '18', '1,18,20' or '1-5'")
    p.add_argument("--ids", default=None,
                   help="restrict to these subject ids, e.g. '801-810' (default: all in zip)")
    p.add_argument("--force", action="store_true",
                   help="re-extract files that are already on disk")
    p.add_argument("--delete-zip", action="store_true",
                   help="delete each zip once all its expressions are extracted "
                        "(irreversible; FaceScape is license-gated)")
    args = p.parse_args()

    expressions = parse_expression_spec(args.expressions)
    ids = parse_ids(args.ids)
    all_expressions = args.expressions.strip().lower() == "all"

    for zp in args.zips:
        Extract(zp, out_root=args.out_root).run(
            expressions, ids=ids, force=args.force,
            delete_zip=args.delete_zip, all_expressions=all_expressions)


if __name__ == "__main__":
    main()

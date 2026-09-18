"""Layout of a training-eval output directory.

    output/<run_name>/
    |-- config.json              run metadata: args, git sha, versions, data root
    |-- split.csv                subject_id,split
    |-- train.csv                one row per epoch
    |-- train.log                console transcript
    |-- checkpoints/
    |   |-- best.pth
    |   `-- last.pth
    |-- eval/
    |   |-- best.json            scalar summary
    |   |-- best_per_subject.csv
    |   `-- best_per_joint.csv
    `-- figures/                 viz_pred_vs_gt

"""
from __future__ import annotations

import csv
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _git_info(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(a, cwd=repo, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    try:
        return {"git_sha": run("git", "rev-parse", "--short", "HEAD"),
                "git_dirty": bool(run("git", "status", "--porcelain"))}
    except Exception:
        return {"git_sha": None, "git_dirty": None}


def collect_env(repo: Path | None = None) -> dict:
    import torch
    env = {
        "started": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    env.update(_git_info(repo or Path(__file__).resolve().parents[2]))
    return env


class OutputDir:
    """Path accessors for one run directory."""

    def __init__(self, root):
        self.root = Path(root)

    def __repr__(self):
        return f"OutputDir({self.root})"

    # -- layout ------------------------------------------------------------
    @property
    def config(self) -> Path:
        return self.root / "config.json"

    @property
    def split(self) -> Path:
        return self.root / "split.csv"

    @property
    def train_csv(self) -> Path:
        return self.root / "train.csv"

    @property
    def train_log(self) -> Path:
        return self.root / "train.log"

    def checkpoint(self, name: str = "best") -> Path:
        return self.root / "checkpoints" / f"{name}.pth"

    def eval_json(self, name: str = "best") -> Path:
        return self.root / "eval" / f"{name}.json"

    def eval_per_subject(self, name: str = "best") -> Path:
        return self.root / "eval" / f"{name}_per_subject.csv"

    def eval_per_joint(self, name: str = "best") -> Path:
        return self.root / "eval" / f"{name}_per_joint.csv"

    def figure(self, name: str) -> Path:
        return self.root / "figures" / name

    @classmethod
    def from_checkpoint(cls, ckpt_path) -> "OutputDir":
        p = Path(ckpt_path).parent
        return cls(p.parent if p.name == "checkpoints" else p)

    def _prompt_on_exists(self) -> str:
        """Ask what to do about an existing run."""
        pr = self.read_progress()
        planned = pr["epochs_planned"]
        planned = (f" of {planned}"
                   if planned and pr["epochs_done"] <= planned else "")
        best = (f"{pr['best_mpjpe']:.2f} mm (epoch {pr['best_epoch']})"
                if pr["best_mpjpe"] is not None else "none recorded")

        print(f"\n{self.root} already contains a run.\n")
        print(f"  started    {pr['started'] or 'unknown'}")
        print(f"  progress   {pr['epochs_done']}{planned} epochs")
        print(f"  best       {best}\n")
        print("  [0] Overwrite   permanently delete this run and start fresh")
        if pr["can_resume"]:
            print(f"  [1] Resume      continue from checkpoints/last.pth "
                  f"(epoch {pr['epochs_done']})")
        print("  [2] Cancel      leave it untouched and exit\n")

        choices = {"0": "force", "2": "cancel"}
        if pr["can_resume"]:
            choices["1"] = "resume"
        for _ in range(3):
            try:
                raw = input("Select [2]: ").strip()
            except EOFError:
                return "cancel"
            if raw == "":
                return "cancel"
            if raw in choices:
                return choices[raw]
            print(f"  '{raw}' is not one of {sorted(choices)}.")
        return "cancel"

    def create(self, on_exists: str = "error") -> "OutputDir":
        """Make the directory tree, deciding what to do if one already exists.

        on_exists:
            "error"   raise (the default, and the fallback when not on a TTY)
            "force"   delete the existing run, then start fresh
            "resume"  keep the existing contents
            "ask"     prompt the user, then act on their choice

        Sets self.action to "created" / "overwritten" / "resumed" so the caller
        knows which path was taken.
        """
        self.action = "created"
        occupied = self.root.exists() and any(self.root.iterdir())

        if occupied:
            if on_exists == "ask":
                # Never block a non-interactive run (nohup, SLURM, a pipe) on a
                # question nobody can see -- fall back to refusing.
                on_exists = self._prompt_on_exists() if sys.stdin.isatty() else "error"

            if on_exists == "cancel":
                raise SystemExit(f"cancelled -- {self.root} left untouched")
            if on_exists == "error":
                raise SystemExit(
                    f"{self.root} is not empty -- refusing to overwrite an existing run.\n"
                    f"Pass --force to overwrite, --resume to continue it, "
                    f"or choose a different --out.")
            if on_exists == "force":
                target = self.root.resolve()
                if target.parent == target or target == Path.home():
                    raise SystemExit(f"refusing to delete {target}")
                shutil.rmtree(target)
                self.action = "overwritten"
            elif on_exists == "resume":
                if not self.checkpoint("last").is_file():
                    raise SystemExit(
                        f"cannot resume {self.root}: no checkpoints/last.pth")
                self.action = "resumed"

        for d in (self.root, self.root / "checkpoints",
                  self.root / "eval", self.root / "figures"):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- writers -----------------------------------------------------------
    def write_config(self, args: dict, **extra) -> Path:
        payload = {"run_name": self.root.name, **collect_env(), **extra, "args": args}
        self.config.write_text(json.dumps(payload, indent=2))
        return self.config

    def write_split(self, train_ids, val_ids) -> Path:
        """One row per subject. Greppable: `grep ,val split.csv | wc -l`."""
        with open(self.split, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "split"])
            for sid in train_ids:
                w.writerow([sid, "train"])
            for sid in val_ids:
                w.writerow([sid, "val"])
        return self.split

    # -- readers -----------------------------------------------------------
    def read_split(self) -> tuple[list[str], list[str]] | None:
        if not self.split.is_file():
            return None
        train, val = [], []
        with open(self.split, newline="") as f:
            for row in csv.DictReader(f):
                (train if row["split"] == "train" else val).append(row["subject_id"])
        return train, val

    def read_progress(self) -> dict:
        """How far an existing run got, for the conflict prompt and reporting."""
        cfg = self.read_config() or {}
        rows = []
        if self.train_csv.is_file():
            with open(self.train_csv, newline="") as f:
                rows = list(csv.DictReader(f))
        best = None
        for r in rows:
            try:
                v = float(r["val_mpjpe"])
            except (KeyError, ValueError):
                continue
            if best is None or v < best[1]:
                best = (int(r["epoch"]), v)
        return {
            "started": cfg.get("started"),
            "epochs_done": int(rows[-1]["epoch"]) if rows else 0,
            "epochs_planned": cfg.get("args", {}).get("epochs"),
            "best_epoch": best[0] if best else None,
            "best_mpjpe": best[1] if best else None,
            "can_resume": self.checkpoint("last").is_file(),
        }

    def read_config(self) -> dict | None:
        return json.loads(self.config.read_text()) if self.config.is_file() else None

    def data_root(self) -> str | None:
        cfg = self.read_config()
        if not cfg:
            return None
        return cfg.get("data_root") or cfg.get("args", {}).get("root")

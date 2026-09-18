from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# MIT Indoor Scenes pool (68 categories, ~15.6k images), moved into this repo
# 2026-08-24. Not tracked by git -- data/ is ignored. Resolved relative to the
# working directory, like --root elsewhere in this repo.
DEFAULT_BG_DIR = "data/backgrounds/indoor/Images"


def photometric(img: np.ndarray, rng: np.random.Generator, log: dict | None = None) -> np.ndarray:
    def rec(name, value):  # record the sampled strength (or None if it didn't fire)
        if log is not None:
            log[name] = value

    pil = Image.fromarray(img)

    # --- appearance / ISP: skin tone + white balance + exposure ---
    fired = rng.random() < 0.5
    f = rng.uniform(0.6, 1.4) if fired else None
    if fired:
        pil = ImageEnhance.Brightness(pil).enhance(f)
    rec('brightness', f)

    fired = rng.random() < 0.7
    f = rng.uniform(0.4, 1.6) if fired else None      # spans probe 0.35 break
    if fired:
        pil = ImageEnhance.Contrast(pil).enhance(f)
    rec('contrast', f)

    fired = rng.random() < 0.7
    f = rng.uniform(0.4, 1.6) if fired else None      # saturation; spans 0.75 break
    if fired:
        pil = ImageEnhance.Color(pil).enhance(f)
    rec('saturation', f)

    fired = rng.random() < 0.3
    s = int(rng.integers(-15, 16)) if fired else None
    if fired:
        hsv = np.asarray(pil.convert('HSV'), dtype=np.int16)
        hsv[..., 0] = (hsv[..., 0] + s) % 256  # hue wraps
        pil = Image.fromarray(hsv.astype(np.uint8), 'HSV').convert('RGB')
    rec('hue', s)

    # --- optics: soft focus / cheap lens (minor per the probe) ---
    fired = rng.random() < 0.3
    r = rng.uniform(0.0, 2.5) if fired else None
    if fired:
        pil = pil.filter(ImageFilter.GaussianBlur(r))
    rec('blur', r)

    # --- resolution: low-res capture then upsample (minor) ---
    fired = rng.random() < 0.2
    f = rng.uniform(0.4, 1.0) if fired else None
    if fired:
        w, h = pil.size
        pil = pil.resize((max(1, int(w * f)), max(1, int(h * f))), Image.BILINEAR)
        pil = pil.resize((w, h), Image.BILINEAR)
    rec('downscale', f)

    arr = np.asarray(pil, dtype=np.float32)

    # --- sensor: additive gaussian noise (THE big lever; fires most often) ---
    fired = rng.random() < 0.85
    sigma = rng.uniform(0.0, 16.0) if fired else None  # spans probe 8 break
    if fired:
        arr = arr + rng.normal(0.0, sigma, arr.shape)
    rec('noise', sigma)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # --- compression: JPEG re-encode (real photos are JPEG; renders are not) ---
    fired = rng.random() < 0.5
    q = int(rng.integers(20, 91)) if fired else None   # spans probe q=12
    if fired:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format='JPEG', quality=q)
        buf.seek(0)
        arr = np.asarray(Image.open(buf).convert('RGB'), dtype=np.uint8)
    rec('jpeg', q)

    return arr



# ---------------------------------------------------------------------------
# Multi-view augmentor
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    bg_dir: str | None = DEFAULT_BG_DIR   # background pool, recursively globbed
    bg_prob: float = 0.0           # per-view probability of compositing a background
    photometric: bool = False      # apply the HRNet photometric ISP pipeline per view

    @property
    def enabled(self) -> bool:
        return (self.bg_prob > 0 and bool(self.bg_dir)) or self.photometric


class MultiViewAugmentor:
    """Holds the background pool; applies bg composite + HRNet photometric jitter
    to one view's RGB."""

    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.bg_paths: list[Path] = []
        if cfg.bg_dir and cfg.bg_prob > 0:
            root = Path(cfg.bg_dir)
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                self.bg_paths.extend(sorted(root.rglob(ext)))
            if not self.bg_paths:
                raise FileNotFoundError(f"no background images found under {root}")

    def _fit_bg(self, bg: np.ndarray, H: int, W: int, rng) -> np.ndarray:
        """Scale-to-cover then random-crop the background to (H, W)."""
        Hb, Wb = bg.shape[:2]
        s = max(H / Hb, W / Wb)
        new_w, new_h = max(W, round(Wb * s)), max(H, round(Hb * s))
        bg = np.asarray(Image.fromarray(bg).resize((new_w, new_h), Image.BILINEAR))
        y0 = int(rng.integers(0, new_h - H + 1))
        x0 = int(rng.integers(0, new_w - W + 1))
        return bg[y0:y0 + H, x0:x0 + W]

    def apply(self, rgb: np.ndarray, face_mask: np.ndarray, rng) -> np.ndarray:
        H, W = rgb.shape[:2]
        # Background composite: keep only the rendered face pixels, swap everything
        # else -- including the interior eye holes -- for a random indoor photo.
        if self.bg_paths and rng.random() < self.cfg.bg_prob:
            bp = self.bg_paths[int(rng.integers(len(self.bg_paths)))]
            bg = np.asarray(Image.open(bp).convert("RGB"), dtype=np.uint8)
            bg = self._fit_bg(bg, H, W, rng).astype(np.float32) / 255.0
            rgb = np.where(face_mask[..., None], rgb, bg).astype(np.float32)
        if self.cfg.photometric:
            u8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            rgb = photometric(u8, rng).astype(np.float32) / 255.0
        return rgb

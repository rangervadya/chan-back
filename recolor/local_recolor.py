"""Deterministic recoloring fallback for Camelion (RGB uint8, OpenCV LAB)."""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from . import ops
from .base import RecolorRequest

_FLAT_MATERIALS = {"no_texture"}
_LIGHT_KEEP = {
    "metal": 1.05, "bronze": 1.05, "wood": 0.85, "leather": 0.90,
    "fabric": 0.85, "plastic": 0.72, "glass": 0.95,
    "ceramic": 0.72, "concrete": 0.90, "no_texture": 0.25,
}
# Keep only a fraction of the original chroma variation. Keeping 100% can
# reintroduce the source hue as visible stains after recoloring.
_TEXTURE_KEEP = {
    "metal": 0.30, "bronze": 0.32, "wood": 0.20, "leather": 0.18,
    "fabric": 0.20, "plastic": 0.12, "glass": 0.18,
    "ceramic": 0.12, "concrete": 0.25, "no_texture": 0.0,
}


def recolor(image_rgb: np.ndarray, mask: np.ndarray, request: RecolorRequest,
            feather_px: float = 2.0) -> np.ndarray:
    """Recolor the selected mask, keeping luminance relief and replacing hue."""
    if cv2 is None:
        raise RuntimeError("opencv-python-headless is required for recoloring")
    src = np.asarray(image_rgb, dtype=np.uint8)
    m = ops.to_mask_array(mask)
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("image_rgb must be an HxWx3 RGB image")
    if m.shape != src.shape[:2]:
        raise ValueError("mask dimensions must match image dimensions")
    if not m.any():
        return src.copy()

    target = tuple(int(np.clip(c, 0, 255)) for c in request.target_rgb)
    target_lab = _target_lab_rgb(target)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    out_lab = lab.copy()
    region = lab[m]
    l_mean = float(region[:, 0].mean())
    material = (request.material or "wood").lower()

    # Retain real luminance structure (shadows, folds, reflections) rather
    # than synthesizing arbitrary noise over the photograph.
    l_keep = _LIGHT_KEEP.get(material, 0.85)
    if material in _FLAT_MATERIALS:
        out_lab[m, 0] = target_lab[0] + (region[:, 0] - l_mean) * 0.20
    else:
        out_lab[m, 0] = target_lab[0] + (region[:, 0] - l_mean) * l_keep

    # Establish the chosen colour first; retain a restrained portion of local
    # chroma variation for believable texture without old-colour blotches.
    if material in _FLAT_MATERIALS:
        out_lab[m, 1] = target_lab[1]
        out_lab[m, 2] = target_lab[2]
    else:
        keep = _TEXTURE_KEEP.get(material, 0.18)
        for c in (1, 2):
            deviation = region[:, c] - float(region[:, c].mean())
            out_lab[m, c] = target_lab[c] + deviation * keep

    # Material-specific light response. All effects are luminance-only.
    out_lab = np.clip(out_lab, 0, 255)
    if material in ("metal", "bronze"):
        _apply_metal_shading(out_lab, m, lab[m], request)
    elif material == "wood":
        _blend_luminance_noise(out_lab, m, amount=2.2, cell=max(5, min(src.shape[:2]) // 100), seed=11)
    elif material == "fabric":
        _blend_luminance_weave(out_lab, m, amount=1.2, spacing=max(4, min(src.shape[:2]) // 160))
    elif material in ("leather", "concrete"):
        _blend_luminance_noise(out_lab, m, amount=1.4 if material == "leather" else 1.8,
                               cell=5 if material == "leather" else 3,
                               seed=31 if material == "leather" else 23)
    elif material in ("plastic", "ceramic"):
        _blend_luminance_gradient(out_lab, m, amount=1.0 if material == "plastic" else 0.7)
    elif material == "glass":
        # Tint the whole selected glass surface. Its transparency/shine is
        # represented by retained luminance; do not retain the old hue.
        pass

    rgb = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    rgb = _match_target_mean(rgb.astype(np.float32), m, target)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Feather only toward the inside of the object. A Gaussian blur by itself
    # spills alpha outside the selection and lightly tints surrounding pixels;
    # keep the exterior exactly identical to the uploaded photo.
    soft = ops.feather_mask(m, max(0.0, float(feather_px)))
    soft = np.clip(soft.astype(np.float32), 0.0, 1.0)
    soft[~m] = 0.0
    result = (rgb.astype(np.float32) * soft[..., None] +
              src.astype(np.float32) * (1.0 - soft[..., None]))
    return np.clip(result, 0, 255).astype(np.uint8)


def _target_lab_rgb(target_rgb: tuple[int, int, int]) -> np.ndarray:
    target = np.uint8([[list(target_rgb)]])
    return cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]


def _match_target_mean(image_rgb_f32: np.ndarray, mask: np.ndarray,
                       target_rgb: tuple[int, int, int]) -> np.ndarray:
    """Shift the region mean toward target while retaining local contrast."""
    if cv2 is None or not mask.any():
        return image_rgb_f32
    out = image_rgb_f32.astype(np.float32).copy()
    target = np.asarray(target_rgb, dtype=np.float32)
    for _ in range(3):
        region = out[mask]
        if not region.size:
            break
        delta = target - region.mean(axis=0)
        if float(np.abs(delta).max()) < 1.0:
            break
        out[mask] = np.clip(region + delta, 0, 255)
    return out


def _apply_metal_shading(lab: np.ndarray, mask: np.ndarray,
                         region: np.ndarray, request: RecolorRequest) -> None:
    vals = region[:, 0]
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    span = max(hi - lo, 1.0)
    stretched = np.clip((lab[mask, 0] - lo) / span, 0.0, 1.0)
    # Mildly enhance only the existing highlight/shadow distribution.
    lab[mask, 0] = np.clip(lab[mask, 0] * 0.75 + (lo + stretched * span * 1.08) * 0.25, 0, 255)
    if getattr(request, "patina", False):
        noise = _value_noise(*mask.shape, cell=max(8, min(mask.shape) // 55), seed=41)
        lab[mask, 0] = np.clip(lab[mask, 0] + (noise[mask] - 0.55) * 5.0, 0, 255)


def _blend_luminance_noise(lab: np.ndarray, mask: np.ndarray, amount: float,
                           cell: int, seed: int) -> None:
    noise = _value_noise(*mask.shape, cell=cell, seed=seed)
    lab[mask, 0] = np.clip(lab[mask, 0] + (noise[mask] - 0.5) * amount, 0, 255)


def _blend_luminance_weave(lab: np.ndarray, mask: np.ndarray, amount: float, spacing: int) -> None:
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    weave = np.sin(2 * np.pi * yy / max(1, spacing)) * np.sin(2 * np.pi * xx / max(1, spacing))
    lab[mask, 0] = np.clip(lab[mask, 0] + weave[mask] * amount, 0, 255)


def _blend_luminance_gradient(lab: np.ndarray, mask: np.ndarray, amount: float) -> None:
    ys, _ = np.nonzero(mask)
    if not ys.size:
        return
    h, w = mask.shape
    y0, y1 = int(ys.min()), int(ys.max())
    span = max(y1 - y0, 1)
    grad = np.clip(1.0 - 2.0 * (np.arange(h, dtype=np.float32) - y0) / span, -1.0, 1.0)
    field = np.repeat(grad[:, None], w, axis=1)
    field -= float(field[mask].mean())
    lab[mask, 0] = np.clip(lab[mask, 0] + field[mask] * amount, 0, 255)


def _value_noise(h: int, w: int, cell: int = 8, seed: int = 0) -> np.ndarray:
    cell = max(1, int(cell))
    gh, gw = max(2, h // cell + 2), max(2, w // cell + 2)
    rng = np.random.default_rng(seed)
    grid = rng.random((gh, gw), dtype=np.float32)
    if cv2 is not None:
        noise = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    else:  # pragma: no cover
        noise = np.resize(grid, (h, w)).astype(np.float32)
    lo, hi = float(noise.min()), float(noise.max())
    return ((noise - lo) / (hi - lo)).astype(np.float32) if hi > lo else np.zeros((h, w), np.float32)

"""Mask / image helpers shared by the recolor backends.

All functions accept and return numpy arrays. Masks are stored as boolean
arrays; ``to_mask_array`` normalises the many mask shapes the pipeline
produces (bool, uint8 0/255, float alpha, PIL-like L) into a single bool array
so callers never have to special-case the source.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def to_mask_array(mask) -> np.ndarray:
    """Return a boolean mask from any of the accepted mask representations."""
    if mask is None:
        raise ValueError("mask must not be None")
    arr = np.asarray(mask)
    if arr.dtype == bool:
        return arr
    if arr.dtype.kind == "f":
        return arr > 0.5
    if arr.dtype.kind in "iu":
        # 0/255 uint8 masks are the common case; anything > 0 counts.
        return arr > 0
    return arr.astype(bool)


def feather_mask(mask, feather_px: float) -> np.ndarray:
    """Soft alpha ramp from 0 at the outer edge to 1 well inside the mask.

    A plain Gaussian blur spills outside the selection and tints neighbouring
    pixels; this version clamps the ramp to 1 inside the eroded core so the
    exterior is untouched by the caller.
    """
    m = to_mask_array(mask)
    if feather_px <= 0 or cv2 is None:
        return m.astype(np.float32)
    feather_px = max(0.0, float(feather_px))
    k = max(1, int(round(feather_px * 2.0)))
    if k % 2 == 0:
        k += 1
    dist = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, k)
    dist = dist.astype(np.float32)
    alpha = np.clip(dist / max(feather_px, 1e-6), 0.0, 1.0)
    return alpha


def dilate_mask(mask, iterations: int) -> np.ndarray:
    """Morphological dilation, safe when cv2 is unavailable."""
    m = to_mask_array(mask)
    if iterations <= 0:
        return m
    if cv2 is None:
        return m
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(m.astype(np.uint8), kernel, iterations=int(iterations)) > 0


def erode_mask(mask, iterations: int) -> np.ndarray:
    m = to_mask_array(mask)
    if iterations <= 0 or cv2 is None:
        return m
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(m.astype(np.uint8), kernel, iterations=int(iterations)) > 0


def morphological_cleanup(mask, close_kernel: int = 3, open_kernel: int = 0,
                          fill_holes: bool = True) -> np.ndarray:
    """Close small gaps, optionally open specks and fill interior holes."""
    m = to_mask_array(mask)
    if cv2 is None:
        return m
    out = m.astype(np.uint8)
    if close_kernel and close_kernel > 0:
        k = max(3, int(close_kernel) | 1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    if open_kernel and open_kernel > 0:
        k = max(3, int(open_kernel) | 1)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    if fill_holes:
        h, w = out.shape
        # Standard hole fill: invert, flood-fill the background from the
        # corner, then re-invert and OR with the original mask. This turns
        # enclosed holes into foreground without touching the border.
        inv = np.where(out > 0, 0, 255).astype(np.uint8)
        ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inv, ff, (0, 0), 255)
        holes = np.where(inv == 0, 255, 0).astype(np.uint8)
        out = np.where((out | holes) > 0, 255, 0).astype(np.uint8)
    return out > 0


def bbox(mask) -> tuple[int, int, int, int] | None:
    """Bounding box (x1, y1, x2, y2) exclusive, or None for an empty mask."""
    m = to_mask_array(mask)
    ys, xs = np.nonzero(m)
    if not ys.size:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def mask_bbox_dims(mask) -> tuple[int, int]:
    b = bbox(mask)
    if b is None:
        return (0, 0)
    return (b[2] - b[0], b[3] - b[1])


def mean_rgb(image, mask) -> np.ndarray:
    """Mean RGB over the mask; zeros outside the mask."""
    m = to_mask_array(mask)
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("mean_rgb expects an HxWx3 image")
    if m.shape != arr.shape[:2]:
        raise ValueError("mask dimensions must match image dimensions")
    if not m.any():
        return np.zeros(3, dtype=np.float32)
    return arr[m].mean(axis=0)


def residual_original_fraction(out, original, mask) -> float:
    """Fraction of masked pixels still within a few units of the original."""
    m = to_mask_array(mask)
    if not m.any():
        return 0.0
    o = np.asarray(original, dtype=np.int16)
    n = np.asarray(out, dtype=np.int16)
    if o.shape != n.shape:
        return 0.0
    diff = np.abs(n[m].astype(np.int16) - o[m].astype(np.int16)).max(axis=1)
    return float((diff <= 10).mean())


def changed_fraction_outside_mask(out, original, mask) -> float:
    """Fraction of pixels outside the mask that changed at all."""
    m = to_mask_array(mask)
    o = np.asarray(original, dtype=np.int16)
    n = np.asarray(out, dtype=np.int16)
    if o.shape != n.shape or m.shape != o.shape[:2]:
        return 0.0
    outside = ~m
    if not outside.any():
        return 0.0
    diff = np.abs(n[outside].astype(np.int16) - o[outside].astype(np.int16)).max(axis=1)
    return float((diff > 2).mean())
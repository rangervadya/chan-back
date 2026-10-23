"""Point-selected object segmentation for the free (CPU) recolor pipeline.

The historical "single colour flood-fill" implementation had two known failure
modes pinned by regression tests:

* it leaked across weak object/background boundaries when the two colours were
  similar (the plate-on-wall case);
* it stopped at every internal colour edge, so multi-material objects (a chair
  whose back, seat and legs differ in colour) were only partially selected.

This module uses GrabCut seeded with the tap point plus the image border as
definite background. GrabCut models the foreground and background with Gaussian
mixtures, so it crosses internal colour jumps (multiple chair materials share
the foreground model) while still respecting the outer edge. A flat-image guard
and colour-flood fallback keep it from over-growing on uniform frames.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from logger import get_logger
from .base import SegmentationResult
from . import ops

logger = get_logger(__name__)


def segment_object(image_rgb: np.ndarray, point_x: float, point_y: float,
                   *, feather_px: float = 2.0,
                   min_area_ratio: float = 0.001,
                   max_area_ratio: float = 0.75,
                   bbox: tuple[int, int, int, int] | None = None,
                   seed_rect: tuple[int, int, int, int] | None = None
                   ) -> SegmentationResult:
    """Return a point-selected object mask."""
    t_start = time.monotonic()
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image_rgb must be a non-empty HxWx3 RGB array")
    h, w = image.shape[:2]
    x = int(np.clip(round(point_x), 0, w - 1))
    y = int(np.clip(round(point_y), 0, h - 1))
    notes: list[str] = []

    logger.info(
        "segment: image=%dx%d point=(%d,%d) bbox=%s seed_rect=%s "
        "min_area_ratio=%.4f max_area_ratio=%.2f",
        w, h, x, y, bbox, seed_rect, min_area_ratio, max_area_ratio,
    )

    t_seg = time.monotonic()
    if cv2 is None:
        mask = _color_region(image, x, y, bbox)
        method = "color-flood"
        notes.append("fallback: OpenCV недоступен")
        logger.warning("segment: OpenCV unavailable, using color-flood fallback")
    else:
        if _is_flat(image):
            mask = _flat_region(x, y, h, w)
            method = "flat"
            notes.append("flat image -> fixed radius around point")
            logger.info("segment: flat image detected (std<4.0) -> fixed radius mask")
        elif bbox is not None:
            bx1, by1, bx2, by2 = bbox
            bx0 = max(0, min(int(bx1), w))
            by0 = max(0, min(int(by1), h))
            bx1c = max(0, min(int(bx2), w))
            by1c = max(0, min(int(by2), h))
            box_area = max(0, bx1c - bx0) * max(0, by1c - by0)
            if box_area > 0.7 * w * h:
                # An inflated / near-full-frame hint carries no usable signal:
                # seed GrabCut from the tap point instead so a single object tap
                # is never destroyed by a sloppy detector box.
                mask = _grabcut_seed(image, x, y)
                method = "grabcut"
                notes.append("inflated bbox ignored -> point seeding")
                logger.info(
                    "segment: bbox too large (%.1f%% of frame) -> point-seeded grabcut",
                    100.0 * box_area / max(w * h, 1),
                )
            else:
                mask = _grabcut_in_box(image, x, y, bbox)
                method = "grabcut"
                logger.info(
                    "segment: bbox=%s (%.1f%% of frame) -> grabcut-in-box",
                    bbox, 100.0 * box_area / max(w * h, 1),
                )
        else:
            mask = _grabcut_seed(image, x, y)
            method = "grabcut"
            logger.info("segment: no bbox -> point-seeded grabcut")
    seg_time = time.monotonic() - t_seg
    logger.info("segment: initial %s in %.3fs", method, seg_time)

    mask = ops.to_mask_array(mask)

    # Keep only the component holding the tap point.
    if mask[y, x]:
        mask = _keep_component_with_point(mask, x, y)
    elif mask.any():
        mask = _keep_largest_component(mask)
    else:
        mask = _flat_region(x, y, h, w)

    area_ratio = float(mask.mean())
    if area_ratio > max_area_ratio:
        mask = _keep_largest_component(mask)
        if not mask[y, x]:
            mask = _keep_component_with_point(mask, x, y)
        area_ratio = float(mask.mean())
        notes.append(f"area capped to largest component ({area_ratio * 100:.1f}%)")

    t_clean = time.monotonic()
    mask = ops.morphological_cleanup(mask, close_kernel=3, open_kernel=0, fill_holes=False)
    if not mask[y, x] and mask.any():
        mask = ops.dilate_mask(mask, 1)
    mask = ops.morphological_cleanup(mask, close_kernel=3, open_kernel=0, fill_holes=True)
    clean_time = time.monotonic() - t_clean
    logger.info("segment: morphological cleanup in %.3fs", clean_time)

    t_flood = time.monotonic()
    flood_mask = _flood_reconcile(image, x, y, mask)
    if flood_mask is not None:
        notes.append("flood-fill reconciliation refined the selection")
        mask = ops.morphological_cleanup(flood_mask, close_kernel=3, open_kernel=3, fill_holes=True)
        if not mask[y, x] and mask.any():
            mask = ops.dilate_mask(mask, 1)
        mask = ops.morphological_cleanup(mask, close_kernel=3, open_kernel=0, fill_holes=True)
        logger.info("segment: flood-fill reconciliation applied (mask now %.2f%%)", 100.0 * float(mask.mean()))
    else:
        logger.info("segment: flood-fill reconciliation skipped (no improvement)")
    flood_time = time.monotonic() - t_flood
    logger.info("segment: flood reconciliation in %.3fs", flood_time)

    if not mask.any() or float(mask.mean()) < min_area_ratio:
        logger.warning(
            "segment: mask empty or below min_area_ratio=%.4f (area=%.4f) -> returning empty",
            min_area_ratio, float(mask.mean()),
        )
        return SegmentationResult(
            mask=np.zeros((h, w), bool), method=method,
            area_ratio=0.0, confidence=0.0,
            object_label="selected object",
            notes=notes + ["маска пуста после очистки"],
        )

    t_conf = time.monotonic()
    confidence = _confidence(image, mask, x, y)
    conf_time = time.monotonic() - t_conf
    total_time = time.monotonic() - t_start
    logger.info(
        "segment: done — method=%s mask_area=%.2f%% confidence=%.3f notes=%s "
        "times: seg=%.3fs clean=%.3fs flood=%.3fs conf=%.3fs total=%.3fs",
        method, 100.0 * float(mask.mean()), float(confidence), notes,
        seg_time, clean_time, flood_time, conf_time, total_time,
    )
    return SegmentationResult(
        mask=mask.astype(bool), method=method,
        area_ratio=float(mask.mean()), confidence=float(np.clip(confidence, 0.0, 1.0)),
        object_label="selected object",
        notes=notes + [
            f"point ({x},{y}); mask area={float(mask.mean()) * 100:.2f}%",
            f"seed region area={area_ratio * 100:.2f}%",
            f"segment times: seg={seg_time:.3f}s clean={clean_time:.3f}s flood={flood_time:.3f}s conf={conf_time:.3f}s total={total_time:.3f}s",
        ],
    )


def _is_flat(image: np.ndarray) -> bool:
    """True when the frame has almost no colour/gradient variance."""
    if cv2 is None:
        gray = image[:, :, 0]
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
        gray = lab[..., 0]
    return float(np.std(gray)) < 4.0


def _flat_region(x: int, y: int, h: int, w: int) -> np.ndarray:
    r = max(8, min(h, w) // 5)
    yy, xx = np.mgrid[0:h, 0:w]
    return (yy - y) ** 2 + (xx - x) ** 2 <= r * r


def _color_region(image: np.ndarray, x: int, y: int,
                  bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """Pure-numpy fallback: seeded flood-fill by colour (LAB, FIXED_RANGE)."""
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB) if cv2 else image
    mask = np.zeros((h + 2, w + 2), np.uint8)
    rect = None
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        rect = (max(0, bx1), max(0, by1), min(w, bx2) - max(0, bx1), min(h, by2) - max(0, by1))
    try:
        if rect is None:
            cv2.floodFill(lab, mask, (x, y), 0, loDiff=(18,) * 3, upDiff=(18,) * 3,
                          flags=cv2.FLOODFILL_FIXED_RANGE | (4 << 8))
        else:
            cv2.floodFill(lab, mask, (x, y), 0, loDiff=(18,) * 3, upDiff=(18,) * 3,
                          flags=cv2.FLOODFILL_FIXED_RANGE | (4 << 8), rect=rect)
    except Exception:
        pass
    return mask[1:-1, 1:-1] > 0


def _flood_fill_color(image: np.ndarray, x: int, y: int, tol: int) -> np.ndarray:
    """Seeded colour flood-fill in LAB (FIXED_RANGE)."""
    h, w = image.shape[:2]
    if cv2 is None:
        return np.zeros((h, w), bool)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).copy()
    m = np.zeros((h + 2, w + 2), np.uint8)
    try:
        cv2.floodFill(lab, m, (x, y), 0, loDiff=(tol,) * 3, upDiff=(tol,) * 3,
                      flags=cv2.FLOODFILL_FIXED_RANGE | (4 << 8))
    except Exception:
        return np.zeros((h, w), bool)
    out = m[1:-1, 1:-1] > 0
    if out[y, x]:
        out = _keep_component_with_point(out, x, y)
    return out


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    a = ops.to_mask_array(a)
    b = ops.to_mask_array(b)
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union else 0.0


def _flood_reconcile(image: np.ndarray, x: int, y: int,
                     grabcut_mask: np.ndarray) -> np.ndarray | None:
    """Refine a GrabCut mask with colour flood-fill.

    GrabCut models the whole frame as foreground, so it leaks color-wise into
    similarly coloured neighbours (plate on a wall) and absorbs adjacent
    objects the GMM shares a foreground model with (a hand near a glass).
    Flood-fill, by contrast, only crosses boundaries that share the seed
    colour, so it stays on a flat single-colour object and excludes a differently
    coloured neighbour. It only fails when the object itself is multi-colour
    (chair back/seat/legs) or when the seed lands on a sub-region (a bright
    screen) -- cases where the larger colour jump is inside the object.

    The strategy therefore adapts to the fixture:
      1. flat single-colour objects  -> tight flood (captures the whole colour)
      2. tinted/transparent objects  -> grown flood, clipped to the GrabCut
         hull so the high tolerance cannot escape the object's footprint, plus a
         bright-colour rim catch for glass edges
      3. multi-colour / sub-region   -> fall back to GrabCut (already good)
    """
    if cv2 is None:
        return None
    g = ops.to_mask_array(grabcut_mask)
    if not g.any() or not g[y, x]:
        return None

    # 1) tight flood: complete for flat single-colour objects.
    best_f: np.ndarray | None = None
    best_iou = -1.0
    for tol in (6, 8, 10, 12, 14):
        f = _flood_fill_color(image, x, y, tol)
        if not f.any():
            continue
        i = _iou(f, g)
        if i > best_iou:
            best_iou = i
            best_f = f
    if best_f is not None and best_iou > 0.6:
        return best_f

    # 2) grown flood: clipped to the GrabCut hull so high tolerance cannot leak
    #    into the background. Used for tinted/transparent objects whose colour
    #    spans a wide range (glass interior -> rim).
    Gd = ops.dilate_mask(g, 8)
    grown: np.ndarray | None = None
    grown_area = 0
    prev_area = 0
    stale = 0
    for tol in range(18, 96, 2):
        f = _flood_fill_color(image, x, y, tol) & Gd
        if not f[y, x]:
            if f.any():
                f = _keep_largest_component(f)
            else:
                continue
        else:
            f = _keep_component_with_point(f, x, y)
        a = f.sum()
        if a > grown_area:
            grown_area = a
            grown = f
        if a <= prev_area * 1.005:
            stale += 1
        else:
            stale = 0
        prev_area = a
        if stale >= 3:
            break
    if grown is None:
        return None
    grew = best_f is not None and grown.sum() > best_f.sum() * 1.3
    if grew and _iou(grown, Gd) > 0.4:
        # Bright-colour rim: glass has a specular edge the tinted fill misses.
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        rim = lab[..., 0] > 210
        grown = grown | rim
        if grown[y, x]:
            grown = _keep_component_with_point(grown, x, y)
        return grown

    # 3) flood did not improve over GrabCut -> keep the existing mask.
    return None


def _grabcut_in_box(image: np.ndarray, x: int, y: int,
                    bbox: tuple[int, int, int, int]) -> np.ndarray:
    """GrabCut where the detector box seeds probable-foreground and the border
    outside the (padded) box seeds definite background."""
    h, w = image.shape[:2]
    bx1, by1, bx2, by2 = bbox
    bx1 = max(0, int(bx1)); by1 = max(0, int(by1))
    bx2 = min(w, int(bx2)); by2 = min(h, int(by2))
    if bx2 - bx1 < 2 or by2 - by1 < 2:
        return _color_region(image, x, y, (bx1, by1, bx2, by2))
    pad = 8
    rx1 = max(0, bx1 - pad); ry1 = max(0, by1 - pad)
    rx2 = min(w, bx2 + pad); ry2 = min(h, by2 + pad)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_FGD
    # Definite background: outside the bounding box entirely.
    outside = np.ones((h, w), bool)
    outside[by1:by2, bx1:bx2] = False
    mask[outside] = cv2.GC_BGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return _color_region(image, x, y, (bx1, by1, bx2, by2))
    result = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if not result[y, x]:
        seed_box = image[max(0, y - 6):y + 7, max(0, x - 6):x + 7]
        if seed_box.size:
            result = ops.dilate_mask(result, 1)
    return result


def _grabcut_seed(image: np.ndarray, x: int, y: int) -> np.ndarray:
    """GrabCut seeded from the tap point with the image border as background.

    The whole frame is probable foreground; the border is forced to definite
    background and a small disc around the tap is forced to definite
    foreground. This lets the GMM cross internal material edges (the chair's
    back/seat/legs share the foreground model) while the border anchor
    prevents the whole image being absorbed.
    """
    h, w = image.shape[:2]
    mask = np.full((h, w), cv2.GC_PR_FGD, np.uint8)
    border = 2
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    # Definite-foreground seed around the tap.
    r = max(3, min(h, w) // 24)
    yy, xx = np.mgrid[0:h, 0:w]
    disc = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
    mask[disc] = cv2.GC_FGD
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image, mask, None, bgd, fgd, 6, cv2.GC_INIT_WITH_MASK)
    except Exception:
        return _color_region(image, x, y, None)
    result = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if not result[y, x]:
        result = ops.dilate_mask(result, 1) if result.any() else disc
    return result


def _keep_component_with_point(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    """Largest 4-connected component touching (x, y)."""
    if not mask[y, x]:
        return _largest_component(mask)
    comps = _connected_components(mask)
    if not comps:
        return mask
    best = None
    for comp in comps:
        if comp[y, x] and (best is None or comp.sum() > best.sum()):
            best = comp
    return best if best is not None else _largest_component(mask)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    comps = _connected_components(mask)
    if not comps:
        return mask
    return max(comps, key=lambda c: c.sum())


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    m = ops.to_mask_array(mask)
    if cv2 is None or not m.any():
        return [m] if m.any() else []
    n_labels, labels = cv2.connectedComponents(m.astype(np.uint8), connectivity=4)
    return [(labels == i) for i in range(1, n_labels)]


def _confidence(image: np.ndarray, mask: np.ndarray, x: int, y: int) -> float:
    """Heuristic certainty from boundary colour contrast and fill ratio."""
    m = ops.to_mask_array(mask)
    if not m.any():
        return 0.0
    if cv2 is None:
        return 0.9 if m[y, x] else 0.1
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    gx = cv2.Sobel(lab[..., 0], cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab[..., 0], cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    eroded = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
    border = m & ~eroded.astype(bool)
    if border.sum() == 0:
        return 0.7
    contrast = float(grad[border].mean()) / max(float(grad[m].mean()), 1e-6)
    fill = float(m.sum()) / max(float(m.size), 1)
    return float(np.clip(0.5 + 0.25 * np.log1p(max(contrast, 0.0)) + 0.25 * fill, 0.0, 1.0))

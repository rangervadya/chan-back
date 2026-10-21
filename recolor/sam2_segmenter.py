"""Prompt-based SAM 2 segmentation for point-selected objects.

SAM 2 is loaded once and cached in the process. This module intentionally does
not silently fall back to colour/contour heuristics: a wrong mask can recolour
neighbouring furniture, so a missing model is an explicit service error.
"""
from __future__ import annotations

import os
import threading
from typing import Any

import numpy as np

from .base import SegmentationResult

_LOCK = threading.Lock()
_PREDICTOR: Any | None = None
_PREDICTOR_MODEL_ID: str | None = None
_PREDICTOR_DEVICE: str | None = None


class Sam2UnavailableError(RuntimeError):
    """SAM 2 cannot be loaded or used in the current runtime."""


def is_loaded() -> bool:
    return _PREDICTOR is not None


def loaded_model_info() -> dict[str, str | bool | None]:
    return {
        "loaded": _PREDICTOR is not None,
        "model": _PREDICTOR_MODEL_ID,
        "device": _PREDICTOR_DEVICE,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


DEFAULT_MODEL = "facebook/sam2.1-hiera-base-plus"


def _model_variant(model_id: str | None) -> str:
    """Extract the Hiera variant suffix from a model path or repo id.

    e.g. ``facebook/sam2.1-hiera-base-plus`` -> ``base-plus``,
    ``/models/facebook--sam2.1-hiera-large`` -> ``large``.
    """
    if not model_id:
        return "base-plus"
    base = os.path.basename(model_id.rstrip("/"))
    name = base.replace("facebook--", "").replace("facebook/", "")
    for prefix in ("sam2.1-hiera-", "sam2-hiera-", "sam2.1-", "sam2-"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _resolve_model(requested: str | None = None) -> tuple[str, bool]:
    """Resolve the effective model path and local_files_only flag.

    Resolution order:
      1. Explicit ``model_name`` argument (used by callers/tests).
      2. ``SAM2_LOCAL_MODEL_PATH`` env var — a local directory path; implies
         ``local_files_only=True`` so no network is attempted.
      3. ``SAM2_MODEL_NAME`` env var (HF repo id or local path).
      4. ``DEFAULT_MODEL``.

    Returns ``(model_path_or_id, local_files_only)``.
    """
    local_path = os.getenv("SAM2_LOCAL_MODEL_PATH", "").strip()
    if local_path:
        return local_path, True

    requested = (requested or os.getenv("SAM2_MODEL_NAME", DEFAULT_MODEL)).strip()
    if not requested:
        requested = DEFAULT_MODEL

    local_files_only = os.getenv("SAM2_LOCAL_FILES_ONLY", "1").strip().lower() in {"1", "true", "yes", "on"}
    return requested, local_files_only


def _get_predictor(model_name: str | None = None):
    global _PREDICTOR, _PREDICTOR_MODEL_ID, _PREDICTOR_DEVICE
    requested, local_files_only = _resolve_model(model_name)
    if _PREDICTOR is not None and _PREDICTOR_MODEL_ID == requested:
        return _PREDICTOR

    with _LOCK:
        if _PREDICTOR is not None and _PREDICTOR_MODEL_ID == requested:
            return _PREDICTOR
        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:  # model package is an installation/runtime dependency
            raise Sam2UnavailableError(
                "SAM 2 runtime is not installed. Rebuild the server image with "
                "server/requirements.sam2.txt; do not switch to the heuristic mask silently."
            ) from exc

        try:
            threads = max(1, int(os.getenv("TORCH_NUM_THREADS", "2")))
        except ValueError:
            threads = 2
        torch.set_num_threads(threads)
        try:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"
        try:
            predictor = SAM2ImagePredictor.from_pretrained(
                requested, device=device, local_files_only=local_files_only
            )
            if getattr(predictor, "model", None) is not None:
                predictor.model = predictor.model.to(device).eval()
        except Exception as exc:
            raise Sam2UnavailableError(
                f"Could not load SAM 2 model '{requested}'. Check server internet access, "
                "Hugging Face availability, cache permissions, and available RAM."
            ) from exc

        _PREDICTOR = predictor
        _PREDICTOR_MODEL_ID = requested
        _PREDICTOR_DEVICE = device
        return _PREDICTOR


def _prediction_candidates(predictor, x: int, y: int) -> list[tuple[np.ndarray, float]]:
    masks, scores, _ = predictor.predict(
        point_coords=np.asarray([[x, y]], dtype=np.float32),
        point_labels=np.asarray([1], dtype=np.int32),
        multimask_output=True,
    )
    mask_array = _as_numpy(masks)
    score_array = _as_numpy(scores).astype(np.float32).reshape(-1)
    if mask_array.ndim == 2:
        mask_array = mask_array[None, ...]
    if mask_array.ndim != 3:
        raise Sam2UnavailableError(f"Unexpected SAM 2 mask dimensions: {mask_array.shape}")

    candidates: list[tuple[np.ndarray, float]] = []
    for index, raw_mask in enumerate(mask_array):
        mask = np.asarray(raw_mask > 0, dtype=bool)
        if not mask.any():
            continue
        score = float(score_array[index]) if index < len(score_array) else 0.0
        candidates.append((mask, score))
    return candidates


def _select_candidate(candidates: list[tuple[np.ndarray, float]], x: int, y: int,
                      reference: np.ndarray | None = None) -> tuple[np.ndarray, float] | None:
    """Select a mask that contains the prompt; for jitter points prefer same instance."""
    eligible = [(mask, score) for mask, score in candidates if mask[y, x]]
    if not eligible:
        eligible = candidates
    if not eligible:
        return None
    if reference is None:
        return max(eligible, key=lambda item: (item[1], int(item[0].sum())))

    ref_area = max(int(reference.sum()), 1)
    scored: list[tuple[float, np.ndarray, float]] = []
    for mask, score in eligible:
        intersection = int(np.logical_and(mask, reference).sum())
        union = int(np.logical_or(mask, reference).sum())
        iou = intersection / max(union, 1)
        # Reject a mask which likely jumped to another instance. Among the
        # candidates aligned to the reference, prefer confidence and overlap.
        if iou < 0.45:
            continue
        area_ratio = min(int(mask.sum()), ref_area) / max(int(mask.sum()), ref_area, 1)
        scored.append((0.70 * iou + 0.25 * score + 0.05 * area_ratio, mask, score))
    if not scored:
        return None
    _, chosen, chosen_score = max(scored, key=lambda item: item[0])
    return chosen, chosen_score


def segment_with_sam2(
    image_rgb: np.ndarray,
    point_x: float,
    point_y: float,
    *,
    model_name: str | None = None,
) -> SegmentationResult:
    """Return a point-selected object mask from SAM 2, with conservative jitter voting."""
    image = np.asarray(image_rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image_rgb must be a non-empty HxWx3 RGB array")
    h, w = image.shape[:2]
    x = int(np.clip(round(point_x), 0, w - 1))
    y = int(np.clip(round(point_y), 0, h - 1))
    predictor = _get_predictor(model_name)

    try:
        import torch
        with torch.inference_mode():
            predictor.set_image(image)
            initial = _select_candidate(_prediction_candidates(predictor, x, y), x, y)
            if initial is None:
                raise Sam2UnavailableError("SAM 2 did not return a mask for the tapped point")
            primary, primary_score = initial

            # Jitter points by a few source pixels to stabilize uncertain edges.
            # Reuse the SAM 2 image embedding; only the lightweight mask decoder runs again.
            offset = max(3, int(round(np.hypot(h, w) * 0.004)))
            point_specs = [
                (min(w - 1, x + offset), y),
                (max(0, x - offset), y),
                (x, min(h - 1, y + offset)),
                (x, max(0, y - offset)),
            ]
            selected = [primary]
            selected_scores = [primary_score]
            for jx, jy in point_specs:
                candidates = _prediction_candidates(predictor, jx, jy)
                chosen = _select_candidate(candidates, jx, jy, reference=primary)
                if chosen is not None:
                    candidate, score = chosen
                    selected.append(candidate)
                    selected_scores.append(score)
    except Sam2UnavailableError:
        raise
    except Exception as exc:
        raise Sam2UnavailableError(f"SAM 2 segmentation inference failed: {type(exc).__name__}: {exc}") from exc

    # Vote only among masks that match the selected instance. Keep the main mask
    # unless the consensus is substantial and agrees with it; avoids both noisy
    # halos and a consensus that chops off furniture parts.
    votes = np.stack(selected, axis=0).sum(axis=0)
    consensus = votes >= (len(selected) / 2.0)
    primary_area = int(primary.sum())
    consensus_area = int(consensus.sum())
    inter = int(np.logical_and(primary, consensus).sum())
    union = int(np.logical_or(primary, consensus).sum())
    iou = inter / max(union, 1)
    if (
        len(selected) >= 3
        and consensus_area >= 0.70 * primary_area
        and consensus_area <= 1.20 * primary_area
        and iou >= 0.70
    ):
        mask = consensus
    else:
        mask = primary

    # Reject obviously invalid masks instead of silently repainting the scene.
    area_ratio = float(mask.mean())
    if not mask[y, x]:
        raise Sam2UnavailableError("SAM 2 mask did not include the selected pixel; retap the object")
    if area_ratio < 0.0005 or area_ratio > 0.78:
        raise Sam2UnavailableError(
            f"SAM 2 produced an implausible object mask ({area_ratio * 100:.1f}% of image); retap the object"
        )

    # Close tiny pinholes only; do not fill broad holes such as the gap between
    # the arms and seat of a chair. The final recolor still feathers the boundary.
    try:
        from .ops import morphological_cleanup
        mask = morphological_cleanup(mask, close_kernel=3, open_kernel=0, fill_holes=False)
    except Exception:
        pass

    return SegmentationResult(
        mask=mask.astype(bool),
        method=f"sam2.1-hiera-{_model_variant(_PREDICTOR_MODEL_ID)}-point",
        area_ratio=float(mask.mean()),
        confidence=float(np.clip(np.mean(selected_scores), 0.0, 1.0)),
        object_label="selected object",
        notes=[
            f"SAM 2 model={_PREDICTOR_MODEL_ID}; device={_PREDICTOR_DEVICE}",
            f"point ({x},{y}); mask area={float(mask.mean()) * 100:.2f}%",
            f"jitter masks={len(selected)}; primary score={primary_score:.3f}; primary/consensus IoU={iou:.3f}",
        ],
    )

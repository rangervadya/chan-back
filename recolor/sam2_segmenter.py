"""
Prompt-based SAM 2 segmentation for point-selected objects.

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
from logger import get_logger


_logger = get_logger(__name__)

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
    """
    Extract the Hiera variant suffix from a model path or repo ID.

    Examples:
        facebook/sam2.1-hiera-base-plus -> base-plus
        /app/models/sam2.1-hiera-base-plus -> base-plus
        facebook/sam2.1-hiera-tiny -> tiny
    """
    if not model_id:
        return "base-plus"

    base = os.path.basename(model_id.rstrip("/"))
    name = base.replace("facebook--", "").replace("facebook/", "")

    for prefix in (
        "sam2.1-hiera-",
        "sam2.1-hira-",
        "sam2-hiera-",
        "sam2.1-",
        "sam2-",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]

    return name


# IMPORTANT:
# build_sam2() resolves config_file using Hydra's package config search path.
# These must be config names relative to the installed sam2 package,
# NOT absolute filesystem paths to YAML files.
_CONFIG_FILES = {
    "base-plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "b+": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "t": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "s": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
    "l": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def _resolve_model(
    requested: str | None = None,
) -> tuple[str, bool]:
    """
    Resolve the effective model path and local_files_only flag.

    Resolution order:
      1. Existing SAM2_LOCAL_MODEL_PATH, if configured.
      2. Explicit model_name supplied by the caller.
      3. SAM2_MODEL_NAME environment variable.
      4. DEFAULT_MODEL.

    Returns:
        (model_path_or_huggingface_repo_id, local_files_only)
    """
    local_path = os.getenv("SAM2_LOCAL_MODEL_PATH", "").strip()
    _logger.info(
        "_resolve_model: requested=%s SAM2_LOCAL_MODEL_PATH=%s SAM2_MODEL_NAME=%s SAM2_LOCAL_FILES_ONLY=%s",
        requested, local_path, os.getenv("SAM2_MODEL_NAME", DEFAULT_MODEL),
        os.getenv("SAM2_LOCAL_FILES_ONLY", "0"),
    )

    # Prefer the explicitly configured local model directory.
    if local_path and os.path.isdir(local_path):
        _logger.info("_resolve_model: using local model dir=%s (local_files_only=True)", local_path)
        return local_path, True

    if local_path:
        _logger.warning(
            "SAM2_LOCAL_MODEL_PATH='%s' does not exist; "
            "will try the configured model ID instead",
            local_path,
        )

    resolved = (
        requested
        if requested is not None
        else os.getenv("SAM2_MODEL_NAME", DEFAULT_MODEL)
    )
    resolved = resolved.strip()

    if not resolved:
        resolved = DEFAULT_MODEL

    # Do not pass a nonexistent local filesystem path to from_pretrained().
    if (
        resolved != DEFAULT_MODEL
        and not os.path.isdir(resolved)
        and (
            resolved.startswith("/")
            or resolved.startswith("./")
            or resolved.startswith("../")
        )
    ):
        _logger.warning(
            "Model path '%s' does not exist; falling back to '%s'",
            resolved,
            DEFAULT_MODEL,
        )
        resolved = DEFAULT_MODEL

    local_files_only = (
        os.getenv("SAM2_LOCAL_FILES_ONLY", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )

    _logger.info("_resolve_model: resolved=%s local_files_only=%s", resolved, local_files_only)
    return resolved, local_files_only


def _find_local_file(
    directory: str,
    patterns: list[str],
) -> str | None:
    """Find the first local file whose name ends with one of the patterns."""
    try:
        for entry in sorted(os.listdir(directory)):
            lower = entry.lower()

            for pattern in patterns:
                if lower.endswith(pattern.lower()):
                    path = os.path.join(directory, entry)

                    if os.path.isfile(path):
                        return path

    except OSError as exc:
        _logger.warning(
            "Could not inspect model directory '%s': %s",
            directory,
            exc,
        )

    return None


def _load_local_predictor(model_path: str, device: str) -> Any:
    """
    Load SAM 2 from local checkpoint weights.

    The checkpoint is loaded from model_path.
    The Hydra configuration is resolved from the installed sam2 package.

    IMPORTANT:
    Passing an absolute YAML filesystem path to build_sam2(config_file=...)
    does not work with the standard SAM 2 Hydra configuration search path.
    """
    import sam2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not os.path.isdir(model_path):
        raise Sam2UnavailableError(
            f"Local SAM 2 model directory does not exist: '{model_path}'"
        )

    # Find the checkpoint on disk.
    checkpoint = _find_local_file(model_path, [".pt"])

    if not checkpoint:
        raise Sam2UnavailableError(
            f"No checkpoint (.pt) file found in '{model_path}'. "
            "Expected a SAM 2 checkpoint such as "
            "'sam2.1_hiera_base_plus.pt'."
        )

    # Determine which official SAM 2.1 configuration to use.
    variant = _model_variant(model_path)
    config_name = _CONFIG_FILES.get(variant)

    if not config_name:
        raise Sam2UnavailableError(
            f"Unsupported SAM 2 model variant '{variant}' "
            f"for local model directory '{model_path}'."
        )

    # Verify that the selected configuration exists in the installed package.
    package_root = os.path.dirname(sam2.__file__)
    config_path = os.path.join(
        package_root,
        *config_name.split("/"),
    )

    if not os.path.isfile(config_path):
        local_yaml = _find_local_file(model_path, [".yaml", ".yml"])

        raise Sam2UnavailableError(
            "The required SAM 2 configuration is missing from the installed "
            f"sam2 package: '{config_path}'. "
            f"Local YAML found: '{local_yaml}'. "
            "The standard build_sam2() function resolves configuration names "
            "through Hydra; it cannot load an arbitrary absolute YAML path "
            "directly as config_file. Check the SAM 2 package installation."
        )

    local_yaml = _find_local_file(model_path, [".yaml", ".yml"])

    _logger.info(
        "Loading SAM 2 from local directory: "
        "checkpoint=%s, config=%s, device=%s",
        os.path.basename(checkpoint),
        config_name,
        device,
    )

    if local_yaml:
        _logger.info(
            "Local YAML file exists: %s; "
            "using the matching bundled SAM 2 configuration",
            os.path.basename(local_yaml),
        )

    # CRITICAL FIXES:
    # 1. config_file must be a Hydra config name, not an absolute YAML path.
    # 2. The official build_sam2() checkpoint argument is ckpt_path.
    sam_model = build_sam2(
        config_file=config_name,
        ckpt_path=checkpoint,
        device=device,
    )

    predictor = SAM2ImagePredictor(sam_model)

    return predictor


def _get_predictor(model_name: str | None = None) -> Any:
    global _PREDICTOR
    global _PREDICTOR_MODEL_ID
    global _PREDICTOR_DEVICE

    requested, local_files_only = _resolve_model(model_name)

    if (
        _PREDICTOR is not None
        and _PREDICTOR_MODEL_ID == requested
    ):
        _logger.info("_get_predictor: cached predictor reused (model=%s)", requested)
        return _PREDICTOR

    with _LOCK:
        # Another request may have loaded the model while we were waiting.
        if (
            _PREDICTOR is not None
            and _PREDICTOR_MODEL_ID == requested
        ):
            _logger.info("_get_predictor: cached predictor reused (model=%s)", requested)
            return _PREDICTOR

        try:
            import torch
            from sam2.sam2_image_predictor import SAM2ImagePredictor

        except Exception as exc:
            raise Sam2UnavailableError(
                "SAM 2 runtime is not installed or could not be imported. "
                "Rebuild the server image with the SAM 2 dependencies."
            ) from exc

        try:
            threads = max(1, int(os.getenv("TORCH_NUM_THREADS", "2")))
        except ValueError:
            threads = 2

        torch.set_num_threads(threads)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

        _logger.info("_get_predictor: loading model=%s device=%s threads=%d", requested, device, threads)

        try:
            if os.path.isdir(requested):
                predictor = _load_local_predictor(requested, device)
            else:
                # This branch is used only when a local directory is not
                # selected and the resolved model is a Hugging Face model ID.
                _logger.info("_get_predictor: loading from HuggingFace repo=%s", requested)
                predictor = SAM2ImagePredictor.from_pretrained(
                    requested,
                    device=device,
                    local_files_only=local_files_only,
                )

            if getattr(predictor, "model", None) is not None:
                predictor.model = predictor.model.to(device).eval()

        except Sam2UnavailableError:
            raise

        except Exception as exc:
            raise Sam2UnavailableError(
                f"Could not load SAM 2 model '{requested}': "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        _PREDICTOR = predictor
        _PREDICTOR_MODEL_ID = requested
        _PREDICTOR_DEVICE = device

        _logger.info(
            "SAM 2 model loaded successfully: model=%s, device=%s",
            requested,
            device,
        )

        return _PREDICTOR


def _prediction_candidates(
    predictor: Any,
    x: int,
    y: int,
) -> list[tuple[np.ndarray, float]]:
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
        raise Sam2UnavailableError(
            f"Unexpected SAM 2 mask dimensions: {mask_array.shape}"
        )

    candidates: list[tuple[np.ndarray, float]] = []

    for index, raw_mask in enumerate(mask_array):
        mask = np.asarray(raw_mask > 0, dtype=bool)

        if not mask.any():
            continue

        score = (
            float(score_array[index])
            if index < len(score_array)
            else 0.0
        )

        candidates.append((mask, score))

    return candidates


def _select_candidate(
    candidates: list[tuple[np.ndarray, float]],
    x: int,
    y: int,
    reference: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """
    Select a mask containing the prompt.

    For jitter points, prefer a mask representing the same object instance
    as the primary mask.
    """
    eligible = [
        (mask, score)
        for mask, score in candidates
        if mask[y, x]
    ]

    if not eligible:
        eligible = candidates

    if not eligible:
        return None

    if reference is None:
        return max(
            eligible,
            key=lambda item: (
                item[1],
                int(item[0].sum()),
            ),
        )

    ref_area = max(int(reference.sum()), 1)

    scored: list[tuple[float, np.ndarray, float]] = []

    for mask, score in eligible:
        intersection = int(
            np.logical_and(mask, reference).sum()
        )
        union = int(
            np.logical_or(mask, reference).sum()
        )

        iou = intersection / max(union, 1)

        # Reject masks that likely jumped to a different instance.
        if iou < 0.45:
            continue

        mask_area = int(mask.sum())

        area_ratio = (
            min(mask_area, ref_area)
            / max(mask_area, ref_area, 1)
        )

        combined_score = (
            0.70 * iou
            + 0.25 * score
            + 0.05 * area_ratio
        )

        scored.append(
            (combined_score, mask, score)
        )

    if not scored:
        return None

    _, chosen, chosen_score = max(
        scored,
        key=lambda item: item[0],
    )

    return chosen, chosen_score


def segment_with_sam2(
    image_rgb: np.ndarray,
    point_x: float,
    point_y: float,
    *,
    model_name: str | None = None,
) -> SegmentationResult:
    """
    Return a point-selected object mask from SAM 2.

    A small number of neighbouring points are used to stabilize uncertain
    mask boundaries. If the primary mask is not reliable, it is retained
    rather than replacing it with an inconsistent consensus.
    """
    image = np.asarray(image_rgb, dtype=np.uint8)

    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.size == 0
    ):
        raise ValueError(
            "image_rgb must be a non-empty HxWx3 RGB array"
        )

    h, w = image.shape[:2]

    x = int(np.clip(round(point_x), 0, w - 1))
    y = int(np.clip(round(point_y), 0, h - 1))

    _logger.info(
        "segment_with_sam2: image=%dx%d point=(%d,%d) model_name=%s",
        w, h, x, y, model_name,
    )

    predictor = _get_predictor(model_name)

    try:
        import torch

        with torch.inference_mode():
            predictor.set_image(image)
            _logger.info("segment_with_sam2: image embedding set")

            initial = _select_candidate(
                _prediction_candidates(predictor, x, y),
                x,
                y,
            )

            if initial is None:
                raise Sam2UnavailableError(
                    "SAM 2 did not return a mask for the tapped point"
                )

            primary, primary_score = initial
            _logger.info(
                "segment_with_sam2: primary mask score=%.3f area=%dpx",
                float(primary_score), int(primary.sum()),
            )

            # Jitter points by a few source pixels to stabilize uncertain
            # boundaries. The image embedding is reused between predictions.
            offset = max(
                3,
                int(round(np.hypot(h, w) * 0.004)),
            )

            point_specs = [
                (min(w - 1, x + offset), y),
                (max(0, x - offset), y),
                (x, min(h - 1, y + offset)),
                (x, max(0, y - offset)),
            ]

            selected = [primary]
            selected_scores = [primary_score]

            for jx, jy in point_specs:
                candidates = _prediction_candidates(
                    predictor,
                    jx,
                    jy,
                )

                chosen = _select_candidate(
                    candidates,
                    jx,
                    jy,
                    reference=primary,
                )

                if chosen is not None:
                    candidate, score = chosen
                    selected.append(candidate)
                    selected_scores.append(score)
                    _logger.info(
                        "segment_with_sam2: jitter point (%d,%d) score=%.3f area=%dpx",
                        jx, jy, float(score), int(candidate.sum()),
                    )
                else:
                    _logger.info(
                        "segment_with_sam2: jitter point (%d,%d) rejected (no match)",
                        jx, jy,
                    )

    except Sam2UnavailableError:
        raise

    except Exception as exc:
        raise Sam2UnavailableError(
            "SAM 2 segmentation inference failed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # Vote only among masks that match the selected object.
    # Keep the primary mask unless the consensus is substantial and agrees
    # with it, avoiding a noisy boundary or an incorrectly truncated object.
    votes = np.stack(selected, axis=0).sum(axis=0)

    consensus = votes >= (len(selected) / 2.0)

    primary_area = int(primary.sum())
    consensus_area = int(consensus.sum())

    intersection = int(
        np.logical_and(primary, consensus).sum()
    )
    union = int(
        np.logical_or(primary, consensus).sum()
    )

    iou = intersection / max(union, 1)

    if (
        len(selected) >= 3
        and consensus_area >= 0.70 * primary_area
        and consensus_area <= 1.20 * primary_area
        and iou >= 0.70
    ):
        mask = consensus
    else:
        mask = primary

    # Reject implausible masks rather than recolouring the wrong object.
    area_ratio = float(mask.mean())

    if not mask[y, x]:
        raise Sam2UnavailableError(
            "SAM 2 mask did not include the selected pixel; "
            "please tap the object again"
        )

    if area_ratio < 0.0005 or area_ratio > 0.78:
        raise Sam2UnavailableError(
            "SAM 2 produced an implausible object mask "
            f"({area_ratio * 100:.1f}% of image); "
            "please tap the object again"
        )

    # Close tiny pinholes only. Preserve large gaps, such as spaces between
    # the seat and arms of a chair.
    try:
        from .ops import morphological_cleanup

        mask = morphological_cleanup(
            mask,
            close_kernel=3,
            open_kernel=0,
            fill_holes=False,
        )

    except Exception as exc:
        _logger.debug(
            "Morphological cleanup skipped: %s",
            exc,
        )

    return SegmentationResult(
        mask=mask.astype(bool),
        method=(
            f"sam2.1-hiera-{_model_variant(_PREDICTOR_MODEL_ID)}-point"
        ),
        area_ratio=float(mask.mean()),
        confidence=float(
            np.clip(
                np.mean(selected_scores),
                0.0,
                1.0,
            )
        ),
        object_label="selected object",
        notes=[
            (
                f"SAM 2 model={_PREDICTOR_MODEL_ID}; "
                f"device={_PREDICTOR_DEVICE}"
            ),
            (
                f"point ({x},{y}); "
                f"mask area={float(mask.mean()) * 100:.2f}%"
            ),
            (
                f"jitter masks={len(selected)}; "
                f"primary score={primary_score:.3f}; "
                f"primary/consensus IoU={iou:.3f}"
            ),
        ],
    )
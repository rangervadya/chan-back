"""Backend orchestration for Camelion recoloring.

Cloudflare mode runs a real generative edit (Klein or explicit inpaint), then
composites the edit inside the segmentation mask. Local recoloring is an
explicit fallback and is reported as such.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from logger import get_logger
from config import METALLIC_VARIANTS
from . import ops
from .base import BackendResult, RecolorRequest, SegmentationResult
from .cloudflare_ai import CloudflareAIError
from .diagnostic import DiagnosticCollector, request_id_from_env

logger = get_logger(__name__)
_diag: DiagnosticCollector | None = None


class BackendConfigurationError(RuntimeError):
    """The configured backend cannot be constructed from current environment."""

_METALLIC_VARIANTS = set(METALLIC_VARIANTS)


class BaseBackend:
    name = "base"

    def run(self, request: RecolorRequest) -> BackendResult:  # pragma: no cover
        raise NotImplementedError

    def _segment(self, request: RecolorRequest, bbox=None) -> SegmentationResult:
        from .segmentation import segment_object
        return segment_object(
            request.image_rgb, request.point_x, request.point_y,
            feather_px=_cfg("MASK_FEATHER_PX", 2.0),
            min_area_ratio=_cfg("MIN_MASK_AREA_RATIO", 0.001),
            max_area_ratio=_cfg("MAX_MASK_AREA_RATIO", 0.75),
            bbox=bbox,
        )


class Sam2LocalBackend(BaseBackend):
    """SAM 2 point segmentation followed by deterministic, texture-preserving recoloring.

    It intentionally does not call an image-generation API. Pixels outside the
    selected object mask remain the original pixels, while the target hue is
    changed locally and the photograph's luminance/texture is retained.

    When SAM 2 fails (model missing, empty mask, implausible mask area, inference
    error) the backend falls back to the local GrabCut segmenter so the request
    still produces a recoloured image instead of a hard 503.
    """
    name = "sam2-local"

    def run(self, request: RecolorRequest) -> BackendResult:
        from .local_recolor import recolor
        from .sam2_segmenter import segment_with_sam2, Sam2UnavailableError

        diag = _diag or DiagnosticCollector(request_id_from_env())
        image = np.asarray(request.image_rgb, dtype=np.uint8)
        diag.capture("00_original_resized", image,
                     note=f"исходная фотография ({image.shape[1]}x{image.shape[0]})")
        logger.info(
            "sam2-local backend: image=%dx%d point=(%.1f,%.1f) material=%s "
            "target_rgb=%s hex_color=%s object_name=%s patina=%s from_pipette=%s",
            image.shape[1], image.shape[0], request.point_x, request.point_y,
            request.material, request.target_rgb, getattr(request, "hex_color", ""),
            request.object_name, bool(getattr(request, "patina", False)),
            bool(getattr(request, "from_pipette", False)),
        )
        t_sam = time.monotonic()
        try:
            seg = segment_with_sam2(
                image, request.point_x, request.point_y,
                model_name=_cfg("SAM2_MODEL_NAME", "facebook/sam2.1-hiera-base-plus"),
                negative_points=getattr(request, "negative_points", []),
            )
        except Sam2UnavailableError as exc:
            if getattr(request, "negative_points", []):
                # GrabCut cannot honour SAM 2 negative clicks. A fallback here
                # could recolour the very details the user explicitly protected.
                raise Sam2UnavailableError(
                    "SAM 2 failed while protected details were specified; "
                    "unsafe fallback was cancelled. Retry or remove protection points."
                ) from exc
            logger.warning(
                "sam2-local: SAM 2 segmentation failed (%s); falling back to local GrabCut segmenter",
                exc,
            )
            return self._local_fallback(request, image, diag, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "sam2-local: SAM 2 unexpected error (%s: %s); falling back to local GrabCut segmenter",
                type(exc).__name__, exc,
            )
            return self._local_fallback(request, image, diag, f"{type(exc).__name__}: {exc}")
        sam_time = time.monotonic() - t_sam
        logger.info("sam2-local: SAM 2 segmentation done in %.3fs", sam_time)

        mask = ops.to_mask_array(seg.mask)
        if not mask.any():
            logger.warning("sam2-local: SAM 2 returned empty mask; falling back to local GrabCut segmenter")
            return self._local_fallback(request, image, diag, "SAM 2 returned an empty object mask")
        _capture_segmentation(diag, image, seg)
        crop = _crop_for_diag(image, mask)
        diag.capture("03_crop", crop, note="выделенный SAM 2 объект")
        diag.metric("crop_size", list(crop.shape[:2]))
        diag.metric("segmentation_method", seg.method)
        diag.metric("segmentation_confidence", round(float(seg.confidence), 4))
        diag.metric("mask_area_ratio", round(float(mask.mean()), 5))

        logger.info(
            "sam2-local: segmentation done — method=%s mask_area=%.2f%% confidence=%.3f",
            seg.method, 100.0 * float(mask.mean()), float(seg.confidence),
        )

        t_recolor = time.monotonic()
        out = recolor(image, mask, request, feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        recolor_time = time.monotonic() - t_recolor
        logger.info("sam2-local: local recolor done in %.3fs", recolor_time)
        diag.capture("04_model_output", out, note="локальная перекраска по SAM 2 маске")
        diag.capture("05_detail_restored", out, note="исходная яркость и текстура сохранены")
        diag.capture("06_final", out, note="итоговая локальная перекраска")
        _record_quality_metrics(diag, out, image, mask, request.target_rgb)
        summary = diag.flush()
        notes = [
            "SAM 2 point segmentation + local color transform; external image generation not used",
            *seg.notes,
        ]
        _append_diag_note(notes, summary)
        logger.info("sam2-local: done — backend=%s total=%.3fs", self.name, time.monotonic() - t_sam - recolor_time + sam_time + recolor_time)
        return BackendResult(
            image_rgb=out, backend=self.name, segmentation=seg,
            object_label=request.object_name or "selected object", notes=notes,
        )

    def _local_fallback(self, request: RecolorRequest, image: np.ndarray,
                        diag: DiagnosticCollector, sam2_error: str) -> BackendResult:
        """Fallback to the local GrabCut segmenter when SAM 2 cannot be used."""
        t_start = time.monotonic()
        from .segmentation import segment_object
        from .local_recolor import recolor

        t_seg = time.monotonic()
        seg = segment_object(
            image, request.point_x, request.point_y,
            feather_px=_cfg("MASK_FEATHER_PX", 2.0),
            min_area_ratio=_cfg("MIN_MASK_AREA_RATIO", 0.001),
            max_area_ratio=_cfg("MAX_MASK_AREA_RATIO", 0.75),
        )
        seg_time = time.monotonic() - t_seg
        logger.info("sam2-local fallback: local segment done in %.3fs", seg_time)
        mask = ops.to_mask_array(seg.mask)
        if not mask.any():
            logger.warning(
                "sam2-local fallback: local segmenter also returned empty mask; "
                "returning original image unchanged (error was: %s)",
                sam2_error,
            )
            diag.capture("06_final", image, note="итог: исходное изображение (сегментация не удалась)")
            summary = diag.flush()
            notes = [
                f"SAM 2 fallback: {sam2_error}",
                "Fallback: локальная сегментация тоже не выделила объект — возвращено исходное изображение",
                *seg.notes,
            ]
            _append_diag_note(notes, summary)
            return BackendResult(
                image_rgb=image.copy(), backend="sam2-local->local-fallback-empty",
                segmentation=seg, object_label=request.object_name or "selected object",
                notes=notes,
            )
        _capture_segmentation(diag, image, seg)
        crop = _crop_for_diag(image, mask)
        diag.capture("03_crop", crop, note="объект выделен локальным segment_object (fallback)")
        diag.metric("crop_size", list(crop.shape[:2]))
        diag.metric("segmentation_method", seg.method)
        diag.metric("segmentation_confidence", round(float(seg.confidence), 4))
        diag.metric("mask_area_ratio", round(float(mask.mean()), 5))
        logger.info(
            "sam2-local fallback: segmentation done — method=%s mask_area=%.2f%% confidence=%.3f",
            seg.method, 100.0 * float(mask.mean()), float(seg.confidence),
        )

        t_recolor = time.monotonic()
        out = recolor(image, mask, request, feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        recolor_time = time.monotonic() - t_recolor
        logger.info("sam2-local fallback: local recolor done in %.3fs", recolor_time)
        diag.capture("04_model_output", out, note="локальная перекраска по fallback маске")
        diag.capture("05_detail_restored", out, note="исходная яркость и текстура сохранены")
        diag.capture("06_final", out, note="итоговая локальная перекраска (fallback)")
        _record_quality_metrics(diag, out, image, mask, request.target_rgb)
        summary = diag.flush()
        notes = [
            f"SAM 2 fallback: {sam2_error}",
            "Fallback: локальная сегментация (GrabCut) + локальный цветовой преобразование",
            *seg.notes,
        ]
        _append_diag_note(notes, summary)
        logger.info("sam2-local: done via fallback — backend=%s total=%.3fs", self.name, time.monotonic() - t_start)
        return BackendResult(
            image_rgb=out, backend="sam2-local->local-fallback", segmentation=seg,
            object_label=request.object_name or "selected object", notes=notes,
        )


class LocalBackend(BaseBackend):
    name = "local"

    def run(self, request: RecolorRequest) -> BackendResult:
        from .local_recolor import recolor
        diag = _diag or DiagnosticCollector(request_id_from_env())
        image = request.image_rgb
        diag.capture("00_original_resized", image, note=f"original {image.shape[1]}x{image.shape[0]}")
        logger.info(
            "local backend: image=%dx%d point=(%.1f,%.1f) material=%s target_rgb=%s",
            image.shape[1], image.shape[0], request.point_x, request.point_y,
            request.material, request.target_rgb,
        )
        t_seg = time.monotonic()
        seg = self._segment(request)
        seg_time = time.monotonic() - t_seg
        logger.info("local: segmentation done in %.3fs method=%s mask_area=%.2f%%",
                    seg_time, seg.method, 100.0 * float(ops.to_mask_array(seg.mask).mean()))
        mask = ops.to_mask_array(seg.mask)
        _capture_segmentation(diag, image, seg)
        crop = _crop_for_diag(image, mask)
        diag.capture("03_crop", crop, note="crop вокруг выбранного объекта")
        diag.metric("crop_size", list(crop.shape[:2]))
        t_recolor = time.monotonic()
        out = recolor(image, mask, request, feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        recolor_time = time.monotonic() - t_recolor
        logger.info("local: recolor done in %.3fs", recolor_time)
        diag.capture("04_model_output", out, note="local recolor result")
        diag.capture("05_detail_restored", out, note="local recolor result")
        diag.capture("06_final", out, note="final local recolor")
        _record_quality_metrics(diag, out, image, mask, request.target_rgb)
        summary = diag.flush()
        notes = ["локальная перекраска внутри маски"]
        _append_diag_note(notes, summary)
        logger.info("local: done — backend=%s total=%.3fs", self.name, seg_time + recolor_time)
        return BackendResult(image_rgb=out, backend=self.name, segmentation=seg,
                             object_label=request.object_name, notes=notes)


class _AnalyzingMixin(BaseBackend):
    analyzer = None
    analyzer_name = "none"

    def _describe(self, request: RecolorRequest) -> tuple[str, list[str]]:
        notes: list[str] = []
        if self.analyzer is None:
            return request.object_name or "object", notes
        try:
            label = self.analyzer.describe(request.image_rgb, request.point_x, request.point_y)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{self.analyzer_name} анализ недоступен: {exc}")
            return request.object_name or "object", notes
        if label:
            notes.append(f"{self.analyzer_name}: объект = '{label}'")
            return label, notes
        notes.append(f"{self.analyzer_name}: пустой ответ")
        return request.object_name or "object", notes


class CloudflareBackend(_AnalyzingMixin):
    """Local SAM 2 segmentation followed by Cloudflare FLUX and masked compositing."""
    name = "cloudflare"

    def __init__(self, client) -> None:
        self.client = client
        self.analyzer = client if getattr(client, "is_configured", False) else None
        self.analyzer_name = "cloudflare-vision"

    def run(self, request: RecolorRequest) -> BackendResult:
        t_start = time.monotonic()
        diag = _diag or DiagnosticCollector(request_id_from_env())
        image = np.asarray(request.image_rgb, dtype=np.uint8)
        diag.capture("00_original_resized", image,
                     note=f"исходная фотография ({image.shape[1]}x{image.shape[0]})")
        logger.info(
            "cloudflare backend: image=%dx%d point=(%.1f,%.1f) material=%s "
            "target_rgb=%s hex_color=%s object_name=%s patina=%s from_pipette=%s",
            image.shape[1], image.shape[0], request.point_x, request.point_y,
            request.material, request.target_rgb, getattr(request, "hex_color", ""),
            request.object_name, bool(getattr(request, "patina", False)),
            bool(getattr(request, "from_pipette", False)),
        )
        # Segmentation is performed locally by SAM 2. The mask constrains the
        # final composite after generation. FLUX.2 Klein's documented API uses
        # input_image_0 as a reference image and does not document a native mask
        # input; cloudflare_ai.py must send its supported multipart field names.
        t_seg = time.monotonic()
        try:
            from .sam2_segmenter import segment_with_sam2

            seg = segment_with_sam2(
                image,
                request.point_x,
                request.point_y,
                model_name=_cfg(
                    "SAM2_MODEL_NAME",
                    "facebook/sam2.1-hiera-base-plus",
                ),
                negative_points=getattr(request, "negative_points", []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "cloudflare: local SAM 2 segmentation failed; FLUX was not called"
            )
            raise CloudflareAIError(
                f"SAM 2 segmentation failed; FLUX generation was not started: {exc}",
                status=503,
                retryable=False,
            ) from exc

        seg_time = time.monotonic() - t_seg
        logger.info(
            "cloudflare: SAM 2 segmentation done in %.3fs method=%s",
            seg_time,
            seg.method,
        )
        diag.metric("segmentation_engine", "SAM 2 local")

        mask = ops.to_mask_array(seg.mask)
        if mask.ndim != 2 or mask.shape != image.shape[:2]:
            raise CloudflareAIError(
                "SAM 2 returned a mask with an invalid shape: "
                f"mask={getattr(mask, 'shape', None)}, image={image.shape[:2]}",
                status=422,
                retryable=False,
            )

        _capture_segmentation(diag, image, seg)
        if not mask.any():
            raise CloudflareAIError(
                "SAM 2 returned an empty object mask; FLUX generation was cancelled",
                status=422,
                retryable=False,
            )

        # Avoid an extra Cloudflare DETR/vision request before FLUX. The object
        # label supplied by the client is enough for the current prompt builder.
        label = (str(request.object_name or "").strip() or "selected object")
        notes = [
            "Объект выделен локальной SAM 2; маска передаётся в FLUX через Cloudflare Workers AI."
        ]
        prompt = build_recolor_prompt(request, label, _color_name(request))
        model_choice = str(_cfg("CLOUDFLARE_RECOLOR_MODEL", "klein")).strip().lower()
        if model_choice not in {"klein", "inpaint"}:
            logger.warning("Неподдерживаемый CLOUDFLARE_RECOLOR_MODEL=%r; использую klein", model_choice)
            model_choice = "klein"
        model_name = (getattr(self.client, "klein_model", "") if model_choice == "klein"
                      else getattr(self.client, "inpaint_model", ""))
        diag.metric("recolor_model", model_choice)
        diag.metric("recolor_model_real", model_name)
        diag.metric("prompt", prompt)
        logger.info("Cloudflare generation: backend=%s model=%s mask=%.2f%%",
                    model_choice, model_name, float(mask.mean()) * 100)
        logger.info("cloudflare: prompt built — label=%s color_name=%s prompt=%s",
                    label, _color_name(request), prompt)
        notes.append(f"prompt: {prompt}")

        t_gen = time.monotonic()
        try:
            painted = self._generate_recolor(
                request, mask, prompt, model_choice,
                attempts=int(_cfg("CLOUDFLARE_KLEIN_ATTEMPTS", 2)), diag=diag,
            )
        except Exception as exc:  # noqa: BLE001
            notes.append(f"ГЕНЕРАЦИЯ НЕ ВЫПОЛНЕНА ({model_choice}, {model_name}): {exc}")
            logger.error("Cloudflare generation failed: %s", exc)
            if _cfg("CLOUDFLARE_ALLOW_LOCAL_FALLBACK", False):
                notes.append("Включён явный CLOUDFLARE_ALLOW_LOCAL_FALLBACK; возвращаю локальную перекраску")
                return self._local_finish(request, seg, notes, object_label=label, fallback=True, diag=diag)
            raise
        gen_time = time.monotonic() - t_gen
        logger.info("cloudflare: generation done in %.3fs", gen_time)

        diag.capture("03_crop", _crop_for_diag(image, mask), note="crop вокруг выбранного объекта")
        diag.metric("crop_size", list(_crop_for_diag(image, mask).shape[:2]))
        diag.capture("04_model_output", painted, note="ответ генеративной модели до композиции")
        diag.metric("model_output_shape", list(painted.shape[:2]))
        coverage_issue = _validate_pixel_coverage(painted, image, mask, request.target_rgb, request.material)
        diag.metric("coverage_issue_before_composite", coverage_issue or "none")
        if coverage_issue:
            notes.append(f"диагностика покрытия: {coverage_issue}")

        out = _hybrid_finish(
            painted_ai=painted, original=image, mask=mask,
            target_rgb=request.target_rgb, material=request.material,
            feather_px=_cfg("MASK_FEATHER_PX", 2.0), request=request,
            ai_texture_influence=float(_cfg("CLOUDFLARE_AI_TEXTURE_INFLUENCE", 0.08)),
        )
        finish_time = time.monotonic() - t_gen - gen_time
        diag.capture("05_detail_restored", out, note="после восстановления деталей и коррекции цвета")
        diag.capture("06_final", out, note="итог после композиции по маске")
        _record_quality_metrics(diag, out, image, mask, request.target_rgb)
        summary = diag.flush()
        _append_diag_note(notes, summary)
        notes.append(f"генеративная перекраска: {model_choice} ({model_name or 'model name unavailable'})")
        logger.info(
            "cloudflare: done — backend=%s total=%.3fs (sam2=%.3fs gen=%.3fs finish=%.3fs)",
            self.name,
            time.monotonic() - t_start,
            seg_time,
            gen_time,
            finish_time,
        )
        return BackendResult(image_rgb=out, backend=self.name, segmentation=seg,
                             object_label=label, notes=notes)

    def _generate_recolor(self, request: RecolorRequest, mask: np.ndarray, prompt: str,
                          recolor_model: str, attempts: int | None = None,
                          diag: DiagnosticCollector | None = None) -> np.ndarray:
        if attempts is None:
            attempts = int(_cfg("CLOUDFLARE_KLEIN_ATTEMPTS", 2))
        attempts = max(1, min(int(attempts), 3))
        last: Exception | None = None
        for attempt in range(attempts):
            seed = int(np.random.default_rng(attempt).integers(1, 2**31 - 1))
            t_attempt = time.monotonic()
            try:
                if recolor_model == "inpaint":
                    painted = self.client.inpaint(
                        request.image_rgb, mask, prompt=prompt,
                        negative_prompt=("do not change background or geometry; no extra objects; "
                                         "preserve original perspective and realistic texture"),
                        strength=float(_cfg("CLOUDFLARE_INPAINT_STRENGTH", 0.82)),
                        num_steps=int(_cfg("CLOUDFLARE_INPAINT_STEPS", 20)),
                        guidance=float(_cfg("CLOUDFLARE_INPAINT_GUIDANCE", 7.5)),
                        size=int(_cfg("CLOUDFLARE_INPAINT_SIZE", 1024)), seed=seed,
                    )
                else:
                    painted = self.client.recolor_klein(
                        request.image_rgb, mask, prompt=prompt,
                        size=int(_cfg("CLOUDFLARE_KLEIN_SIZE", 768)), seed=seed,
                    )
            except Exception as exc:  # noqa: BLE001
                last = exc
                logger.warning("Cloudflare %s attempt %d/%d: %s", recolor_model, attempt + 1, attempts, exc)
                # Credentials, account/model permissions and quota failures do not improve by retrying.
                if isinstance(exc, CloudflareAIError) and not exc.retryable and exc.status in {400, 401, 403, 404, 429}:
                    break
                continue

            issue = _validate_generated(painted, request.image_rgb, mask, request.target_rgb)
            if issue is None:
                # Retry only when a large fraction of eligible pixels clearly
                # remain unchanged. Pixels that changed to the wrong hue are
                # still recoverable by the final chroma correction.
                issue = _validate_pixel_coverage(
                    painted, request.image_rgb, mask, request.target_rgb,
                    request.material, residual_threshold=0.62,
                )
            if issue is None:
                logger.info("cloudflare: %s attempt %d/%d ok in %.3fs",
                            recolor_model, attempt + 1, attempts, time.monotonic() - t_attempt)
                return np.clip(np.asarray(painted), 0, 255).astype(np.uint8)
            last = CloudflareAIError(issue, status=0)
            logger.warning("Cloudflare %s attempt %d/%d rejected: %s", recolor_model, attempt + 1, attempts, issue)
            if diag is not None:
                diag.metric(f"generation_attempt_{attempt + 1}", issue)
        message = f"{recolor_model}: все попытки дали некорректный результат" + (f"; последняя ошибка: {last}" if last else "")
        if isinstance(last, CloudflareAIError):
            raise CloudflareAIError(message, status=last.status, retryable=last.retryable) from last
        raise CloudflareAIError(message) from last

    def _local_finish(self, request: RecolorRequest, seg: SegmentationResult,
                      notes: list[str], object_label: str = "", fallback: bool = False,
                      diag: DiagnosticCollector | None = None) -> BackendResult:
        from .local_recolor import recolor
        t0 = time.monotonic()
        out = recolor(request.image_rgb, seg.mask, request,
                      feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        logger.info("cloudflare: local finish recolor done in %.3fs", time.monotonic() - t0)
        if fallback:
            notes.append("ВНИМАНИЕ: генеративная модель не сработала; использована локальная перекраска")
        else:
            notes.append("локальная перекраска внутри маски")
        if diag is not None:
            diag.capture("06_final", out, note="итоговое изображение (локальный fallback)")
            _record_quality_metrics(diag, out, request.image_rgb, seg.mask, request.target_rgb)
            summary = diag.flush()
            _append_diag_note(notes, summary)
        return BackendResult(
            image_rgb=out,
            backend="cloudflare->local-fallback" if fallback else "cloudflare",
            segmentation=seg,
            object_label=object_label or request.object_name,
            notes=notes,
        )


class GeminiBackend(_AnalyzingMixin):
    name = "gemini"

    def __init__(self, client) -> None:
        self.client = client
        self.analyzer = client if getattr(client, "is_configured", False) else None
        self.analyzer_name = "gemini"

    def run(self, request: RecolorRequest) -> BackendResult:
        t_start = time.monotonic()
        image = np.asarray(request.image_rgb, dtype=np.uint8)
        logger.info(
            "gemini backend: image=%dx%d point=(%.1f,%.1f) material=%s target_rgb=%s",
            image.shape[1], image.shape[0], request.point_x, request.point_y,
            request.material, request.target_rgb,
        )
        t_seg = time.monotonic()
        seg = self._segment(request)
        seg_time = time.monotonic() - t_seg
        t_desc = time.monotonic()
        label, notes = self._describe(request)
        desc_time = time.monotonic() - t_desc
        logger.info("gemini: segmentation done in %.3fs method=%s mask_area=%.2f%% label=%r",
                    seg_time, seg.method, 100.0 * float(ops.to_mask_array(seg.mask).mean()), label)
        from .local_recolor import recolor
        t_recolor = time.monotonic()
        out = recolor(request.image_rgb, seg.mask, request,
                      feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        recolor_time = time.monotonic() - t_recolor
        if request.from_pipette:
            out = _match_mean(out, seg.mask, request.target_rgb)
        logger.info("gemini: done — backend=%s total=%.3fs (seg=%.3fs desc=%.3fs recolor=%.3fs)",
                    self.name, time.monotonic() - t_start, seg_time, desc_time, recolor_time)
        return BackendResult(image_rgb=out, backend=self.name, segmentation=seg,
                             object_label=label, notes=notes + ["перекраска локальным алгоритмом"])


class ZaiBackend(BaseBackend):
    name = "zai"

    def __init__(self, client) -> None:
        self.client = client

    def _segment(self, request: RecolorRequest) -> SegmentationResult:
        from .segmentation import segment_object
        box = None
        if getattr(self.client, "is_available", True):
            t_box = time.monotonic()
            try:
                box = self.client.object_box(request.image_rgb, request.point_x, request.point_y)
                logger.info("zai: object_box returned bbox=%s in %.3fs", box, time.monotonic() - t_box)
            except Exception as exc:  # noqa: BLE001
                logger.info("Z.AI недоступен -> локальная сегментация: %s", exc)
        return segment_object(
            request.image_rgb, request.point_x, request.point_y,
            feather_px=_cfg("MASK_FEATHER_PX", 2.0),
            min_area_ratio=_cfg("MIN_MASK_AREA_RATIO", 0.001),
            max_area_ratio=_cfg("MAX_MASK_AREA_RATIO", 0.75), bbox=box,
        )

    def run(self, request: RecolorRequest) -> BackendResult:
        t_start = time.monotonic()
        image = np.asarray(request.image_rgb, dtype=np.uint8)
        logger.info(
            "zai backend: image=%dx%d point=(%.1f,%.1f) material=%s target_rgb=%s",
            image.shape[1], image.shape[0], request.point_x, request.point_y,
            request.material, request.target_rgb,
        )
        t_seg = time.monotonic()
        seg = self._segment(request)
        seg_time = time.monotonic() - t_seg
        from .local_recolor import recolor
        t_recolor = time.monotonic()
        out = recolor(request.image_rgb, seg.mask, request,
                      feather_px=_cfg("MASK_FEATHER_PX", 2.0))
        recolor_time = time.monotonic() - t_recolor
        if request.from_pipette:
            out = _match_mean(out, seg.mask, request.target_rgb)
        notes = [n for n in seg.notes if n.startswith("box-") or "bbox" in n or "маску" in n]
        if self.client is not None and not getattr(self.client, "is_configured", False):
            notes.append("Z.AI не сконфигурирован -> локальная сегментация")
        logger.info("zai: done — backend=%s total=%.3fs (seg=%.3fs recolor=%.3fs)",
                    self.name, time.monotonic() - t_start, seg_time, recolor_time)
        return BackendResult(image_rgb=out, backend=self.name, segmentation=seg,
                             object_label=request.object_name, notes=notes)


class LegacyBackend(BaseBackend):
    name = "legacy"

    def __init__(self, runner) -> None:
        self.runner = runner

    def run(self, request: RecolorRequest) -> BackendResult:
        return self.runner(request)


_BACKENDS = {"local": LocalBackend, "cloudflare": CloudflareBackend,
             "gemini": GeminiBackend, "zai": ZaiBackend}


def get_backend(name: str | None = None):
    import config
    chosen = (name or getattr(config, "AI_BACKEND", "auto") or "auto").strip().lower()
    logger.info("get_backend: requested=%s resolved=%s", name, chosen)
    if chosen == "legacy":
        from app import run_legacy_recolor_job
        return LegacyBackend(run_legacy_recolor_job)
    if chosen == "cloudflare":
        return _build_cloudflare(required=True)
    if chosen == "gemini":
        return _build_gemini(required=True)
    if chosen == "zai":
        return _build_zai(required=True)
    if chosen in {"sam2-local", "local-sam2", "sam2"}:
        return Sam2LocalBackend()
    if chosen == "local":
        return LocalBackend()
    # Auto mode promises image generation. Analysis-only APIs/local color math
    # are not equivalent to FLUX, so fail loudly instead of silently returning a
    # flat-looking fallback to the end user.
    cf = _build_cloudflare(required=False)
    if cf is not None:
        logger.info("AI backend: cloudflare (generative recoloring)")
        return cf
    raise BackendConfigurationError(
        "AI_BACKEND=auto требует CLOUDFLARE_ACCOUNT_ID и CLOUDFLARE_API_TOKEN. "
        "Для перекраски без платы за генерацию используйте AI_BACKEND=sam2-local "
        "(SAM 2 CPU-сегментация + локальное сохранение фактуры); для генеративного редактирования настройте Cloudflare Workers AI."
    )


def _build_cloudflare(required: bool):
    import config
    account_id = getattr(config, "CLOUDFLARE_ACCOUNT_ID", "")
    token = getattr(config, "CLOUDFLARE_API_TOKEN", "")
    logger.info("_build_cloudflare: account_id=%s token_present=%s required=%s",
                account_id, bool(token), required)
    if not (account_id and token):
        if required:
            raise BackendConfigurationError("AI_BACKEND=cloudflare, но не заданы CLOUDFLARE_ACCOUNT_ID и/или CLOUDFLARE_API_TOKEN")
        return None
    from .cloudflare_ai import CloudflareAI
    client = CloudflareAI(
        account_id=account_id, api_token=token,
        inpaint_model=getattr(config, "CLOUDFLARE_INPAINT_MODEL", ""),
        vision_model=getattr(config, "CLOUDFLARE_VISION_MODEL", ""),
        klein_model=getattr(config, "CLOUDFLARE_KLEIN_MODEL", ""),
        klein_guidance=getattr(config, "CLOUDFLARE_KLEIN_GUIDANCE", 3.5),
        timeout=getattr(config, "CLOUDFLARE_TIMEOUT", 75),
        max_retries=getattr(config, "CLOUDFLARE_MAX_RETRIES", 0),
        detection_model=getattr(config, "CLOUDFLARE_DETECTION_MODEL", "@cf/facebook/detr-resnet-50"),
        detection_min_score=getattr(config, "CLOUDFLARE_DETECTION_MIN_SCORE", 0.35),
    )
    return CloudflareBackend(client)


def _build_zai(required: bool):
    import config
    api_key = getattr(config, "ZAI_API_KEY", "")
    logger.info("_build_zai: api_key_present=%s required=%s", bool(api_key), required)
    if not api_key:
        if required:
            raise RuntimeError("AI_BACKEND=zai, но не задан ZAI_API_KEY")
        return None
    from .zai_ai import ZaiVision
    client = ZaiVision(api_key=api_key,
                       model=getattr(config, "ZAI_MODEL", "glm-4.6v-flash"),
                       base_url=getattr(config, "ZAI_BASE_URL", "https://api.z.ai/api/paas/v4"),
                       timeout=getattr(config, "ZAI_TIMEOUT", 25),
                       max_retries=getattr(config, "ZAI_MAX_RETRIES", 1))
    return ZaiBackend(client)


def _build_gemini(required: bool):
    import config
    api_key = getattr(config, "GEMINI_API_KEY", "")
    logger.info("_build_gemini: api_key_present=%s required=%s", bool(api_key), required)
    if not api_key:
        if required:
            raise RuntimeError("AI_BACKEND=gemini, но не задан GEMINI_API_KEY")
        return None
    from .gemini_ai import GeminiAI
    client = GeminiAI(api_key=api_key,
                      model=getattr(config, "GEMINI_VISION_MODEL", "gemini-2.5-flash"),
                      timeout=getattr(config, "GEMINI_TIMEOUT", 60),
                      max_retries=getattr(config, "GEMINI_MAX_RETRIES", 2))
    return GeminiBackend(client)


def _cfg(name: str, default):
    import config
    return getattr(config, name, default)


def _color_name(request: RecolorRequest) -> str:
    """Return a readable color name, or the valid HEX itself as a fallback.

    Color-name tables usually contain only a limited palette. A valid HEX that
    is absent from the table must still be usable by the prompt builder.
    """
    provided = str(getattr(request, "color_name", "") or "").strip()
    if provided:
        logger.info("_color_name: using provided color_name=%s", provided)
        return provided

    raw_hex = getattr(request, "color_hex", None)
    try:
        if isinstance(raw_hex, str):
            value_str = raw_hex.strip()
            if value_str.startswith("#"):
                value_str = value_str[1:]
            if value_str.lower().startswith("0x"):
                value_str = value_str[2:]
            value = int(value_str, 16)
        else:
            value = int(raw_hex)
        if not 0 <= value <= 0xFFFFFF:
            raise ValueError("HEX color must be between 0x000000 and 0xFFFFFF")
        hex_code = f"#{value:06X}"
    except (TypeError, ValueError):
        logger.warning("_color_name: invalid color_hex=%r", raw_hex)
        return ""

    try:
        from config import get_color_hex_name
        name = get_color_hex_name(value)
        if name and str(name).strip():
            name = str(name).strip()
            logger.info("_color_name: resolved %s -> %s", hex_code, name)
            return name
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "_color_name: lookup failed for %s (%s); using HEX fallback",
            hex_code, exc,
        )

    logger.info("_color_name: using HEX fallback %s", hex_code)
    return hex_code


def build_recolor_prompt(request: RecolorRequest, object_label: str, color_name: str) -> str:
    """Build a conservative color-edit prompt that prioritizes source preservation.

    FLUX is a generative model, so the prompt explicitly treats the supplied
    photograph as the reference and forbids redesign. The SAM 2 mask is still
    used by the backend's post-generation composition; this prompt does not
    imply that Klein supports a native inpainting-mask parameter.
    """
    material = (str(getattr(request, "material", "object") or "object")).strip().lower()
    color = str(color_name or getattr(request, "hex_color", "") or f"#{int(request.color_hex):06X}")
    hex_color = f"#{int(request.color_hex):06X}"
    try:
        rgb = tuple(int(v) for v in (request.color_r, request.color_g, request.color_b))
    except (AttributeError, TypeError, ValueError):
        rgb = tuple(int(v) for v in request.target_rgb)

    label = str(object_label or "").strip()
    if not label or label.lower() in {"object", "selected object", "unknown object"}:
        label = f"{material} object" if material not in {"", "object"} else "selected object"

    # Only describe the material details that must remain intact. Avoid asking
    # the model to create a new finish, grain pattern, or design.
    if material == "no_texture":
        material_rule = (
            "Apply a uniform matte color to the existing surface. Do not add texture, "
            "patterns, grain, gloss, or decorative details. Preserve original shading and edges."
        )
    elif material == "wood":
        material_rule = (
            "Preserve the exact existing wood-grain pattern, knots, scratches, wear, "
            "and finish; do not invent or redraw the grain."
        )
    elif material == "fabric":
        material_rule = (
            "Preserve the exact existing weave, folds, seams, stitching, and fabric finish; "
            "do not invent a new textile pattern."
        )
    elif material == "leather":
        material_rule = (
            "Preserve the existing leather grain, creases, stitching, wear, and finish; "
            "do not create new texture or decorative details."
        )
    elif material == "glass":
        material_rule = (
            "Tint only the existing visible glass. Preserve transparency, refraction, "
            "reflections, rim highlights, shape, and the original scene visible through it; "
            "do not tint the background seen through the glass."
        )
    elif material in {"metal", "bronze"} or material in _METALLIC_VARIANTS:
        material_rule = (
            "Preserve the existing metal finish, roughness, reflections, specular highlights, "
            "scratches, and surface wear; change its color without redesigning the finish."
        )
    elif material == "plastic":
        material_rule = (
            "Preserve the existing plastic gloss or matte finish, molded details, reflections, "
            "and surface wear. Do not change the object's design."
        )
    elif material == "ceramic":
        material_rule = (
            "Preserve the existing ceramic glaze, reflections, small imperfections, and finish; "
            "do not add patterns or decorations."
        )
    elif material == "concrete":
        material_rule = (
            "Preserve the existing pores, aggregate, cracks, wear, and surface relief; "
            "do not generate new texture."
        )
    else:
        material_rule = (
            "Preserve the object's existing material appearance, texture, finish, and wear; "
            "do not invent surface details."
        )

    prompt = (
        "STRICT PHOTO EDIT. input_image_0 is the original photograph. input_image_1 is a "
        "binary selection map: white means the only area allowed to change, black means "
        "preserve the original. Use input_image_0 as the source and keep the same photograph; "
        "do not create a new scene or reinterpret the object. Change only the surface color "
        f"inside the white selection area on the existing {label} to {color} "
        f"(HEX {hex_color}, RGB {rgb}). "
        "Preserve the exact identity, silhouette, proportions, geometry, components, joints, "
        "edges, seams, accessories, position, orientation, camera angle, and perspective. "
        f"{material_rule} "
        "Keep the original lighting, luminance gradients, natural shadows, highlights, and "
        "reflections. Do not add, remove, replace, move, reshape, smooth, sharpen, or redesign "
        "anything. Do not alter any unselected area or the background. No stylization and no "
        "new details. If instructions conflict, preserving the source object's structure and "
        "the original photograph takes priority over creative changes."
    )
    if material in {"metal", "bronze"} and bool(getattr(request, "patina", False)):
        prompt += " Add only a subtle, natural aged patina; do not change the geometry or design."

    logger.info(
        "build_recolor_prompt: material=%s color=%s object_label=%s prompt_len=%d",
        material, color, label, len(prompt),
    )
    return prompt


# Backwards compatibility with the earlier internal function name.
build_inpaint_prompt = build_recolor_prompt


def _validate_generated(painted: np.ndarray, original: np.ndarray, mask: np.ndarray,
                        target_rgb: tuple[int, int, int]) -> str | None:
    if not isinstance(painted, np.ndarray) or painted.ndim != 3 or painted.shape[2] != 3:
        return "результат модели не является RGB-изображением"
    if painted.shape[:2] != original.shape[:2]:
        return f"размер ответа {painted.shape[:2]} не совпадает с исходным {original.shape[:2]}"
    m = ops.to_mask_array(mask)
    if not m.any():
        return "пустая маска"
    region = painted[m].astype(np.float32)
    if region.size == 0 or not np.isfinite(region).all():
        return "пустая или некорректная область результата"
    if float(region.mean()) < 2 or float(region.max()) < 8:
        return "практически чёрный результат"
    if np.abs(region.mean(axis=0) - np.asarray(original[m], dtype=np.float32).mean(axis=0)).max() < 2:
        return "модель практически не изменила выбранную область"
    return None


def _validate_pixel_coverage(painted: np.ndarray, original: np.ndarray, mask: np.ndarray,
                             target_rgb: tuple[int, int, int], material: str = "object",
                             residual_threshold: float = 0.62) -> str | None:
    """Detect a gross partial edit using LAB chroma progress, with glass band check.

    This is diagnostic/retry guidance, not a proof of semantic segmentation.
    It deliberately ignores pixels whose original chroma already matches the
    target, to avoid rejecting a good edit where the object started similar in colour.
    """
    if cv2 is None:
        return None
    m = ops.to_mask_array(mask)
    if not m.any() or painted.shape[:2] != original.shape[:2]:
        return "пустая маска или несовпадающий размер" if not m.any() else "несовпадающий размер"
    tgt = np.uint8([[list(tuple(int(np.clip(c, 0, 255)) for c in target_rgb))]])
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0, 1:]
    orig_lab = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    gen_lab = cv2.cvtColor(np.clip(painted, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    orig_ab, gen_ab = orig_lab[m, 1:], gen_lab[m, 1:]
    delta = tgt_lab[None, :] - orig_ab
    den = np.sum(delta * delta, axis=1)
    eligible = den > (9.0 * 9.0)
    if eligible.sum() > max(20, int(m.sum() * 0.04)):
        progress = np.sum((gen_ab - orig_ab) * delta, axis=1) / np.maximum(den, 1.0)
        residual = float(np.mean(progress[eligible] < 0.22))
        if residual > residual_threshold:
            return f"перекрашено неравномерно: {residual * 100:.0f}% информативных пикселей почти не сдвинуты к целевому оттенку"
    if material == "glass":
        ys, _ = np.nonzero(m)
        if ys.size > 30:
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            span = max(1, y1 - y0)
            # Correct bands: top_band is the upper third; bot_band is the lower third.
            top_band = m.copy()
            top_band[y0 + span // 3:] = False
            bot_band = m.copy()
            bot_band[:y1 - span // 3] = False
            top_issue = _coverage_fraction_unchanged(painted, original, top_band, tgt_lab)
            bot_issue = _coverage_fraction_unchanged(painted, original, bot_band, tgt_lab)
            if top_issue is not None and bot_issue is not None and top_issue > 0.70 and bot_issue < 0.35:
                return f"стекло изменено преимущественно снизу; верхняя треть почти не изменила оттенок ({top_issue * 100:.0f}%)"
    return None


def _coverage_fraction_unchanged(painted: np.ndarray, original: np.ndarray,
                                 mask: np.ndarray, target_ab: np.ndarray) -> float | None:
    if cv2 is None or not mask.any():
        return None
    ori = cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[mask, 1:]
    gen = cv2.cvtColor(np.clip(painted, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[mask, 1:]
    delta = target_ab[None, :] - ori
    elig = np.linalg.norm(delta, axis=1) > 9.0
    if elig.sum() < 8:
        return 0.0
    # Count pixels whose chroma stayed close to the source, not all pixels
    # that failed to match the target exactly.
    changed = np.linalg.norm(gen - ori, axis=1) >= 8.0
    return float(np.mean(~changed[elig]))


def _hybrid_finish(
    painted_ai: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    target_rgb: tuple[int, int, int],
    material: str,
    feather_px: float,
    request: RecolorRequest | None = None,
    ai_texture_influence: float = 0.08,
) -> np.ndarray:
    """Recolor only selected pixels while preserving source geometry and relief.

    Cloudflare FLUX.2 Klein is a reference-image generator, not a masked
    inpainting endpoint. Its full RGB output can redraw lettering, cameras,
    seams or product geometry. The final result therefore starts from the
    original photo: source luminance is kept, only LAB chroma is shifted toward
    the requested color, and FLUX may contribute a very small low-frequency
    chroma variation. Pixels outside the binary SAM 2 mask are copied exactly.
    """
    m = ops.to_mask_array(mask)
    src = np.asarray(original, dtype=np.uint8)
    if not m.any():
        return src.copy()

    if cv2 is None:
        # OpenCV is required by this project for production recoloring. If it
        # is absent, prefer a source-based fallback over inserting AI geometry.
        if request is not None:
            from .local_recolor import recolor
            return recolor(src, m, request, feather_px=feather_px)
        return src.copy()

    material_key = (material or "object").lower()
    target = np.uint8([[list(tuple(int(np.clip(c, 0, 255)) for c in target_rgb))]])
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    source_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
    base_lab = source_lab.copy()

    # Keep original luminance exactly. Rebase only chroma on the chosen color;
    # a small fraction of existing material chroma variation preserves natural
    # local differences without retaining large blotches of the old hue.
    source_chroma_keep = {
        "metal": 0.18, "bronze": 0.18, "wood": 0.08, "leather": 0.06,
        "fabric": 0.06, "plastic": 0.03, "glass": 0.02,
        "ceramic": 0.03, "concrete": 0.08, "no_texture": 0.0,
    }.get(material_key, 0.04)
    for channel in (1, 2):
        values = source_lab[..., channel][m]
        median = float(np.median(values))
        deviation = source_lab[..., channel] - median
        if values.size:
            q = float(np.percentile(np.abs(deviation[m]), 90))
            if q > 0:
                deviation = np.clip(deviation, -q, q)
        base_lab[m, channel] = np.clip(
            target_lab[channel] + deviation[m] * source_chroma_keep,
            0,
            255,
        )

    influence = float(np.clip(ai_texture_influence, 0.0, 0.20))
    if influence > 0 and material_key != "no_texture":
        ai = np.asarray(painted_ai, dtype=np.uint8)
        if ai.shape != src.shape:
            ai = cv2.resize(ai, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        ai_lab = cv2.cvtColor(ai, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Only use a tiny amount of smoothed AI chroma; never take its luminance
        # or pixel edges. This avoids importing model-created text/geometry.
        sigma = max(2.0, min(src.shape[:2]) / 120.0)
        ai_a = cv2.GaussianBlur(ai_lab[..., 1], (0, 0), sigmaX=sigma, sigmaY=sigma)
        ai_b = cv2.GaussianBlur(ai_lab[..., 2], (0, 0), sigmaX=sigma, sigmaY=sigma)
        ai_scale = {
            "metal": 0.14, "bronze": 0.14, "wood": 0.08, "leather": 0.06,
            "fabric": 0.06, "plastic": 0.04, "glass": 0.03,
            "ceramic": 0.04, "concrete": 0.07,
        }.get(material_key, 0.05) * influence / 0.08
        for channel, smooth in ((1, ai_a), (2, ai_b)):
            vals = smooth[m]
            if vals.size == 0:
                continue
            deviation = smooth - float(np.median(vals))
            q = float(np.percentile(np.abs(deviation[m]), 90))
            if q > 0:
                deviation = np.clip(deviation, -q, q)
            base_lab[m, channel] = np.clip(
                base_lab[m, channel] + deviation[m] * ai_scale,
                0,
                255,
            )

    # The L channel was copied from the original and was never changed.
    recolored_rgb = cv2.cvtColor(
        np.clip(base_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
    )
    out = src.copy()
    out[m] = recolored_rgb[m]
    return out.astype(np.uint8)


def _restore_detail(painted: np.ndarray, original: np.ndarray, mask: np.ndarray,
                    feather_px: float, amount: float = 1.0) -> np.ndarray:
    """Restore high-frequency luminance detail, not original RGB colour noise.

    Only the L (luminance) channel of the original is injected into the
    painted result — chroma (a, b) stays untouched so the model's colour is
    never pulled back toward the original hue by the blur.
    """
    if cv2 is None:
        return painted
    m = ops.to_mask_array(mask)
    if not m.any():
        return original.copy()
    p = np.asarray(painted, dtype=np.float32)
    o = np.asarray(original, dtype=np.float32)
    # Avoid background colours bleeding into a crop blur near a hard mask edge.
    src = np.where(m[..., None], p, o).astype(np.uint8)
    # Reduced blur radius: the old sigma=1.35 smeared the AI model's colour
    # transitions near the mask boundary. sigma=0.8 keeps texture while
    # preserving sharp recolouring inside the mask.
    low = cv2.GaussianBlur(src, (0, 0), sigmaX=0.8, sigmaY=0.8)
    orig_u8 = np.clip(o, 0, 255).astype(np.uint8)
    orig_lab = cv2.cvtColor(orig_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    blur_orig = cv2.GaussianBlur(orig_u8, (0, 0), sigmaX=0.8, sigmaY=0.8)
    blur_lab = cv2.cvtColor(blur_orig, cv2.COLOR_RGB2LAB).astype(np.float32)
    detail_l = orig_lab[..., 0] - blur_lab[..., 0]
    low_lab = cv2.cvtColor(low, cv2.COLOR_RGB2LAB).astype(np.float32)
    scale = float(np.clip(np.std(low_lab[..., 0]) / max(float(np.std(orig_lab[..., 0])), 1e-6), 0.55, 1.35))
    soft = np.clip(ops.feather_mask(m, max(0.0, float(feather_px))).astype(np.float32), 0, 1)
    # ONLY modify the L channel inside the soft mask. The a,b channels
    # (chroma from the painted AI result) are preserved exactly.
    low_lab[..., 0] = np.clip(low_lab[..., 0] + detail_l * float(amount) * scale * soft, 0, 255)
    restored = cv2.cvtColor(np.clip(low_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    # Exact original outside soft mask; only edge transition gets a blend.
    result = restored * soft[..., None] + o * (1.0 - soft[..., None])
    return np.clip(result, 0, 255).astype(np.uint8)


def _enforce_coverage(image: np.ndarray, mask: np.ndarray,
                      target_rgb: tuple[int, int, int], material: str,
                      tolerance: float = 22.0, max_rounds: int = 3,
                      original: np.ndarray | None = None) -> np.ndarray:
    """Normalise mask chroma toward the requested colour while retaining relief.

    The base hue is forced to the selected colour. Only restrained deviations
    from the generated region median are retained as material highlights, so
    an old hue cannot survive as a large coloured patch. Luminance is preserved.
    """
    if cv2 is None:
        return _shift_mean_to(image, mask, target_rgb)
    m = ops.to_mask_array(mask)
    if not m.any():
        return image
    out_rgb = np.clip(image, 0, 255).astype(np.uint8).copy()
    lab = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(np.uint8([[list(target_rgb)]]), cv2.COLOR_RGB2LAB).astype(np.float32)[0, 0]
    material = (material or "object").lower()
    # Preserve more generated chroma structure on metal; less on glass/plastic.
    detail_scale = {"metal": 0.38, "bronze": 0.38, "wood": 0.24, "leather": 0.22,
                    "fabric": 0.24, "plastic": 0.12, "glass": 0.12,
                    "ceramic": 0.15, "concrete": 0.28, "no_texture": 0.0}.get(material, 0.20)
    for channel in (1, 2):
        vals = lab[m, channel]
        med = float(np.median(vals))
        deviation = vals - med
        # Robust limit avoids a rare model-generated coloured blemish dominating.
        if deviation.size:
            q = float(np.percentile(np.abs(deviation), 90))
            if q > 0:
                deviation = np.clip(deviation, -2.0 * q, 2.0 * q)
        lab[m, channel] = np.clip(target_lab[channel] + deviation * detail_scale, 0, 255)
    corrected = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
    corrected = _shift_mean_to(corrected, m, target_rgb)
    # Hard blend toward target for pixels that are still far from target in
    # RGB space — handles AI output that left large original-colour patches
    # untouched. We blend, not replace, to keep material highlights.
    target = np.asarray(target_rgb, dtype=np.float32)
    diff = np.sqrt(np.sum((corrected - target) ** 2, axis=2))
    blend = np.clip((diff - 45.0) / (120.0 - 45.0), 0.0, 0.55)
    blended = corrected * (1.0 - blend[..., None]) + target * blend[..., None]
    # Never let colour correction alter the background.
    out = out_rgb.astype(np.float32)
    out[m] = blended[m]
    return np.clip(out, 0, 255).astype(np.uint8)


def _shift_mean_to(image: np.ndarray, mask: np.ndarray,
                   target_rgb: tuple[int, int, int]) -> np.ndarray:
    m = ops.to_mask_array(mask)
    if not m.any():
        return image
    out = np.asarray(image, dtype=np.float32).copy()
    target = np.asarray(target_rgb, dtype=np.float32)
    # Iterative offset is less destructive than z-score scaling and deals with
    # clipping on bright/dark selected colours.
    for _ in range(3):
        region = out[m]
        delta = target - region.mean(axis=0)
        if float(np.abs(delta).max()) <= 1.0:
            break
        out[m] = np.clip(region + delta, 0, 255)
    return np.clip(out, 0, 255).astype(np.uint8)


def _composite(painted: np.ndarray, original: np.ndarray, mask: np.ndarray,
               feather_px: float) -> np.ndarray:
    soft = np.clip(ops.feather_mask(ops.to_mask_array(mask), feather_px).astype(np.float32), 0, 1)
    return np.clip(painted.astype(np.float32) * soft[..., None] +
                   original.astype(np.float32) * (1.0 - soft[..., None]), 0, 255).astype(np.uint8)


def _mask_of(mask: np.ndarray) -> np.ndarray:
    return ops.to_mask_array(mask)


def _match_mean(image: np.ndarray, mask: np.ndarray, target_rgb) -> np.ndarray:
    return _shift_mean_to(image, mask, tuple(int(x) for x in target_rgb))


def _is_sensible_recolor(mean_rgb: np.ndarray, target_luma: float) -> bool:
    if mean_rgb is None or np.asarray(mean_rgb).size != 3:
        return False
    mean = np.asarray(mean_rgb, dtype=float)
    if float(mean.max()) < 8.0:
        return False
    luma = float(0.299 * mean[0] + 0.587 * mean[1] + 0.114 * mean[2])
    if luma < 0.25 * max(float(target_luma), 1.0):
        return False
    return not (target_luma < 90.0 and luma > 235.0)


def _is_not_black(mean_rgb: np.ndarray) -> bool:
    return mean_rgb is not None and np.asarray(mean_rgb).size == 3 and float(np.max(mean_rgb)) >= 8.0


def _overlay_mask(image: np.ndarray, mask: np.ndarray,
                  color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    if cv2 is None:
        return image
    m = ops.to_mask_array(mask)
    overlay = image.copy()
    overlay[m] = color
    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def _crop_for_diag(image: np.ndarray, mask: np.ndarray, pad: int = 8) -> np.ndarray:
    box = ops.bbox(mask)
    if box is None:
        return image
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    x1, y1, x2, y2 = max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)
    return image[y1:y2, x1:x2].copy() if x2 > x1 and y2 > y1 else image


def _capture_segmentation(diag, image: np.ndarray, seg: SegmentationResult) -> None:
    mask = ops.to_mask_array(seg.mask)
    diag.capture("01_mask", mask, note=f"маска ({mask.mean() * 100:.2f}% кадра)")
    diag.capture("02_mask_overlay", _overlay_mask(image, mask), note="маска поверх оригинала")
    diag.metric("mask_area_ratio", float(seg.area_ratio))
    diag.metric("mask_bbox", list(ops.bbox(mask)) if mask.any() else None)
    diag.metric("mask_dims", list(ops.mask_bbox_dims(mask)) if mask.any() else None)
    diag.metric("mask_pixels", int(mask.sum()))
    diag.metric("segmentation_method", seg.method)
    diag.metric("segmentation_confidence", float(seg.confidence))


def _record_quality_metrics(diag, out: np.ndarray, original: np.ndarray,
                            mask: np.ndarray, target_rgb) -> None:
    m = ops.to_mask_array(mask)
    diag.metric("residual_original_fraction", float(ops.residual_original_fraction(out, original, m)))
    diag.metric("changed_inside_mask", float(
        (np.abs(out[m].astype(int) - original[m].astype(int)).max(axis=1) > 12).mean()
    ) if m.any() else 0.0)
    diag.metric("changed_outside_mask", float(ops.changed_fraction_outside_mask(out, original, m)))
    diag.metric("final_mean_rgb", [float(v) for v in ops.mean_rgb(out, m)])
    diag.metric("target_rgb", list(target_rgb))


def _append_diag_note(notes: list[str], summary: dict) -> None:
    notes.append(f"диагностика: {'включена' if summary.get('enabled') else 'выключена'}")
    if summary.get("enabled"):
        metrics = summary.get("metrics", {})
        notes.append(
            f"маска {metrics.get('mask_area_ratio', 0) * 100:.2f}%; "
            f"остаточные отличия от оригинала {metrics.get('residual_original_fraction', 0) * 100:.1f}%; "
            f"изменения вне маски {metrics.get('changed_outside_mask', 0) * 100:.2f}%"
        )


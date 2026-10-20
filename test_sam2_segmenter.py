"""Focused tests for SAM 2 mask selection; no real model download needed."""
from __future__ import annotations

import numpy as np

from recolor import sam2_segmenter


class FakePredictor:
    def __init__(self, image_shape):
        self.image_shape = image_shape
        self.calls = 0

    def set_image(self, image):
        self.image_shape = image.shape[:2]

    def predict(self, point_coords, point_labels, multimask_output=True):
        h, w = self.image_shape
        x, y = np.rint(point_coords[0]).astype(int)
        base = np.zeros((h, w), dtype=bool)
        base[max(2, y - 12):min(h - 2, y + 13), max(2, x - 10):min(w - 2, x + 11)] = True
        smaller = np.zeros_like(base)
        smaller[max(2, y - 5):min(h - 2, y + 6), max(2, x - 5):min(w - 2, x + 6)] = True
        # Return the correct object and two plausible but smaller alternatives.
        self.calls += 1
        return np.stack((base, smaller, base)), np.asarray([0.94, 0.82, 0.90], dtype=np.float32), None


def test_sam2_point_segmentation_selects_mask_and_tracks_jitter(monkeypatch):
    image = np.full((80, 96, 3), 150, dtype=np.uint8)
    predictor = FakePredictor(image.shape[:2])
    monkeypatch.setattr(sam2_segmenter, "_get_predictor", lambda model_name=None: predictor)

    result = sam2_segmenter.segment_with_sam2(image, 48, 40, model_name="fake-model")

    assert result.mask.shape == image.shape[:2]
    assert result.mask.dtype == np.bool_
    assert result.mask[40, 48]
    assert 0.02 < result.area_ratio < 0.30
    assert result.method.startswith("sam2")
    assert predictor.calls >= 3


def test_sam2_refuses_implausibly_large_mask(monkeypatch):
    class LargePredictor(FakePredictor):
        def predict(self, point_coords, point_labels, multimask_output=True):
            h, w = self.image_shape
            full = np.ones((h, w), dtype=bool)
            return np.stack((full, full, full)), np.asarray([0.99, 0.98, 0.97]), None

    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    monkeypatch.setattr(sam2_segmenter, "_get_predictor", lambda model_name=None: LargePredictor(image.shape[:2]))

    try:
        sam2_segmenter.segment_with_sam2(image, 32, 32, model_name="fake-model")
    except sam2_segmenter.Sam2UnavailableError as exc:
        assert "implausible" in str(exc)
    else:
        raise AssertionError("A full-frame mask must be rejected rather than recoloring the whole scene")


def test_sam2_local_backend_uses_mask_and_preserves_pixels_outside(monkeypatch):
    from recolor.backends import Sam2LocalBackend
    from recolor.base import RecolorRequest, SegmentationResult

    image = np.full((48, 64, 3), (120, 100, 80), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[10:38, 18:46] = True
    seg = SegmentationResult(mask=mask, method="mock-sam2", area_ratio=float(mask.mean()), confidence=0.99)
    monkeypatch.setattr(sam2_segmenter, "segment_with_sam2", lambda *args, **kwargs: seg)

    request = RecolorRequest(
        image_rgb=image, point_x=32, point_y=24, material="leather",
        color_r=30, color_g=120, color_b=210, color_hex=0x1E78D2,
    )
    result = Sam2LocalBackend().run(request)

    assert result.backend == "sam2-local"
    assert result.segmentation is seg
    assert np.array_equal(result.image_rgb[~mask], image[~mask])
    assert np.any(result.image_rgb[mask] != image[mask])


def test_get_backend_selects_sam2_local_without_provider_keys(monkeypatch):
    import config
    from recolor.backends import Sam2LocalBackend, get_backend

    monkeypatch.setattr(config, "AI_BACKEND", "sam2-local")
    monkeypatch.setattr(config, "CLOUDFLARE_ACCOUNT_ID", "")
    monkeypatch.setattr(config, "CLOUDFLARE_API_TOKEN", "")
    assert isinstance(get_backend(), Sam2LocalBackend)

"""Shared data contracts for the Camelion recolor pipeline.

These plain dataclasses keep the backends independent of the HTTP layer and
the segmentation implementation. They intentionally accept either a tuple or
a numpy array for ``target_rgb`` so the same request object can be reused by
the local recolor math and the Cloudflare prompt builder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np


def _as_target_rgb(value: Any) -> tuple[int, int, int]:
    """Normalise a colour spec (tuple/list/ndarray/ARGB int) to an RGB tuple."""
    if value is None:
        return (0, 0, 0)
    if isinstance(value, (int, float)):
        iv = int(value)
        return ((iv >> 16) & 0xFF, (iv >> 8) & 0xFF, iv & 0xFF)
    arr = list(value)
    if len(arr) != 3:
        raise ValueError(f"target_rgb must have 3 components, got {arr!r}")
    return tuple(int(np.clip(c, 0, 255)) for c in arr)  # type: ignore[name-defined]


@dataclass
class RecolorRequest:
    """Everything a backend needs to recolor one selected object."""

    image_rgb: Any
    point_x: float
    point_y: float
    material: str = "wood"
    color_r: int = 0
    color_g: int = 0
    color_b: int = 0
    color_hex: int = 0
    color_name: str = ""
    object_name: str = "object"
    patina: bool = False
    from_pipette: bool = False
    strength: float = 1.0
    guidance_scale: float = 5.0
    num_inference_steps: int = 30
    # Convenience aliases used by the prompt builder and local recolor.
    target_rgb: tuple[int, int, int] | None = None
    hex_color: str = ""

    def __post_init__(self) -> None:
        if self.target_rgb is None:
            self.target_rgb = (int(self.color_r), int(self.color_g), int(self.color_b))
        if not self.hex_color:
            self.hex_color = f"#{int(self.color_hex) & 0xFFFFFF:06X}"

    @property
    def target(self) -> tuple[int, int, int]:
        return self.target_rgb


@dataclass
class SegmentationResult:
    """Object mask plus provenance, returned by every segmentation path."""

    mask: Any
    method: str = "unknown"
    area_ratio: float = 0.0
    confidence: float = 0.0
    object_label: str = "selected object"
    notes: list[str] = field(default_factory=list)


@dataclass
class BackendResult:
    """Final recolor output plus the backend/segmentation provenance."""

    image_rgb: Any
    backend: str = "local"
    segmentation: SegmentationResult | None = None
    object_label: str = "object"
    notes: list[str] = field(default_factory=list)
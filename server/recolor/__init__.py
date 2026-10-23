"""Camelion recolor package: segmentation, local recolor, and AI backends.

Public surface mirrors the historical layout the rest of the server expects:

* ``RecolorRequest`` / ``BackendResult`` / ``SegmentationResult`` live in
  :mod:`recolor.base`.
* ``get_backend`` is re-exported here for callers that used the flat import
  ``from recolor import get_backend``.
"""
from __future__ import annotations

from .base import BackendResult, RecolorRequest, SegmentationResult
from .backends import get_backend

__all__ = [
    "BackendResult",
    "RecolorRequest",
    "SegmentationResult",
    "get_backend",
]
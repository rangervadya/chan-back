"""Optional diagnostic collector: saves intermediate frames + metrics.

Production keeps this disabled (RECOLOR_DIAGNOSTIC=0). When enabled it writes
PNG frames into ``debug_output/<request_id>/`` so the recolour pipeline can be
audited frame-by-frame: original, mask, crop, model output, detail-restored and
final. It also records scalar metrics (mask area, colour error, etc.) and
cleans up directories older than ``RETENTION_SECONDS``.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Any

import numpy as np

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

from logger import get_logger

logger = get_logger(__name__)

RETENTION_SECONDS = 12 * 3600  # keep the last ~12h of debug frames


def request_id_from_env() -> str:
    """Stable request id for tracing, overridable per test."""
    return os.environ.get("RECOLOR_REQUEST_ID", f"req-{uuid.uuid4().hex[:12]}")


def _project_root() -> str:
    """Resolve the project root so debug_output is consistent across CWDs."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def cleanup_expired(base_dir: str | None = None) -> int:
    """Remove diagnostic directories older than the retention window."""
    root = os.path.join(base_dir or _project_root(), "debug_output")
    if not os.path.isdir(root):
        return 0
    removed = 0
    now = time.time()
    try:
        for name in os.listdir(root):
            path = os.path.join(root, name)
            if not os.path.isdir(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if now - mtime > RETENTION_SECONDS:
                _remove_tree(path)
                removed += 1
    except OSError:
        pass
    return removed


def _remove_tree(path: str) -> None:
    import shutil
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


class DiagnosticCollector:
    """Collects intermediate images + metrics for a single request."""

    def __init__(self, request_id: str, enabled: bool | None = None,
                 base_dir: str | None = None) -> None:
        self.request_id = request_id
        if enabled is None:
            enabled = os.environ.get("RECOLOR_DIAGNOSTIC", "0").strip().lower() in {
                "1", "true", "yes", "on"
            }
        self.enabled = bool(enabled)
        self.images: dict[str, np.ndarray] = {}
        self.metrics: dict[str, Any] = {}
        self.notes: dict[str, str] = {}
        if self.enabled:
            cleanup_expired()
            root = os.path.join(base_dir or _project_root(), "debug_output")
            self.dir = os.path.join(root, self.request_id)
            os.makedirs(self.dir, exist_ok=True)
        else:
            self.dir = None

    def capture(self, key: str, image: np.ndarray, note: str | None = None) -> None:
        """Store an image buffer and (when enabled) persist it as PNG."""
        if not self.enabled:
            return
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 3:
            display = arr
        elif arr.ndim == 3 and arr.shape[2] == 4:
            display = arr[..., :3]
        elif arr.ndim == 2:
            display = np.stack([arr] * 3, axis=-1)
        else:
            display = arr
        self.images[key] = display
        if self.dir is not None and Image is not None:
            try:
                self._save_png(key, display)
            except Exception as exc:  # noqa: BLE001
                logger.debug("diagnostic save failed for %s: %s", key, exc)
        if note:
            self.notes[key] = note

    def _save_png(self, key: str, image: np.ndarray) -> None:
        path = os.path.join(self.dir, f"{key}.png")
        Image.fromarray(image, mode="RGB").save(path)

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def flush(self) -> dict[str, Any]:
        """Return a serializable summary used to append to BackendResult.notes."""
        summary = {
            "enabled": self.enabled,
            "request_id": self.request_id,
            "dir": self.dir,
            "images": sorted(self.images.keys()),
            "metrics": dict(self.metrics),
            "notes": dict(self.notes),
        }
        return summary

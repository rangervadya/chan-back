"""Z.AI (GLM-4V) vision client: returns an object bounding box for the tap.

Z.AI is used purely as a segmentation *hint*: it returns a bounding box, which
``segment_object`` then turns into a precise mask via GrabCut. If the API is
unavailable or fails, the local colour-segmenter still runs (see ``ZaiBackend``).
"""
from __future__ import annotations

import io
import json
import base64
import time

import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

from logger import get_logger
from recolor.cloudflare_ai import CloudflareAIError

logger = get_logger(__name__)


class ZaiAIError(RuntimeError):
    def __init__(self, message: str, status: int = 0, retryable: bool = True) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


def _parse_box(text: str, image_w: int, image_h: int) -> tuple[int, int, int, int] | None:
    """Parse a bounding box from a GLM free-text or JSON response.

    Handles:
    * JSON dicts ``{"box": [xmin, ymin, xmax, ymax]}`` (possibly fenced in
      ```json ... ```) and bare ``{"box": null}``;
    * coordinates on the 0..1000 normalised scale, scaled back to the image;
    * swapped coordinate order (sorted to min/max).
    """
    if not text:
        return None
    cleaned = text.strip()
    # Drop markdown code fences (```json ... ```), keeping only inner lines.
    if cleaned.startswith("```"):
        lines = [ln for ln in cleaned.splitlines() if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    # Try strict JSON first.
    data: dict | None = None
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            data = parsed
    except Exception:
        data = None
    if data is not None:
        box = data.get("box", data.get("bbox"))
        if box is None:
            return None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return None
        coords = [float(c) for c in box]
    else:
        return None
    coords = _normalise_coords(coords, image_w, image_h)
    if coords is None:
        return None
    x0, y0, x1, y1 = coords
    x0, x1 = sorted([int(round(x0)), int(round(x1))])
    y0, y1 = sorted([int(round(y0)), int(round(y1))])
    return (x0, y0, x1, y1)


def _normalise_coords(coords: list[float], image_w: int, image_h: int
                      ) -> list[float] | None:
    x0, y0, x1, y1 = coords
    mx = max(x0, y0, x1, y1)
    # The 0..1000 normalised scale is used by GLM for larger than 1000px crops.
    if mx <= 1000 and max(image_w, image_h) > 1000:
        scale_x = image_w / 1000.0
        scale_y = image_h / 1000.0
        return [x0 * scale_x, y0 * scale_y, x1 * scale_x, y1 * scale_y]
    # Absolute pixel coordinates (or already-scaled); clamp on return elsewhere.
    return [x0, y0, x1, y1]


class ZaiVision:
    def __init__(self, api_key: str, model: str = "glm-4.6v-flash",
                 base_url: str = "https://api.z.ai/api/paas/v4",
                 timeout: int = 25, max_retries: int = 1) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self._http_session: requests.Session | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def is_available(self) -> bool:
        return self.is_configured

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"}

    def _session(self) -> "requests.Session":
        if self._http_session is None:
            if requests is None:
                raise ZaiAIError("requests not installed", status=0, retryable=False)
            self._http_session = requests.Session()
        return self._http_session

    def object_box(self, image_rgb: np.ndarray, point_x: float, point_y: float
                   ) -> tuple[int, int, int, int] | None:
        """Return ``(x1, y1, x2, y2)`` in source pixels, or None."""
        t0 = time.monotonic()
        if not self.is_configured:
            raise ZaiAIError("ZaiVision not configured", status=401, retryable=False)
        img = np.asarray(image_rgb, dtype=np.uint8)
        h, w = img.shape[:2]
        b64 = _encode_png(img)
        x = int(np.clip(round(point_x), 0, w - 1))
        y = int(np.clip(round(point_y), 0, h - 1))
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Draw a bounding box around the object the user tapped at pixel "
                                            f"({x},{y}). Return STRICT json: {{\"box\": [x1,y1,x2,y2]}}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
            "max_tokens": 512,
            "temperature": 0.2,
        }
        attempt = 0
        last: Exception | None = None
        while True:
            try:
                resp = self._session().post(f"{self.base_url}/chat/completions",
                                            headers=self._headers, json=payload, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                last = ZaiAIError(f"network error: {exc}", status=0, retryable=True)
                if attempt >= self.max_retries:
                    logger.warning("zai: object_box network error after %.2fs: %s", time.monotonic() - t0, exc)
                    raise last from exc
                attempt += 1
                continue
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    raise ZaiAIError("invalid JSON from Z.AI", status=502, retryable=True)
                text = _extract_text(body)
                box = _parse_box(text, w, h)
                logger.info("zai: object_box done in %.2fs box=%s", time.monotonic() - t0, box)
                if box is not None:
                    return box
                return None
            message, retryable = _zai_parse_error(resp)
            err = ZaiAIError(message, status=resp.status_code, retryable=retryable)
            if not retryable or attempt >= self.max_retries:
                logger.warning("zai: object_box error after %.2fs: %s", time.monotonic() - t0, err)
                raise err
            attempt += 1
            last = err
            continue
        if last:
            raise last  # pragma: no cover


def _extract_text(body) -> str:
    """Pull the assistant message text out of an Anthropic-compatible reply."""
    if isinstance(body, str):
        return body
    try:
        choices = body.get("choices", []) or []
        if choices:
            msg = choices[0].get("message", {})
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if "msg" in body:
            return str(body.get("msg", ""))
    except Exception:  # noqa: BLE001
        pass
    return ""


def _zai_parse_error(resp) -> tuple[str, bool]:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return f"Z.AI error (HTTP {resp.status_code})", resp.status_code in {429, 503}
    err = body.get("error", {})
    if isinstance(err, dict):
        msg = err.get("message", err.get("detail", str(err)))
    else:
        msg = str(err) or resp.text[:300]
    retryable = resp.status_code in {429, 503}
    return str(msg), retryable


def _encode_png(image: np.ndarray) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required for Z.AI image payloads")
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

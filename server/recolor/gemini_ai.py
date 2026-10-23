"""Google Gemini Vision client (object labelling only).

Gemini is an analysis-only backend: it can *describe* the tapped object so the
prompt builder can name it, but it does not perform image generation. The local
or Cloudflare backends handle the actual recolour.
"""
from __future__ import annotations

import io
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

_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiAI:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash",
                 timeout: int = 60, max_retries: int = 2) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self._http_session: requests.Session | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _post(self, payload: dict) -> str:
        url = f"{_GEMINI_ENDPOINT}/{self.model}:generateContent"
        params = {"key": self.api_key}
        headers = {"Content-Type": "application/json"}
        attempt = 0
        while True:
            try:
                resp = self._session().post(url, params=params, headers=headers,
                                            json=payload, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    raise CloudflareAIError(f"Gemini network error: {exc}", status=0, retryable=True) from exc
                attempt += 1
                continue
            if resp.status_code != 200:
                err = self._parse(resp)
                raise CloudflareAIError(err, status=resp.status_code, retryable=resp.status_code in {429, 503})
            return resp

    @staticmethod
    def _parse(resp) -> str:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return f"Gemini error (HTTP {resp.status_code}): {resp.text[:300]}"
        err = body.get("error", {})
        if isinstance(err, dict) and err.get("message"):
            return err["message"]
        return f"Gemini error (HTTP {resp.status_code})"

    def describe(self, image_rgb: np.ndarray, point_x: float, point_y: float) -> str:
        t0 = time.monotonic()
        img = np.asarray(image_rgb, dtype=np.uint8)
        h, w = img.shape[:2]
        x = int(np.clip(round(point_x), 0, w - 1))
        y = int(np.clip(round(point_y), 0, h - 1))
        b64 = self._encode_png(img)
        prompt = (
            f"What single object is at pixel ({x},{y}) in this photo? "
            f"Reply with only 2-4 words describing that object, no background."
        )
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": b64}},
                ]
            }],
            "generationConfig": {"maxOutputTokens": 60, "temperature": 0.2},
        }
        resp = self._post(payload)
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return ""
        parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        # Strip any markdown fencing the model emits.
        text = text.strip().strip("```").strip()
        if "\n" in text:
            text = text.splitlines()[0]
        label = text.strip()
        logger.info("gemini: describe done in %.2fs label=%r", time.monotonic() - t0, label)
        return label

    @staticmethod
    def _encode_png(image: np.ndarray) -> str:
        if Image is None:
            raise RuntimeError("Pillow is required for Gemini image payloads")
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _session(self) -> "requests.Session":
        if self._http_session is None:
            if requests is None:
                raise RuntimeError("requests is required for Gemini backend")
            self._http_session = requests.Session()
        return self._http_session

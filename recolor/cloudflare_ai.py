"""Cloudflare Workers AI client for Camelion.

Wraps the three models the recolor pipeline actually exercises:

* the DETR object detector that returns a bounding box for the tapped point;
* the vision LLM (Llama-3.2-11B-Vision) used only to *name* the object;
* FLUX.2 [klein] 4B and the SD-1.5 inpaint model that perform the edit.

The transport is a bare ``requests.Session`` so the contract is fully exercised
by the unit tests without a live account.
"""
from __future__ import annotations

import io
import os
import struct
import time
from typing import Any

import numpy as np

try:
    import requests
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
MAX_SIDE = 768


class CloudflareAIError(RuntimeError):
    """Raised for any Cloudflare AI failure that is not a network blip."""

    def __init__(self, message: str, status: int = 0, retryable: bool = True) -> None:
        super().__init__(message)
        self.status = status
        # Auth / quota / permission / not-found failures never improve by
        # retrying (matches the policy used by _parse_error).
        if status in {400, 401, 403, 404, 429, 3036, 3037, 2200, 9103, 9105, 7003, 81044, 9100}:
            retryable = False
        self.retryable = bool(retryable)


def _parse_error(response: Any) -> tuple[str, bool]:
    """Map a Cloudflare error response onto ``(message, retryable)``.

    The daily-free-neuron limit (error 3036) and quota errors are never
    retried: retrying burns the same 429 and only delays the failure.
    """
    status = getattr(response, "status_code", 0) or 0
    text = getattr(response, "text", "") or ""
    try:
        body = response.json()
    except Exception:  # noqa: BLE001
        body = None
    message_parts: list[str] = []
    retryable = True
    codes: list[int] = []
    if isinstance(body, dict):
        for err in body.get("errors", []) or []:
            if isinstance(err, dict):
                if "code" in err:
                    codes.append(int(err["code"]))
                if "message" in err:
                    message_parts.append(str(err["message"]))
    # Hard, non-retryable conditions -------------------------------------------------
    non_retry_codes = {3036, 3037, 2200, 9103, 9105, 7003, 81044, 9100}
    if any(c in non_retry_codes for c in codes):
        retryable = False
    if status == 429:
        retryable = False
    if status in {400, 401, 403, 404}:
        retryable = False
    message = " | ".join(message_parts) if message_parts else text
    if not message:
        message = f"Cloudflare AI error (HTTP {status})"
    if codes:
        message = f"[error {codes[0]}] {message}" if len(codes) == 1 else f"[errors {codes}] {message}"
    return message, retryable


def _fit_klein_input_size(box_w: int, box_h: int, cap: int) -> tuple[int, int]:
    """Scale a bounding-box size to fit within ``cap`` keeping aspect ratio.

    Output sides are multiples of 32 (Flux tokenizer grid) and at least 64.
    """
    box_w = max(1, int(box_w))
    box_h = max(1, int(box_h))
    scale = min(cap / box_w, cap / box_h, 1.0)
    cw = max(64, int(round(box_w * scale / 32.0)) * 32)
    ch = max(64, int(round(box_h * scale / 32.0)) * 32)
    cw = min(cw, cap)
    ch = min(ch, cap)
    return cw, ch


def _resize_with_ratio(image: np.ndarray, size: int) -> tuple[bytes, tuple[int, int]]:
    """Resize to fit within ``size`` square, return PNG bytes + original size."""
    h, w = image.shape[:2]
    if max(w, h) <= size and w % 32 == 0 and h % 32 == 0:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
        return buf.getvalue(), (w, h)
    scale = min(size / w, size / h, 1.0)
    nw = max(64, int(round(w * scale / 32)) * 32)
    nh = max(64, int(round(h * scale / 32)) * 32)
    nw = min(nw, size)
    nh = min(nh, size)
    if cv2 is not None:
        resized = cv2.resize(np.asarray(image, np.uint8), (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        resized = np.asarray(Image.fromarray(np.asarray(image, np.uint8)).resize((nw, nh), Image.LANCZOS))
    buf = io.BytesIO()
    Image.fromarray(resized).save(buf, format="PNG")
    return buf.getvalue(), (nw, nh)


def _mask_to_png_bytes(mask: np.ndarray) -> bytes:
    m = np.asarray(mask)
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255
    if m.ndim == 2:
        m = np.stack([m] * 3, axis=-1)
    buf = io.BytesIO()
    Image.fromarray(m).save(buf, format="PNG")
    return buf.getvalue()


class CloudflareAI:
    """Client for the Cloudflare Workers AI text/image models."""

    def __init__(self, account_id: str, api_token: str,
                 inpaint_model: str = "",
                 vision_model: str = "",
                 klein_model: str = "",
                 klein_guidance: float = 3.5,
                 timeout: int = 75,
                 max_retries: int = 0,
                 detection_model: str = "@cf/facebook/detr-resnet-50",
                 detection_min_score: float = 0.35) -> None:
        self.account_id = account_id
        self.api_token = api_token
        self.inpaint_model = inpaint_model
        self.vision_model = vision_model
        self.klein_model = klein_model
        self.klein_guidance = float(klein_guidance)
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.detection_model = detection_model
        self.detection_min_score = float(detection_min_score)
        self._http_session: requests.Session | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.account_id) and bool(self.api_token)

    def _session(self) -> "requests.Session":
        if self._http_session is None:
            self._http_session = requests.Session()
        return self._http_session

    @property
    def base_url(self) -> str:
        return f"{BASE_URL}/{self.account_id}/ai"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_token}"}

    def _run(self, model: str, payload: dict[str, Any], raw: bool = False) -> Any:
        url = f"{self.base_url}/run/{model}"
        headers = self._headers()
        if raw:
            headers["Content-Type"] = "application/octet-stream"
        attempt = 0
        last: Exception | None = None
        t0 = time.monotonic()
        while True:
            try:
                resp = self._session().post(url, headers=headers, json=payload, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= self.max_retries:
                    logger.warning("cloudflare: %s network error after %.2fs: %s", model, time.monotonic() - t0, exc)
                    raise CloudflareAIError(f"network error calling {model}: {exc}", status=0, retryable=True) from exc
                attempt += 1
                continue
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    if raw:
                        return resp.content
                    raise CloudflareAIError(f"invalid response from {model}", status=502, retryable=True)
                if body.get("success", True) and not body.get("errors"):
                    logger.info("cloudflare: %s ok in %.2fs", model, time.monotonic() - t0)
                    return body.get("result", body)
                message, retryable = _parse_error(resp)
                err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
                if not retryable or attempt >= self.max_retries:
                    logger.warning("cloudflare: %s error in %.2fs: %s", model, time.monotonic() - t0, err)
                    raise err
                attempt += 1
                continue
            if resp.status_code in {429}:
                _, retryable = _parse_error(resp)
                err = CloudflareAIError(_parse_error(resp)[0], status=429, retryable=False)
                logger.warning("cloudflare: %s 429 in %.2fs", model, time.monotonic() - t0)
                raise err
            message, retryable = _parse_error(resp)
            err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
            if not retryable or attempt >= self.max_retries:
                logger.warning("cloudflare: %s error in %.2fs: %s", model, time.monotonic() - t0, err)
                raise err
            attempt += 1
            continue
        if last:  # pragma: no cover
            raise CloudflareAIError(f"failed after retries: {last}", status=0, retryable=True) from last
        raise CloudflareAIError(f"call to {model} failed", status=0, retryable=True)  # pragma: no cover

    def _run_multipart(self, model: str, data: dict[str, Any],
                      files: dict[str, tuple[str, bytes, str]]) -> Any:
        """Post a ``multipart/form-data`` request (used by Flux image-gen models)."""
        url = f"{self.base_url}/run/{model}"
        headers = self._headers()
        attempt = 0
        last: Exception | None = None
        t0 = time.monotonic()
        while True:
            try:
                resp = self._session().post(url, headers=headers, data=data,
                                            files=files, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if attempt >= self.max_retries:
                    logger.warning("cloudflare: %s multipart network error after %.2fs: %s", model, time.monotonic() - t0, exc)
                    raise CloudflareAIError(f"network error calling {model}: {exc}", status=0, retryable=True) from exc
                attempt += 1
                continue
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    return resp.content
                if body.get("success", True) and not body.get("errors"):
                    logger.info("cloudflare: %s multipart ok in %.2fs", model, time.monotonic() - t0)
                    return body.get("result", body)
                message, retryable = _parse_error(resp)
                err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
                if not retryable or attempt >= self.max_retries:
                    logger.warning("cloudflare: %s multipart error in %.2fs: %s", model, time.monotonic() - t0, err)
                    raise err
                attempt += 1
                continue
            if resp.status_code in {429}:
                err = CloudflareAIError(_parse_error(resp)[0], status=429, retryable=False)
                logger.warning("cloudflare: %s multipart 429 in %.2fs", model, time.monotonic() - t0)
                raise err
            message, retryable = _parse_error(resp)
            err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
            if not retryable or attempt >= self.max_retries:
                logger.warning("cloudflare: %s multipart error in %.2fs: %s", model, time.monotonic() - t0, err)
                raise err
            attempt += 1
            continue

    def describe(self, image_rgb: np.ndarray, point_x: float, point_y: float) -> str:
        """Ask the vision LLM what object the tap selects; empty string if unknown."""
        t0 = time.monotonic()
        img = np.asarray(image_rgb, dtype=np.uint8)
        h, w = img.shape[:2]
        x = int(np.clip(round(point_x), 0, w - 1))
        y = int(np.clip(round(point_y), 0, h - 1))
        if not self.vision_model:
            return ""
        b64 = _image_to_base64(img)
        color = img[y, x].tolist()
        prompt = (
            f"What is the object the user tapped at pixel ({x},{y}) on this photo? "
            f"Describe only the single object at that point in 2-4 words, "
            f"no background. The pixel color at the tap is rgb({color})."
        )
        result = self._run(self.vision_model, {"prompt": prompt, "image": b64})
        if isinstance(result, dict):
            resp = result.get("response", "")
        else:
            resp = ""
        label = _clean_label(str(resp))
        logger.info("cloudflare: vision describe done in %.2fs label=%r", time.monotonic() - t0, label)
        return label

    def detect_object(self, image_rgb: np.ndarray, point_x: float, point_y: float):
        """Return (box, label, score) for the detected object containing the tap."""
        t0 = time.monotonic()
        if not self.detection_model:
            return None
        img = np.asarray(image_rgb, dtype=np.uint8)
        h, w = img.shape[:2]
        x = int(np.clip(round(point_x), 0, w - 1))
        y = int(np.clip(round(point_y), 0, h - 1))
        scale = min(MAX_SIDE / w, MAX_SIDE / h, 1.0) if max(w, h) > MAX_SIDE else 1.0
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        if cv2 is not None:
            small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            small = np.asarray(Image.fromarray(img).resize((nw, nh), Image.LANCZOS))
        sx, sy = (x * scale if scale else x), (y * scale if scale else y)
        blob = _mask_to_png_bytes(small)
        try:
            resp = self._raw_post(self.detection_model, blob)
        except CloudflareAIError:
            raise
        result = _as_list(resp)
        best = None
        bx, by = sx, sy
        for item in result:
            label = item.get("label", "")
            score = float(item.get("score", 0.0))
            box = item.get("box", item.get("bbox", {}))
            if isinstance(box, dict):
                xmin = float(box.get("xmin", box.get("left", 0)))
                ymin = float(box.get("ymin", box.get("top", 0)))
                xmax = float(box.get("xmax", box.get("right", 0)))
                ymax = float(box.get("ymax", box.get("bottom", 0)))
            elif isinstance(box, (list, tuple)) and len(box) == 4:
                xmin, ymin, xmax, ymax = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            else:
                continue
            if not (xmin <= bx <= xmax and ymin <= by <= ymax):
                continue
            if score >= self.detection_min_score and (best is None or score > best[2]):
                best = (
                    _scale_box(xmin, ymin, xmax, ymax, scale),
                    _clean_label(str(label)),
                    score,
                )
        logger.info("cloudflare: DETR detect_object done in %.2fs best=%s", time.monotonic() - t0, best)
        return best

    def recolor_klein(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str,
                      size: int = 1024, seed: int | None = None) -> np.ndarray:
        """FLUX.2 [klein] 4B guided edit: send image+mask+prompt, return RGB."""
        t0 = time.monotonic()
        if not self.klein_model:
            raise CloudflareAIError(
                "CLOUDFLARE_KLEIN_MODEL is not configured; cannot run FLUX.2 [klein]. "
                "Set CLOUDFLARE_RECOLOR_MODEL=inpaint to use inpaint instead.",
                status=500, retryable=False)
        img = np.asarray(image_rgb, dtype=np.uint8)
        m = mask if mask is not None else np.zeros(img.shape[:2], bool)
        m = m.astype(bool)
        box_w, box_h = ops_bbox_size(m)
        cw, ch = _fit_klein_input_size(box_w, box_h, size)
        payload_size = max(cw, ch)
        img_bytes, (rw, rh) = _resize_with_ratio(img, payload_size)
        # Scale mask to the payload size as a PNG.
        if cv2 is not None:
            mask_resized = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((rw, rh), Image.NEAREST))
        mask_bytes = _mask_to_png_bytes(mask_resized)
        # FLUX.2 [klein] via Workers AI rejects a JSON body: its schema at the
        # top level requires `multipart`. Send image+mask as file uploads and the
        # scalar knobs as form fields (checked live: JSON "image" byte arrays
        # return 400 "required properties at '/' are 'multipart'").
        fields = {
            "prompt": prompt,
            "width": str(rw),
            "height": str(rh),
            "seed": str(seed or 0),
            "guidance": str(self.klein_guidance),
        }
        files = {
            "input_image_0": (
                "input_image_0.png",
                img_bytes,
                "image/png",
            ),
        }
        result = self._run_multipart(self.klein_model, fields, files)
        out = _decode_image_result(result, img.shape)
        logger.info("cloudflare: klein recolor done in %.2fs (model=%s)", time.monotonic() - t0, self.klein_model)
        return out

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, prompt: str,
                negative_prompt: str = "", strength: float = 0.82,
                num_steps: int = 20, guidance: float = 7.5,
                size: int = 1024, seed: int | None = None) -> np.ndarray:
        """SD-1.5 inpainting: image+mask as byte arrays per the live API contract."""
        t0 = time.monotonic()
        if not self.inpaint_model:
            raise CloudflareAIError(
                "CLOUDFLARE_INPAINT_MODEL is not configured; cannot run inpaint.",
                status=500, retryable=False)
        img = np.asarray(image_rgb, dtype=np.uint8)
        m = np.asarray(mask, dtype=bool) if mask is not None else np.zeros(img.shape[:2], bool)
        box_w, box_h = ops_bbox_size(m)
        cw, ch = _fit_klein_input_size(box_w, box_h, size)
        img_bytes, (rw, rh) = _resize_with_ratio(img, max(cw, ch))
        if cv2 is not None:
            mask_resized = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((rw, rh), Image.NEAREST))
        mask_bytes = _mask_to_png_bytes(mask_resized)
        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "image": list(img_bytes),
            "mask": list(mask_bytes),
            "width": rw,
            "height": rh,
            "strength": float(strength),
            "num_steps": int(num_steps),
            "guidance_scale": float(guidance),
            "seed": seed or 0,
        }
        result = self._run(self.inpaint_model, payload)
        out = _decode_image_result(result, img.shape)
        logger.info("cloudflare: inpaint done in %.2fs (model=%s)", time.monotonic() - t0, self.inpaint_model)
        return out

    def _raw_post(self, model: str, blob: bytes) -> Any:
        """Post raw binary (image) to a model endpoint, e.g. DETR detection."""
        t0 = time.monotonic()
        url = f"{self.base_url}/run/{model}"
        headers = self._headers()
        headers["Content-Type"] = "image/png"
        attempt = 0
        while True:
            try:
                resp = self._session().post(url, headers=headers, data=blob, timeout=self.timeout)
            except Exception as exc:  # noqa: BLE001
                if attempt >= self.max_retries:
                    logger.warning("cloudflare: %s raw network error after %.2fs: %s", model, time.monotonic() - t0, exc)
                    raise CloudflareAIError(f"network error calling {model}: {exc}", status=0, retryable=True) from exc
                attempt += 1
                continue
            if resp.status_code == 200:
                try:
                    body = resp.json()
                except Exception:  # noqa: BLE001
                    logger.info("cloudflare: %s raw ok in %.2fs (binary)", model, time.monotonic() - t0)
                    return resp.content
                if body.get("success", True) and not body.get("errors"):
                    logger.info("cloudflare: %s raw ok in %.2fs", model, time.monotonic() - t0)
                    return body.get("result", body)
                message, retryable = _parse_error(resp)
                err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
                if not retryable or attempt >= self.max_retries:
                    logger.warning("cloudflare: %s raw error in %.2fs: %s", model, time.monotonic() - t0, err)
                    raise err
                attempt += 1
                continue
            message, retryable = _parse_error(resp)
            err = CloudflareAIError(message, status=resp.status_code, retryable=retryable)
            if not retryable or attempt >= self.max_retries:
                logger.warning("cloudflare: %s raw error in %.2fs: %s", model, time.monotonic() - t0, err)
                raise err
            attempt += 1
            continue
        raise CloudflareAIError("unreachable", status=0)  # pragma: no cover


def _image_to_base64(image: np.ndarray) -> str:
    if Image is None:
        raise RuntimeError("Pillow is required for Cloudflare image payloads")
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(buf, format="PNG")
    import base64
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _decode_image_result(result: Any, original_shape: tuple[int, ...]) -> np.ndarray:
    """Decode a Cloudflare image-generation result back to an RGB uint8 array."""
    if isinstance(result, dict):
        b64 = result.get("image") or result.get("base64") or result.get("data", "")
        if isinstance(b64, str) and b64:
            data = _b64_to_bytes(b64)
        else:
            data = None
    elif isinstance(result, (bytes, bytearray)):
        data = bytes(result)
    elif isinstance(result, str):
        data = _b64_to_bytes(result)
    else:
        data = None
    if data is not None:
        img = Image.open(io.BytesIO(data))
        if cv2 is not None:
            decoded = np.array(img.convert("RGB"))
        else:
            decoded = np.array(img.convert("RGB"))
        if decoded.shape[:2] == original_shape[:2]:
            return decoded
        # Resize back to the source resolution.
        if cv2 is not None:
            return cv2.resize(decoded, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return np.array(Image.fromarray(decoded).resize((original_shape[1], original_shape[0]), Image.LANCZOS))
    # Fallback: return a copy of the original so the caller can post-process.
    return np.zeros(original_shape[:2] + (3,), np.uint8)


def _b64_to_bytes(value: str) -> bytes:
    import base64
    v = value.strip()
    if v.startswith("data:") and "base64," in v:
        v = v.split("base64,", 1)[1]
    return base64.b64decode(v)


def _as_list(result: Any) -> list[dict]:
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        first = next(iter(result.values()))
        if isinstance(first, list):
            return [item for item in first if isinstance(item, dict)]
    return []


def _scale_box(xmin: float, ymin: float, xmax: float, ymax: float, inv: float
              ) -> tuple[int, int, int, int]:
    return (int(round(xmin / inv)), int(round(ymin / inv)),
            int(round(xmax / inv)), int(round(ymax / inv)))


def _clean_label(text: str) -> str:
    text = text.strip().strip("```").strip()
    text = text.replace("\n", " ").strip()
    if not text:
        return ""
    low = text.lower()
    for token in ("the ", "a ", "an ", "object", "item", "thing"):
        if low.startswith(token):
            text = text[len(token):].strip()
    return text


def ops_bbox_size(mask: np.ndarray) -> tuple[int, int]:
    from . import ops as _ops
    b = _ops.bbox(mask)
    if b is None:
        h, w = mask.shape
        return w, h
    return b[2] - b[0], b[3] - b[1]


def _image_to_png_bytes(image: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image, np.uint8)).save(buf, format="PNG")
    return buf.getvalue()

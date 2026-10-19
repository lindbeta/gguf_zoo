"""Remote image generation via an OpenAI-compatible API (FR-20).

Network access happens ONLY when the user explicitly passes the --image-*
arguments (FR-0.4 exception). The HTTP transport is a thin seam
(_post_json / _get_bytes) so tests can run fully offline.
"""
from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request

TIMEOUT = 180


class ImageGenError(Exception):
    pass


def sanitize_name(name: str | None) -> str:
    """Filesystem-safe model name for default output filenames (FR-20.4)."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return safe or "model"


def default_paths(model_name: str | None, ext: str = "png"):
    base = sanitize_name(model_name)
    return (f"{base}_species_image.{ext}",
            f"{base}_species_description.md")


def endpoint_for(url: str) -> str:
    u = url.rstrip("/")
    if u.endswith("/images/generations"):
        return u
    if u.endswith("/v1"):
        return u + "/images/generations"
    return u + "/v1/images/generations"


def _post_json(url: str, payload: dict, key: str) -> dict:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, ValueError) as exc:
        raise ImageGenError(f"image API request failed: {exc}") from exc


def _get_bytes(url: str) -> bytes:
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            return resp.read()
    except urllib.error.URLError as exc:
        raise ImageGenError(f"image download failed: {exc}") from exc


def generate_image(prompt: str, url: str, model: str, key: str = "",
                   size: str = "1024x1024") -> tuple[bytes, str]:
    """Send the prompt to an OpenAI-compatible images API.

    Returns (image bytes, extension). Accepts b64_json or url responses.
    """
    data = _post_json(endpoint_for(url),
                      {"model": model, "prompt": prompt, "n": 1,
                       "size": size, "response_format": "b64_json"}, key)
    try:
        item = data["data"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise ImageGenError("image API returned no image data") from exc
    if item.get("b64_json"):
        return base64.b64decode(item["b64_json"]), "png"
    if item.get("url"):
        img_url = item["url"]
        ext = img_url.rsplit(".", 1)[-1].lower() if "." in img_url.rsplit("/", 1)[-1] else "png"
        return _get_bytes(img_url), ext
    raise ImageGenError("image API response contained neither b64_json nor url")

"""
prefilter.py  —  STAGE 0: open-source quality gate
==================================================
Cheap, deterministic image checks that reject junk BEFORE any paid VLM call.
This is the "open-source" half of the hybrid: it costs ~nothing, runs in
milliseconds, and is MORE accurate than an LLM at "is this too blurry / too
dark / too small". Every image rejected here is a paid Gemini call saved.

Implemented for real with numpy + PIL (runs in this prototype as-is):
  - blur     : variance of the Laplacian (focus measure)
  - exposure : mean luma (too dark / blown out)
  - size     : megapixels and aspect ratio

PLUGGABLE deep detectors (interface below) — in production wire these to
open-source models so NSFW/group rejects also skip the LLM:
  - NSFW            -> e.g. CLIP-based NSFW classifier / Falconsai nsfw_image_detection
  - Face count      -> e.g. RetinaFace / MediaPipe / YOLO-face  (group detection)
  - Face presence   -> same detector (reject "no human face")
The prototype ships a no-op DeepDetector so it runs without heavy weights;
swap in a real one without touching the pipeline.
"""

from __future__ import annotations

import io
from typing import Optional, Protocol

import numpy as np
from PIL import Image

from .schemas import PreFilterResult


def _laplacian_variance(gray: np.ndarray) -> float:
    """Focus measure: convolve with a 3x3 Laplacian kernel, return variance."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    # 'valid' 2D convolution via stride tricks (no scipy dependency).
    sub = np.lib.stride_tricks.sliding_window_view(g, (3, 3))
    conv = np.einsum("ijkl,kl->ij", sub, k)
    return float(conv.var())


def load_image(image_bytes: bytes) -> Image.Image:
    """Bytes -> RGB PIL image with stable MIME handling."""
    img = Image.open(io.BytesIO(image_bytes))
    return img.convert("RGB")


class DeepDetector(Protocol):
    """Optional open-source ML gate. Return dict of flags (1/0) and face_count."""
    def detect(self, img: Image.Image) -> dict: ...


class NoOpDeepDetector:
    """Default: defers NSFW/group/face decisions to the VLM stage."""
    def detect(self, img: Image.Image) -> dict:
        return {"nsfw": 0, "face_count": 1}


def run_prefilter(
    image_bytes: bytes,
    cfg: dict,
    deep: Optional[DeepDetector] = None,
) -> tuple[PreFilterResult, Image.Image]:
    pf_cfg = cfg["prefilter"]
    img = load_image(image_bytes)
    arr = np.asarray(img)

    w, h = img.size
    megapixels = (w * h) / 1_000_000.0
    aspect = w / h if h else 0.0
    gray = arr[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    brightness = float(gray.mean())
    blur = _laplacian_variance(gray)

    reasons = []
    if megapixels < pf_cfg["min_megapixels"]:
        reasons.append(f"too_small ({megapixels:.2f}MP)")
    if blur < pf_cfg["min_blur_score"]:
        reasons.append(f"too_blurry (lapvar={blur:.1f})")
    if brightness < pf_cfg["min_brightness"]:
        reasons.append(f"too_dark (luma={brightness:.0f})")
    if brightness > pf_cfg["max_brightness"]:
        reasons.append(f"overexposed (luma={brightness:.0f})")
    if not (pf_cfg["min_aspect_ratio"] <= aspect <= pf_cfg["max_aspect_ratio"]):
        reasons.append(f"bad_aspect ({aspect:.2f})")

    # Optional deep detectors (NSFW / group) — also reject pre-LLM.
    detector = deep or NoOpDeepDetector()
    deep_out = detector.detect(img)
    if deep_out.get("nsfw", 0) == 1:
        reasons.append("nsfw_prefilter")
    if deep_out.get("face_count", 1) == 0:
        reasons.append("no_face")

    result = PreFilterResult(
        passed=len(reasons) == 0,
        blur_score=round(blur, 2),
        brightness=round(brightness, 2),
        megapixels=round(megapixels, 3),
        aspect_ratio=round(aspect, 3),
        reasons=reasons,
    )
    return result, img

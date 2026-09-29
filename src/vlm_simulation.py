"""
vlm_simulation.py  —  STAGE 1 + 2: deterministic mock brain
===========================================================
Lets the WHOLE pipeline run end-to-end with zero credentials and zero network,
so the prototype is demonstrably working *right now*. It is deterministic
(seeded by image hash) and reflects real image features extracted in Stage 0,
so its output is plausible rather than random — perfect for a live demo and
for unit-testing the blend/cache/fallback logic.

Swap `--mode simulation` -> `--mode gemini` to hit the real model. Same return
contract as score_with_gemini().
"""

from __future__ import annotations

import hashlib

from PIL import Image
import numpy as np

from .schemas import QualitativeAttributes, FlagSet

MODEL_VERSION = "sim-v1"


def _seed(img_hash: str, salt: str) -> float:
    h = hashlib.sha256((img_hash + salt).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # 0..1, stable per (image, salt)


def image_pixel_hash(img: Image.Image) -> str:
    """Stable hash of pixel content — identical images seed identical sim scores."""
    return hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def qualitative_raw(img: Image.Image, gender: str, img_hash: str) -> dict:
    """Deterministic raw qualitative dict (16 CamelCase keys), feature-aware."""
    arr = np.asarray(img).astype(np.float32)
    gray = arr[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    return {
        "Expression": _clip(0.4 + _seed(img_hash, "ex") * 0.55),
        "FaceOrientationVisibility": _clip(0.45 + _seed(img_hash, "fo") * 0.5),
        "EyeContact": 1 if _seed(img_hash, "ey") > 0.4 else 0,
        "GroomingNeatness": _clip(0.45 + _seed(img_hash, "gr") * 0.5),
        "Centering": _clip(0.4 + _seed(img_hash, "ce") * 0.55),
        "PoseCameraAngle": _clip(0.5 + _seed(img_hash, "po") * 0.45),
        "BackgroundForegroundSeparation": _clip(0.35 + contrast * 0.6 + (_seed(img_hash, "bg") - 0.5) * 0.3),
        "LightingOnFace": _clip(0.45 + (brightness - 0.5) * 1.2 + (_seed(img_hash, "lt") - 0.5) * 0.3),
        "Attire": _clip(0.35 + _seed(img_hash, "at") * 0.6),
        "CelebrityStock": 1 if _seed(img_hash, "cs") > 0.93 else 0,
        "NSFW": 0,
        "AdvertisementsOrText": 1 if _seed(img_hash, "ad") > 0.9 else 0,
        "GroupImage": 1 if _seed(img_hash, "gi") > 0.92 else 0,
        "Inappropriate": 0,
        "RephotographOrScreenshot": 1 if _seed(img_hash, "rp") > 0.94 else 0,
        "GenderMismatch": 0,
    }


def aesthetic_raw(img: Image.Image, img_hash: str) -> dict:
    arr = np.asarray(img).astype(np.float32)
    gray = arr[..., :3].dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))
    brightness = float(gray.mean()) / 255.0
    contrast = float(gray.std()) / 128.0
    return {
        "photogenic": _clip(0.30 + brightness * 0.20 + contrast * 0.15 + _seed(img_hash, "aes") * 0.45),
        "reasoning": f"[sim] luma={brightness:.2f} contrast={contrast:.2f}; deterministic photogenic estimate.",
    }


class SimulationAdapter:
    """Deterministic, credential-free adapter for demos/tests."""
    name = MODEL_VERSION

    def __init__(self, img_hash_of=None):
        # img_hash is per-image; resolved at call time via a closure the
        # pipeline supplies, or computed by the caller. We accept it per call.
        self._hash_fn = img_hash_of

    def _h(self, img):
        return self._hash_fn(img) if self._hash_fn else "0" * 64

    def qualitative_raw(self, img, gender):
        return qualitative_raw(img, gender, self._h(img))

    def aesthetic_raw(self, img):
        return aesthetic_raw(img, self._h(img))


class PartialSimulationAdapter(SimulationAdapter):
    """A simulation 'primary' that deliberately FAILS to score some parameters
    (drops keys / returns nulls) — used to demonstrate and test that the
    open-source fallback repairs exactly those parameters, nothing more."""
    name = "sim-primary-partial"

    def __init__(self, drop_qual=(), drop_aes=(), img_hash_of=None):
        super().__init__(img_hash_of)
        self._drop_qual = set(drop_qual)
        self._drop_aes = set(drop_aes)

    def qualitative_raw(self, img, gender):
        d = super().qualitative_raw(img, gender)
        for k in self._drop_qual:
            d[k] = None  # Gemini "couldn't score this parameter"
        return d

    def aesthetic_raw(self, img):
        d = super().aesthetic_raw(img)
        for k in self._drop_aes:
            d[k] = None
        return d


def score_with_simulation(img: Image.Image, gender: str, img_hash: str):
    """Back-compat shim: full deterministic score as (attrs, flags, aes, reason)."""
    from .fusion import ResilientScorer

    class _Fixed(SimulationAdapter):
        def _h(self, _img):
            return img_hash
    attrs, flags, aes, reason, _prov = ResilientScorer(_Fixed()).score(img, gender, img_hash)
    return attrs, flags, aes, reason

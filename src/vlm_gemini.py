"""
vlm_gemini.py  —  STAGE 1 + 2: the Gemini brain (PRODUCTION path)
================================================================
One vendor, one SDK. Two structured calls per image, run in parallel:
  * qualitative + flags  (the 16-key schema)
  * aesthetic / photogenic score (reframed, non-sexual -> Gemini answers it
    reliably with NO refusals; this is the whole reason Grok is no longer needed)

Determinism levers (fixes PRD bug #6):
  temperature = 0, response_mime_type = application/json, response_schema set.

Requires:  pip install google-genai   and   env GEMINI_API_KEY
This module is import-safe even if google-genai is absent (raises only on use),
so the simulation path keeps working with zero dependencies.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

import os

from .config_loader import load_qualitative_prompt, load_aesthetic_prompt, load_config
from .resilience import CircuitBreaker, retry_call
from .schemas import QualitativeAttributes, FlagSet

# Pick the Gemini model via env, e.g. `export GEMINI_MODEL=gemini-2.5-pro` for
# the higher-tier model. Defaults to the fast model. The value is stamped as the
# model_version on every record, so scores remain reproducible per model.
MODEL_VERSION = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# One breaker shared across every image and both concurrent calls, so a Gemini
# incident trips the WHOLE pipeline once instead of per-image. Built lazily from
# config so importing this module stays side-effect-free (simulation path).
_BREAKER: "CircuitBreaker | None" = None
_RETRY_CFG: dict = {}


def _resilience():
    global _BREAKER, _RETRY_CFG
    if _BREAKER is None:
        rc = load_config().get("resilience", {})
        cb = rc.get("circuit_breaker", {})
        _RETRY_CFG = rc.get("retry", {})
        _BREAKER = CircuitBreaker(
            fail_max=cb.get("fail_max", 5),
            reset_timeout=cb.get("reset_timeout", 30.0),
            name="gemini",
        )
    return _BREAKER, _RETRY_CFG


def _resilient(fn, *args, **kwargs):
    """Run a Gemini SDK call behind retry + the shared circuit breaker."""
    breaker, rcfg = _resilience()
    return retry_call(
        fn, *args,
        max_attempts=rcfg.get("max_attempts", 3),
        base_delay=rcfg.get("base_delay", 0.5),
        max_delay=rcfg.get("max_delay", 8.0),
        jitter=rcfg.get("jitter", 0.25),
        breaker=breaker,
        **kwargs,
    )

_QUAL_KEYS = [
    "Expression", "FaceOrientationVisibility", "EyeContact", "GroomingNeatness",
    "Centering", "PoseCameraAngle", "BackgroundForegroundSeparation",
    "LightingOnFace", "Attire",
]
_FLAG_KEYS = [
    "CelebrityStock", "NSFW", "AdvertisementsOrText", "GroupImage",
    "Inappropriate", "RephotographOrScreenshot", "GenderMismatch",
]


def _client():
    try:
        from google import genai
    except ImportError as e:
        raise RuntimeError(
            "google-genai not installed. `pip install google-genai` and set "
            "GEMINI_API_KEY, or run with --mode simulation."
        ) from e
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set. Use --mode simulation to demo.")
    # A per-request timeout (ms) is ESSENTIAL: without it a stalled response hangs
    # on a blocking SSL read forever and the retry/breaker never fire (they catch
    # exceptions, not hangs). With it, a stall raises -> retry -> breaker.
    from google.genai import types
    timeout_ms = int(os.environ.get("GEMINI_TIMEOUT_MS", "60000"))
    return genai.Client(api_key=api_key,
                        http_options=types.HttpOptions(timeout=timeout_ms))


def _gen_config(schema: dict):
    from google.genai import types
    return types.GenerateContentConfig(
        temperature=0.0,                       # determinism
        response_mime_type="application/json",
        response_schema=schema,
    )


def _qual_schema() -> dict:
    props = {k: {"type": "number"} for k in _QUAL_KEYS}
    props.update({k: {"type": "integer"} for k in _FLAG_KEYS})
    return {"type": "object", "properties": props, "required": _QUAL_KEYS + _FLAG_KEYS}


def _aesthetic_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "photogenic": {"type": "number"},
            "reasoning": {"type": "string"},
        },
        "required": ["photogenic", "reasoning"],
    }


def _call_qualitative(client, img: Image.Image, gender: str):
    prompt = load_qualitative_prompt(gender)
    resp = _resilient(
        client.models.generate_content,
        model=MODEL_VERSION,
        contents=[prompt, img],
        config=_gen_config(_qual_schema()),
    )
    return json.loads(resp.text)


def _call_aesthetic(client, img: Image.Image):
    prompt = load_aesthetic_prompt()
    resp = _resilient(
        client.models.generate_content,
        model=MODEL_VERSION,
        contents=[prompt, img],
        config=_gen_config(_aesthetic_schema()),
    )
    return json.loads(resp.text)


class GeminiAdapter:
    """Fusion adapter (name + qualitative_raw + aesthetic_raw). Both raw calls
    are already wrapped in retry + the shared circuit breaker. Returns the raw
    JSON dicts so the fusion layer can validate/repair per parameter."""
    name = MODEL_VERSION

    def __init__(self):
        self._client = None

    def _c(self):
        if self._client is None:
            self._client = _client()
        return self._client

    def qualitative_raw(self, img: Image.Image, gender: str) -> dict:
        return _call_qualitative(self._c(), img, gender)

    def aesthetic_raw(self, img: Image.Image) -> dict:
        return _call_aesthetic(self._c(), img)


def score_with_gemini(img: Image.Image, gender: str):
    """Returns (QualitativeAttributes, FlagSet, aesthetic_float, reasoning_str)."""
    client = _client()
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_qual = ex.submit(_call_qualitative, client, img, gender)
        f_aes = ex.submit(_call_aesthetic, client, img)
        qual = f_qual.result()
        aes = f_aes.result()

    attrs = QualitativeAttributes(
        expression=float(qual["Expression"]),
        face_orientation_visibility=float(qual["FaceOrientationVisibility"]),
        eye_contact=float(qual["EyeContact"]),
        grooming_neatness=float(qual["GroomingNeatness"]),
        centering=float(qual["Centering"]),
        pose_camera_angle=float(qual["PoseCameraAngle"]),
        background_foreground_separation=float(qual["BackgroundForegroundSeparation"]),
        lighting_on_face=float(qual["LightingOnFace"]),
        attire=float(qual["Attire"]),
    )
    flags = FlagSet(
        nsfw=int(qual["NSFW"]),
        celebrity_stock=int(qual["CelebrityStock"]),
        advertisements_or_text=int(qual["AdvertisementsOrText"]),
        group_image=int(qual["GroupImage"]),
        inappropriate=int(qual["Inappropriate"]),
        rephotograph_or_screenshot=int(qual["RephotographOrScreenshot"]),
        gender_mismatch=int(qual["GenderMismatch"]),
    )
    return attrs, flags, float(aes["photogenic"]), str(aes.get("reasoning", ""))

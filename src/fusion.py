"""
fusion.py — Gemini-primary, open-source-fallback, PER-PARAMETER
================================================================
The scoring brain. Gemini scores ALL parameters. For any parameter Gemini
could not produce a valid value (missing key, null, NaN, out-of-range, or the
whole call failed), we fall back to the best OPEN-SOURCE VLM — Qwen2.5-VL by
default — and fill ONLY those parameters. Not all-or-nothing: a single bad key
is repaired without paying for a full re-score of the good ones.

Every value carries PROVENANCE: the record records, per parameter, whether it
came from `gemini`, the `fallback` model, or a neutral `default`. That makes
the system fully auditable — "why is this profile ranked here, and which model
decided each factor?" is answerable down to the attribute.

An adapter is anything with:  name: str,  qualitative_raw(img, gender) -> dict,
aesthetic_raw(img) -> dict.  Gemini, the open-source VLM, and the simulation all
implement it, so this layer is provider-agnostic.

Total failure (neither model returned anything usable) RAISES, so the pipeline's
existing safe-fallback path (status=FAILED, low neutral score, retry) handles it
— we never fabricate a confident score out of nothing.
"""

from __future__ import annotations

import logging
import math
import numbers
from typing import Callable, Optional

from PIL import Image

from .schemas import QualitativeAttributes, FlagSet

log = logging.getLogger(__name__)

# --- Canonical parameter sets (single source of truth for every adapter) ---- #
QUAL_GRADED_KEYS = [
    "Expression", "FaceOrientationVisibility", "EyeContact", "GroomingNeatness",
    "Centering", "PoseCameraAngle", "BackgroundForegroundSeparation",
    "LightingOnFace", "Attire",
]
QUAL_FLAG_KEYS = [
    "CelebrityStock", "NSFW", "AdvertisementsOrText", "GroupImage",
    "Inappropriate", "RephotographOrScreenshot", "GenderMismatch",
]
ALL_QUAL_KEYS = QUAL_GRADED_KEYS + QUAL_FLAG_KEYS

# CamelCase (model/JSON) -> dataclass field (snake_case).
_GRADED_FIELD = {
    "Expression": "expression",
    "FaceOrientationVisibility": "face_orientation_visibility",
    "EyeContact": "eye_contact",
    "GroomingNeatness": "grooming_neatness",
    "Centering": "centering",
    "PoseCameraAngle": "pose_camera_angle",
    "BackgroundForegroundSeparation": "background_foreground_separation",
    "LightingOnFace": "lighting_on_face",
    "Attire": "attire",
}
_FLAG_FIELD = {
    "CelebrityStock": "celebrity_stock",
    "NSFW": "nsfw",
    "AdvertisementsOrText": "advertisements_or_text",
    "GroupImage": "group_image",
    "Inappropriate": "inappropriate",
    "RephotographOrScreenshot": "rephotograph_or_screenshot",
    "GenderMismatch": "gender_mismatch",
}

# Neutral defaults if BOTH models fail a single parameter (partial degradation).
_GRADED_DEFAULT = 0.5      # neutral — neither rewards nor punishes the unknown
_FLAG_DEFAULT = 0          # absence of evidence is not a blocking flag
_AES_DEFAULT = 0.5


def _is_num(v) -> bool:
    # numbers.Real covers Python int/float AND numpy scalars (np.float32/64),
    # so values from any adapter validate correctly; bool is excluded on purpose.
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def validate_qualitative(d: dict) -> tuple[dict, list]:
    """Return (valid_subset, missing_keys) over the 16 qualitative keys.

    A graded key is valid if it's a finite number (clamped to [0,1]); a flag is
    valid only if it resolves cleanly to 0 or 1. Everything else is 'missing'
    and becomes a fallback target — this is the per-parameter granularity.
    """
    valid: dict = {}
    missing: list = []
    for k in QUAL_GRADED_KEYS:
        v = d.get(k)
        if _is_num(v):
            valid[k] = _clamp01(v)
        else:
            missing.append(k)
    for k in QUAL_FLAG_KEYS:
        v = d.get(k)
        if isinstance(v, bool):
            valid[k] = int(v)
        elif _is_num(v) and round(float(v)) in (0, 1):
            valid[k] = int(round(float(v)))
        else:
            missing.append(k)
    return valid, missing


def validate_aesthetic(d: dict) -> tuple[dict, list]:
    """Return (valid_subset, missing_keys) for the aesthetic call. Only
    'photogenic' is required; 'reasoning' is best-effort free text."""
    valid: dict = {}
    missing: list = []
    if _is_num(d.get("photogenic")):
        valid["photogenic"] = _clamp01(d["photogenic"])
    else:
        missing.append("photogenic")
    return valid, missing


def attrs_flags_from_raw(qual_raw: dict) -> tuple:
    """Parse a raw qualitative dict into (QualitativeAttributes, FlagSet),
    defaulting any invalid/missing key neutrally. Used by the comparison harness
    where the qualitative engine is held constant (Gemini) across providers."""
    valid, missing = validate_qualitative(qual_raw)
    merged = dict(valid)
    for k in missing:
        merged[k] = _GRADED_DEFAULT if k in QUAL_GRADED_KEYS else _FLAG_DEFAULT
    attrs = QualitativeAttributes(**{
        _GRADED_FIELD[k]: float(merged[k]) for k in QUAL_GRADED_KEYS})
    flags = FlagSet(**{
        _FLAG_FIELD[k]: int(merged[k]) for k in QUAL_FLAG_KEYS})
    return attrs, flags


def photogenic_from_raw(aes_raw: dict) -> float:
    """Extract a validated photogenic score (neutral default if unscorable)."""
    valid, _ = validate_aesthetic(aes_raw)
    return float(valid.get("photogenic", _AES_DEFAULT))


class ResilientScorer:
    """Primary VLM + per-parameter open-source fallback + provenance."""

    def __init__(self, primary, fallback=None):
        self.primary = primary
        self.fallback = fallback

    # ------------------------------------------------------------------ #
    def _fuse(self, primary_call, fallback_call, validate, defaults: dict):
        """Run primary, validate per key, repair missing keys from fallback,
        then default any stragglers. Returns (merged_values, source_map,
        primary_raw, fallback_raw)."""
        merged: dict = {}
        source: dict = {}

        try:
            p_raw = primary_call() or {}
        except Exception as e:  # noqa: BLE001 - degrade, don't crash
            log.warning("primary VLM (%s) failed: %s", self.primary.name, e)
            p_raw = {}
        p_valid, missing = validate(p_raw)
        for k, v in p_valid.items():
            merged[k], source[k] = v, self.primary.name

        f_raw: dict = {}
        if missing and self.fallback is not None:
            try:
                f_raw = fallback_call() or {}
            except Exception as e:  # noqa: BLE001
                log.warning("fallback VLM (%s) failed: %s", self.fallback.name, e)
                f_raw = {}
            f_valid, _ = validate(f_raw)
            for k in list(missing):
                if k in f_valid:
                    merged[k], source[k] = f_valid[k], self.fallback.name
                    missing.remove(k)

        for k in missing:                       # neither model produced it
            merged[k], source[k] = defaults[k], "default"
        return merged, source, p_raw, f_raw

    # ------------------------------------------------------------------ #
    def score(self, img: Image.Image, gender: str, img_hash: str):
        """Returns (attrs, flags, aesthetic, reasoning, provenance).

        provenance = {
          "sources": {param: 'gemini'|'<fallback>'|'default', ...},
          "fallback_used": bool,   # any param repaired by the fallback model
          "degraded": bool,        # any param fell through to a neutral default
        }
        Raises RuntimeError on TOTAL failure (nothing usable from either model).
        """
        qual_defaults = {**{k: _GRADED_DEFAULT for k in QUAL_GRADED_KEYS},
                         **{k: _FLAG_DEFAULT for k in QUAL_FLAG_KEYS}}
        q_merged, q_src, _, _ = self._fuse(
            lambda: self.primary.qualitative_raw(img, gender),
            (lambda: self.fallback.qualitative_raw(img, gender)) if self.fallback else None,
            validate_qualitative, qual_defaults,
        )
        a_merged, a_src, a_praw, a_fraw = self._fuse(
            lambda: self.primary.aesthetic_raw(img),
            (lambda: self.fallback.aesthetic_raw(img)) if self.fallback else None,
            validate_aesthetic, {"photogenic": _AES_DEFAULT},
        )

        sources = {**q_src, "Photogenic": a_src["photogenic"]}

        # TOTAL failure guard: if EVERY qualitative key AND photogenic came from
        # 'default', neither model said anything — let the pipeline FAIL safely
        # rather than emit a confident-looking fabricated score.
        all_defaulted = (
            all(sources[k] == "default" for k in ALL_QUAL_KEYS)
            and sources["Photogenic"] == "default"
        )
        if all_defaulted:
            raise RuntimeError("both primary and fallback VLM produced no usable output")

        attrs = QualitativeAttributes(**{
            _GRADED_FIELD[k]: float(q_merged[k]) for k in QUAL_GRADED_KEYS
        })
        flags = FlagSet(**{
            _FLAG_FIELD[k]: int(q_merged[k]) for k in QUAL_FLAG_KEYS
        })
        aesthetic = float(a_merged["photogenic"])
        reasoning = str(a_praw.get("reasoning") or a_fraw.get("reasoning") or "")

        provenance = {
            "sources": sources,
            "fallback_used": any(
                v not in (self.primary.name, "default") for v in sources.values()
            ),
            "degraded": any(v == "default" for v in sources.values()),
        }
        return attrs, flags, aesthetic, reasoning, provenance

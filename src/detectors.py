"""
detectors.py — STAGE 0 deep detectors (open-source ML gate)
===========================================================
The pluggable ML half of the quality gate. A detector takes a PIL image and
returns a dict of safety signals — e.g. {"nsfw": 0/1, "face_count": N}. Anything
flagged here is rejected BEFORE the paid Gemini call, so NSFW/group/no-face
images cost ~nothing instead of two vision calls.

THREE things make this "ultimate" rather than a TODO:

  1) A real production adapter — `TransformersNSFWDetector` wraps the
     open-source Falconsai/nsfw_image_detection classifier. One config flag
     turns it on; `pip install transformers torch` is the only new dependency.

  2) Composability — `CompositeDeepDetector` fans an image across several
     detectors (NSFW classifier + a face detector + ...) and merges their
     verdicts: flags are OR'd (any detector can reject), face_count takes the
     strictest (min) reading. Add a detector without touching the pipeline.

  3) A config-driven factory with GRACEFUL FALLBACK — `build_detector(cfg)`
     reads `prefilter.deep_detector` and, if a heavy dependency is missing,
     logs a warning and falls back to the safe NoOp instead of crashing the
     pipeline. Same resilience philosophy as the VLM layer.

A DELIBERATE FAIRNESS DECISION
------------------------------
We intentionally do NOT ship a skin-tone / skin-ratio heuristic for NSFW or
face presence. Classic YCbCr skin rules systematically under-detect darker
skin; on a matrimony product where colorism is the explicit risk, a tone-based
gate would re-introduce exactly the bias the system is built to avoid. Deep
decisions use a LEARNED classifier or defer to the VLM — judgment over
cleverness.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from PIL import Image

# Re-export the protocol + safe default so callers have one import home.
from .prefilter import DeepDetector, NoOpDeepDetector

log = logging.getLogger(__name__)

# Neutral baseline every merge starts from: not-NSFW, one face assumed present.
_DEFAULT_SIGNALS = {"nsfw": 0, "face_count": 1}


class FakeDeepDetector:
    """Deterministic detector for tests/wiring — returns a fixed verdict."""

    def __init__(self, **signals):
        self._signals = {**_DEFAULT_SIGNALS, **signals}

    def detect(self, img: Image.Image) -> dict:
        return dict(self._signals)


class TransformersNSFWDetector:
    """Real NSFW gate via the open-source Falconsai image classifier.

    Lazy, learned, and deterministic at inference. Construction loads the model
    once; `detect()` returns {"nsfw": 0/1, "nsfw_score": float}. The pipeline's
    blocking logic already treats nsfw=1 as a hard reject.
    """

    def __init__(
        self,
        model: str = "Falconsai/nsfw_image_detection",
        threshold: float = 0.7,
    ):
        self.model = model
        self.threshold = float(threshold)
        try:
            from transformers import pipeline  # heavy, optional dependency
        except ImportError as e:
            raise RuntimeError(
                "transformers not installed. `pip install transformers torch` "
                "to enable the NSFW gate, or set prefilter.deep_detector.type "
                "to 'none' to defer NSFW to the VLM."
            ) from e
        self._pipe = pipeline("image-classification", model=model)

    def detect(self, img: Image.Image) -> dict:
        preds = self._pipe(img)  # [{"label": "nsfw"/"normal", "score": ...}, ...]
        score = 0.0
        for p in preds:
            if str(p.get("label", "")).lower() == "nsfw":
                score = float(p.get("score", 0.0))
                break
        return {"nsfw": int(score >= self.threshold), "nsfw_score": round(score, 4)}


class CompositeDeepDetector:
    """Run several detectors and merge their verdicts conservatively.

    Merge rule: every binary flag is OR'd (ANY detector may reject), face_count
    takes the strictest reading (min). Auxiliary fields (scores) are carried
    through. A detector that errors is logged and SKIPPED — one bad model never
    takes the gate down.
    """

    def __init__(self, detectors: List[DeepDetector]):
        if not detectors:
            raise ValueError("CompositeDeepDetector needs at least one detector")
        self._detectors = list(detectors)

    def detect(self, img: Image.Image) -> dict:
        merged = dict(_DEFAULT_SIGNALS)
        for d in self._detectors:
            try:
                out = d.detect(img)
            except Exception as e:  # noqa: BLE001 - isolate a failing detector
                log.warning("detector %s failed, skipping: %s",
                            type(d).__name__, e)
                continue
            for k, v in out.items():
                if k == "face_count":
                    merged[k] = min(merged.get(k, v), v)
                elif isinstance(v, int) and v in (0, 1) and k in merged:
                    merged[k] = max(merged.get(k, 0), v)  # OR binary flags
                else:
                    merged[k] = v  # scores / new keys pass through
        return merged


def build_detector(cfg: dict) -> Optional[DeepDetector]:
    """Factory: build the Stage-0 deep detector from config, with fallback.

    config (under prefilter.deep_detector):
        type: none | transformers | composite
        nsfw_threshold: 0.7
        composite: [ {type: transformers, ...}, ... ]

    On a missing heavy dependency we log and fall back to NoOp rather than
    crash — the basic numeric gate (blur/exposure/size) always still runs.
    """
    dd = (cfg.get("prefilter", {}) or {}).get("deep_detector", {}) or {}
    kind = str(dd.get("type", "none")).lower()
    try:
        return _build_one(kind, dd)
    except Exception as e:  # noqa: BLE001 - never let gate config crash startup
        log.warning("deep_detector '%s' unavailable (%s); falling back to NoOp. "
                    "NSFW/face are deferred to the VLM stage.", kind, e)
        return NoOpDeepDetector()


def _build_one(kind: str, dd: dict) -> Optional[DeepDetector]:
    if kind in ("none", ""):
        return None  # pipeline uses NoOp; NSFW/face deferred to the VLM
    if kind == "noop":
        return NoOpDeepDetector()
    if kind == "transformers":
        return TransformersNSFWDetector(
            model=dd.get("model", "Falconsai/nsfw_image_detection"),
            threshold=dd.get("nsfw_threshold", 0.7),
        )
    if kind == "composite":
        subs = [_build_one(str(s.get("type")).lower(), s)
                for s in dd.get("composite", [])]
        subs = [s for s in subs if s is not None]
        if not subs:
            return None
        return CompositeDeepDetector(subs)
    raise ValueError(f"unknown deep_detector type: {kind!r}")

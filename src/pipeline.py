"""
pipeline.py  —  orchestration
=============================
Wires the stages together with the cross-cutting guarantees:

  Stage 0  prefilter (OSS gate)         -> PREFILTERED (no paid call) on junk
  Stage 1+2 VLM brain (gemini | sim)    -> attributes, flags, aesthetic
  Stage 3  blend                        -> SCORED / BLOCKED
  cache    image-hash + version key     -> idempotency, cost, re-score trigger
  fallback technical failure            -> FAILED w/ low neutral score + retry

Plus profile_feed_score(): the feed consumes the FIRST VALID image's score
(an improvement on "first image even if blocked" — blocked lead photos are
skipped rather than tanking the whole profile).
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from . import config_loader
from .cache import ScoreCache, image_hash, cache_key
from .prefilter import run_prefilter, DeepDetector
from .detectors import build_detector
from .blender import blend
from .fairness import AuditLog
from .schemas import (
    ScoreRecord, ScoreStatus, Versions, QualitativeAttributes, FlagSet,
)


class RatingPipeline:
    def __init__(
        self,
        mode: str = "simulation",
        cache_path: Optional[Path] = None,
        audit_path: Optional[Path] = None,
        deep_detector: Optional[DeepDetector] = None,
    ):
        self.mode = mode
        self.cfg = config_loader.load_config()
        self.cache = ScoreCache(cache_path)
        self.audit = AuditLog(audit_path) if audit_path else None
        # Explicit detector wins; otherwise build from config (falls back to
        # NoOp if a heavy ML dependency is absent — the gate never crashes).
        self.deep_detector = deep_detector or build_detector(self.cfg)

        self._scorer = self._build_scorer(mode)
        # _score_fn returns (attrs, flags, aesthetic, reasoning, provenance).
        self._score_fn: Callable = lambda img, g, h: self._scorer.score(img, g, h)

        self.versions = Versions(
            pipeline_version=config_loader.PIPELINE_VERSION,
            model_version=self._model_version,
            prompt_version=config_loader.prompt_version(),
            weight_version=self.cfg["weight_version"],
        )

    # ------------------------------------------------------------------ #
    def _build_scorer(self, mode: str):
        """Assemble the Gemini-primary + open-source-fallback fusion scorer.

        The model_version fingerprint records BOTH models when a fallback is
        wired, so a score's provenance is reproducible from its version stamp.
        """
        from .fusion import ResilientScorer

        fb_cfg = (self.cfg.get("fallback", {}) or {}).get("vlm", {}) or {}
        res_cfg = self.cfg.get("resilience", {}) or {}

        def _make_oss():
            if not fb_cfg.get("enabled"):
                return None
            try:
                from .vlm_opensource import OpenSourceAdapter
                return OpenSourceAdapter(
                    model=fb_cfg.get("model", "Qwen/Qwen2.5-VL-72B-Instruct"),
                    base_url_env=fb_cfg.get("base_url_env", "OSS_VLM_BASE_URL"),
                    api_key_env=fb_cfg.get("api_key_env", "OSS_VLM_API_KEY"),
                    retry_cfg=res_cfg.get("retry", {}),
                )
            except Exception:  # adapter import/config issues never block startup
                return None

        if mode == "gemini":
            from .vlm_gemini import GeminiAdapter, MODEL_VERSION
            oss = _make_oss()
            self._model_version = (
                f"{MODEL_VERSION}+{oss.name}" if oss else MODEL_VERSION
            )
            return ResilientScorer(GeminiAdapter(), fallback=oss)

        if mode == "simulation":
            from .vlm_simulation import SimulationAdapter, image_pixel_hash, MODEL_VERSION
            self._model_version = MODEL_VERSION
            return ResilientScorer(SimulationAdapter(img_hash_of=image_pixel_hash))

        raise ValueError("mode must be 'gemini' or 'simulation'")

    # ------------------------------------------------------------------ #
    def score_image(
        self,
        image_bytes: bytes,
        image_id: str,
        gender_feed: str,
        segment: Optional[str] = None,
    ) -> ScoreRecord:
        t0 = time.time()
        h = image_hash(image_bytes)
        key = cache_key(h, self.versions)

        cached = self.cache.get(key)
        if cached:
            cached["_cache_hit"] = True
            rec = _record_from_dict(cached)
            # A cache hit is still a real scoring EVENT: it must hit the audit
            # log and the fairness tally. Skipping it here silently blinds bias
            # monitoring for every repeat image — i.e. most of production.
            if self.audit:
                self.audit.write(rec.as_dict(), segment)
            return rec

        try:
            # ---- Stage 0: OSS quality gate ---------------------------- #
            pf, img = run_prefilter(image_bytes, self.cfg, self.deep_detector)
            if not pf.passed:
                rec = ScoreRecord(
                    image_id=image_id, image_hash=h, gender_feed=gender_feed,
                    status=ScoreStatus.PREFILTERED,
                    final_score=self.cfg["fallback"]["prefiltered_score"],
                    prefilter=pf, versions=self.versions,
                    latency_ms=round((time.time() - t0) * 1000, 1),
                    reasoning="prefilter_reject: " + ", ".join(pf.reasons),
                )
                return self._finalize(key, rec, segment)

            # ---- Stage 1+2: VLM brain (Gemini primary + OSS fallback) --- #
            attrs, flags, aesthetic, reasoning, provenance = self._score_fn(
                img, gender_feed, h
            )

            # ---- Stage 3: blend --------------------------------------- #
            status, q, final = blend(aesthetic, attrs, flags, gender_feed, self.cfg)
            rec = ScoreRecord(
                image_id=image_id, image_hash=h, gender_feed=gender_feed,
                status=status, aesthetic_score=round(aesthetic, 4),
                qualitative_score=round(q, 4), final_score=final,
                attributes=attrs, flags=flags, prefilter=pf,
                versions=self.versions, reasoning=reasoning,
                provenance=provenance,
                latency_ms=round((time.time() - t0) * 1000, 1),
            )
            return self._finalize(key, rec, segment)

        except Exception as e:  # ---- fallback (fixes bug #5) ------------ #
            rec = ScoreRecord(
                image_id=image_id, image_hash=h, gender_feed=gender_feed,
                status=ScoreStatus.FAILED,
                final_score=self.cfg["fallback"]["failed_score"],
                versions=self.versions,
                latency_ms=round((time.time() - t0) * 1000, 1),
                error=f"{type(e).__name__}: {e}",
                reasoning="technical_failure -> low neutral fallback + retry",
            )
            # Failures are NOT cached (so a retry can succeed later).
            if self.audit:
                self.audit.write(rec.as_dict(), segment)
            return rec

    # ------------------------------------------------------------------ #
    def _finalize(self, key: str, rec: ScoreRecord, segment) -> ScoreRecord:
        self.cache.put(key, rec.as_dict())
        if self.audit:
            self.audit.write(rec.as_dict(), segment)
        return rec

    # ------------------------------------------------------------------ #
    def profile_feed_score(self, records: list[ScoreRecord]) -> float:
        """Feed weight for a profile = first VALID image's score."""
        for r in records:
            if r.status == ScoreStatus.SCORED:
                return r.final_score
        # No valid image -> lowest priority but not a hard zero.
        return self.cfg["fallback"]["failed_score"]

    # ------------------------------------------------------------------ #
    def score_profile(
        self,
        images: list[tuple],          # [(image_bytes, image_id), ...] in upload order
        gender_feed: str,
        segment: Optional[str] = None,
    ) -> tuple[float, list[ScoreRecord]]:
        """PRD scope E — score every uploaded image INDEPENDENTLY and return the
        profile's feed weight (the first VALID image's score) plus all records.

        The PRD says "use the first image"; we use the first VALID image so a
        blocked/failed lead photo doesn't tank an otherwise good profile.
        """
        records = [
            self.score_image(b, image_id, gender_feed, segment=segment)
            for (b, image_id) in images
        ]
        return self.profile_feed_score(records), records

    # ------------------------------------------------------------------ #
    def needs_rescore(self, record_versions: dict) -> bool:
        """PRD scope D — is a stored record stale vs the current config?

        True iff ANY version (pipeline/model/prompt/weight) differs from what
        this pipeline would stamp now. Bumping weights.yaml or editing a prompt
        flips this for the affected records, which is exactly the re-score
        trigger — no separate bookkeeping required.
        """
        return record_versions != self.versions.as_dict()


def _record_from_dict(d: dict) -> ScoreRecord:
    attrs = QualitativeAttributes(**d["attributes"]) if d.get("attributes") else QualitativeAttributes()
    flags = FlagSet(**d["flags"]) if d.get("flags") else FlagSet()
    versions = Versions(**d["versions"]) if d.get("versions") else None
    rec = ScoreRecord(
        image_id=d["image_id"], image_hash=d["image_hash"],
        gender_feed=d["gender_feed"], status=ScoreStatus(d["status"]),
        aesthetic_score=d.get("aesthetic_score", 0.0),
        qualitative_score=d.get("qualitative_score", 0.0),
        final_score=d.get("final_score", 0.0),
        attributes=attrs, flags=flags, versions=versions,
        scored_at=d.get("scored_at", 0.0), latency_ms=d.get("latency_ms", 0.0),
        error=d.get("error"), reasoning=d.get("reasoning"),
        provenance=d.get("provenance"),
    )
    return rec

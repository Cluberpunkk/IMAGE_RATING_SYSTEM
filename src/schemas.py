"""
schemas.py
==========
Typed records for the image-rating pipeline.

Design note (fixes PRD gap #7 — missing versioning):
Every score record carries the *exact* versions that produced it
(model, prompt, weights, pipeline). This is the backbone of the
re-scoring workflow: an image needs re-scoring iff any of these
versions has changed. It also makes every score reproducible and
auditable ("why is this profile here?").
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time


class ScoreStatus(str, Enum):
    SCORED = "SCORED"          # full pipeline succeeded
    BLOCKED = "BLOCKED"        # a blocking flag fired -> excluded from feed
    FAILED = "FAILED"          # technical failure -> safe fallback, retry later
    PREFILTERED = "PREFILTERED"  # rejected by Stage-0 quality gate (no VLM call)


@dataclass
class QualitativeAttributes:
    """The 9 presentation attributes the VLM actually outputs (all 0..1).

    NOTE (fixes PRD bug #3): 'Attractiveness' is NOT in this list.
    Attractiveness is the *aesthetic* signal and is a separate top-level
    score. In the original PRD the weight table referenced an
    'Attractiveness' qualitative parameter that no model output produced.
    """
    expression: float = 0.0
    face_orientation_visibility: float = 0.0
    eye_contact: float = 0.0
    grooming_neatness: float = 0.0
    centering: float = 0.0
    pose_camera_angle: float = 0.0
    background_foreground_separation: float = 0.0
    lighting_on_face: float = 0.0
    attire: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlagSet:
    """Binary safety / validity flags (1 = true)."""
    nsfw: int = 0
    celebrity_stock: int = 0
    advertisements_or_text: int = 0
    group_image: int = 0
    inappropriate: int = 0
    rephotograph_or_screenshot: int = 0
    gender_mismatch: int = 0  # subject gender != feed gender

    def as_dict(self) -> dict:
        return asdict(self)

    def any_set(self, keys) -> bool:
        d = self.as_dict()
        return any(int(d.get(k, 0)) == 1 for k in keys)


@dataclass
class PreFilterResult:
    """Output of the open-source Stage-0 technical-quality gate."""
    passed: bool
    blur_score: float          # Laplacian variance (higher = sharper)
    brightness: float          # mean luma 0..255
    megapixels: float
    aspect_ratio: float
    reasons: list = field(default_factory=list)  # why it failed, if it did

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Versions:
    """The version fingerprint stamped onto every score record."""
    pipeline_version: str
    model_version: str         # e.g. "gemini-2.5-flash" or "sim-v1"
    prompt_version: str        # hash/tag of the prompt files used
    weight_version: str        # tag from weights.yaml

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScoreRecord:
    """The complete, storable result for a single image."""
    image_id: str
    image_hash: str
    gender_feed: str           # "female" | "male"
    status: ScoreStatus

    aesthetic_score: float = 0.0       # reframed "photogenic" score (replaces Grok)
    qualitative_score: float = 0.0     # weighted avg of the 9 attributes
    final_score: float = 0.0           # blended, what ranking consumes

    attributes: QualitativeAttributes = field(default_factory=QualitativeAttributes)
    flags: FlagSet = field(default_factory=FlagSet)
    prefilter: Optional[PreFilterResult] = None

    versions: Optional[Versions] = None
    scored_at: float = field(default_factory=lambda: time.time())
    latency_ms: float = 0.0
    error: Optional[str] = None
    reasoning: Optional[str] = None    # short model rationale (audit/debug)

    # Per-parameter provenance (fusion layer): which model produced each value.
    # {"sources": {param: 'gemini'|'<oss-model>'|'default'}, "fallback_used":
    # bool, "degraded": bool}. This is what makes the score auditable per factor.
    provenance: Optional[dict] = None

    def as_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

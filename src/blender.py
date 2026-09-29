"""
blender.py  —  STAGE 3: scoring & blend
=======================================
Turns raw model output into the final feed score, correctly.

Pipeline of math:
  1. Blocking flags  -> if any fire, status=BLOCKED, final=0 (out of feed).
  2. Qualitative score Q = weighted avg of the 9 attributes (weights from config,
     already normalised to sum 1.0).
  3. Final = w_aes * aesthetic + w_qual * Q     (per-gender blend weights)
  4. Soft penalties subtracted (e.g. ad/text overlay) and clamped to [0,1].

Fixes baked in:
  * #3  Q uses only real model attributes; 'Attractiveness' is the separate
        `aesthetic` term, not a phantom qualitative key.
  * blocking set is explicit & configurable (no ambiguity about what blocks).
"""

from __future__ import annotations

from .schemas import QualitativeAttributes, FlagSet, ScoreStatus


def qualitative_score(attrs: QualitativeAttributes, cfg: dict) -> float:
    w = cfg["qualitative_weights"]
    d = attrs.as_dict()
    return float(sum(w[k] * d[k] for k in w))


def is_blocked(flags: FlagSet, cfg: dict) -> bool:
    return flags.any_set(cfg["blocking_flags"])


def soft_penalty(flags: FlagSet, cfg: dict) -> float:
    fd = flags.as_dict()
    pen = 0.0
    for key, amount in cfg.get("soft_penalties", {}).items():
        if int(fd.get(key, 0)) == 1:
            pen += float(amount)
    return pen


def blend(
    aesthetic: float,
    attrs: QualitativeAttributes,
    flags: FlagSet,
    gender: str,
    cfg: dict,
) -> tuple[ScoreStatus, float, float]:
    """Returns (status, qualitative_score, final_score)."""
    q = qualitative_score(attrs, cfg)

    if is_blocked(flags, cfg):
        return ScoreStatus.BLOCKED, q, 0.0

    bw = cfg["blend_weights"][gender]
    final = bw["aesthetic"] * aesthetic + bw["qualitative"] * q
    final -= soft_penalty(flags, cfg)
    final = max(0.0, min(1.0, final))
    return ScoreStatus.SCORED, q, round(final, 4)

"""
fairness.py
===========
Two governance pieces the original PRD lacked:

1) Audit log  — append-only JSONL of every score (versions + raw factors). This
   is what lets you answer "why is this profile ranked here?" and run the
   manual flag-sanity audit the PRD asks for.

2) Cohort calibration — the PRD's success metric "Score Distribution Health
   (not clustered)" needs an actual mechanism. We percentile-rank final scores
   within a cohort so the feed weight always spreads across 0..1 instead of
   bunching. Calibration is applied at ranking time, leaving raw scores intact
   for audit.

Skin-tone / region drift monitoring hook is included: pass an optional
`segment` label and the logger tallies mean score per segment so you can watch
for systematic bias (colorism is a specific, real risk for this product).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seg_sum: dict = defaultdict(float)
        self._seg_n: dict = defaultdict(int)

    def write(self, record: dict, segment: Optional[str] = None) -> None:
        line = dict(record)
        if segment:
            line["_segment"] = segment
            # Only SCORED images inform the bias signal. Prefilter/technical
            # rejects (final=0) are not a fairness signal and would skew it.
            if record.get("status") == "SCORED":
                self._seg_sum[segment] += record.get("final_score", 0.0)
                self._seg_n[segment] += 1
        with open(self.path, "a") as f:
            f.write(json.dumps(line) + "\n")

    def segment_report(self) -> dict:
        """Mean final score per segment — watch for bias drift."""
        return {
            seg: round(self._seg_sum[seg] / self._seg_n[seg], 4)
            for seg in self._seg_sum if self._seg_n[seg]
        }


def distribution_health(
    scores: list[float],
    healthy_band: tuple = (0.2, 0.8),
) -> dict:
    """PRD success metric 'Score Distribution Health (not clustered)', made
    measurable. A healthy feed signal SPREADS across 0..1 rather than bunching.

    Returns:
      n, mean, std, spread (max-min), coverage (fraction of [0,1] deciles hit),
      clustered (bool: std too low / coverage too thin to differentiate the feed).
    """
    if not scores:
        return {"n": 0, "mean": 0.0, "std": 0.0, "spread": 0.0,
                "coverage": 0.0, "clustered": True}
    arr = np.asarray(scores, dtype=np.float64)
    deciles = np.unique(np.clip((arr * 10).astype(int), 0, 9))
    coverage = len(deciles) / 10.0
    std = float(arr.std())
    # "Clustered" = the distribution can't differentiate the feed: little spread
    # AND few deciles touched. Tunable; this is the metric the PRD only named.
    clustered = bool(std < 0.10 and coverage < 0.30)
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 4),
        "std": round(std, 4),
        "spread": round(float(arr.max() - arr.min()), 4),
        "coverage": round(coverage, 4),
        "clustered": clustered,
    }


def percentile_calibrate(scores: list[float]) -> list[float]:
    """Map raw final scores to percentile ranks in [0,1] within the cohort."""
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    order = arr.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(arr)
    ranks[order] = np.arange(n)
    return list(np.round(ranks / max(n - 1, 1), 4))

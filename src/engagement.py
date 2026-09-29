"""
engagement.py — Layer 2: rank the feed for UNLOCK SPEND (the business goal)
===========================================================================
The cold-start VLM score answers "is this a good photo?". This layer answers the
question that actually drives revenue: "will a viewer SPEND AN UNLOCK on this
profile?" — and ranks each feed to maximise that, so unlocks deplete fast and
viewers re-buy (repeat rate).

It is a learned P(unlock) model on YOUR engagement log, NOT a hand-tuned "hotness"
score. That is the only honest way to optimise for the business outcome — and it
keeps us out of the brand/legal trap of rating desirability directly.

Three pieces:
  1. EngagementModel — logistic P(unlock | features), trained on impressions
     labelled unlocked/not. Pure numpy (no sklearn dependency), deterministic.
  2. rank_feed()    — orders a viewer's feed by predicted P(unlock), with a
     cold-start EXPLORATION bonus so brand-new profiles (no history) still get
     shown and gather signal instead of being starved by the model.
  3. fairness_by_segment() — watches predicted P(unlock) across tone/region
     segments. If the model is really just ranking by skin tone, this is where
     it shows up — the colorism guardrail, made measurable.

Cold start: a profile with no engagement history is ranked on its VLM appeal
score (the Layer-1 prior) plus the exploration bonus, until it earns real signal.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

# Features that plausibly drive an unlock. vlm_appeal is the Layer-1 score; the
# rest are the qualitative attributes (a warm expression / eye contact / good
# lighting tend to convert). Extend with profile/viewer metadata when available.
FEATURE_KEYS = [
    "vlm_appeal", "expression", "eye_contact", "lighting_on_face",
    "grooming_neatness", "face_orientation_visibility", "centering",
    "pose_camera_angle", "background_foreground_separation", "attire",
]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def featurize(rec: dict) -> list:
    """Build a feature vector from a scored record:
    {'vlm_appeal': float, 'attributes': {attr: float, ...}}."""
    a = rec.get("attributes", {}) or {}
    return [float(rec.get("vlm_appeal", 0.5))] + [
        float(a.get(k, 0.5)) for k in FEATURE_KEYS[1:]
    ]


class _Logistic:
    """Standardised logistic regression via gradient descent (deterministic)."""

    def __init__(self):
        self.w = None
        self.b = 0.0
        self.mu = None
        self.sd = None

    def fit(self, X, y, weights=None, epochs: int = 600, lr: float = 0.2, l2: float = 1e-3):
        """Weighted logistic regression. `y` may be hard 0/1 OR a soft rate in
        [0,1] (e.g. unlock_rate); `weights` (e.g. impressions) scale each row's
        gradient — so 200 aggregated photos train directly, no 50M-row expansion."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        ws = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
        ws = ws / (ws.mean() + 1e-9)
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(epochs):
            p = _sigmoid(Xs @ self.w + self.b)
            g = (p - y) * ws
            self.w -= lr * (Xs.T @ g / n + l2 * self.w)
            self.b -= lr * g.mean()
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.mu) / self.sd
        return _sigmoid(Xs @ self.w + self.b)


class EngagementModel:
    """P(unlock | features). Train on labelled impressions, then score profiles."""

    def __init__(self):
        self._m = _Logistic()
        self.trained = False

    def train(self, examples: list):
        """examples: [{'features': [...], 'unlocked': 0|1}, ...]."""
        X = [e["features"] for e in examples]
        y = [e["unlocked"] for e in examples]
        self._m.fit(X, y)
        self.trained = True
        return self

    def p_unlock(self, features: list) -> float:
        if not self.trained:
            # Cold model: fall back to the VLM appeal score as the prior.
            return float(features[0])
        return float(self._m.predict_proba([features])[0])

    def feature_weights(self) -> dict:
        """Standardised weights — which features the model leans on (audit)."""
        if not self.trained:
            return {}
        return {k: round(float(w), 4) for k, w in zip(FEATURE_KEYS, self._m.w)}

    def save(self, path) -> None:
        """Persist the trained model to JSON (portable, no pickle)."""
        import json
        from pathlib import Path
        m = self._m
        Path(path).write_text(json.dumps({
            "feature_keys": FEATURE_KEYS,
            "w": list(map(float, m.w)), "b": float(m.b),
            "mu": list(map(float, m.mu)), "sd": list(map(float, m.sd)),
            "trained": self.trained,
        }, indent=2))

    @classmethod
    def load(cls, path):
        import json
        d = json.loads(open(path).read())
        obj = cls()
        obj._m.w = np.asarray(d["w"]); obj._m.b = d["b"]
        obj._m.mu = np.asarray(d["mu"]); obj._m.sd = np.asarray(d["sd"])
        obj.trained = d.get("trained", True)
        return obj


def shrink_to_prior(observed_unlocks: float, impressions: float,
                    prior_rate: float, prior_strength: float = 20000.0) -> float:
    """Empirical-Bayes estimate of a profile's true unlock rate.

    A brand-new profile (impressions≈0) is pulled to the VLM-implied prior; a
    profile with lots of impressions trusts its observed rate. `prior_strength`
    is the pseudo-impression count of the prior (≈ how much data it takes to
    override the photo score). This is the cold→warm transition done right —
    no cliff between "no data" and "some data".
    """
    obs_rate = (observed_unlocks / impressions) if impressions else 0.0
    w = impressions / (impressions + prior_strength)
    return w * obs_rate + (1.0 - w) * prior_rate


def rank_feed(profiles: list, model: EngagementModel,
              explore: float = 0.10) -> list:
    """Rank a viewer's feed by predicted P(unlock), with a cold-start bonus.

    profiles: [{'profile_id', 'features': [...], 'impressions': int,
                'segment'?: str}, ...]
    The UCB-style bonus ~ 1/sqrt(impressions+1) keeps fresh profiles visible so
    the model keeps learning instead of locking in early winners.
    """
    out = []
    for p in profiles:
        base = model.p_unlock(p["features"])
        bonus = explore * float(np.sqrt(1.0 / (p.get("impressions", 0) + 1)))
        out.append({**p, "p_unlock": round(base, 4),
                    "rank_score": round(base + bonus, 4)})
    out.sort(key=lambda x: -x["rank_score"])
    return out


def fairness_by_segment(ranked: list, segment_key: str = "segment") -> dict:
    """Mean predicted P(unlock) per segment — the colorism guardrail. A large
    gap between tone/region segments means the model may be ranking on a
    sensitive proxy; investigate before it ships."""
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in ranked:
        seg = p.get(segment_key)
        if seg is not None:
            buckets[seg].append(p["p_unlock"])
    report = {k: round(sum(v) / len(v), 4) for k, v in buckets.items() if v}
    if len(report) >= 2:
        report["_max_gap"] = round(max(report.values()) - min(report.values()), 4)
    return report


# Map the detailed-benchmark CSV columns -> engagement feature keys, so we can
# reuse the VLM scores we already computed as features (no re-scoring).
_DETAIL_COL = {
    "vlm_appeal": "aesthetic:Gemini fully", "expression": "q:Expression",
    "eye_contact": "q:EyeContact", "lighting_on_face": "q:LightingOnFace",
    "grooming_neatness": "q:GroomingNeatness",
    "face_orientation_visibility": "q:FaceOrientationVisibility",
    "centering": "q:Centering", "pose_camera_angle": "q:PoseCameraAngle",
    "background_foreground_separation": "q:BackgroundForegroundSeparation",
    "attire": "q:Attire",
}


def load_features_from_detail(path) -> dict:
    """Read comparison_detailed.csv -> {photo_id: feature_list}. Reuses the VLM
    scores already produced by the benchmark — no re-scoring needed."""
    import csv
    out = {}
    for r in csv.DictReader(open(path)):
        try:
            out[r["image"]] = [float(r[_DETAIL_COL[k]]) for k in FEATURE_KEYS]
        except (KeyError, ValueError, TypeError):
            continue
    return out


def build_examples(engagement_path, features: dict) -> list:
    """Join engagement rows to VLM features by photo id. Accepts either:
      * per-impression rows:  photo_id, unlocked(0/1) [, segment]
      * aggregated rows:      photo_id, impressions, unlocks [, segment]
    Aggregated rows are expanded to `unlocks` positives + (impressions-unlocks)
    negatives. Returns [{features, unlocked, segment, photo_id}, ...]."""
    import csv
    from pathlib import Path
    examples = []
    for r in csv.DictReader(open(engagement_path)):
        pid = (r.get("photo_id") or r.get("image") or r.get("profile_id") or "").strip()
        if pid not in features and (r.get("photo_url") or r.get("url")):
            pid = Path((r.get("photo_url") or r.get("url")).split("?")[0]).stem  # url -> stem
        feats = features.get(pid)
        if feats is None:
            continue
        seg = r.get("segment") or r.get("profile_segment")
        if "unlocked" in r and r["unlocked"] != "":
            examples.append({"features": feats, "unlocked": int(float(r["unlocked"])),
                             "segment": seg, "photo_id": pid, "vlm_appeal": feats[0]})
        else:
            imp = int(float(r.get("impressions") or r.get("shown_count") or 0))
            unl = int(float(r.get("unlocks") or r.get("unlock_count") or 0))
            for _ in range(unl):
                examples.append({"features": feats, "unlocked": 1, "segment": seg,
                                 "photo_id": pid, "vlm_appeal": feats[0]})
            for _ in range(max(0, imp - unl)):
                examples.append({"features": feats, "unlocked": 0, "segment": seg,
                                 "photo_id": pid, "vlm_appeal": feats[0]})
    return examples


def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs); i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a: list, b: list) -> float:
    """Rank correlation. +1 = scoring perfectly orders by the metric, 0 = none,
    -1 = inverted. The honest test of 'does our image rating predict unlocks'."""
    if len(a) < 3:
        return float("nan")
    ra, rb = np.asarray(_rank(a)), np.asarray(_rank(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def auc(scores: list, labels: list) -> float:
    """Mann-Whitney AUC of P(unlock) vs actual unlocked — model quality."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    order = np.argsort(np.asarray(scores))
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    rsum = sum(r for r, l in zip(ranks, labels) if l == 1)
    return float((rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

# The Beast: ranking the feed for unlock spend (and repeat)

**Goal (yours):** the feed surfaces the profiles a viewer is most likely to spend
an **unlock** on → unlocks deplete fast → viewers re-buy → **repeat rate** climbs.
Hinge-style engagement ranking, not a "nice photo" score.

This is a **learned P(unlock) model on your data**, not a hand-tuned desirability
score — the only honest way to hit the business metric, and it keeps us out of
the "rate hotness" brand/legal trap.

---

## The 3 layers

```
LAYER 1  Cold start — VLM appeal score (built: vlm_gemini + fusion)
         A brand-new profile with no history is ranked on its photo score.

LAYER 2  Engagement model — P(unlock | features)   ← src/engagement.py (built)
         Learns from your impression+unlock log. The actual beast.

LAYER 3  Personalised ranking — rank_feed()        ← src/engagement.py (built)
         Each viewer's feed ordered by predicted P(unlock) FOR THEM, plus a
         cold-start exploration bonus so new profiles aren't starved, plus a
         fairness guard on tone/region segments.
```

Layer 1 is live. Layers 2–3 are **built and tested on synthetic data** — they go
live the moment you give them real engagement logs. That's the whole gap.

---

## What you must log (the only blocker)

### A. Impression log — one row per profile shown in a feed
This is the training data. Target = `unlocked`.

| field | example | why |
|---|---|---|
| `impression_id` | imp_8831 | key |
| `viewer_id` | u_55 | who saw it (personalisation later) |
| `viewer_gender` | male | which feed |
| `profile_id` | p_902 | who was shown |
| `profile_gender` | female | |
| `profile_segment` | tone_deep / region_telugu | **fairness monitoring** |
| `photo_id` | enhanced_144414 | the enhanced image scored |
| `vlm_appeal` | 0.71 | Layer-1 score (we compute) |
| `attributes` | {expression:0.8,...} | the 16 VLM params (we compute) |
| `shown_at` | 2026-06-29T10:00 | recency/decay |
| **`unlocked`** | **1** | **THE LABEL — did they spend an unlock** |
| `connected` / `replied` | 0/1 | optional downstream signals |

### B. Purchase log — for the north-star (repeat)
| field | example |
|---|---|
| `viewer_id` | u_55 |
| `unlock_purchase_at` | 2026-06-29T11:00 |
| `unlocks_bought` | 10 |

`unlocked` trains the model; the purchase log measures whether ranking-for-unlocks
actually lifts **repeat rate** (the real KPI) — validate with an A/B holdout.

> **Minimum to start:** even just `(profile_id, photo_id, shown_count, unlock_count)`
> per profile lets us train a first P(unlock) model. The richer log above unlocks
> personalisation + fairness.

---

## How it works (code that exists today)

```python
from src.engagement import EngagementModel, featurize, rank_feed, fairness_by_segment

# 1) Train on your impression log (each row -> features + unlocked 0/1)
examples = [{"features": featurize(row), "unlocked": row["unlocked"]} for row in log]
model = EngagementModel().train(examples)

# 2) Rank a viewer's candidate feed by P(unlock), with cold-start exploration
feed = rank_feed(candidate_profiles, model, explore=0.10)   # sorted, best first

# 3) Watch the colorism guardrail before shipping
print(fairness_by_segment(feed))   # mean P(unlock) per tone/region segment + gap
print(model.feature_weights())     # which features the model leans on (audit)
```

---

## Guardrails (your marriage-product brand)
- **Fairness monitor on by default.** `fairness_by_segment` surfaces if the model
  is really ranking by skin tone/region. A large `_max_gap` = stop and investigate.
- **Exploration** so the model doesn't lock in early winners and starve new users.
- **Audit:** `feature_weights()` shows exactly what drives the score — defensible
  to a director or regulator.
- We predict **behaviour (unlocks)**, never desirability directly.

---

## The path
1. **Now:** ship Layer-1 VLM appeal as the cold-start score (done, benchmarked).
2. **Instrument:** log Table A + B (above). This is the real work — it's data, not code.
3. **Train Layer 2** on 2–4 weeks of impressions → P(unlock) model.
4. **A/B holdout:** ranked-by-unlock feed vs control; measure **repeat rate**, not
   just clicks. Ship if it wins.
5. **Iterate:** add viewer personalisation, recency decay, per-segment calibration.

# Image Rating System — Key Findings (team one-pager)

*Matrimony feed: can a photo score predict who gets unlocked? Tested on real data.*

---

## 1. Three engines compared (40 photos, each with reasoning)
| Engine | Avg score | Won (best per photo) |
|---|---|---|
| Gemini fully | 0.67 | 0 — most conservative, sharpest at naming defects |
| Gemini + Grok | 0.77 | **23** |
| Gemini + Llama-4 (open-source) | 0.78 | **17** |

**Takeaway:** engines *agree on what's wrong* in a photo; they differ on *how harshly* they score. Gemini is strictest. Sheet: `comparison_3engine_detailed.csv`.

## 2. Does the rating predict unlocks? — YES, moderately
- **+0.28** held-out correlation (image rating → real unlock rate). Moderate, usable.
- **1.5×** — the model's top-25% photos get 1.5× the unlock rate of its bottom-25%.
- Trained on **real impressions + unlocks** (GA4 + warehouse, validated).

## 3. What actually drives unlocks (actionable)
| Factor | Correlation |
|---|---|
| **Lighting on face** | **+0.20** ← #1 lever |
| Overall appeal | +0.14 |
| Eye contact | +0.03 |
| "Fancy" attire | −0.18 (the VLM over-rewards it) |

**👉 Lighting is the single biggest driver — prioritise it in the photo-enhancement product.**

## 4. Fairness — clean
~0 gap in predicted unlock probability across Tamil / Telugu / Kannada / Bharat.
We rank on the **photo**, never on caste/religion. Defensible.

## 5. Bottom line
- Photo rating is a **real, defensible** unlock signal (+0.28), with a **1.5× ranking lift**.
- **Lighting** is the key, actionable driver.
- Fairness audited — no community bias.
- Pipeline: Gemini-primary + open-source fallback, live data via lokal-lens, cold-start ready, 47 automated tests.

**Next to make it stronger:** scale the photo set + add behavioural features (feed position, recency) → push the signal past +0.3.

---
*Generated from real engagement data; GA4 impressions validated against actual volume (the original export had a ~23× inflation, since corrected).*

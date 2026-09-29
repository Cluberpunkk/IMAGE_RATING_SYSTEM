# Image Rating System v3 — "Gemini Brain + Open-Source Gate"

A production-shaped prototype for scoring matrimony profile photos and feeding
a single, defensible quality score into feed ranking.

It runs **end-to-end right now with zero credentials** (`--mode simulation`),
and flips to the **real model** with one flag (`--mode gemini`).

---

## 1. The architecture decision (and why)

You asked: *Gemini-only, or Gemini + an open-source model?* The answer that
beats both pure options is a **hybrid that puts each tool where it's strongest**:

```
Image ─► STAGE 0: OSS Gate ─► STAGE 1+2: Gemini Brain ─► STAGE 3: Blend ─► Feed score
         (numpy/PIL CV)        (scores ALL params)         (versioned)
                                      │
                          any param Gemini can't score?
                                      │ yes
                          └─► Qwen2.5-VL fills ONLY those params (per-parameter)
                                      │
                              every value carries PROVENANCE (which model decided it)
```

**Gemini-primary, open-source-fallback, per parameter.** Gemini scores all 18
parameters. For any parameter Gemini returns invalid/missing (or if the call
fails), the best open-source VLM — **Qwen2.5-VL**, *not* Grok — fills *exactly*
those parameters, and the record stamps per-parameter **provenance**
(`gemini` / `qwen2.5-vl` / `default`). See `src/fusion.py`. The fallback is lazy
(only called when Gemini actually misses something) and behind the same retry +
circuit breaker as the primary.

- **Open-Source Gate (Stage 0).** Blur, exposure, resolution, and (pluggable)
  NSFW/face-count checks run in milliseconds for ~zero cost. *Every junk image
  rejected here is a paid Gemini call saved.* CV models are also simply **more
  accurate** than an LLM at "is this blurry / too dark / has no face."
- **Gemini Brain (Stage 1+2).** One vendor, one SDK, two structured JSON calls
  in parallel: the 9 presentation attributes + flags, and the photogenic score.
  Gemini's instruction-following and structured output beat a raw open VLM here,
  and you already run Gemini in production.

**Why not pure Gemini?** It burns paid vision calls on garbage images and is
weaker/over-expensive at the cheap technical checks.
**Why not pure open-source?** Noisier on nuanced judgement, more MLOps to stand
up. We keep an **open-VLM swap path** (Qwen2.5-VL / Llama-4 via Groq) behind the
same adapter for when backfill volume makes cost the priority — see §6.

### Why Grok is gone
The original system used xAI's **Grok** purely to score "hotness / sexual
magnetism" — almost certainly because **Gemini's safety layer refuses that
framing.** We **reframed the task** to a neutral *photogenic / presentation
quality* score (rate the *photograph*, not the person's desirability). That one
change (a) lets Gemini do it reliably with no refusals, and (b) removes a serious
brand/legal liability on a family-oriented matrimony product. Two birds, one fix.

> Note on names: **Grok** = xAI's model (what was used). **Groq** = an inference
> platform that serves open-source models fast/cheap (the natural host for the
> Option-2 swap path). They are different things.

---

## 2. What was broken in the PRD, and how this fixes it

| # | PRD issue | Fix in this prototype |
|---|-----------|----------------------|
| 1 | "Sexual magnetism / hotness" prompt on a matrimony product (brand/legal risk) | Reframed to neutral **photogenic** scoring (`prompts/aesthetic_neutral.txt`) |
| 2 | Female feed weighted **80%** on attractiveness (fairness/optics) | Kept configurable, **recommend 0.55–0.65**; tune in `weights.yaml`, never deploy |
| 3 | **"Attractiveness" weight had no model source** (would break impl) | Removed from qualitative weights; it's the separate top-level `aesthetic` term; qualitative weights renormalised over the 9 real attributes |
| 4 | Ghost **Quantitative engine** (diagram says 3 engines, text says 2) | Purged. Clean 2-call design (qualitative + aesthetic) |
| 5 | **Fallback = 0.5** → failed images out-rank real mediocre ones | Fallback = **0.25** (below median), status `FAILED`, flagged for retry, not cached |
| 6 | "Deterministic" claimed but not enforced | `temperature=0` + response schema **and** image-hash cache → identical input = identical score |
| 7 | Score record **missing versioning** (can't drive re-scoring) | Every record stamps `pipeline / model / prompt / weight` versions |
| 8 | Simulation reshuffled **90%** of the feed, no business-metric validation | Cohort **percentile calibration** + fairness audit + (recommended) A/B holdout on real metrics |
| 9 | No **cost/scale** plan for backfilling millions of images | Stage-0 gate + hash cache/dedup; open-VLM swap path for volume |
| 10 | "Blocking flags" never enumerated; first-image-only ranking brittle | Blocking set explicit in config; feed uses **first VALID** image, skipping blocked lead photos |

Plus added: **fairness/bias drift monitoring** by segment (colorism is a real,
specific risk here), and an **append-only audit log** to answer "why is this
profile ranked here?"

---

## 3. Run it

```bash
pip install -r requirements.txt          # numpy, Pillow, PyYAML (+ google-genai for real mode)

# Full pipeline on privacy-safe synthetic images, no credentials needed:
python run.py demo

# Same pipeline, hitting the real model:
export GEMINI_API_KEY=...                 # see .env.example
python run.py demo --mode gemini

# Score a single image:
python run.py score path/to/photo.jpg --gender female --mode gemini

# Benchmark the aesthetic engines into ONE sheet (output/comparison.csv):
#   Gemini fully | Gemini + Grok | Gemini + Open-Source (Qwen2.5-VL)
python run.py compare                          # synthetic samples, no creds
python run.py compare img1.jpg img2.jpg --mode real   # real APIs (see keys below)

# Build a human ground-truth set: rate each image 1-5 (resumable, crash-safe):
python run.py label /path/to/photos --out output/labels.csv

# Rank the engines by AGREEMENT WITH HUMAN JUDGEMENT (not just who scores high):
#   labels CSV is two columns:  image,truth   (truth = your 0-1 rating or a rank)
python run.py compare *.jpg --mode real --truth output/labels.csv   # -> Spearman ρ + winner

# Tests (blend math, blocking, fallback, caching, prefilter):
python tests/test_pipeline.py
```

The `demo` proves, in one run: scoring, prefilter rejects, cache/dedup hit,
percentile calibration, and the fairness segment report.

---

## 4. The scoring math (exact)

1. **Stage 0 gate** — reject too-blurry / too-dark / too-small (no paid call).
2. **Qualitative score** `Q = Σ wᵢ·attrᵢ` over the 9 attributes (weights sum to 1).
3. **Blocking flags** — if any fire → `BLOCKED`, final = 0 (out of feed).
4. **Blend** `final = w_aes · aesthetic + w_qual · Q` (per-gender weights).
5. **Soft penalty** (e.g. ad/text overlay) subtracted, clamp to [0,1].
6. **Calibration** (ranking time) — percentile-rank within cohort for a healthy
   0–1 spread (the PRD's "scores not clustered" metric, made real).

Feed weight for a profile = the **first VALID image's** final score.

---

## 5. File map

```
image_rating_system/
├── run.py                      # CLI: demo / score
├── make_samples.py             # privacy-safe synthetic test images
├── config/weights.yaml         # versioned weights — edit & bump, no deploy
├── prompts/
│   ├── aesthetic_neutral.txt   # reframed photogenic prompt (no "hotness")
│   └── qualitative_base.txt    # 9 attributes + flags, gender-parameterised
├── src/
│   ├── schemas.py              # typed records + version fingerprint
│   ├── config_loader.py        # weight normalisation, prompt hashing
│   ├── prefilter.py            # STAGE 0 — OSS quality gate (blur/exposure/size)
│   ├── detectors.py            # STAGE 0 — pluggable ML gate (NSFW/face), factory
│   ├── vlm_gemini.py           # STAGE 1+2 — Gemini PRIMARY (temp=0, JSON schema)
│   ├── vlm_opensource.py       # STAGE 1+2 — Qwen2.5-VL FALLBACK (OpenAI-compatible)
│   ├── vlm_simulation.py       # STAGE 1+2 — deterministic mock + fusion adapters
│   ├── fusion.py               # Gemini-primary + per-parameter OSS fallback + provenance
│   ├── resilience.py           # retry + circuit breaker around the paid calls
│   ├── blender.py              # STAGE 3 — blend, blocking, penalties
│   ├── cache.py                # hash+version idempotency cache
│   ├── fairness.py             # audit log + calibration + bias report
│   └── pipeline.py             # orchestration, fallback, feed score
└── tests/test_pipeline.py      # 19 logic tests
```

---

## 6. The moat (where this goes from "good" to "undefeatable")

Phase the rollout:

- **Phase A — ship now.** This prototype: Gemini brain + OSS gate, reframed
  scoring, all bugs fixed. Validate end-to-end on the Telugu female cohort
  **with an A/B holdout measuring real business metrics** (profile views,
  connects, D0/D7 engagement) — not just latency.
  Production hardening already in the box: **retry + circuit breaker** around
  the paid calls (`resilience.py` — one Gemini incident fast-fails to the safe
  fallback instead of timing out per image), and a **pluggable learned NSFW/
  face gate** (`detectors.py`) that rejects unsafe images before the paid call.
  We deliberately use a *learned* classifier, not a skin-tone heuristic — a
  tone-based gate would re-introduce the exact colorism risk we guard against.
- **Phase B — scale cheaply.** Swap the aesthetic call to an **open VLM via
  Groq** (same adapter) once backfill volume makes per-call cost dominate.
- **Phase C — own it.** **Fine-tune a small model on your own engagement
  signal** — predict `P(this first photo earns a connect/view)` from your data.
  This replaces a subjective "attractiveness" guess with an **objective,
  business-aligned, defensible** prediction that is cheap, deterministic, and
  *yours*. It's both more accurate for the real goal (engagement) and something
  you can put in front of a Director, a regulator, or a journalist without
  flinching.

The headline: **stop renting someone's idea of "hot." Predict engagement, on
your own data, behind a cheap deterministic gate.**

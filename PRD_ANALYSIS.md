# PRD v2 → System: Gap Analysis & "Unbeatable" Plan

This maps the **actual PRD (v2)** against the implementation, line by line, and
records every fix applied. It is the document to put in front of a tech lead.

---

## 1. PRD requirement coverage (Section 3 — In Scope)

| PRD scope item | Status | Where |
|---|---|---|
| **A. Image scoring via AI APIs** | ✅ | `vlm_gemini.py` (2 structured calls, temp=0) |
| **A. Final blended score** | ✅ | `blender.py` per-gender blend |
| **A. Configurable weight sets** | ✅ | `config/weights.yaml`, normalised at load |
| **B. Store param-level + final + flags** | ✅ | `schemas.ScoreRecord`, `output/demo_results.json` |
| **B. Expose final to feed-ranking** | ✅ | `pipeline.profile_feed_score()` |
| **B. Backward compatibility** | ✅ | version fingerprint on every record |
| **C. Weight mgmt without code deploy** | ✅ | edit `weights.yaml` + bump `weight_version` |
| **C. Allow prompt changes** | ✅ | `prompt_version()` hash (now incl. fragments) |
| **D. Re-scoring on weight/model update** | ✅ | cache key = hash+versions; `needs_rescore()` |
| **E. Score all images independently** | ✅ | `pipeline.score_profile()` |
| **E. Feed ranking uses first image** | ✅⁺ | uses first **VALID** image (improvement) |

---

## 2. PRD bugs fixed (the original v3 round — see README §2)

Grok "sexual magnetism" framing → neutral photogenic; 0.5 fallback → 0.25;
phantom Quantitative engine purged; unsourced "Attractiveness" weight removed;
determinism enforced (temp=0 + hash cache); versioning added; blocking flags
enumerated; first-VALID-image ranking.

---

## 3. NEW issues found by reading the PRD's own prompt text (this round)

These could only be found from the real PRD, not the README summary.

### 3.1 🔴 Fairness landmines in the PRD rubric
- **"both ears visible"** for high `FaceOrientationVisibility` (PRD p.19/24)
  penalises **hijab / dupatta / ghoonghat / turban** and ear-covering hair.
  → **Fixed** in `prompts/qualitative_base.txt`: judge face visibility only;
  head coverings explicitly NOT penalised.
- **Female attire rubric was generic** while the male rubric named kurta/
  sherwani — risk of under-scoring **saree / salwar / lehenga**.
  → **Fixed** in `config_loader._GENDER_FRAGMENTS["female"]`: explicit cultural
  attire bands on equal footing with Western formal.
- **No anti-bias clause.** → **Added** a FAIRNESS RULES block (no inference on
  caste/religion/region/complexion; traditional dress is valid formal; skin
  tone is never a scoring factor).

### 3.2 🟠 Versioning correctness bug
- `prompt_version()` hashed only `.txt` files, but the **gender fragments live
  in code** and change the prompt. Editing a fragment would NOT bump the
  version → stale scores never re-score → silently breaks PRD scope D.
  → **Fixed**: `prompt_version()` now also hashes `_GENDER_FRAGMENTS`.
  Guarded by `test_prompt_version_covers_gender_fragments`.

### 3.3 🟡 PRD items that worked but weren't first-class / tested
- **Scope E multi-image** → `score_profile()` + test.
- **Success metric "Distribution Health (not clustered)"** (PRD §7) was only a
  printout → real `distribution_health()` metric (std, spread, decile coverage,
  `clustered` bool) + test, shown in the demo.
- **Scope D re-scoring** → concrete `needs_rescore()` + test.

### 3.4 Deliberate, defensible deviation from the PRD
- PRD §6 says fallback = **0.5**. We use **0.25** on purpose: 0.5 is above the
  median real photo, so failed images would out-rank genuine mediocre ones.
  Documented, tested (`test_fallback_is_low_not_half`).

---

## 4. Production hardening added beyond the PRD

- **`fusion.py` — Gemini-primary + open-source-fallback, PER PARAMETER.** Gemini
  scores all 18 parameters; for any parameter Gemini returns invalid/missing,
  the best open-source VLM (**Qwen2.5-VL**, not Grok) fills *exactly* those, and
  every value carries **provenance** (`gemini` / `qwen2.5-vl` / `default`) on the
  record. Lazy (fallback only fires on a real miss), behind the same retry +
  breaker, total-failure raises to the safe FAILED path. `vlm_opensource.py`
  speaks the OpenAI-compatible API → OpenRouter / Groq / vLLM all work by env.
- **`resilience.py`** — retry (exp backoff + jitter, transient-only) + circuit
  breaker around the paid calls. One Gemini incident fast-fails to the safe
  fallback instead of paying full timeout latency per image.
- **`detectors.py`** — pluggable, composable, learned NSFW/face gate with a
  config-driven factory and graceful fallback. **No skin-tone heuristics** (they
  are tone-biased — the exact risk this product guards against).
- **Golden-set regression guard** — deterministic test that identical inputs +
  versions give identical scores, making prompt/weight edits reviewable.
- **Audit log records cache hits too** — fixed a bug where bias monitoring went
  blind on repeat images (most of production).

---

## 5. Roadmap to "truly unbeatable" (needs your data / infra)

1. **A/B holdout on real business metrics** (profile views, connects, D0/D7) —
   the PRD's simulation only measured feed *reordering* (10% overlap), not
   whether reordering improves engagement. This is the missing proof.
2. **Fine-tune on your own engagement signal** — predict `P(first photo earns a
   connect)` from your data. Replaces a rented notion of "attractive" with an
   objective, defensible, cheap, owned model.
3. **Swap the aesthetic call to an open VLM via Groq** once backfill volume
   makes per-call cost dominate (same adapter).
4. **Persist calibration percentiles per cohort** so the feed layer reads a
   stable, monotonic 0..1 weight.

---

## 6. Test coverage

30 logic tests, all green: blend math, blocking, soft penalty, fallback,
cache/determinism, prefilter, retry, circuit breaker, detector composition +
factory fallback, NSFW pre-block, multi-image profile, re-score trigger,
prompt-version integrity, distribution health, calibration spread, golden-set
regression, **fusion per-parameter fallback, validation, total-failure handling,
and provenance recording.** Run: `python tests/test_pipeline.py`.

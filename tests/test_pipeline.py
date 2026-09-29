"""
test_pipeline.py
================
Proves the logic that matters (the stuff a tech reviewer will poke at):
  - qualitative weighted-average math is exact
  - per-gender blend is exact
  - blocking flags force BLOCKED / final=0
  - soft penalty subtracts correctly
  - fallback is the LOW value, never 0.5
  - cache makes identical bytes idempotent (determinism)
  - prefilter rejects junk before any scoring

Run:  python -m pytest tests/ -q     (or)     python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import config_loader
from src.blender import qualitative_score, blend, is_blocked, soft_penalty
from src.schemas import QualitativeAttributes, FlagSet, ScoreStatus
from src.fairness import distribution_health, percentile_calibrate
from src.pipeline import RatingPipeline
from src.resilience import (
    retry_call, CircuitBreaker, CircuitOpenError, is_transient,
)
from src.detectors import (
    build_detector, FakeDeepDetector, CompositeDeepDetector, NoOpDeepDetector,
)
import make_samples


def _cfg():
    return config_loader.load_config()


def test_qualitative_weights_normalised():
    cfg = _cfg()
    assert abs(sum(cfg["qualitative_weights"].values()) - 1.0) < 1e-9


def test_qualitative_score_all_ones_is_one():
    cfg = _cfg()
    attrs = QualitativeAttributes(*([1.0] * 9))
    assert abs(qualitative_score(attrs, cfg) - 1.0) < 1e-9


def test_blend_female_math_exact():
    cfg = _cfg()
    # All attributes = 0.5 -> Q = 0.5. Aesthetic = 1.0.
    attrs = QualitativeAttributes(*([0.5] * 9))
    status, q, final = blend(1.0, attrs, FlagSet(), "female", cfg)
    assert status == ScoreStatus.SCORED
    assert abs(q - 0.5) < 1e-9
    bw = cfg["blend_weights"]["female"]
    expected = bw["aesthetic"] * 1.0 + bw["qualitative"] * 0.5
    assert abs(final - round(expected, 4)) < 1e-9


def test_blend_male_weights_differ_from_female():
    cfg = _cfg()
    attrs = QualitativeAttributes(*([0.2] * 9))
    _, _, female = blend(0.9, attrs, FlagSet(), "female", cfg)
    _, _, male = blend(0.9, attrs, FlagSet(), "male", cfg)
    assert female != male  # 80/20 vs 50/50 must diverge


def test_blocking_flag_forces_zero():
    cfg = _cfg()
    attrs = QualitativeAttributes(*([0.9] * 9))
    flags = FlagSet(nsfw=1)
    assert is_blocked(flags, cfg)
    status, _, final = blend(0.95, attrs, flags, "female", cfg)
    assert status == ScoreStatus.BLOCKED
    assert final == 0.0


def test_soft_penalty_subtracts():
    cfg = _cfg()
    attrs = QualitativeAttributes(*([0.6] * 9))
    clean = blend(0.6, attrs, FlagSet(), "female", cfg)[2]
    penalised = blend(0.6, attrs, FlagSet(advertisements_or_text=1), "female", cfg)[2]
    assert abs((clean - penalised) - cfg["soft_penalties"]["advertisements_or_text"]) < 1e-9


def test_fallback_is_low_not_half():
    cfg = _cfg()
    assert cfg["fallback"]["failed_score"] < 0.5
    assert cfg["fallback"]["failed_score"] <= 0.3  # below median real photo


def test_cache_idempotent_identical_bytes():
    specs = make_samples.generate(ROOT / "sample_images")
    pipe = RatingPipeline(mode="simulation")  # in-memory cache
    clean = next(s for s in specs if s["id"] == "img_clean")
    b = Path(clean["path"]).read_bytes()
    r1 = pipe.score_image(b, "a", "female")
    n1 = pipe.cache.stats()["entries"]
    r2 = pipe.score_image(b, "b", "female")  # identical bytes, different id
    n2 = pipe.cache.stats()["entries"]
    assert n1 == n2                      # no new entry
    assert r1.final_score == r2.final_score  # deterministic


def test_prefilter_rejects_blurry_and_tiny():
    specs = make_samples.generate(ROOT / "sample_images")
    pipe = RatingPipeline(mode="simulation")
    for sid in ("img_blurry", "img_tiny", "img_dark"):
        s = next(x for x in specs if x["id"] == sid)
        rec = pipe.score_image(Path(s["path"]).read_bytes(), sid, s["gender"])
        assert rec.status == ScoreStatus.PREFILTERED
        assert rec.final_score == 0.0


# --------------------------------------------------------------------------- #
# Resilience layer (retry + circuit breaker). Deterministic: fake clock, no
# real sleeps, so these run instantly and never flake.
# --------------------------------------------------------------------------- #

class _Boom(Exception):
    """Transient-looking error (message carries a retryable marker)."""
    def __init__(self):
        super().__init__("503 service unavailable")


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise _Boom()
        return "ok"

    slept = []
    out = retry_call(flaky, max_attempts=3, sleep=slept.append, rng=lambda: 0.5)
    assert out == "ok"
    assert calls["n"] == 3          # original + 2 retries
    assert len(slept) == 2          # one backoff before each retry


def test_retry_does_not_retry_permanent_error():
    calls = {"n": 0}

    def auth_fail():
        calls["n"] += 1
        raise ValueError("invalid api key")   # no transient marker -> fail fast

    slept = []
    try:
        retry_call(auth_fail, max_attempts=5, sleep=slept.append, rng=lambda: 0.5)
        assert False, "should have raised"
    except ValueError:
        pass
    assert calls["n"] == 1          # never retried
    assert slept == []


def test_retry_backoff_is_exponential():
    seq = []

    def always():
        raise _Boom()

    try:
        retry_call(always, max_attempts=4, base_delay=1.0, jitter=0.0,
                   sleep=seq.append, rng=lambda: 0.5)
    except _Boom:
        pass
    assert seq == [1.0, 2.0, 4.0]   # doubling, jitter disabled


def test_circuit_breaker_opens_after_fail_max():
    clock = {"t": 0.0}
    cb = CircuitBreaker(fail_max=3, reset_timeout=30.0, now=lambda: clock["t"])
    for _ in range(3):
        cb.before_call()
        cb.on_failure()
    assert cb.state == CircuitBreaker.OPEN
    # While OPEN, calls fast-fail instead of executing.
    try:
        cb.before_call()
        assert False, "open breaker must fast-fail"
    except CircuitOpenError:
        pass
    assert cb.trip_count == 1


def test_circuit_breaker_half_opens_then_closes_on_recovery():
    clock = {"t": 0.0}
    cb = CircuitBreaker(fail_max=2, reset_timeout=10.0, now=lambda: clock["t"])
    for _ in range(2):
        cb.before_call(); cb.on_failure()
    assert cb.state == CircuitBreaker.OPEN
    clock["t"] = 11.0               # cooldown elapsed
    cb.before_call()               # -> HALF_OPEN probe allowed
    assert cb.state == CircuitBreaker.HALF_OPEN
    cb.on_success()                # probe ok -> back to normal
    assert cb.state == CircuitBreaker.CLOSED


def test_circuit_breaker_fast_fail_is_not_retried():
    # A CircuitOpenError must NOT be retried (the breaker already decided).
    cb = CircuitBreaker(fail_max=1, reset_timeout=60.0, now=lambda: 0.0)
    cb.before_call(); cb.on_failure()          # trip it
    assert cb.state == CircuitBreaker.OPEN
    attempts = {"n": 0}

    def guarded():
        attempts["n"] += 1
        raise RuntimeError("unreached")

    slept = []
    try:
        retry_call(guarded, breaker=cb, max_attempts=5, sleep=slept.append,
                   rng=lambda: 0.5)
        assert False, "should fast-fail"
    except CircuitOpenError:
        pass
    assert attempts["n"] == 0       # breaker blocked entry; fn never ran
    assert not is_transient(CircuitOpenError("x"))


# --------------------------------------------------------------------------- #
# Stage-0 deep detectors (src/detectors.py): composition, factory, and the
# end-to-end reject (a detector flag must keep the image out of the VLM stage).
# --------------------------------------------------------------------------- #

def test_composite_ORs_flags_and_takes_strictest_face_count():
    safe = FakeDeepDetector(nsfw=0, face_count=1)
    bad = FakeDeepDetector(nsfw=1, face_count=2)   # nsfw + a group photo
    merged = CompositeDeepDetector([safe, bad]).detect(img=None)
    assert merged["nsfw"] == 1                      # any detector can reject
    assert merged["face_count"] == 1               # strictest (min) reading


def test_composite_skips_a_failing_detector():
    class Broken:
        def detect(self, img):
            raise RuntimeError("model blew up")
    merged = CompositeDeepDetector([Broken(), FakeDeepDetector(nsfw=1)]).detect(None)
    assert merged["nsfw"] == 1                      # survivor's verdict still counts


def test_factory_none_and_unknown_fall_back_safely():
    assert build_detector({"prefilter": {"deep_detector": {"type": "none"}}}) is None
    # An unknown / unavailable type must degrade to NoOp, never crash.
    fb = build_detector({"prefilter": {"deep_detector": {"type": "does_not_exist"}}})
    assert isinstance(fb, NoOpDeepDetector)


def test_nsfw_detector_flag_blocks_before_vlm():
    # An injected detector that flags NSFW must PREFILTER the image -> no VLM,
    # final score 0. This is the cost-saving + safety guarantee end to end.
    specs = make_samples.generate(ROOT / "sample_images")
    clean = next(s for s in specs if s["id"] == "img_clean")
    pipe = RatingPipeline(mode="simulation",
                          deep_detector=FakeDeepDetector(nsfw=1))
    rec = pipe.score_image(Path(clean["path"]).read_bytes(), "x", "female")
    assert rec.status == ScoreStatus.PREFILTERED
    assert rec.final_score == 0.0
    assert "nsfw_prefilter" in (rec.reasoning or "")


# --------------------------------------------------------------------------- #
# PRD scope coverage: multi-image profiles (E), re-scoring trigger (D),
# distribution-health success metric, and prompt-version integrity.
# --------------------------------------------------------------------------- #

def test_profile_uses_first_valid_image_not_first_image():
    # Lead photo is blurry (PREFILTERED); profile score must come from the
    # first VALID image, not the tanked lead. (PRD scope E, improved.)
    specs = make_samples.generate(ROOT / "sample_images")
    blurry = next(s for s in specs if s["id"] == "img_blurry")
    clean = next(s for s in specs if s["id"] == "img_clean")
    pipe = RatingPipeline(mode="simulation")
    feed, recs = pipe.score_profile(
        [(Path(blurry["path"]).read_bytes(), "lead"),
         (Path(clean["path"]).read_bytes(), "second")],
        gender_feed="female",
    )
    assert recs[0].status == ScoreStatus.PREFILTERED
    assert recs[1].status == ScoreStatus.SCORED
    assert feed == recs[1].final_score          # skipped the junk lead photo


def test_needs_rescore_triggers_on_version_change():
    pipe = RatingPipeline(mode="simulation")
    current = pipe.versions.as_dict()
    assert pipe.needs_rescore(current) is False            # same versions -> fresh
    stale = dict(current, weight_version="v0.0-old")
    assert pipe.needs_rescore(stale) is True               # weight bump -> re-score
    stale2 = dict(current, prompt_version="prompt-deadbeef")
    assert pipe.needs_rescore(stale2) is True              # prompt edit -> re-score


def test_prompt_version_covers_gender_fragments():
    # Editing a gender fragment MUST change prompt_version, or stale scores
    # would never re-score (the bug this guards against).
    base = config_loader.prompt_version()
    frags = config_loader._GENDER_FRAGMENTS
    saved = frags["female"]["ATTIRE_EXTRA_HIGH"]
    try:
        frags["female"]["ATTIRE_EXTRA_HIGH"] = saved + " CHANGED"
        assert config_loader.prompt_version() != base
    finally:
        frags["female"]["ATTIRE_EXTRA_HIGH"] = saved        # restore
    assert config_loader.prompt_version() == base


def test_distribution_health_flags_clustered_vs_healthy():
    clustered = distribution_health([0.5, 0.5, 0.51, 0.49, 0.5])
    assert clustered["clustered"] is True
    healthy = distribution_health([0.05, 0.25, 0.45, 0.65, 0.85, 0.95])
    assert healthy["clustered"] is False
    assert healthy["coverage"] > clustered["coverage"]


def test_calibration_spreads_scores_to_full_range():
    cal = percentile_calibrate([0.71, 0.72, 0.73, 0.74])
    assert min(cal) == 0.0 and max(cal) == 1.0   # clustered raw -> full spread


def test_golden_set_regression_guard():
    # A deterministic guard: identical inputs + identical versions must yield
    # identical scores. This is the safety net that makes prompt/weight edits
    # reviewable — re-run, diff the golden scores, ship only if intended.
    specs = make_samples.generate(ROOT / "sample_images")
    pipe = RatingPipeline(mode="simulation")
    golden = {}
    for sid in ("img_clean", "img_deeptone", "img_male"):
        s = next(x for x in specs if x["id"] == sid)
        rec = pipe.score_image(Path(s["path"]).read_bytes(), sid, s["gender"])
        golden[sid] = rec.final_score
    # Re-score: must be bit-identical (determinism / no drift).
    for sid in ("img_clean", "img_deeptone", "img_male"):
        s = next(x for x in specs if x["id"] == sid)
        rec = pipe.score_image(Path(s["path"]).read_bytes(), sid, s["gender"])
        assert rec.final_score == golden[sid]


# --------------------------------------------------------------------------- #
# Gemini-primary + open-source-fallback FUSION (src/fusion.py): per-parameter
# repair, provenance, validation, total-failure handling.
# --------------------------------------------------------------------------- #

from src.fusion import (
    ResilientScorer, validate_qualitative, validate_aesthetic, ALL_QUAL_KEYS,
)
from src.vlm_simulation import SimulationAdapter, PartialSimulationAdapter, image_pixel_hash


def _clean_image():
    specs = make_samples.generate(ROOT / "sample_images")
    clean = next(s for s in specs if s["id"] == "img_clean")
    return Image.open(clean["path"]).convert("RGB")


def _h(img):
    return image_pixel_hash(img)


def test_validate_qualitative_rejects_bad_values():
    good = {k: 0.5 for k in ALL_QUAL_KEYS}
    valid, missing = validate_qualitative(good)
    assert missing == []                                    # all present & valid
    bad = dict(good, Expression=None, NSFW="yes", LightingOnFace=float("nan"))
    valid, missing = validate_qualitative(bad)
    assert set(missing) == {"Expression", "NSFW", "LightingOnFace"}


def test_fusion_fills_only_missing_params_from_fallback():
    img = _clean_image()
    # Primary (Gemini stand-in) fails to score 2 qual params + photogenic.
    primary = PartialSimulationAdapter(
        drop_qual=("Expression", "LightingOnFace"), drop_aes=("photogenic",),
        img_hash_of=_h,
    )
    fallback = SimulationAdapter(img_hash_of=_h)
    attrs, flags, aes, reason, prov = ResilientScorer(primary, fallback).score(
        img, "female", _h(img))
    src = prov["sources"]
    # Exactly the dropped params come from the fallback; everything else primary.
    assert src["Expression"] == fallback.name
    assert src["LightingOnFace"] == fallback.name
    assert src["Photogenic"] == fallback.name
    assert src["Centering"] == primary.name                 # untouched
    assert prov["fallback_used"] is True
    assert prov["degraded"] is False                        # nothing defaulted


def test_fusion_defaults_when_no_fallback_available():
    img = _clean_image()
    primary = PartialSimulationAdapter(drop_qual=("Attire",), img_hash_of=_h)
    attrs, flags, aes, reason, prov = ResilientScorer(primary, fallback=None).score(
        img, "female", _h(img))
    assert prov["sources"]["Attire"] == "default"
    assert prov["degraded"] is True                         # a neutral default used
    assert attrs.attire == 0.5                              # neutral, not 0


def test_fusion_total_failure_raises():
    img = _clean_image()
    # Primary drops EVERYTHING and there is no fallback -> total failure -> raise,
    # so the pipeline's safe FAILED path (not a fabricated score) takes over.
    dead = PartialSimulationAdapter(
        drop_qual=tuple(ALL_QUAL_KEYS), drop_aes=("photogenic",), img_hash_of=_h)
    try:
        ResilientScorer(dead, fallback=None).score(img, "female", _h(img))
        assert False, "total failure must raise"
    except RuntimeError:
        pass


def test_pipeline_records_provenance():
    specs = make_samples.generate(ROOT / "sample_images")
    clean = next(s for s in specs if s["id"] == "img_clean")
    pipe = RatingPipeline(mode="simulation")
    rec = pipe.score_image(Path(clean["path"]).read_bytes(), "p", "female")
    assert rec.provenance is not None
    assert set(rec.provenance["sources"]) >= set(ALL_QUAL_KEYS)
    # Simulation primary scores everything -> no fallback, no degradation.
    assert rec.provenance["fallback_used"] is False
    assert rec.provenance["degraded"] is False


# --------------------------------------------------------------------------- #
# Comparison harness (src/compare.py): the benchmark sheet.
# --------------------------------------------------------------------------- #

def test_compare_produces_three_diverging_columns_and_csv(tmp_path=None):
    from src.compare import compare_images, PROVIDERS
    specs = make_samples.generate(ROOT / "sample_images")
    images = [(Path(s["path"]).read_bytes(), s["id"])
              for s in specs if s["id"] in ("img_clean", "img_deeptone", "img_male")]
    out_csv = ROOT / "output" / "comparison_test.csv"
    rows = compare_images(images, gender="female", mode="simulation", csv_path=out_csv)
    assert len(rows) == 3
    for r in rows:
        assert r["status"] == "SCORED"
        for p in PROVIDERS:
            assert r[p] is not None
    # The three engines must actually diverge on at least one image (not a
    # degenerate sheet where every column is identical).
    assert any(len({r["gemini"], r["grok"], r["opensource"]}) > 1 for r in rows)
    assert out_csv.exists()
    out_csv.unlink()


def test_spearman_monotonic_and_reversed():
    from src.compare import _spearman
    assert abs(_spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9   # perfect
    assert abs(_spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9   # inverse


def test_engine_correlations_picks_best_against_truth():
    from src.compare import engine_correlations, PROVIDERS
    # Truth aligns with engine 'opensource' ordering; it must win.
    rows = [
        {"image": "a", "status": "SCORED", "gemini": 0.9, "grok": 0.5, "opensource": 0.1},
        {"image": "b", "status": "SCORED", "gemini": 0.8, "grok": 0.6, "opensource": 0.5},
        {"image": "c", "status": "SCORED", "gemini": 0.7, "grok": 0.4, "opensource": 0.9},
    ]
    truth = {"a": 0.1, "b": 0.5, "c": 0.9}     # matches 'opensource' monotonically
    summary = engine_correlations(rows, truth)
    assert summary["best"] == "opensource"
    assert abs(summary["opensource"]["spearman"] - 1.0) < 1e-9


def test_auc_and_binary_winner_approved_vs_rejected():
    from src.compare import engine_correlations
    # 2 approved (1.0) + 2 rejected (0.0). 'opensource' perfectly separates them.
    rows = [
        {"image": "g1", "status": "SCORED", "gemini": 0.6, "grok": 0.55, "opensource": 0.9},
        {"image": "g2", "status": "SCORED", "gemini": 0.4, "grok": 0.50, "opensource": 0.8},
        {"image": "b1", "status": "SCORED", "gemini": 0.7, "grok": 0.45, "opensource": 0.2},
        {"image": "b2", "status": "SCORED", "gemini": 0.3, "grok": 0.60, "opensource": 0.1},
    ]
    truth = {"g1": 1.0, "g2": 1.0, "b1": 0.0, "b2": 0.0}    # binary good/bad
    summary = engine_correlations(rows, truth)
    assert summary["binary"] is True
    assert summary["metric"] == "auc"
    assert summary["opensource"]["auc"] == 1.0             # perfect separation
    assert summary["best"] == "opensource"
    assert summary["opensource"]["separation"] > 0


def test_load_truth_parses_and_skips_header():
    from src.compare import load_truth
    p = ROOT / "labels_example.csv"
    t = load_truth(p)
    assert "img_clean" in t and isinstance(t["img_clean"], float)
    assert "image" not in t                    # header skipped


def test_compare_marks_prefiltered_images():
    from src.compare import compare_images, PROVIDERS
    specs = make_samples.generate(ROOT / "sample_images")
    blurry = next(s for s in specs if s["id"] == "img_blurry")
    rows = compare_images([(Path(blurry["path"]).read_bytes(), "img_blurry")],
                          mode="simulation")
    assert rows[0]["status"] == "PREFILTERED"
    assert all(rows[0][p] is None for p in PROVIDERS)


# --------------------------------------------------------------------------- #
# Labeling helper (src/labeler.py): resumable ground-truth collection.
# --------------------------------------------------------------------------- #

def test_normalize_rating_maps_1to5_onto_0to1():
    from src.labeler import normalize_rating
    assert normalize_rating(1) == 0.0
    assert normalize_rating(3) == 0.5
    assert normalize_rating(5) == 1.0


def test_label_images_writes_skips_and_resumes():
    from src.labeler import label_images, load_existing
    out = ROOT / "output" / "labels_test.csv"
    if out.exists():
        out.unlink()
    images = [("a.jpg", "a"), ("b.jpg", "b"), ("c.jpg", "c")]
    # Rate a=5, skip b, then quit before c.
    script = {"a": 5, "b": "skip", "c": "quit"}
    labels = label_images(images, out, show=None, ask=lambda i, p: script[i])
    assert labels == {"a": 1.0}                          # only a saved
    # Resume: a is skipped (already labeled), now rate c=1.
    labels2 = label_images(images, out, show=None,
                           ask=lambda i, p: 1 if i == "c" else "skip")
    assert load_existing(out) == {"a": 1.0, "c": 0.0}    # persisted + merged
    out.unlink()


def test_discover_images_expands_dir():
    from src.labeler import discover_images
    make_samples.generate(ROOT / "sample_images")
    found = discover_images([str(ROOT / "sample_images")])
    ids = {i for _, i in found}
    assert "img_clean" not in ids  # ids are file stems, e.g. "01_clean_portrait"
    assert any("clean_portrait" in i for i in ids)
    assert len(found) >= 6


# --------------------------------------------------------------------------- #
# Sheet ingestion (src/fetch.py): URL extraction, id derivation, auto-labels.
# --------------------------------------------------------------------------- #

def test_extract_urls_and_ids_from_csv_text():
    from src.fetch import extract_urls_from_csv, url_to_id
    sample = ROOT / "output" / "sheet_sample.csv"
    sample.parent.mkdir(parents=True, exist_ok=True)
    sample.write_text(
        's.no,url,original,enhanced\n'
        '1,https://matrimony-cdn.getlokalapp.com/x/2993926/enhanced_3502017.webp,a,b\n'
        '2,"https://matrimony-cdn.getlokalapp.com/x/141373/enhanced_144414.webp",a,b\n'
        '2,https://matrimony-cdn.getlokalapp.com/x/141373/enhanced_144414.webp,dup,row\n'
    )
    urls = extract_urls_from_csv(sample)
    assert len(urls) == 2                                  # de-duped
    assert url_to_id(urls[0]) == "enhanced_3502017"
    assert url_to_id(urls[1]) == "enhanced_144414"
    sample.unlink()


def test_parse_sheet_url_and_export_link():
    from src.fetch import parse_sheet_url, sheet_csv_url
    url = ("https://docs.google.com/spreadsheets/d/12xRAOvANVH_oe-SNPMGiK13acyqYZ"
           "--Wut1HAfNtoUE/edit?gid=563815522#gid=563815522")
    sid, gid = parse_sheet_url(url)
    assert sid == "12xRAOvANVH_oe-SNPMGiK13acyqYZ--Wut1HAfNtoUE"
    assert gid == "563815522"
    assert "format=csv" in sheet_csv_url(sid, gid) and "gid=563815522" in sheet_csv_url(sid, gid)


def test_extract_urls_from_text_dedupes():
    from src.fetch import extract_urls_from_text
    txt = ("a https://cdn.x.com/p/1/enhanced_11.webp b "
           "https://cdn.x.com/p/1/enhanced_11.webp c https://cdn.x.com/p/2/img_22.jpg")
    assert extract_urls_from_text(txt) == [
        "https://cdn.x.com/p/1/enhanced_11.webp", "https://cdn.x.com/p/2/img_22.jpg"]


def test_label_value_and_append_labels():
    from src.fetch import label_value, append_labels
    from src.labeler import load_existing
    assert label_value("approved") == 1.0
    assert label_value("rejected") == 0.0
    assert label_value("0.7") == 0.7
    out = ROOT / "output" / "labels_fetch_test.csv"
    if out.exists():
        out.unlink()
    append_labels([("enhanced_1", 1.0), ("enhanced_2", 0.0)], out)
    append_labels([("enhanced_3", 1.0)], out)              # merges, not overwrites
    assert load_existing(out) == {"enhanced_1": 1.0, "enhanced_2": 0.0, "enhanced_3": 1.0}
    out.unlink()


# --------------------------------------------------------------------------- #
# Engagement model (src/engagement.py): the unlock-prediction beast (Layer 2).
# --------------------------------------------------------------------------- #

def _synth_engagement(n=400):
    """Synthetic impressions where unlock truly depends on appeal+eye_contact+
    lighting (+ noise). A good model must recover that and rank by it."""
    import numpy as np
    from src.engagement import featurize
    rng = np.random.RandomState(0)
    ex = []
    for i in range(n):
        appeal = rng.uniform(0, 1)
        eye = float(rng.uniform(0, 1) > 0.5)
        light = rng.uniform(0, 1)
        attrs = {"expression": rng.uniform(0, 1), "eye_contact": eye,
                 "lighting_on_face": light, "grooming_neatness": rng.uniform(0, 1),
                 "face_orientation_visibility": rng.uniform(0, 1),
                 "centering": rng.uniform(0, 1), "pose_camera_angle": rng.uniform(0, 1),
                 "background_foreground_separation": rng.uniform(0, 1),
                 "attire": rng.uniform(0, 1)}
        logit = 3.0 * appeal + 1.5 * eye + 1.2 * light - 3.0 + rng.normal(0, 0.4)
        unlocked = int(1 / (1 + np.exp(-logit)) > rng.uniform(0, 1))
        ex.append({"features": featurize({"vlm_appeal": appeal, "attributes": attrs}),
                   "unlocked": unlocked})
    return ex


def test_engagement_model_learns_to_predict_unlocks():
    from src.engagement import EngagementModel, auc
    data = _synth_engagement(500)
    train, test = data[:400], data[400:]
    m = EngagementModel().train(train)
    scores = [m.p_unlock(e["features"]) for e in test]
    labels = [e["unlocked"] for e in test]
    assert auc(scores, labels) > 0.75          # recovers the real signal


def test_cold_start_falls_back_to_vlm_appeal():
    from src.engagement import EngagementModel, featurize
    m = EngagementModel()                       # untrained
    feats = featurize({"vlm_appeal": 0.77, "attributes": {}})
    assert m.p_unlock(feats) == 0.77            # uses the Layer-1 prior


def test_rank_feed_orders_by_unlock_and_explores_cold_start():
    from src.engagement import EngagementModel, featurize, rank_feed
    m = EngagementModel()                        # untrained -> ranks on appeal
    profiles = [
        {"profile_id": "hi", "features": featurize({"vlm_appeal": 0.9}), "impressions": 500},
        {"profile_id": "lo", "features": featurize({"vlm_appeal": 0.3}), "impressions": 500},
        {"profile_id": "new", "features": featurize({"vlm_appeal": 0.5}), "impressions": 0},
    ]
    ranked = rank_feed(profiles, m, explore=0.1)
    assert ranked[0]["profile_id"] == "hi"       # best appeal on top
    # The brand-new profile gets an exploration lift above its raw appeal.
    new = next(p for p in ranked if p["profile_id"] == "new")
    assert new["rank_score"] > new["p_unlock"]


def test_shrink_to_prior_cold_to_warm():
    from src.engagement import shrink_to_prior
    prior = 0.001                                   # VLM-implied rate for a new profile
    # No impressions -> fully the prior.
    assert abs(shrink_to_prior(0, 0, prior) - prior) < 1e-9
    # Tons of impressions with a high observed rate -> trusts the data.
    warm = shrink_to_prior(observed_unlocks=900, impressions=100000, prior_rate=prior)
    assert warm > 0.005                             # pulled toward observed 0.009
    # A few impressions -> mostly prior still (no overreaction to noise).
    early = shrink_to_prior(observed_unlocks=2, impressions=100, prior_rate=prior)
    assert abs(early - prior) < 0.001


def test_fairness_guard_flags_segment_gap():
    from src.engagement import fairness_by_segment
    ranked = [
        {"p_unlock": 0.8, "segment": "tone_light"},
        {"p_unlock": 0.8, "segment": "tone_light"},
        {"p_unlock": 0.4, "segment": "tone_deep"},
        {"p_unlock": 0.4, "segment": "tone_deep"},
    ]
    rep = fairness_by_segment(ranked)
    assert rep["tone_light"] == 0.8 and rep["tone_deep"] == 0.4
    assert rep["_max_gap"] == 0.4                # big gap surfaced for review


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()

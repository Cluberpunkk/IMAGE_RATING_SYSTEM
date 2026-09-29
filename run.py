#!/usr/bin/env python3
"""
run.py — entry point
====================
Usage:
  python run.py demo                      # synthetic batch, full pipeline (no creds)
  python run.py demo --mode gemini        # same, hitting real Gemini (needs key)
  python run.py score <path> --gender female [--mode gemini]

`demo` builds privacy-safe synthetic images, runs the whole pipeline, then
prints: per-image results, prefilter rejects, a BLOCKED example, cache-hit on a
duplicate, cohort percentile calibration, and the fairness segment report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.pipeline import RatingPipeline                       # noqa: E402
from src.fairness import percentile_calibrate, distribution_health  # noqa: E402
from src.schemas import ScoreStatus                           # noqa: E402
import make_samples                              # noqa: E402

OUT = ROOT / "output"


def _print_row(rec):
    f = rec.flags.as_dict() if rec.flags else {}
    fired = [k for k, v in f.items() if v == 1]
    line = (
        f"  {rec.image_id:<14} {rec.status.value:<11} "
        f"final={rec.final_score:<6} aes={rec.aesthetic_score:<6} "
        f"qual={rec.qualitative_score:<6} {rec.latency_ms:>5}ms"
    )
    if fired:
        line += f"  flags={fired}"
    if rec.status in (ScoreStatus.PREFILTERED, ScoreStatus.FAILED):
        line += f"  <- {rec.reasoning}"
    print(line)


def _demo_fusion(specs):
    """Show per-parameter fallback: a primary (Gemini stand-in) that fails to
    score 3 parameters, with the open-source model (Qwen stand-in) repairing
    EXACTLY those — and the provenance the system records for each value."""
    from PIL import Image
    from src.fusion import ResilientScorer
    from src.vlm_simulation import (
        SimulationAdapter, PartialSimulationAdapter, image_pixel_hash,
    )

    clean = next(s for s in specs if s["id"] == "img_clean")
    img = Image.open(clean["path"]).convert("RGB")
    primary = PartialSimulationAdapter(
        drop_qual=("Expression", "LightingOnFace"), drop_aes=("photogenic",),
        img_hash_of=image_pixel_hash,
    )
    primary.name = "gemini-2.5-flash"                         # label for the demo
    fallback = SimulationAdapter(img_hash_of=image_pixel_hash)
    fallback.name = "qwen2.5-vl-72b"
    _, _, _, _, prov = ResilientScorer(primary, fallback).score(
        img, "female", image_pixel_hash(img))
    by_model = {}
    for param, model in prov["sources"].items():
        by_model.setdefault(model, []).append(param)
    for model, params in by_model.items():
        print(f"  {model:<18} scored {len(params):>2}: {', '.join(sorted(params))}")
    print(f"  fallback_used={prov['fallback_used']}  degraded={prov['degraded']}"
          "  (Gemini missed 3 params; the open-source model filled exactly those)")


def cmd_demo(args):
    print(f"\n=== Image Rating System v3 — demo (mode={args.mode}) ===\n")
    specs = make_samples.generate(ROOT / "sample_images")

    pipe = RatingPipeline(
        mode=args.mode,
        cache_path=OUT / "score_cache.json",
        audit_path=OUT / "audit_log.jsonl",
    )
    print("Version fingerprint stamped on every record:")
    print("  " + json.dumps(pipe.versions.as_dict()) + "\n")

    records = []
    print("Per-image results:")
    for s in specs:
        if s["id"] == "img_dupe":
            continue  # shown separately in the cache check below
        img_bytes = Path(s["path"]).read_bytes()
        rec = pipe.score_image(img_bytes, s["id"], s["gender"], segment=s["segment"])
        records.append((s, rec))
        _print_row(rec)

    # ---- Cache demonstration: re-score the duplicate, expect a cache hit ----
    print("\nCache / idempotency check (re-running identical bytes):")
    dupe = next(s for s in specs if s["id"] == "img_dupe")
    before = pipe.cache.stats()["entries"]
    rec2 = pipe.score_image(Path(dupe["path"]).read_bytes(), "img_dupe", "female")
    after = pipe.cache.stats()["entries"]
    print(f"  img_dupe -> final={rec2.final_score}  cache_entries {before}->{after} "
          f"(no new paid call; identical bytes served from cache)")

    # ---- Cohort calibration (healthy spread, fixes 'clustered scores') ----
    scored = [r for _, r in records if r.status == ScoreStatus.SCORED]
    raw = [r.final_score for r in scored]
    cal = percentile_calibrate(raw)
    print("\nCohort percentile calibration (raw -> calibrated feed weight):")
    for r, c in zip(scored, cal):
        print(f"  {r.image_id:<14} raw={r.final_score:<7} -> calibrated={c}")

    # ---- Distribution health (PRD success metric: 'scores not clustered') ----
    print("\nScore Distribution Health (PRD success metric):")
    print("  raw        " + json.dumps(distribution_health(raw)))
    print("  calibrated " + json.dumps(distribution_health(cal)))
    print("  (calibration lifts coverage/spread so the feed signal differentiates.)")

    # ---- Fusion demo: Gemini primary + open-source fallback, per parameter ----
    print("\nPer-parameter fusion (Gemini primary + open-source fallback):")
    _demo_fusion(specs)

    # ---- Fairness / bias drift report ----
    print("\nFairness segment report (mean final score by tone segment):")
    print("  " + json.dumps(pipe.audit.segment_report()))
    print("  (light vs deep tone should be CLOSE; a large gap = investigate.)")

    # ---- Persist a clean machine-readable result ----
    dump = [
        {"spec": s, "record": r.as_dict()}
        for s, r in records
    ]
    (OUT / "demo_results.json").write_text(json.dumps(dump, indent=2))
    print(f"\nFull structured output written to: {OUT / 'demo_results.json'}")
    print("Audit trail (append-only):           " + str(OUT / "audit_log.jsonl"))
    print("Score cache:                          " + str(OUT / "score_cache.json"))
    print("\nSummary:")
    counts = {}
    for _, r in records:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    print("  " + json.dumps(counts))
    print()


def cmd_score(args):
    pipe = RatingPipeline(mode=args.mode, cache_path=OUT / "score_cache.json")
    img_bytes = Path(args.path).read_bytes()
    rec = pipe.score_image(img_bytes, Path(args.path).stem, args.gender)
    print(json.dumps(rec.as_dict(), indent=2))


def cmd_compare(args):
    """Benchmark sheet: Gemini fully vs Gemini+Grok vs Gemini+OpenSource."""
    from src.compare import (
        compare_images, format_table, load_truth, engine_correlations,
        format_correlation,
    )

    print(f"\n=== Aesthetic-engine benchmark (mode={args.mode}, gender={args.gender}) ===\n")
    truth = load_truth(Path(args.truth)) if args.truth else None

    if args.urls:
        # Stream from sheet URLs — NOTHING saved to disk.
        from src.fetch import read_url_manifest, stream_images
        manifest = read_url_manifest(args.urls)
        if truth is None:                       # manifest can carry the labels too
            truth = {iid: t for iid, _u, t in manifest if t is not None} or None
        if args.limit_per_class and truth:
            from collections import defaultdict
            buckets = defaultdict(list)
            for iid, url, _t in manifest:
                if iid in truth:
                    buckets[truth[iid]].append((iid, url, _t))
            manifest = [x for lst in buckets.values() for x in lst[:args.limit_per_class]]
            print(f"(balanced pilot: {len(manifest)} images, <= {args.limit_per_class}/class)\n")
        print(f"Streaming {len(manifest)} image(s) from URLs (no local copies)...\n")
        images = stream_images(manifest)        # lazy generator
    else:
        if args.images:
            images = [(Path(p).read_bytes(), Path(p).stem) for p in args.images]
        else:
            specs = make_samples.generate(ROOT / "sample_images")
            images = [(Path(s["path"]).read_bytes(), s["id"])
                      for s in specs if s["id"] != "img_dupe"]
        if args.limit_per_class and truth:
            from collections import defaultdict
            buckets = defaultdict(list)
            for b, iid in images:
                if iid in truth:
                    buckets[truth[iid]].append((b, iid))
            images = [item for lst in buckets.values() for item in lst[:args.limit_per_class]]
            print(f"(balanced pilot: {len(images)} images, <= {args.limit_per_class}/class)\n")

    csv_path = Path(args.out) if args.out else OUT / "comparison.csv"
    detail_path = Path(args.detail_out) if args.detail_out else OUT / "comparison_detailed.csv"
    rows = compare_images(images, gender=args.gender, mode=args.mode,
                          csv_path=csv_path, truth=truth,
                          no_prefilter=args.no_prefilter,
                          no_blocking=args.no_blocking,
                          detail_csv_path=detail_path)
    print(format_table(rows))

    if truth:
        print()
        summary = engine_correlations(rows, truth)
        print(format_correlation(summary))
    else:
        print("\n(no --truth given: 'best' = highest score, NOT necessarily best."
              " Pass --truth labels.csv to rank engines against human judgement.)")

    if args.mode == "simulation":
        print("\n(simulation: numbers are deterministic stand-ins to prove the sheet;"
              " run --mode real with API keys for true scores.)")
    print(f"\nSummary sheet:  {csv_path}")
    print(f"Detailed sheet: {detail_path}  (every parameter, reasoning, winner)")


def cmd_optimize(args):
    """Optimize Gemini-only scoring against a REAL quality label, held-out
    validated. Trains the learned model on Gemini's 16 attributes and compares
    it (and the simple blend, and Grok) against the label. The honest optimizer."""
    import csv as _csv
    import numpy as np
    from src.engagement import (load_features_from_detail, _Logistic, spearman,
                                EngagementModel)
    from src.compare import load_truth, _auc
    truth = load_truth(Path(args.labels))
    feats = load_features_from_detail(args.scores)
    fin_gem, fin_grok = {}, {}
    for r in _csv.DictReader(open(args.scores)):
        if r.get('FINAL:Gemini fully'): fin_gem[r['image']] = float(r['FINAL:Gemini fully'])
        if r.get('FINAL:Gemini + Grok'): fin_grok[r['image']] = float(r['FINAL:Gemini + Grok'])
    ids = [s for s in truth if s in feats]
    if len(ids) < 10:
        print(f"Only {len(ids)} labeled+scored photos. Label ~30+ for a real result "
              "(python run.py label label_photos --out output/quality_labels.csv).")
        return
    X = [feats[s] for s in ids]; y = [truth[s] for s in ids]
    uniq = sorted(set(y))
    if len(uniq) == 2:                      # already binary (e.g. good/bad)
        hi = uniq[-1]; lab = [1 if v == hi else 0 for v in y]
    else:                                   # 1-5 ratings -> split at median
        med = float(np.median(y)); lab = [1 if v > med else 0 for v in y]
    print(f"{len(ids)} photos labeled | good={sum(lab)}  bad={len(lab)-sum(lab)}\n")

    # 5-fold CV held-out predictions for the learned model (no peeking)
    n = len(ids); rng = np.random.RandomState(0); idx = rng.permutation(n)
    folds = np.array_split(idx, min(5, n)); preds = [0.0]*n
    for k in range(len(folds)):
        te = set(folds[k].tolist()); tr = [i for i in range(n) if i not in te]
        m = _Logistic().fit([X[i] for i in tr], [y[i] for i in tr])
        for i in folds[k]: preds[i] = float(m.predict_proba([X[i]])[0])

    print("=== OPTIMIZED GEMINI vs alternatives, on YOUR quality labels ===")
    print(f"  Learned model (Gemini-only, HELD-OUT) : AUC {_auc(preds,lab):.3f}  ρ {spearman(preds,y):+.3f}")
    print(f"  Gemini-fully simple blend             : AUC {_auc([fin_gem[s] for s in ids],lab):.3f}")
    if all(s in fin_grok for s in ids):
        print(f"  Gemini + Grok                         : AUC {_auc([fin_grok[s] for s in ids],lab):.3f}")
    print("  (higher AUC = better at telling good photos from bad)")

    em = EngagementModel(); em._m = _Logistic().fit(X, y); em.trained = True
    em.save(OUT / "quality_model.json")
    print(f"\nSaved optimized Gemini-only model -> {OUT / 'quality_model.json'}")
    print("What Gemini signals best predict quality (learned weights):")
    for kf, w in sorted(em.feature_weights().items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"   {kf:30} {w:+.3f}")


def cmd_rank(args):
    """Rank a feed with the trained P(unlock) model (cold-start + exploration)."""
    from src.engagement import EngagementModel, load_features_from_detail, rank_feed
    model = EngagementModel.load(args.model)
    feats = load_features_from_detail(args.scores)
    profiles = [{"profile_id": pid, "features": f, "impressions": 0}
                for pid, f in feats.items()]
    ranked = rank_feed(profiles, model, explore=0.0)
    print(f"\n=== Feed ranked by P(unlock)  ({len(ranked)} profiles) ===")
    print("Top 10 (surface first):")
    for p in ranked[:10]:
        print(f"   {p['profile_id']:22} P(unlock)={p['p_unlock']:.4f}")
    print("Bottom 5:")
    for p in ranked[-5:]:
        print(f"   {p['profile_id']:22} P(unlock)={p['p_unlock']:.4f}")


def cmd_analyze(args):
    """Does the image rating predict real unlocks? Join VLM scores to engagement,
    report rank-correlation, train a weighted P(unlock) model, write a sheet."""
    import csv as _csv
    from pathlib import Path
    from src.engagement import (
        load_features_from_detail, spearman, _Logistic, FEATURE_KEYS,
        fairness_by_segment, EngagementModel,
    )
    feats = load_features_from_detail(args.scores)
    print(f"VLM features loaded for {len(feats)} photos\n")

    joined = []
    for r in _csv.DictReader(open(args.engagement)):
        stem = Path((r.get("enhanced_photo_url") or r.get("photo_url") or "").split("?")[0]).stem
        f = feats.get(stem)
        if f is None:
            continue
        imp = float(r.get("impressions") or 0); unl = float(r.get("unlocks") or 0)
        rate = (unl / imp) if imp else 0.0
        joined.append({"photo_id": r.get("photo_id", stem), "stem": stem,
                       "community": r.get("community"), "features": f,
                       "impressions": imp, "unlocks": unl, "rate": rate,
                       "vlm_appeal": f[0]})
    if len(joined) < 5:
        print(f"Only {len(joined)} photos joined — wait for scoring to finish "
              "(output/score_eng.log) or check id matching.")
        return
    print(f"Joined {len(joined)} photos (scored ∩ engagement).\n")

    # Headline: does the photo score rank-correlate with the real unlock rate?
    appeal = [j["vlm_appeal"] for j in joined]; rate = [j["rate"] for j in joined]
    print("=== DOES THE IMAGE RATING PREDICT UNLOCKS? ===")
    print(f"  Spearman(vlm_appeal, unlock_rate) = {spearman(appeal, rate):+.3f}")
    print("  (+1 perfect, 0 none, -1 inverted)\n")
    print("Per-attribute correlation with unlock_rate (what actually drives unlocks):")
    for i, k in enumerate(FEATURE_KEYS):
        c = spearman([j["features"][i] for j in joined], rate)
        print(f"   {k:30} {c:+.3f}")

    # Weighted P(unlock) model: target=rate, weight=impressions (no row blow-up).
    X = [j["features"] for j in joined]
    m = _Logistic().fit(X, rate, weights=[j["impressions"] for j in joined])
    pred = list(m.predict_proba(X))
    print(f"\nModel: Spearman(predicted P(unlock), actual rate) = {spearman(pred, rate):+.3f}")
    # Persist the trained production model for reuse by `rank`.
    em = EngagementModel(); em._m = m; em.trained = True
    em.save(OUT / "unlock_model.json")
    print(f"Trained model saved -> {OUT / 'unlock_model.json'}")

    # Output sheet, sorted by predicted P(unlock).
    for j, p in zip(joined, pred):
        j["pred"] = round(float(p), 5)
    joined.sort(key=lambda j: -j["pred"])
    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["rank", "photo_id", "community", "vlm_appeal"]
                   + FEATURE_KEYS[1:] + ["impressions", "unlocks",
                   "actual_unlock_rate", "predicted_p_unlock"])
        for i, j in enumerate(joined, 1):
            w.writerow([i, j["photo_id"], j["community"], round(j["vlm_appeal"], 4)]
                       + [round(x, 4) for x in j["features"][1:]]
                       + [int(j["impressions"]), int(j["unlocks"]),
                          round(j["rate"] * 1000, 3), j["pred"]])
    print(f"\nOutput sheet -> {out}")
    fr = fairness_by_segment([{"p_unlock": j["pred"], "segment": j["community"]} for j in joined])
    print("Fairness (mean predicted P(unlock) by community): " + json.dumps(fr))


def cmd_engage(args):
    """Train P(unlock) on REAL engagement data (joined to VLM scores) and report
    whether the scoring effectively predicts unlocks."""
    from src.engagement import (
        EngagementModel, rank_feed, fairness_by_segment, auc,
        load_features_from_detail, build_examples,
    )
    feats = load_features_from_detail(args.scores)
    print(f"VLM features loaded for {len(feats)} photos from {args.scores}")
    ex = build_examples(args.data, feats)
    if not ex:
        print("No examples — check that photo ids in your engagement CSV match "
              "the 'image' column in the scores CSV.")
        return
    pos = sum(e["unlocked"] for e in ex)
    print(f"Training examples: {len(ex)}  (unlocked={pos}, not={len(ex)-pos})\n")

    # Held-out split by photo so we test generalisation, not memorisation.
    photos = sorted({e["photo_id"] for e in ex})
    k = max(1, len(photos) // 5)
    test_ids = set(photos[:k])
    tr = [e for e in ex if e["photo_id"] not in test_ids]
    te = [e for e in ex if e["photo_id"] in test_ids]
    model = EngagementModel().train(tr)

    if te:
        s = [model.p_unlock(e["features"]) for e in te]
        y = [e["unlocked"] for e in te]
        a = auc(s, y)
        print(f"Effectiveness: P(unlock) AUC on held-out photos = {a:.3f}")
        print("  (>0.5 = scoring predicts unlocks; 0.5 = no signal; <0.5 = inverted)")
    print("\nWhat drives unlocks (feature weights):")
    for kf, w in sorted(model.feature_weights().items(), key=lambda kv: -abs(kv[1]))[:6]:
        print(f"   {kf:30} {w:+.3f}")

    # Rank the photos by predicted P(unlock) — the actual feed order.
    uniq = {}
    for e in ex:
        uniq[e["photo_id"]] = {"profile_id": e["photo_id"], "features": e["features"],
                               "impressions": 0, "segment": e.get("segment")}
    ranked = rank_feed(list(uniq.values()), model, explore=0.0)
    print("\nTop 5 profiles the model would surface (highest P(unlock)):")
    for p in ranked[:5]:
        print(f"   {p['profile_id']:20} P(unlock)={p['p_unlock']:.3f}")
    print("Bottom 3:")
    for p in ranked[-3:]:
        print(f"   {p['profile_id']:20} P(unlock)={p['p_unlock']:.3f}")
    fr = fairness_by_segment(ranked)
    if fr:
        print("\nFairness guard (mean P(unlock) by segment):")
        print("  " + json.dumps(fr))


def cmd_engage_demo(args):
    """Demo Layer 2: train P(unlock) on synthetic engagement, rank, show fairness."""
    import numpy as np
    from src.engagement import (
        EngagementModel, featurize, rank_feed, fairness_by_segment, auc,
    )
    rng = np.random.RandomState(0)
    log = []
    for _ in range(800):
        appeal = rng.uniform(0, 1); eye = float(rng.uniform(0, 1) > 0.5)
        light = rng.uniform(0, 1); seg = rng.choice(["tone_light", "tone_deep"])
        attrs = {"expression": rng.uniform(0, 1), "eye_contact": eye,
                 "lighting_on_face": light, "grooming_neatness": rng.uniform(0, 1),
                 "face_orientation_visibility": rng.uniform(0, 1), "centering": rng.uniform(0, 1),
                 "pose_camera_angle": rng.uniform(0, 1),
                 "background_foreground_separation": rng.uniform(0, 1), "attire": rng.uniform(0, 1)}
        logit = 3*appeal + 1.5*eye + 1.2*light - 3 + rng.normal(0, .4)
        unlocked = int(1/(1+np.exp(-logit)) > rng.uniform(0, 1))
        log.append({"features": featurize({"vlm_appeal": appeal, "attributes": attrs}),
                    "unlocked": unlocked, "segment": seg, "vlm_appeal": appeal})

    print("\n=== Layer 2: engagement model demo (synthetic data) ===\n")
    tr, te = log[:600], log[600:]
    model = EngagementModel().train(tr)
    s = [model.p_unlock(e["features"]) for e in te]; y = [e["unlocked"] for e in te]
    print(f"Model quality: P(unlock) AUC on held-out = {auc(s, y):.3f}  (0.5=random)")
    print("Feature weights (what drives unlocks):")
    for k, w in sorted(model.feature_weights().items(), key=lambda kv: -abs(kv[1]))[:5]:
        print(f"   {k:30} {w:+.3f}")
    feed = [{"profile_id": f"p{i}", "features": e["features"],
             "impressions": int(rng.choice([0, 50, 500])), "segment": e["segment"]}
            for i, e in enumerate(te[:8])]
    ranked = rank_feed(feed, model, explore=0.1)
    print("\nRanked feed (by predicted P(unlock), cold-start gets exploration lift):")
    for p in ranked[:6]:
        print(f"   {p['profile_id']:5} P(unlock)={p['p_unlock']:.3f} "
              f"rank={p['rank_score']:.3f} impressions={p['impressions']:>3} {p['segment']}")
    print("\nFairness guard (mean P(unlock) by tone segment):")
    print("  " + json.dumps(fairness_by_segment(ranked)))
    print("  (light vs deep should be CLOSE; _max_gap large => investigate.)\n")


def cmd_fetch(args):
    """Download images from a Sheet (API by tab name, URL, or CSV) + auto-label."""
    from src.fetch import (
        extract_urls_from_csv, extract_urls_from_text, fetch_sheet_text,
        read_tab_via_api, parse_sheet_url, download_images, append_labels,
        label_value,
    )

    # Build a list of (source_text, label_as) jobs so each source gets its own
    # label (approved tab -> 1.0, rejected tab -> 0.0).
    jobs = []
    if args.approved_tab or args.rejected_tab:
        if not (args.service_account and args.sheet_url):
            print("--approved-tab/--rejected-tab need --service-account and --sheet-url")
            return
        sid, _ = parse_sheet_url(args.sheet_url)
        if args.approved_tab:
            jobs.append((read_tab_via_api(sid, args.approved_tab, args.service_account), "approved"))
        if args.rejected_tab:
            jobs.append((read_tab_via_api(sid, args.rejected_tab, args.service_account), "rejected"))
    elif args.sheet_url:
        jobs.append((fetch_sheet_text(args.sheet_url), args.label_as))
    for csv_path in args.csv:
        jobs.append((Path(csv_path).read_text(errors="ignore"), args.label_as))

    if not jobs:
        print("Nothing to fetch. Pass CSV file(s), --sheet-url, or --approved-tab/--rejected-tab.")
        return

    label_pairs, total = [], 0
    for text, label_as in jobs:
        urls = list(dict.fromkeys(extract_urls_from_text(text)))
        tag = label_as or "source"
        print(f"  {tag}: {len(urls)} image URL(s) -> downloading")
        results = download_images(urls, args.out)
        total += len(results)
        if label_as:
            val = label_value(label_as)
            label_pairs += [(i, val) for _, i in results]

    print(f"\nGot {total} image(s) -> {args.out}")
    if label_pairs:
        labels = append_labels(label_pairs, args.labels)
        print(f"Wrote {len(labels)} labels (approved=1.0 / rejected=0.0) -> {args.labels}")
        print(f"\nNext:  python run.py compare {args.out}/* --mode real --truth {args.labels}")
    else:
        print(f"\nNext:  python run.py compare {args.out}/* --mode real")


def cmd_check_oss(args):
    """Verify the open-source VLM endpoint (free or paid) in one command."""
    import os
    from PIL import Image
    from src import config_loader

    cfg = config_loader.load_config()
    fb = (cfg.get("fallback", {}) or {}).get("vlm", {}) or {}
    base = os.environ.get("OSS_VLM_BASE_URL")
    key = os.environ.get("OSS_VLM_API_KEY")
    model = args.model or os.environ.get("OSS_VLM_MODEL") or fb.get("model")

    print("\n=== Open-source VLM connectivity check ===")
    print(f"  endpoint : {base or '(unset)'}")
    print(f"  model    : {model}")
    print(f"  api key  : {'set (' + key[:6] + '…)' if key else '(unset)'}\n")
    if not base or not key:
        print("✗ Set OSS_VLM_BASE_URL and OSS_VLM_API_KEY first. Examples:")
        print("    Groq (free):  export OSS_VLM_BASE_URL=https://api.groq.com/openai/v1")
        print("    HF router:    export OSS_VLM_BASE_URL=https://router.huggingface.co/v1")
        return

    specs = make_samples.generate(ROOT / "sample_images")
    clean = next(s for s in specs if s["id"] == "img_clean")
    img = Image.open(clean["path"]).convert("RGB")
    try:
        from src.vlm_opensource import OpenSourceAdapter
        out = OpenSourceAdapter(model=model).aesthetic_raw(img)
        print(f"✓ SUCCESS — model answered: {json.dumps(out)[:160]}")
        print("\nYou're connected. Now run:")
        print("  python run.py compare sample_images/*.jpg --mode real")
    except Exception as e:
        print(f"✗ FAILED — {type(e).__name__}: {e}\n")
        print("  Common causes: wrong base URL, model name not valid for THIS")
        print("  provider, no billing/credits, or the token lacks inference scope.")


def cmd_label(args):
    """Interactively rate images 1-5 -> labels.csv (for `compare --truth`)."""
    from src.labeler import discover_images, label_images, open_in_viewer

    images = discover_images(args.images)
    if not images:
        print("No images found. Pass image files or a directory.")
        return
    out_csv = Path(args.out)
    print(f"\nLabeling {len(images)} image(s). Each opens in your viewer.")
    print("Rate 1 (worst) - 5 (best).  [s]=skip  [q]=save & quit\n")

    def ask(image_id, path):
        while True:
            raw = input(f"  {image_id}: rate 1-5 (s/q) > ").strip().lower()
            if raw in ("q", "quit"):
                return "quit"
            if raw in ("s", "skip", ""):
                return "skip"
            if raw in ("1", "2", "3", "4", "5"):
                return int(raw)
            print("    please enter 1-5, or s / q")

    show = (lambda p: None) if args.no_view else open_in_viewer
    labels = label_images(images, out_csv, show=show, ask=ask)
    print(f"\nSaved {len(labels)} label(s) to: {out_csv}")
    print(f"Now run:  python run.py compare {' '.join(p for p, _ in images)} "
          f"--mode real --truth {out_csv}")


def main():
    ap = argparse.ArgumentParser(description="Image Rating System v3")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="run synthetic batch through full pipeline")
    d.add_argument("--mode", choices=["simulation", "gemini"], default="simulation")
    d.set_defaults(func=cmd_demo)

    s = sub.add_parser("score", help="score a single image file")
    s.add_argument("path")
    s.add_argument("--gender", choices=["female", "male"], default="female")
    s.add_argument("--mode", choices=["simulation", "gemini"], default="simulation")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare",
                       help="benchmark sheet: Gemini vs Gemini+Grok vs Gemini+OpenSource")
    c.add_argument("images", nargs="*", help="image paths (default: synthetic samples)")
    c.add_argument("--urls", help="manifest CSV (image,url[,truth]) to STREAM from "
                                  "the sheet links — nothing saved to disk")
    c.add_argument("--gender", choices=["female", "male"], default="female")
    c.add_argument("--mode", choices=["simulation", "real"], default="simulation")
    c.add_argument("--truth", help="labels CSV (image,truth) to rank engines by "
                                   "agreement with human judgement")
    c.add_argument("--no-prefilter", action="store_true",
                   help="skip the Stage-0 gate so EVERY image is VLM-scored")
    c.add_argument("--no-blocking", action="store_true",
                   help="skip the blocking gate (e.g. GenderMismatch on a mixed-gender "
                        "set) so flags don't zero the continuous quality score")
    c.add_argument("--limit-per-class", type=int,
                   help="with --truth: score at most N images per label (cheap pilot)")
    c.add_argument("--out", help="summary CSV output path (default output/comparison.csv)")
    c.add_argument("--detail-out", help="detailed CSV output path (per-param + reasoning)")
    c.set_defaults(func=cmd_compare)

    fe = sub.add_parser("fetch", help="download images from a Sheet (URL or CSV) + auto-label")
    fe.add_argument("csv", nargs="*", help="CSV file(s) exported from Google Sheets")
    fe.add_argument("--sheet-url", help="Google Sheets URL (link-shared Viewer for direct "
                                        "read, or any URL of the sheet when using --service-account)")
    fe.add_argument("--service-account", help="path to service-account JSON (Sheets API; "
                                              "works on PRIVATE sheets shared with the SA email)")
    fe.add_argument("--approved-tab", help="tab name of approved/good photos (-> truth 1.0)")
    fe.add_argument("--rejected-tab", help="tab name of rejected/bad photos (-> truth 0.0)")
    fe.add_argument("--out", default="real_images", help="folder to save images into")
    fe.add_argument("--label-as", help="'approved' (->1.0), 'rejected' (->0.0), or a float")
    fe.add_argument("--labels", default=str(OUT / "labels.csv"), help="labels CSV to write")
    fe.set_defaults(func=cmd_fetch)

    ed = sub.add_parser("engage-demo", help="demo the P(unlock) engagement model (Layer 2)")
    ed.set_defaults(func=cmd_engage_demo)

    an = sub.add_parser("analyze", help="does the image rating predict real unlocks? (rate-based)")
    an.add_argument("--engagement", default=str(OUT / "engagement.csv"),
                    help="engagement CSV with impressions,unlocks per photo")
    an.add_argument("--scores", default=str(OUT / "comparison_detailed.csv"),
                    help="VLM scores CSV (from compare)")
    an.add_argument("--out", default=str(OUT / "image_rating_vs_unlocks.csv"),
                    help="output sheet path")
    an.set_defaults(func=cmd_analyze)

    op = sub.add_parser("optimize", help="optimize Gemini-only scoring vs a real quality label (held-out)")
    op.add_argument("--labels", default=str(OUT / "quality_labels.csv"),
                    help="quality labels CSV (image,truth) from `run.py label`")
    op.add_argument("--scores", default=str(OUT / "comparison_beast_100_detailed.csv"),
                    help="Gemini scores CSV (per-attribute) to optimize over")
    op.set_defaults(func=cmd_optimize)

    rk = sub.add_parser("rank", help="rank a feed with the trained P(unlock) model")
    rk.add_argument("--model", default=str(OUT / "unlock_model.json"))
    rk.add_argument("--scores", default=str(OUT / "comparison_detailed.csv"))
    rk.set_defaults(func=cmd_rank)

    eg = sub.add_parser("engage", help="train P(unlock) on REAL engagement data + report")
    eg.add_argument("--data", required=True,
                    help="engagement CSV: photo_id,impressions,unlocks[,segment]")
    eg.add_argument("--scores", default=str(OUT / "comparison_detailed.csv"),
                    help="VLM scores CSV (from `compare`) to use as features")
    eg.set_defaults(func=cmd_engage)

    ck = sub.add_parser("check-oss", help="smoke-test the open-source VLM endpoint")
    ck.add_argument("--model", help="override model id (else OSS_VLM_MODEL env / weights.yaml)")
    ck.set_defaults(func=cmd_check_oss)

    l = sub.add_parser("label", help="rate images 1-5 -> labels.csv for compare --truth")
    l.add_argument("images", nargs="+", help="image files or a directory")
    l.add_argument("--out", default=str(OUT / "labels.csv"), help="output CSV path")
    l.add_argument("--no-view", action="store_true",
                   help="don't open images in a viewer (label from filenames only)")
    l.set_defaults(func=cmd_label)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

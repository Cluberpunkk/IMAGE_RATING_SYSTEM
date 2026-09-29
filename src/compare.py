"""
compare.py — head-to-head benchmark of aesthetic engines
========================================================
Produces ONE sheet comparing the final feed score for each image under three
configurations, with Gemini holding the 16 qualitative attributes CONSTANT and
only the AESTHETIC / photogenic engine swapped:

    image | Gemini fully | Gemini + Grok | Gemini + Open-Source (Qwen2.5-VL)

This is the original PRD's two-engine split (qualitative engine + aesthetic
engine), turned into an apples-to-apples A/B/C: same qualitative scores, same
blend weights, same prefilter — the ONLY variable is which model produced the
aesthetic term. With the female blend at 0.8 aesthetic, that term dominates, so
the sheet is a real measurement of "which aesthetic engine ranks better."

Modes:
  simulation  — three distinct deterministic engines, no credentials (proves the
                sheet + lets you eyeball the format).
  real        — Gemini (GEMINI_API_KEY) qualitative + aesthetic; Grok aesthetic
                (GROK_API_KEY, xAI); Qwen2.5-VL aesthetic (OSS_VLM_* / OpenRouter).
                A provider with no key is reported as 'no-key', never crashes.

Output: a printed table AND output/comparison.csv (open in Sheets/Excel).
"""

from __future__ import annotations

import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from . import config_loader
from .blender import blend
from .fusion import attrs_flags_from_raw, photogenic_from_raw
from .prefilter import run_prefilter, load_image
from .schemas import ScoreStatus

# Column order = the sheet the user asked for.
PROVIDERS = ["gemini", "grok", "opensource"]
COLUMN_LABEL = {
    "gemini": "Gemini fully",
    "grok": "Gemini + Grok",
    "opensource": "Gemini + OpenSource",
}


# --------------------------------------------------------------------------- #
# Simulation engines — three distinct "personalities" so the demo sheet shows
# real divergence (each engine weights brightness/contrast/seed differently,
# the way different real VLMs disagree). Deterministic per image.
# --------------------------------------------------------------------------- #
class _SimAesthetic:
    def __init__(self, name, bias, salt):
        self.name, self._bias, self._salt = name, bias, salt

    def aesthetic_raw(self, img):
        import numpy as np
        from .vlm_simulation import _seed, _clip, image_pixel_hash
        arr = np.asarray(img.convert("RGB")).astype(np.float32)
        gray = arr.dot(np.array([0.299, 0.587, 0.114], dtype=np.float32))
        brightness, contrast = gray.mean() / 255.0, gray.std() / 128.0
        h = image_pixel_hash(img)
        val = _clip(self._bias + brightness * 0.25 + contrast * 0.15
                    + _seed(h, self._salt) * 0.4)
        return {"photogenic": val, "reasoning": f"[sim:{self.name}]"}


def _build(mode: str, cfg: dict):
    """Return (qualitative_adapter, {provider: aesthetic_adapter | None})."""
    if mode == "simulation":
        from .vlm_simulation import SimulationAdapter, image_pixel_hash
        qual = SimulationAdapter(img_hash_of=image_pixel_hash)
        aes = {
            "gemini": _SimAesthetic("gemini", 0.30, "aes"),
            "grok": _SimAesthetic("grok", 0.38, "grok"),      # runs a touch hot
            "opensource": _SimAesthetic("qwen", 0.26, "qwen"),  # a touch conservative
        }
        return qual, aes

    # real mode
    from .vlm_gemini import GeminiAdapter
    from .vlm_opensource import OpenSourceAdapter
    qual = _safe(lambda: GeminiAdapter())
    fb = (cfg.get("fallback", {}) or {}).get("vlm", {}) or {}
    aes = {
        "gemini": _safe(lambda: GeminiAdapter()),
        "grok": _safe(lambda: OpenSourceAdapter(
            # GROK_MODEL env wins; default is a current multimodal Grok.
            model=os.environ.get("GROK_MODEL", "grok-4.3"),
            base_url_env="GROK_BASE_URL", api_key_env="GROK_API_KEY")),
        "opensource": _safe(lambda: OpenSourceAdapter(
            # OSS_VLM_MODEL env wins, so you can swap free<->paid models without
            # editing any file: `export OSS_VLM_MODEL=...`.
            model=os.environ.get("OSS_VLM_MODEL") or fb.get("model", "Qwen/Qwen2.5-VL-72B-Instruct"),
            base_url_env=fb.get("base_url_env", "OSS_VLM_BASE_URL"),
            api_key_env=fb.get("api_key_env", "OSS_VLM_API_KEY"))),
    }
    return qual, aes


def _safe(factory):
    try:
        return factory()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
def compare_images(images: list, gender: str = "female",
                   mode: str = "simulation",
                   csv_path: Optional[Path] = None,
                   truth: Optional[dict] = None,
                   no_prefilter: bool = False,
                   no_blocking: bool = False,
                   detail_csv_path: Optional[Path] = None,
                   details_out: Optional[list] = None) -> list:
    """Score each image under all three configs. images = [(bytes, id), ...]
    (a list OR a lazy generator — supports URL streaming with nothing on disk).

    Writes the summary sheet (csv_path) and, if detail_csv_path is given, a rich
    sheet with every qualitative parameter, each engine's aesthetic + final +
    reasoning, and the per-image winner. `details_out` (if a list) is filled
    with the same detail dicts for programmatic use.
    `no_prefilter=True` skips the Stage-0 gate so every image is VLM-scored.
    """
    from .blender import qualitative_score, soft_penalty
    import sys
    truth = truth or {}
    cfg = config_loader.load_config()
    qual_adapter, aes_adapters = _build(mode, cfg)
    rows, details = [], []

    # Incremental writers: each photo's row is flushed to disk AS IT'S SCORED,
    # so a hang/kill never loses prior work and progress is visible live.
    sum_f = open(Path(csv_path), "w", newline="") if csv_path else None
    det_f = open(Path(detail_csv_path), "w", newline="") if detail_csv_path else None
    sum_w = csv.writer(sum_f) if sum_f else None
    det_w = csv.writer(det_f) if det_f else None
    if sum_w:
        sum_w.writerow(_summary_header()); sum_f.flush()
    if det_w:
        det_w.writerow(_detail_header()); det_f.flush()

    for img_bytes, image_id in images:
        if no_prefilter:
            img = load_image(img_bytes)                # score everything
        else:
            pf, img = run_prefilter(img_bytes, cfg)
            if not pf.passed:
                r = {"image": image_id, "status": "PREFILTERED",
                     **{p: None for p in PROVIDERS}, "best": "-",
                     "reason": ", ".join(pf.reasons), "truth": truth.get(image_id)}
                rows.append(r)
                if sum_w:
                    sum_w.writerow(_summary_row(r)); sum_f.flush()
                continue

        # Qualitative once (Gemini), held constant across providers.
        try:
            qual_raw = qual_adapter.qualitative_raw(img, gender) if qual_adapter else {}
        except Exception:
            qual_raw = {}
        attrs, flags = attrs_flags_from_raw(qual_raw)
        q = round(qualitative_score(attrs, cfg), 4)

        # Aesthetic per provider, in parallel. Capture score + reasoning.
        def _score(provider):
            ad = aes_adapters.get(provider)
            if ad is None:
                return provider, {"final": None, "aesthetic": None, "reasoning": ""}
            try:
                raw = ad.aesthetic_raw(img)
            except Exception:
                return provider, {"final": None, "aesthetic": None, "reasoning": ""}
            aes = photogenic_from_raw(raw)
            if no_blocking:
                # Pure quality score — skip the blocking gate so a mis-fired flag
                # (e.g. GenderMismatch on a mixed-gender set) can't zero photos.
                bw = cfg["blend_weights"][gender]
                final = bw["aesthetic"] * aes + bw["qualitative"] * q
                final = round(max(0.0, min(1.0, final - soft_penalty(flags, cfg))), 4)
            else:
                _, _, final = blend(aes, attrs, flags, gender, cfg)
            return provider, {"final": final, "aesthetic": round(aes, 4),
                              "reasoning": str(raw.get("reasoning") or "")[:300]}

        with ThreadPoolExecutor(max_workers=len(PROVIDERS)) as ex:
            eng = dict(ex.map(_score, PROVIDERS))

        finals = {p: eng[p]["final"] for p in PROVIDERS}
        scored = {p: v for p, v in finals.items() if v is not None}
        best = max(scored, key=scored.get) if scored else "-"
        best_label = COLUMN_LABEL.get(best, best)
        r = {"image": image_id, "status": "SCORED", **finals,
             "best": best_label, "reason": "", "truth": truth.get(image_id)}
        d = {"image": image_id, "truth": truth.get(image_id),
             "attributes": attrs.as_dict(), "flags": flags.as_dict(),
             "qualitative_score": q, "engines": eng, "best": best_label}
        rows.append(r); details.append(d)
        if sum_w:
            sum_w.writerow(_summary_row(r)); sum_f.flush()
        if det_w:
            det_w.writerow(_detail_row(d)); det_f.flush()
        if len(rows) % 10 == 0:                       # live progress
            print(f"  ...scored {len(rows)} photos", file=sys.stderr, flush=True)

    if sum_f:
        sum_f.close()
    if det_f:
        det_f.close()
    if details_out is not None:
        details_out.extend(details)
    return rows


# --------------------------------------------------------------------------- #
# Ground truth: rank-correlate each engine against human labels so the sheet
# says which engine is BETTER (matches human judgement), not just HIGHER.
# --------------------------------------------------------------------------- #
def load_truth(path: Path) -> dict:
    """Read a labels CSV -> {image_id: truth_value}. Format: two columns
    `image,truth` (header optional). truth may be a score or a rank — Spearman
    is rank-based, so either works."""
    truth = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            key, val = row[0].strip(), row[1].strip()
            if not key or key.lower() in ("image", "image_id", "id"):
                continue  # header
            try:
                truth[key] = float(val)
            except ValueError:
                continue
    return truth


def _rankdata(xs: list) -> list:
    """Average-rank of each element (ties share the mean rank), 1-based."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(a: list, b: list) -> float:
    """Spearman rank correlation = Pearson on ranks. NaN if <2 pts or no variance."""
    if len(a) < 2:
        return float("nan")
    ra, rb = np.asarray(_rankdata(a)), np.asarray(_rankdata(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (ra.std() * rb.std()))


def _auc(scores: list, labels: list) -> float:
    """Mann-Whitney AUC: P(a positive scores above a negative). labels are
    1=positive(approved/good), 0=negative(rejected/bad). 0.5=coin flip, 1.0=perfect."""
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    rank_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def engine_correlations(rows: list, truth: dict) -> dict:
    """Per-engine agreement with the labels, over SCORED images that have one.

    Always reports Spearman ρ. If the labels are BINARY (e.g. approved=1.0 /
    rejected=0.0), also reports AUC and mean separation (good-minus-bad), and
    picks the winner by AUC — the right metric for 'separates good from bad'.
    """
    all_truth = sorted({truth[r["image"]] for r in rows if r["image"] in truth})
    binary = len(all_truth) == 2
    hi = all_truth[-1] if all_truth else None

    summary = {"binary": binary}
    for p in PROVIDERS:
        pairs = [(r[p], truth[r["image"]]) for r in rows
                 if r["status"] == "SCORED" and r[p] is not None and r["image"] in truth]
        if len(pairs) < 2:
            summary[p] = {"n": len(pairs), "spearman": float("nan"),
                          "auc": float("nan"), "separation": float("nan")}
            continue
        eng, tru = [list(x) for x in zip(*pairs)]
        rec = {"n": len(pairs), "spearman": round(_spearman(eng, tru), 4),
               "auc": float("nan"), "separation": float("nan")}
        if binary:
            labels = [1 if t == hi else 0 for t in tru]
            rec["auc"] = round(_auc(eng, labels), 4)
            good = [s for s, l in zip(eng, labels) if l == 1]
            bad = [s for s, l in zip(eng, labels) if l == 0]
            rec["separation"] = round(float(np.mean(good) - np.mean(bad)), 4)
        summary[p] = rec

    metric = "auc" if binary else "spearman"
    ranked = [(p, summary[p][metric]) for p in PROVIDERS
              if not np.isnan(summary[p][metric])]
    summary["best"] = (max(ranked, key=lambda kv: kv[1])[0] if ranked else None)
    summary["metric"] = metric
    return summary


def format_correlation(summary: dict) -> str:
    if summary.get("binary"):
        lines = ["Ground-truth agreement (labels are good/bad):",
                 "  AUC = P(good photo scores above bad) — 0.5 coin-flip, 1.0 perfect."]
        for p in PROVIDERS:
            s = summary[p]
            auc = "n/a" if np.isnan(s["auc"]) else f"{s['auc']:.4f}"
            sep = "n/a" if np.isnan(s["separation"]) else f"{s['separation']:+.4f}"
            lines.append(f"  {COLUMN_LABEL[p]:<22} AUC={auc}  separation={sep}  (n={s['n']})")
    else:
        lines = ["Ground-truth agreement (Spearman ρ — higher = better):"]
        for p in PROVIDERS:
            s = summary[p]
            rho = "n/a" if np.isnan(s["spearman"]) else f"{s['spearman']:+.4f}"
            lines.append(f"  {COLUMN_LABEL[p]:<22} ρ={rho}  (n={s['n']})")
    if summary["best"]:
        lines.append(f"  --> BEST at separating good from bad: {COLUMN_LABEL[summary['best']]}")
    return "\n".join(lines)


def _summary_header():
    return (["Image"] + [COLUMN_LABEL[p] for p in PROVIDERS]
            + ["Truth", "Best engine", "Status/Reason"])


def _summary_row(r: dict):
    return ([r["image"]]
            + [("" if r[p] is None else r[p]) for p in PROVIDERS]
            + [("" if r.get("truth") is None else r["truth"]), r["best"],
               r["reason"] or r["status"]])


def _detail_header():
    from .fusion import QUAL_GRADED_KEYS, QUAL_FLAG_KEYS
    return (["image", "truth"]
            + [f"q:{k}" for k in QUAL_GRADED_KEYS]
            + [f"flag:{k}" for k in QUAL_FLAG_KEYS]
            + ["qualitative_score"]
            + [f"aesthetic:{COLUMN_LABEL[p]}" for p in PROVIDERS]
            + [f"FINAL:{COLUMN_LABEL[p]}" for p in PROVIDERS]
            + ["BEST_engine"]
            + [f"reasoning:{COLUMN_LABEL[p]}" for p in PROVIDERS])


def _detail_row(d: dict):
    from .fusion import QUAL_GRADED_KEYS, QUAL_FLAG_KEYS, _GRADED_FIELD, _FLAG_FIELD
    a, fl, e = d["attributes"], d["flags"], d["engines"]
    return ([d["image"], "" if d["truth"] is None else d["truth"]]
            + [a.get(_GRADED_FIELD[k]) for k in QUAL_GRADED_KEYS]
            + [fl.get(_FLAG_FIELD[k]) for k in QUAL_FLAG_KEYS]
            + [d["qualitative_score"]]
            + [("" if e[p]["aesthetic"] is None else e[p]["aesthetic"]) for p in PROVIDERS]
            + [("" if e[p]["final"] is None else e[p]["final"]) for p in PROVIDERS]
            + [d["best"]]
            + [e[p]["reasoning"] for p in PROVIDERS])


def _write_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_summary_header())
        for r in rows:
            w.writerow(_summary_row(r))


def _write_detail_csv(details: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_detail_header())
        for d in details:
            w.writerow(_detail_row(d))


def format_table(rows: list) -> str:
    """Pretty fixed-width table for the console."""
    head = f"{'image':<16}{'Gemini fully':>14}{'Gemini+Grok':>14}{'Gemini+OSS':>14}   best"
    lines = [head, "-" * len(head)]
    for r in rows:
        def cell(p):
            v = r[p]
            return "PREFILT" if r["status"] != "SCORED" else (
                "no-key" if v is None else f"{v:.4f}")
        lines.append(
            f"{r['image']:<16}{cell('gemini'):>14}{cell('grok'):>14}"
            f"{cell('opensource'):>14}   {r['best']}")
    return "\n".join(lines)

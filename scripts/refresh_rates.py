#!/usr/bin/env python3
"""
refresh_rates.py — pull the 3 engagement rates (paid/unpaid) for a set of profiles
==================================================================================
Live via lokal-lens (BigQuery ds=39). Computes, per profile, split by the viewer's
`is_paid` user-property (paid / unpaid / unknown):

  a) tap_unlock_rate   = distinct users who fired `tap_unlock`         / unique impressions
  b) unlock_rate       = distinct users who fired `contact_unlocked`   / unique impressions
  c) profile_view_rate = distinct users who fired `viewed_screen_profile` / unique impressions

unique impressions = COUNT(DISTINCT user_pseudo_id) on `card_impression`.

Use this to (re)populate the engagement columns for the 100 sheet profiles once
they have accrued real traffic (they were created 2026-07-02; give them 1-2 weeks).

Usage:
  python scripts/refresh_rates.py IDS_FILE START_YYYYMMDD END_YYYYMMDD [OUT_CSV]
    IDS_FILE : text file, comma- or newline-separated profile_ids
  e.g. python scripts/refresh_rates.py ids.txt 20260702 20260716 output/rates_100.csv

NOTE: live engagement currently lives ONLY in `events_intraday_*` (the daily
`events_*` export is stale at 2024). Keep END within the intraday retention window.
"""
import csv
import os
import sys
from pathlib import Path

LOKAL = Path("/Users/mohdshamshulqamar/Downloads/lokal-lens")
EVENTS = ("card_impression", "tap_unlock", "contact_unlocked", "viewed_screen_profile")


def _adapter():
    for line in (LOKAL / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    sys.path.insert(0, str(LOKAL))
    from lokal_lens.redash_adapter import RedashAdapter
    return RedashAdapter(
        base_url=os.environ.get("REDASH_URL") or os.environ.get("REDASH_BASE_URL"),
        api_key=os.environ.get("REDASH_API_KEY") or os.environ.get("REDASH_API_TOKEN"),
        query_ids_path=str(LOKAL / "config" / "query_ids.yaml"))


def build_sql(ids, start, end):
    inlist = ",".join("'%s'" % str(i).strip() for i in ids)
    return f"""
WITH ev AS (
  SELECT event_name, user_pseudo_id,
    COALESCE(CAST((SELECT ep.value.int_value FROM UNNEST(event_params) ep WHERE ep.key='profile_id') AS STRING),
             (SELECT ep.value.string_value FROM UNNEST(event_params) ep WHERE ep.key='profile_id')) AS pid,
    CASE WHEN (SELECT up.value.string_value FROM UNNEST(user_properties) up WHERE up.key='is_paid')='true'  THEN 'paid'
         WHEN (SELECT up.value.string_value FROM UNNEST(user_properties) up WHERE up.key='is_paid')='false' THEN 'unpaid'
         ELSE 'unknown' END AS seg
  FROM `analytics_440871731.events_intraday_*`
  WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}'
    AND event_name IN {EVENTS}
)
SELECT pid,
  CASE seg WHEN 'paid' THEN 'paid' WHEN 'unpaid' THEN 'unpaid' ELSE 'unknown' END AS seg,
  COUNT(DISTINCT IF(event_name='card_impression',      user_pseudo_id, NULL)) AS uniq_impr,
  COUNT(DISTINCT IF(event_name='tap_unlock',           user_pseudo_id, NULL)) AS tap_unlock_u,
  COUNT(DISTINCT IF(event_name='contact_unlocked',     user_pseudo_id, NULL)) AS unlock_u,
  COUNT(DISTINCT IF(event_name='viewed_screen_profile',user_pseudo_id, NULL)) AS profview_u
FROM ev WHERE pid IN ({inlist}) GROUP BY pid, seg
"""


def rate(n, d):
    return round(n / d, 5) if d else 0.0


def main():
    a = sys.argv
    if len(a) < 4:
        print(__doc__)
        return
    ids = [x for x in Path(a[1]).read_text().replace("\n", ",").split(",") if x.strip()]
    start, end = a[2], a[3]
    out = a[4] if len(a) > 4 else "output/rates.csv"
    ad = _adapter()
    print(f"pulling rates for {len(ids)} profiles, {start}..{end} ...")
    rows = ad.run_raw_sql(build_sql(ids, start, end), 39).get("rows", [])
    from collections import defaultdict
    prof = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        for k in ("uniq_impr", "tap_unlock_u", "unlock_u", "profview_u"):
            prof[r["pid"]][r["seg"]][k] = r[k]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["profile_id"]
    for seg in ("all", "paid", "unpaid"):
        cols += [f"impr_{seg}", f"tap_unlock_rate_{seg}", f"unlock_rate_{seg}", f"profile_view_rate_{seg}"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for pid in ids:
            segs = prof.get(pid, {})
            def g(s, k):
                if s == "all":
                    return sum(segs.get(x, {}).get(k, 0) for x in ("paid", "unpaid", "unknown"))
                return segs.get(s, {}).get(k, 0)
            row = {"profile_id": pid}
            for seg in ("all", "paid", "unpaid"):
                imp = g(seg, "uniq_impr")
                row[f"impr_{seg}"] = imp
                row[f"tap_unlock_rate_{seg}"] = rate(g(seg, "tap_unlock_u"), imp)
                row[f"unlock_rate_{seg}"] = rate(g(seg, "unlock_u"), imp)
                row[f"profile_view_rate_{seg}"] = rate(g(seg, "profview_u"), imp)
            w.writerow(row)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

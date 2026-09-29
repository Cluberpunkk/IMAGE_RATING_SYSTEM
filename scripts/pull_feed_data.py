#!/usr/bin/env python3
"""
pull_feed_data.py — reproduce & SCALE the enhanced-photo feed-performance dataset
=================================================================================
Runs the exact BigQuery join (ds=39) behind enhanced_photo_feed_performance.csv,
live via lokal-lens — NO Redash query id needed, NO Claude Code reload.

Joins:
  GA4 card_impression (impressions/profile)  +  profiles_profileimage (enhanced
  photo)  +  profiles_profileunlock (unlocks)  +  profiles_info (community/gender/
  age/qualification). Writes output/feed_data.csv ready for scoring + the model.

Usage:
  python scripts/pull_feed_data.py [START_YYYYMMDD] [END_YYYYMMDD] [MIN_IMPRESSIONS] [LIMIT]
  default: 20260427 20260626 2000 1000
"""
import csv
import os
import sys
from pathlib import Path

LOKAL = Path("/Users/mohdshamshulqamar/Downloads/lokal-lens")
OUT = Path(__file__).resolve().parent.parent / "output" / "feed_data.csv"


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


def build_sql(start, end, min_impr, limit):
    d1 = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    d2 = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    return f"""
WITH impr AS (
  SELECT CAST((SELECT ep.value.int_value FROM UNNEST(event_params) ep WHERE ep.key='profile_id') AS STRING) AS pid,
         COUNT(*) AS impressions
  FROM `analytics_440871731.events_intraday_*`
  WHERE _TABLE_SUFFIX BETWEEN '{start}' AND '{end}' AND event_name='card_impression'
  GROUP BY pid ),
img AS (
  SELECT CAST(profile_id AS STRING) AS pid, ANY_VALUE(enhanced_image) AS enhanced_image
  FROM `lokal-matrimony.matrimony_pg_datastreampublic.profiles_profileimage`
  WHERE enhancement_status=4 AND is_display_image=TRUE AND enhanced_image IS NOT NULL
  GROUP BY pid ),
unl AS (
  SELECT CAST(unlocked_profile AS STRING) AS pid, COUNT(*) AS unlocks
  FROM `lokal-matrimony.matrimony_pg_datastreampublic.profiles_profileunlock`
  WHERE DATE(created_at + INTERVAL 330 MINUTE) BETWEEN '{d1}' AND '{d2}'
  GROUP BY pid ),
info AS (
  SELECT CAST(profile_id AS STRING) AS pid, locale AS community, gender, age, qualification
  FROM `lokal-matrimony.ref_tables.mrityunjay_mehta_profiles_info` )
SELECT i.pid AS profile_id, img.enhanced_image AS enhanced_photo_url,
       info.community, info.gender, info.age, info.qualification,
       i.impressions, COALESCE(u.unlocks,0) AS unlocks,
       ROUND(COALESCE(u.unlocks,0)/i.impressions*1000, 3) AS unlock_rate_per_1000
FROM impr i JOIN img USING(pid)
LEFT JOIN unl u USING(pid) LEFT JOIN info USING(pid)
WHERE i.impressions >= {min_impr}
ORDER BY i.impressions DESC LIMIT {limit}
"""


def main():
    a = sys.argv
    start = a[1] if len(a) > 1 else "20260427"
    end = a[2] if len(a) > 2 else "20260626"
    min_impr = int(a[3]) if len(a) > 3 else 2000
    limit = int(a[4]) if len(a) > 4 else 1000
    ad = _adapter()
    print(f"pulling feed data {start}..{end}, min_impr={min_impr}, limit={limit} ...")
    res = ad.run_raw_sql(build_sql(start, end, min_impr, limit), 39)
    rows = res.get("rows", [])
    print(f"got {len(rows)} rows (latency {res.get('latency_ms')}ms)")
    if not rows:
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ["profile_id", "enhanced_photo_url", "community", "gender", "age",
            "qualification", "impressions", "unlocks", "unlock_rate_per_1000"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

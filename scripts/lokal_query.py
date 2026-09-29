#!/usr/bin/env python3
"""
lokal_query.py — pull live metrics from lokal-lens (Redash) WITHOUT the MCP/reload
==================================================================================
Uses the lokal-lens Python library directly (same data layer the MCP server uses),
so it works in this Claude Code session with no restart. Loads Redash creds from
lokal-lens/.env.

Usage:
  python scripts/lokal_query.py list
  python scripts/lokal_query.py <metric> [start_date] [end_date]
  e.g. python scripts/lokal_query.py profile_unlocks 2026-05-01 2026-05-31
"""
import json
import os
import sys
from pathlib import Path

LOKAL = Path("/Users/mohdshamshulqamar/Downloads/lokal-lens")


def _load_env():
    env = LOKAL / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    sys.path.insert(0, str(LOKAL))


def _adapter():
    from lokal_lens.redash_adapter import RedashAdapter
    return RedashAdapter(
        base_url=os.environ.get("REDASH_URL") or os.environ.get("REDASH_BASE_URL"),
        api_key=os.environ.get("REDASH_API_KEY") or os.environ.get("REDASH_API_TOKEN"),
        query_ids_path=str(LOKAL / "config" / "query_ids.yaml"),
    )


def main():
    _load_env()
    ad = _adapter()
    if len(sys.argv) < 2 or sys.argv[1] == "list":
        print(json.dumps(list(ad.query_map.keys()), indent=2))
        return
    metric = sys.argv[1]
    params = {}
    if len(sys.argv) >= 4:
        params = {"start_date": sys.argv[2], "end_date": sys.argv[3]}
    rows = ad.run_metric(metric, params)
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()

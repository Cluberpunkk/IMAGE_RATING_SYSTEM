"""
fetch.py — pull real images from a Google-Sheet CSV export + auto-label
=======================================================================
Your sheet's image cells are CDN URLs (matrimony-cdn.getlokalapp.com/.../*.webp).
Export a tab to CSV (File -> Download -> CSV in Sheets), and this scans every
cell for image URLs, downloads them locally, and (optionally) writes ground-truth
labels straight from which tab they came from:

    post_implementation_approved  -> truth 1.0  (a good photo)
    post_implementation_rejected  -> truth 0.0  (a bad photo)

That turns your existing review decisions into the exact labels.csv that
`compare --truth` needs — no manual re-rating. Stdlib only (urllib), so no new
dependency; images are de-duped and cached (re-runs skip already-downloaded).
"""

from __future__ import annotations

import csv
import re
import urllib.parse
import urllib.request
from pathlib import Path

# Match http(s) URLs ending in a common image extension, inside any CSV cell.
IMG_URL_RE = re.compile(r'https?://[^\s,"\'<>)]+?\.(?:webp|jpe?g|png)', re.IGNORECASE)

_LABEL_MAP = {"approved": 1.0, "rejected": 0.0}


def extract_urls_from_text(text: str) -> list:
    """Return de-duped image URLs found anywhere in text (order preserved)."""
    seen, out = set(), []
    for u in IMG_URL_RE.findall(text):
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def extract_urls_from_csv(path) -> list:
    """Return de-duped image URLs found anywhere in a CSV file."""
    return extract_urls_from_text(Path(path).read_text(errors="ignore"))


# --- Read a Google Sheet directly (no manual export) ----------------------- #
_SHEET_ID_RE = re.compile(r"/spreadsheets/d/([A-Za-z0-9_-]+)")
_GID_RE = re.compile(r"[#&?]gid=(\d+)")


def parse_sheet_url(url: str) -> tuple:
    """Pull (spreadsheet_id, gid) out of a Google Sheets URL."""
    m = _SHEET_ID_RE.search(url)
    if not m:
        raise ValueError("not a Google Sheets URL")
    g = _GID_RE.search(url)
    return m.group(1), (g.group(1) if g else "0")


def sheet_csv_url(spreadsheet_id: str, gid: str) -> str:
    """The CSV-export endpoint for one tab (works if link-shared 'Viewer')."""
    return (f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            f"/export?format=csv&gid={gid}")


def read_tab_via_api(spreadsheet_id: str, tab_name: str, service_account_json: str) -> str:
    """Read one tab BY NAME using the Google Sheets API + a service account.
    Works on PRIVATE sheets (the account just needs Viewer access). Returns the
    tab flattened to CSV-ish text so the URL extractor can scan it."""
    try:
        import gspread  # optional dependency for the real Sheets API path
    except ImportError as e:
        raise RuntimeError(
            "Sheets API path needs gspread: `pip install gspread`."
        ) from e
    gc = gspread.service_account(filename=service_account_json)
    ws = gc.open_by_key(spreadsheet_id).worksheet(tab_name)
    return "\n".join(",".join(row) for row in ws.get_all_values())


def fetch_sheet_text(url: str, timeout: int = 20) -> str:
    """Download one Sheet tab as CSV text via the export URL. Raises a helpful
    error if the sheet is private (Google returns a login page, not CSV)."""
    sid, gid = parse_sheet_url(url)
    req = urllib.request.Request(sheet_csv_url(sid, gid),
                                 headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        ctype = r.headers.get("Content-Type", "")
        data = r.read().decode("utf-8", "ignore")
    if ("text/html" in ctype or data.lstrip()[:15].lower().startswith("<!doctype html")
            or "accounts.google.com" in data):
        raise RuntimeError(
            "got a login page, not CSV — the sheet is private. Either share it "
            "'Anyone with the link: Viewer', or use a service account (see README).")
    return data


def url_to_id(url: str) -> str:
    """CDN url -> stable image id = filename stem (e.g. enhanced_3502017)."""
    return Path(urllib.parse.urlparse(url).path).stem


def download_images(urls: list, out_dir, timeout: int = 20) -> list:
    """Download each URL to out_dir. Returns [(local_path, image_id), ...].
    Skips URLs already on disk; logs and skips failures (never raises)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for u in urls:
        image_id = url_to_id(u)
        ext = Path(urllib.parse.urlparse(u).path).suffix or ".img"
        dest = out / (image_id + ext)
        if dest.exists() and dest.stat().st_size > 0:
            results.append((str(dest), image_id))
            continue
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                dest.write_bytes(r.read())
            results.append((str(dest), image_id))
        except Exception as e:  # noqa: BLE001 - one bad URL never stops the batch
            print(f"  skip {u}  ({type(e).__name__}: {e})")
    return results


def stream_bytes(url_or_path: str, timeout: int = 30) -> bytes:
    """Fetch image bytes WITHOUT saving to disk. Local paths are read directly;
    URLs are fetched via urllib, falling back to curl (robust behind SSL
    proxies and on any machine)."""
    if not url_or_path.startswith(("http://", "https://")):
        return Path(url_or_path).read_bytes()
    try:
        req = urllib.request.Request(url_or_path, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        import subprocess
        r = subprocess.run(["curl", "-s", "--max-time", str(timeout), url_or_path],
                           capture_output=True)
        if r.returncode == 0 and r.stdout:
            return r.stdout
        raise


def read_url_manifest(path) -> list:
    """Read a manifest CSV (image,url[,truth]) -> [(image_id, url, truth_or_None)]."""
    out = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2 or row[0].strip().lower() in ("image", "image_id", "id"):
                continue
            iid, url = row[0].strip(), row[1].strip()
            truth = None
            if len(row) >= 3 and row[2].strip():
                try:
                    truth = float(row[2].strip())
                except ValueError:
                    truth = None
            out.append((iid, url, truth))
    return out


def stream_images(manifest: list):
    """Lazy generator: yield (bytes, image_id) for each (id, url) — nothing saved."""
    for iid, url, _truth in manifest:
        try:
            yield stream_bytes(url), iid
        except Exception as e:  # noqa: BLE001
            print(f"  skip {iid} ({type(e).__name__}: {e})")


def label_value(label_as: str) -> float:
    """'approved'->1.0, 'rejected'->0.0, or a literal float string."""
    if label_as in _LABEL_MAP:
        return _LABEL_MAP[label_as]
    return float(label_as)


def append_labels(pairs: list, labels_path) -> dict:
    """Merge [(image_id, truth), ...] into a labels CSV (image,truth)."""
    from .labeler import load_existing, _write
    labels = load_existing(labels_path)
    for image_id, val in pairs:
        labels[image_id] = val
    _write(labels, Path(labels_path))
    return labels

"""
labeler.py — collect human ground-truth ratings into labels.csv
===============================================================
A tiny, resumable labeling tool so you can build the ~50-100 image truth set the
benchmark needs. For each image it opens the file in your OS viewer, asks for a
1-5 star rating in the terminal, and appends a normalised 0-1 score to a CSV in
the exact `image,truth` format `run.py compare --truth` consumes.

Design for trust:
  * RESUMABLE — already-labeled images are skipped; writes after EVERY rating so
    a crash or quit never loses work.
  * Injectable `show` (viewer) and `ask` (prompt) callables -> fully unit-tested
    without a real screen or keyboard.
  * 1-5 stars are stored as (stars-1)/4 -> {0, .25, .5, .75, 1.0}, same 0-1 scale
    as the engine scores (Spearman is rank-based, so the mapping is monotonic and
    correlation is unaffected — this just keeps the sheet tidy).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def normalize_rating(stars: int, scale: int = 5) -> float:
    """1..scale -> 0.0..1.0 (linear, monotonic)."""
    stars = max(1, min(scale, int(stars)))
    return round((stars - 1) / (scale - 1), 4)


def load_existing(path: Path) -> dict:
    """Read an existing labels CSV -> {image_id: truth}. Empty if absent."""
    out: dict = {}
    if not Path(path).exists():
        return out
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            k, v = row[0].strip(), row[1].strip()
            if not k or k.lower() in ("image", "image_id", "id"):
                continue
            try:
                out[k] = float(v)
            except ValueError:
                continue
    return out


def _write(labels: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "truth"])
        for k, v in labels.items():
            w.writerow([k, v])


def open_in_viewer(path: str) -> None:
    """Open an image in the OS default viewer (best-effort, never fatal)."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", path], check=False)
        elif os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
    except Exception:
        pass  # labeling can continue from the filename alone


def discover_images(paths: list) -> list:
    """Expand files/dirs into [(path_str, image_id), ...] sorted, de-duped."""
    found: list = []
    seen = set()
    for p in paths:
        p = Path(p)
        candidates = sorted(p.iterdir()) if p.is_dir() else [p]
        for c in candidates:
            if c.suffix.lower() in IMAGE_EXTS and c.stem not in seen:
                seen.add(c.stem)
                found.append((str(c), c.stem))
    return found


def label_images(
    images: list,
    out_csv: Path,
    show: Optional[Callable[[str], None]] = open_in_viewer,
    ask: Callable[[str, str], object] = None,
    scale: int = 5,
) -> dict:
    """Drive the labeling loop. `ask(image_id, path)` returns an int rating, the
    string 'skip', or the string 'quit'. Returns the full labels dict.

    Writes after every accepted rating, so the run is resumable and crash-safe.
    """
    if ask is None:
        raise ValueError("label_images needs an `ask` callable")
    labels = load_existing(out_csv)
    todo = [(p, i) for (p, i) in images if i not in labels]
    for n, (path, image_id) in enumerate(todo, 1):
        if show:
            show(path)
        ans = ask(image_id, path)
        if ans == "quit":
            break
        if ans == "skip" or ans is None:
            continue
        labels[image_id] = normalize_rating(int(ans), scale)
        _write(labels, out_csv)               # persist immediately
    return labels

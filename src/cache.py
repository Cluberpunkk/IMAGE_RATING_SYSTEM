"""
cache.py
========
Idempotency / score cache (fixes PRD bugs #6 and #9 together).

Key = sha256(image_bytes) + model_version + prompt_version + weight_version.

Consequences of this one design choice:
  * Determinism: identical image + identical versions -> identical cached score.
    No more feed wobble from re-running the same photo.
  * Cost: duplicate uploads (same bytes) are scored once, ever.
  * Re-scoring trigger: bump ANY version -> key misses -> image is re-scored.
    That is exactly the re-scoring workflow, for free.

This reference impl is a JSON file (swap for Redis/DynamoDB/Postgres in prod;
the interface is just get()/put()).
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

from .schemas import Versions


def image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


def cache_key(img_hash: str, versions: Versions) -> str:
    raw = f"{img_hash}|{versions.model_version}|{versions.prompt_version}|{versions.weight_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


class ScoreCache:
    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._store: dict = {}
        if self.path and self.path.exists():
            try:
                self._store = json.loads(self.path.read_text())
            except Exception:
                self._store = {}

    def get(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._store.get(key)

    def put(self, key: str, record: dict) -> None:
        with self._lock:
            self._store[key] = record
            if self.path:
                self.path.write_text(json.dumps(self._store, indent=2))

    def stats(self) -> dict:
        return {"entries": len(self._store)}

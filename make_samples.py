"""
make_samples.py
===============
Generates synthetic test images that exercise every pipeline branch without
using real people's photos (privacy-safe demo). Each image is crafted to hit a
specific path: a clean portrait, a blurry one, a too-dark one, a tiny one, etc.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def _portrait(bg, face_rgb, size=(600, 750), draw_face=True):
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    # textured background so contrast/bg-separation isn't degenerate
    for i in range(0, size[0], 40):
        d.line([(i, 0), (i, size[1])], fill=tuple(max(0, c - 12) for c in bg), width=1)
    if draw_face:
        cx, cy, rw, rh = size[0] // 2, int(size[1] * 0.42), 120, 150
        d.ellipse([cx - rw, cy - rh, cx + rw, cy + rh], fill=face_rgb)       # head
        d.ellipse([cx - 55, cy - 30, cx - 15, cy + 5], fill=(255, 255, 255)) # eyes
        d.ellipse([cx + 15, cy - 30, cx + 55, cy + 5], fill=(255, 255, 255))
        d.ellipse([cx - 45, cy - 22, cx - 27, cy - 4], fill=(40, 40, 40))
        d.ellipse([cx + 27, cy - 22, cx + 45, cy - 4], fill=(40, 40, 40))
        d.arc([cx - 60, cy + 20, cx + 60, cy + 90], 20, 160, fill=(60, 30, 30), width=6)
        d.rectangle([cx - 150, cy + rh, cx + 150, size[1]], fill=(70, 90, 140))  # torso
    return img


def generate(out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = []

    # 1. Clean, well-lit portrait (should SCORE high-ish)
    img = _portrait((180, 200, 220), (225, 195, 170))
    p = out_dir / "01_clean_portrait.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_clean", "path": str(p), "gender": "female",
                  "segment": "tone_light", "note": "clean well-lit portrait"})

    # 2. Darker-skin clean portrait (must NOT be penalised for tone)
    img = _portrait((150, 170, 190), (120, 85, 60))
    p = out_dir / "02_clean_portrait_deeptone.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_deeptone", "path": str(p), "gender": "female",
                  "segment": "tone_deep", "note": "deeper skin tone, well lit"})

    # 3. Blurry (should be PREFILTERED: too_blurry)
    img = _portrait((180, 200, 220), (225, 195, 170)).filter(ImageFilter.GaussianBlur(8))
    p = out_dir / "03_blurry.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_blurry", "path": str(p), "gender": "female",
                  "segment": "tone_light", "note": "out of focus"})

    # 4. Too dark (should be PREFILTERED: too_dark)
    arr = np.asarray(_portrait((20, 22, 26), (40, 35, 30))).astype(np.float32) * 0.5
    img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    p = out_dir / "04_too_dark.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_dark", "path": str(p), "gender": "male",
                  "segment": "tone_deep", "note": "severely underexposed"})

    # 5. Tiny / low-res (should be PREFILTERED: too_small)
    img = _portrait((180, 200, 220), (225, 195, 170)).resize((180, 220))
    p = out_dir / "05_tiny.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_tiny", "path": str(p), "gender": "male",
                  "segment": "tone_light", "note": "resolution too low"})

    # 6. Male clean portrait (SCORE via male blend weights)
    img = _portrait((160, 185, 175), (210, 180, 150))
    p = out_dir / "06_male_portrait.jpg"; img.save(p, quality=92)
    specs.append({"id": "img_male", "path": str(p), "gender": "male",
                  "segment": "tone_light", "note": "clean male portrait"})

    # 7. Duplicate of #1 bytes (demonstrates cache hit / dedup)
    p2 = out_dir / "07_duplicate_of_01.jpg"
    p2.write_bytes((out_dir / "01_clean_portrait.jpg").read_bytes())
    specs.append({"id": "img_dupe", "path": str(p2), "gender": "female",
                  "segment": "tone_light", "note": "identical bytes to img_clean"})

    return specs

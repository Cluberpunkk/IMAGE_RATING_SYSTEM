"""
config_loader.py
================
Loads weights.yaml, normalises qualitative weights, and computes a stable
prompt_version fingerprint from the prompt files. The fingerprint is what
makes re-scoring deterministic: change a prompt -> hash changes -> records
with the old hash are known-stale.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "weights.yaml"
PROMPTS_DIR = ROOT / "prompts"

PIPELINE_VERSION = "pipeline-1.0.0"

# Gender-specific fragments injected into the qualitative prompt template.
_GENDER_FRAGMENTS = {
    "female": {
        "GROOMING_EXTRA": "",
        "GROOMING_EXTRA_POS": "",
        # Cultural attire context so traditional Indian wear is scored on the
        # SAME footing as Western formal — mirrors the specificity the male
        # rubric already has, and prevents under-scoring saree/lehenga/etc.
        "ATTIRE_EXTRA_LOW": ", nightwear/housecoat",
        "ATTIRE_EXTRA_MID": " (simple kurti, plain top)",
        "ATTIRE_EXTRA_OK": " (salwar kameez, churidar, everyday saree)",
        "ATTIRE_EXTRA_HIGH": " (silk saree, lehenga, festive/designer wear, formal Western)",
        "WRONG_GENDER": "male-presenting",
    },
    "male": {
        "GROOMING_EXTRA": ", unkempt beard/hair",
        "GROOMING_EXTRA_POS": ", clean shave or well-groomed beard",
        "ATTIRE_EXTRA_LOW": ", vest/banyan",
        "ATTIRE_EXTRA_MID": " (basic T-shirt, casual shirt)",
        "ATTIRE_EXTRA_OK": " (collared shirt, kurta)",
        "ATTIRE_EXTRA_HIGH": " (suit, sherwani, blazer, dress shirt)",
        "WRONG_GENDER": "female-presenting",
    },
}


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    # Normalise qualitative weights so they always sum to 1.0.
    qw = cfg["qualitative_weights"]
    total = sum(qw.values())
    if total <= 0:
        raise ValueError("qualitative_weights must sum to > 0")
    cfg["qualitative_weights"] = {k: v / total for k, v in qw.items()}

    # Validate each blend pair sums to 1.0 (within tolerance); renormalise.
    for gender, pair in cfg["blend_weights"].items():
        s = pair["aesthetic"] + pair["qualitative"]
        if abs(s - 1.0) > 1e-6:
            cfg["blend_weights"][gender] = {
                "aesthetic": pair["aesthetic"] / s,
                "qualitative": pair["qualitative"] / s,
            }
    return cfg


def prompt_version() -> str:
    """Hash the prompt files AND the gender fragments -> short stable tag.

    The fragments live in code, not in the .txt files, but they materially
    change the prompt the model sees. Hashing them too means editing a fragment
    bumps the version -> stale scores are correctly re-scored (PRD scope D).
    Hashing ONLY the .txt files (the obvious version) would silently skip that.
    """
    h = hashlib.sha256()
    for name in sorted(os.listdir(PROMPTS_DIR)):
        if name.endswith(".txt"):
            h.update((PROMPTS_DIR / name).read_bytes())
    # Canonical, order-stable serialisation of the fragments.
    import json
    h.update(json.dumps(_GENDER_FRAGMENTS, sort_keys=True).encode())
    return "prompt-" + h.hexdigest()[:10]


def load_qualitative_prompt(gender: str) -> str:
    template = (PROMPTS_DIR / "qualitative_base.txt").read_text()
    frags = _GENDER_FRAGMENTS[gender]
    for key, val in frags.items():
        template = template.replace("{" + key + "}", val)
    return template


def load_aesthetic_prompt() -> str:
    return (PROMPTS_DIR / "aesthetic_neutral.txt").read_text()

"""
vlm_opensource.py — the open-source FALLBACK brain (Qwen2.5-VL by default)
==========================================================================
When Gemini cannot score one or more parameters, the fusion layer asks the best
open-source VLM to fill exactly those. Qwen2.5-VL-72B-Instruct is the strongest
open vision-language model for structured visual judgement, and it speaks the
OpenAI-compatible chat API — so this one adapter works through OpenRouter, Groq,
Together, Fireworks, or your OWN vLLM/TGI box, just by pointing the base URL.

Deliberately NOT Grok: Grok was only ever used for the "sexual magnetism" score
we removed; the open-source path here scores the same neutral, defensible
parameters as Gemini, behind the same prompts.

Determinism: temperature=0 + a JSON-only instruction. Same retry + circuit
breaker as the Gemini path. Import-safe: needs `pip install openai` and the
endpoint env vars only when actually used.

Config (weights.yaml -> fallback.vlm):
  enabled: true
  model: "Qwen/Qwen2.5-VL-72B-Instruct"
  base_url_env: OSS_VLM_BASE_URL     # e.g. https://openrouter.ai/api/v1
  api_key_env:  OSS_VLM_API_KEY
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

from PIL import Image

from .config_loader import load_qualitative_prompt, load_aesthetic_prompt
from .resilience import CircuitBreaker, retry_call

_JSON_INSTRUCTION = (
    "\n\nReturn ONLY a single valid JSON object, no markdown, no commentary."
)


def _img_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _extract_json(text: str) -> dict:
    """Open models sometimes wrap JSON in prose/fences — recover the object."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise ValueError("no JSON object in open-source VLM response")
        return json.loads(m.group(0))


class OpenSourceAdapter:
    """Fusion adapter for an OpenAI-compatible vision endpoint."""

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        base_url_env: str = "OSS_VLM_BASE_URL",
        api_key_env: str = "OSS_VLM_API_KEY",
        breaker: "CircuitBreaker | None" = None,
        retry_cfg: dict | None = None,
    ):
        self.name = model.split("/")[-1].lower()   # e.g. "qwen2.5-vl-72b-instruct"
        self.model = model
        self._base_url_env = base_url_env
        self._api_key_env = api_key_env
        self._client = None
        self._breaker = breaker or CircuitBreaker(name="oss-vlm")
        self._retry = retry_cfg or {}

    def _c(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(
                    "openai client not installed. `pip install openai` to enable "
                    "the open-source VLM fallback, or disable fallback.vlm."
                ) from e
            base_url = os.environ.get(self._base_url_env)
            api_key = os.environ.get(self._api_key_env)
            if not base_url or not api_key:
                raise RuntimeError(
                    f"open-source VLM needs {self._base_url_env} and "
                    f"{self._api_key_env} set."
                )
            self._client = OpenAI(base_url=base_url, api_key=api_key)
        return self._client

    def _chat_json(self, prompt: str, img: Image.Image) -> dict:
        def _call():
            resp = self._c().chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + _JSON_INSTRUCTION},
                        {"type": "image_url", "image_url": {"url": _img_data_url(img)}},
                    ],
                }],
            )
            return _extract_json(resp.choices[0].message.content)

        return retry_call(
            _call,
            max_attempts=self._retry.get("max_attempts", 3),
            base_delay=self._retry.get("base_delay", 0.5),
            max_delay=self._retry.get("max_delay", 8.0),
            jitter=self._retry.get("jitter", 0.25),
            breaker=self._breaker,
        )

    def qualitative_raw(self, img: Image.Image, gender: str) -> dict:
        return self._chat_json(load_qualitative_prompt(gender), img)

    def aesthetic_raw(self, img: Image.Image) -> dict:
        return self._chat_json(load_aesthetic_prompt(), img)

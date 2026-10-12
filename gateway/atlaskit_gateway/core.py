"""Pure-Python core of the AtlasKit gateway.

Everything in this module is framework-free so it can be unit tested without a
GPU, a model download or a web server. The HTTP layer (``app.py``) is a thin
wrapper around these functions.
"""
from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Languages supported by N-ATLaS and its ASR models
# ---------------------------------------------------------------------------

LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "Nigerian English", "asr_model": "NCAIR1/NigerianAccentedEnglish"},
    "yo": {"name": "Yoruba", "asr_model": "NCAIR1/Yoruba-ASR"},
    "ha": {"name": "Hausa", "asr_model": "NCAIR1/Hausa-ASR"},
    "ig": {"name": "Igbo", "asr_model": "NCAIR1/Igbo-ASR"},
}

LANGUAGE_ALIASES = {
    "english": "en", "eng": "en", "en-ng": "en", "pidgin": "en",
    "yoruba": "yo", "yor": "yo", "yorùbá": "yo",
    "hausa": "ha", "hau": "ha",
    "igbo": "ig", "ibo": "ig",
}

DEFAULT_MODEL_ID = "NCAIR1/N-ATLaS"
MODEL_ALIASES = {"n-atlas", "n-atlas-8b", "natlas", "n-atlas-llm", DEFAULT_MODEL_ID.lower()}

DEFAULT_SYSTEM_PROMPT = (
    "you are a large language model trained by Awarri AI technologies. "
    "You are a friendly assistant and you are here to help."
)

# Generation defaults recommended on the N-ATLaS model card.
DEFAULT_GENERATION = {"temperature": 0.1, "repetition_penalty": 1.12, "max_tokens": 1000}


class AtlasError(ValueError):
    """Raised for invalid client input. ``status`` maps to an HTTP status code."""

    def __init__(self, message: str, status: int = 400, code: str = "invalid_request"):
        super().__init__(message)
        self.status = status
        self.code = code


def normalize_language(lang: str | None, *, required: bool = False) -> str | None:
    """Map user-supplied language names/codes to one of en/yo/ha/ig."""
    if lang is None or str(lang).strip() == "":
        if required:
            raise AtlasError("`language` is required (one of: en, yo, ha, ig)")
        return None
    key = str(lang).strip().lower()
    key = LANGUAGE_ALIASES.get(key, key)
    if key not in LANGUAGES:
        raise AtlasError(
            f"Unsupported language '{lang}'. Supported: "
            + ", ".join(f"{c} ({v['name']})" for c, v in LANGUAGES.items())
        )
    return key


# ---------------------------------------------------------------------------
# Lightweight language identification
# ---------------------------------------------------------------------------
# A fast, dependency-free heuristic used for routing (e.g. picking an ASR model
# or a reply language). It is not a replacement for a trained LID model; the
# eval harness measures its accuracy so users know where it stands.

_YO_MARKS = set("ẹọṣẸỌṢ")
_IG_MARKS = set("ịọụṅỊỌỤṄ")
_HA_MARKS = set("ɓɗƙƴƁƊƘƳ")

_STOPWORDS = {
    "yo": {"ni", "ti", "si", "awọn", "awon", "mo", "o", "wa", "ṣe", "se", "kini", "bawo",
           "jẹ", "je", "fun", "pẹlu", "pelu", "ati", "ko", "kò", "ẹ", "e", "ojo", "ọjọ", "ilé", "ile"},
    "ha": {"da", "ne", "ce", "na", "ba", "ya", "ta", "su", "shi", "ita", "wannan", "yaya",
           "menene", "kuma", "amma", "zan", "za", "ina", "kai", "mun", "sun", "domin", "don", "akwai"},
    "ig": {"na", "nke", "ya", "ka", "bụ", "bu", "m", "anyị", "anyi", "ha", "gị", "gi", "kedu",
           "ndị", "ndi", "maka", "ma", "ọ", "o", "nwere", "dị", "di", "ihe", "otu"},
    "en": {"the", "and", "is", "are", "what", "how", "to", "of", "in", "for", "you", "i",
           "my", "with", "this", "that", "please", "can", "do", "abeg", "dey", "wetin"},
}


def detect_language(text: str) -> dict[str, Any]:
    """Return ``{"language": code, "confidence": float, "scores": {...}}``."""
    if not text or not text.strip():
        raise AtlasError("`text` must be a non-empty string")
    scores = {code: 0.0 for code in LANGUAGES}
    chars = set(text)
    scores["ha"] += 3.0 * len(chars & _HA_MARKS)
    # ọ is shared by Yoruba and Igbo; ẹ/ṣ are Yoruba-only, ị/ụ/ṅ Igbo-only.
    scores["yo"] += 3.0 * len(chars & (_YO_MARKS - _IG_MARKS))
    scores["ig"] += 3.0 * len(chars & (_IG_MARKS - _YO_MARKS))

    # Tonal diacritics (grave/acute on vowels) are common in written Yoruba.
    decomposed = unicodedata.normalize("NFD", text)
    tone_marks = sum(1 for c in decomposed if c in "̀́")
    scores["yo"] += min(tone_marks, 6) * 0.5

    words = re.findall(r"[\w’']+", text.lower())
    for w in words:
        for code, sw in _STOPWORDS.items():
            if w in sw:
                scores[code] += 1.0
    total = sum(scores.values())
    if total == 0:
        return {"language": "en", "confidence": 0.0, "scores": scores}
    best = max(scores, key=scores.get)
    return {"language": best, "confidence": round(scores[best] / total, 3), "scores": scores}


# ---------------------------------------------------------------------------
# Chat request normalisation
# ---------------------------------------------------------------------------

def today_string(now: datetime | None = None) -> str:
    """Date string in the format the N-ATLaS chat template expects (``12 Oct 2026``)."""
    return (now or datetime.now(timezone.utc)).strftime("%d %b %Y")


def _validate_messages(messages: Any) -> list[dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise AtlasError("`messages` must be a non-empty list")
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) or m.get("role") not in {"system", "user", "assistant"}:
            raise AtlasError(f"messages[{i}] must have role system|user|assistant")
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts -> plain text
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        if not isinstance(content, str):
            raise AtlasError(f"messages[{i}].content must be a string")
        out.append({"role": m["role"], "content": content})
    return out


def language_instruction(lang: str) -> str:
    name = LANGUAGES[lang]["name"]
    return f"Always reply in {name}, even if the user writes in another language."


@dataclass
class ChatPlan:
    """The upstream request plus metadata the gateway logs."""
    upstream_body: dict[str, Any]
    stream: bool
    language: str | None
    requested_model: str
    extras: dict[str, Any] = field(default_factory=dict)


def build_chat_request(body: dict[str, Any], upstream_model: str = DEFAULT_MODEL_ID,
                       now: datetime | None = None, adapters: tuple[str, ...] = ()) -> ChatPlan:
    """Turn an OpenAI-style chat request into the request sent to vLLM.

    * accepts friendly model aliases (``n-atlas``) and the names of LoRA
      adapters fine-tuned on N-ATLaS (``adapters``, served by vLLM on top of
      the same base weights) and rejects every other model, so a deployment
      can never silently wrap a different foundation model;
    * injects the N-ATLaS default system prompt when none is supplied;
    * supports an ``atlas_language`` extension that pins the reply language;
    * passes ``date_string`` to the N-ATLaS chat template;
    * applies the model card's generation defaults unless overridden.
    """
    if not isinstance(body, dict):
        raise AtlasError("Request body must be a JSON object")
    requested = str(body.get("model") or "n-atlas")
    if requested in adapters:
        target = requested
    elif requested.lower() in MODEL_ALIASES | {upstream_model.lower()}:
        target = upstream_model
    else:
        options = ", ".join(["'n-atlas'", *(f"'{a}'" for a in adapters)])
        raise AtlasError(f"Model '{requested}' is not served here. Use {options}.", 404, "model_not_found")

    messages = _validate_messages(body.get("messages"))
    lang = normalize_language(body.get("atlas_language"))

    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
    if lang:
        messages[0] = {**messages[0], "content": messages[0]["content"].rstrip() + "\n" + language_instruction(lang)}

    upstream: dict[str, Any] = {"model": target, "messages": messages}
    for k, v in DEFAULT_GENERATION.items():
        upstream[k] = body.get(k, v)
    passthrough = ("top_p", "stop", "seed", "n", "presence_penalty", "frequency_penalty",
                   "stream_options", "user", "max_completion_tokens")
    for k in passthrough:
        if k in body:
            upstream[k] = body[k]
    if "max_completion_tokens" in upstream:
        upstream["max_tokens"] = upstream.pop("max_completion_tokens")

    for k in ("temperature", "repetition_penalty"):
        if not isinstance(upstream[k], (int, float)):
            raise AtlasError(f"`{k}` must be a number")
    if not isinstance(upstream["max_tokens"], int) or not (1 <= upstream["max_tokens"] <= 4096):
        raise AtlasError("`max_tokens` must be an integer between 1 and 4096")

    tmpl = dict(body.get("chat_template_kwargs") or {})
    tmpl.setdefault("date_string", today_string(now))
    upstream["chat_template_kwargs"] = tmpl

    stream = bool(body.get("stream", False))
    upstream["stream"] = stream
    if stream:
        upstream.setdefault("stream_options", {"include_usage": True})
    return ChatPlan(upstream, stream, lang, requested)


# ---------------------------------------------------------------------------
# Task prompts built on the same model (translation, summarisation)
# ---------------------------------------------------------------------------

def translation_messages(text: str, target: str, source: str | None = None) -> list[dict[str, str]]:
    if not text or not text.strip():
        raise AtlasError("`text` must be a non-empty string")
    tgt = normalize_language(target, required=True)
    src = normalize_language(source)
    src_name = LANGUAGES[src]["name"] if src else "the source language"
    tgt_name = LANGUAGES[tgt]["name"]
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Translate the following text from {src_name} to {tgt_name}. "
            f"Return only the translation, with no explanation.\n\n{text.strip()}")},
    ]


def summary_messages(text: str, language: str | None = None, max_words: int = 80) -> list[dict[str, str]]:
    if not text or not text.strip():
        raise AtlasError("`text` must be a non-empty string")
    lang = normalize_language(language)
    tail = f" Write the summary in {LANGUAGES[lang]['name']}." if lang else ""
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarise the following in at most {max_words} words.{tail}\n\n{text.strip()}"},
    ]


def clean_completion(text: str) -> str:
    """Strip Llama-3 special tokens that occasionally leak into decoded output."""
    text = re.sub(r"<\|(?:eot_id|end_of_text|start_header_id|end_header_id|begin_of_text)\|>", "", text or "")
    text = re.sub(r"^(assistant)\s*\n", "", text.strip())
    return text.strip()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    upstream_url: str = "http://127.0.0.1:8000/v1"
    upstream_model: str = DEFAULT_MODEL_ID
    api_keys: frozenset[str] = frozenset()
    enable_asr: bool = True
    asr_device: str = "auto"
    log_path: str = "./logs/usage.jsonl"
    cors_origins: tuple[str, ...] = ("*",)
    request_timeout: float = 120.0
    adapters: tuple[str, ...] = ()   # LoRA adapters on N-ATLaS that vLLM serves (--lora-modules)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Settings":
        e = os.environ if env is None else env
        keys = frozenset(k.strip() for k in e.get("ATLAS_API_KEYS", "").split(",") if k.strip())
        return cls(
            upstream_url=e.get("ATLAS_UPSTREAM_URL", cls.upstream_url).rstrip("/"),
            upstream_model=e.get("ATLAS_UPSTREAM_MODEL", cls.upstream_model),
            api_keys=keys,
            enable_asr=e.get("ATLAS_ENABLE_ASR", "1") not in {"0", "false", "no"},
            asr_device=e.get("ATLAS_ASR_DEVICE", "auto"),
            log_path=e.get("ATLAS_LOG_PATH", cls.log_path),
            cors_origins=tuple(o.strip() for o in e.get("ATLAS_CORS_ORIGINS", "*").split(",")),
            request_timeout=float(e.get("ATLAS_TIMEOUT", "120")),
            adapters=tuple(a.strip() for a in e.get("ATLAS_ADAPTERS", "").split(",") if a.strip()),
        )

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlaskit_gateway.core import (
    DEFAULT_SYSTEM_PROMPT, AtlasError, Settings, build_chat_request, clean_completion,
    detect_language, normalize_language, translation_messages,
)

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_language_aliases():
    assert normalize_language("Yoruba") == "yo"
    assert normalize_language("HAU") == "ha"
    assert normalize_language("ibo") == "ig"
    assert normalize_language(None) is None
    with pytest.raises(AtlasError):
        normalize_language("fr")
    with pytest.raises(AtlasError):
        normalize_language("", required=True)


def test_language_detection_on_smoke_set():
    items = [json.loads(l) for l in (ROOT / "eval/data/smoke.jsonl").read_text(encoding="utf-8").splitlines()]
    lid = [i for i in items if i["task"] == "lid"]
    correct = sum(detect_language(i["input"])["language"] == i["label"] for i in lid)
    assert correct == len(lid), [(i["input"], detect_language(i["input"])) for i in lid]


def test_build_chat_request_defaults():
    now = datetime(2026, 10, 12, tzinfo=timezone.utc)
    plan = build_chat_request({"model": "n-atlas", "messages": [{"role": "user", "content": "Sannu"}]}, now=now)
    body = plan.upstream_body
    assert body["model"] == "NCAIR1/N-ATLaS"
    assert body["messages"][0] == {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
    assert body["repetition_penalty"] == 1.12 and body["temperature"] == 0.1
    assert body["chat_template_kwargs"] == {"date_string": "12 Oct 2026"}
    assert body["stream"] is False and not plan.stream


def test_language_pin_appends_instruction_to_custom_system():
    plan = build_chat_request({"messages": [{"role": "system", "content": "You are a farm advisor."},
                                            {"role": "user", "content": "hi"}], "atlas_language": "hausa"})
    sys_msg = plan.upstream_body["messages"][0]["content"]
    assert sys_msg.startswith("You are a farm advisor.") and "Hausa" in sys_msg
    assert plan.language == "ha"


def test_rejects_other_models_and_bad_input():
    with pytest.raises(AtlasError) as e:
        build_chat_request({"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]})
    assert e.value.status == 404
    for bad in ({"messages": []}, {"messages": [{"role": "tool", "content": "x"}]},
                {"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}):
        with pytest.raises(AtlasError):
            build_chat_request(bad)


def test_openai_content_parts_and_streaming_flags():
    plan = build_chat_request({"stream": True, "max_completion_tokens": 50,
                               "messages": [{"role": "user", "content": [{"type": "text", "text": "Kedu"}]}]})
    b = plan.upstream_body
    assert b["messages"][-1]["content"] == "Kedu"
    assert b["max_tokens"] == 50 and "max_completion_tokens" not in b
    assert b["stream_options"] == {"include_usage": True}


def test_fine_tuned_adapters_are_the_only_other_models_allowed():
    msgs = [{"role": "user", "content": "x"}]
    plan = build_chat_request({"model": "natlas-mt", "messages": msgs}, adapters=("natlas-mt",))
    assert plan.upstream_body["model"] == "natlas-mt" and plan.requested_model == "natlas-mt"
    assert build_chat_request({"model": "n-atlas", "messages": msgs}, adapters=("natlas-mt",)).upstream_body["model"] == "NCAIR1/N-ATLaS"
    with pytest.raises(AtlasError) as e:
        build_chat_request({"model": "natlas-mt", "messages": msgs})
    assert e.value.status == 404
    assert Settings.from_env({"ATLAS_ADAPTERS": "natlas-mt, b"}).adapters == ("natlas-mt", "b")


def test_clean_completion_strips_special_tokens():
    assert clean_completion("assistant\nBawo ni<|eot_id|>") == "Bawo ni"


def test_translation_prompt():
    msgs = translation_messages("Good morning", "yo", "en")
    assert "from Nigerian English to Yoruba" in msgs[-1]["content"]


def test_settings_from_env():
    s = Settings.from_env({"ATLAS_API_KEYS": "a, b,", "ATLAS_ENABLE_ASR": "0",
                           "ATLAS_UPSTREAM_URL": "http://x:8000/v1/"})
    assert s.api_keys == frozenset({"a", "b"}) and not s.enable_asr and s.upstream_url == "http://x:8000/v1"

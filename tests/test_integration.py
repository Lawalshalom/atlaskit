"""End-to-end: mock vLLM <- real gateway (uvicorn) <- Python SDK, JS SDK and eval harness."""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from atlaskit import AtlasKit, AtlasKitError
from atlaskit_gateway.app import create_app
from atlaskit_gateway.core import Settings

import mock_vllm

ROOT = Path(__file__).resolve().parents[1]
KEY = "test-key"


class FakeTranscriber:
    def __init__(self):
        self.calls = []

    def loaded(self):
        return []

    def transcribe(self, audio, language):
        self.calls.append((len(audio), language))
        return {"text": "Bawo ni oja se n lo", "language": language, "duration": 2.5,
                "model": f"NCAIR1/{language}-ASR"}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def gateway(tmp_path_factory):
    srv, up_port = mock_vllm.start()
    log = tmp_path_factory.mktemp("logs") / "usage.jsonl"
    settings = Settings(upstream_url=f"http://127.0.0.1:{up_port}/v1", api_keys=frozenset({KEY}), adapters=("natlas-mt",),
                        enable_asr=False, log_path=str(log))
    asr = FakeTranscriber()
    app = create_app(settings, transcriber=asr)
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield {"url": f"http://127.0.0.1:{port}", "asr": asr, "log": log}
    server.should_exit = True
    t.join(5)
    srv.shutdown()


@pytest.fixture
def atlas(gateway):
    return AtlasKit(gateway["url"], api_key=KEY, max_retries=0)


def test_health_and_models(atlas):
    h = atlas.health()
    assert h["status"] == "ok" and h["upstream"] is True
    ids = [m["id"] for m in atlas.models()]
    assert ids[0] == "n-atlas" and "NCAIR1/Yoruba-ASR" in ids


def test_auth_required(gateway):
    with pytest.raises(AtlasKitError) as e:
        AtlasKit(gateway["url"], api_key="wrong", max_retries=0).chat("hi")
    assert e.value.status == 401


def test_chat_and_upstream_shape(atlas):
    r = atlas.chat("Sannu", language="ha")
    assert r.text == "echo: Sannu"           # special tokens stripped
    assert r.raw["model"] == "n-atlas"
    sent = mock_vllm.LAST_REQUEST
    assert sent["model"] == "NCAIR1/N-ATLaS" and "Hausa" in sent["messages"][0]["content"]
    assert "date_string" in sent["chat_template_kwargs"]


def test_stream(atlas):
    text = "".join(atlas.stream("Kedu ka i mere"))
    assert text.strip() == "echo: Kedu ka i mere"


def test_conversation_keeps_history(atlas):
    c = atlas.conversation(language="yo", system="You are a tutor.")
    c.say("first")
    c.say("second")
    roles = [m["role"] for m in mock_vllm.LAST_REQUEST["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_other_model_rejected(gateway):
    with pytest.raises(AtlasKitError) as e:
        AtlasKit(gateway["url"], api_key=KEY, max_retries=0).chat("x", model="gpt-4o")
    assert e.value.status == 404


def test_fine_tuned_adapter_is_routed(atlas):
    assert {"id": "natlas-mt", "parent": "n-atlas"}.items() <= next(m for m in atlas.models() if m["id"] == "natlas-mt").items()
    assert atlas.chat("Sannu", model="natlas-mt").raw["model"] == "natlas-mt"
    assert mock_vllm.LAST_REQUEST["model"] == "natlas-mt"
    atlas.translate("Good morning", "yo", "en", model="natlas-mt")
    assert mock_vllm.LAST_REQUEST["model"] == "natlas-mt"
    atlas.translate("Good morning", "yo", "en")
    assert mock_vllm.LAST_REQUEST["model"] == "NCAIR1/N-ATLaS"
    with pytest.raises(AtlasKitError) as e:
        atlas.chat("x", model="natlas-other")
    assert e.value.status == 404


def test_translate_detect_summarize(atlas):
    assert atlas.translate("Good morning", "yo", "en").startswith("echo:")
    assert atlas.detect_language("Ina kwana? Lafiya lau.")["language"] == "ha"
    assert atlas.summarize("long text", language="ig").startswith("echo:")


def test_transcribe_and_voice_chat(atlas, gateway, tmp_path):
    wav = tmp_path / "note.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    assert atlas.transcribe(wav, "yo")["text"] == "Bawo ni oja se n lo"
    vc = atlas.voice_chat(wav.read_bytes(), "yo")
    assert vc["transcript"] and vc["reply"] == "echo: Bawo ni oja se n lo"
    assert gateway["asr"].calls[-1][1] == "yo"
    with pytest.raises(AtlasKitError):
        atlas.transcribe(wav, "fr")


def test_feedback_and_stats(atlas, gateway):
    atlas.feedback(1, language="yo", tag="test")
    s = atlas.stats()
    assert s["feedback"]["positive"] >= 1 and s["total_requests"] > 5
    logged = gateway["log"].read_text(encoding="utf-8")
    assert "Sannu" not in logged  # no prompt text is ever logged


def test_openai_compatible_raw_http(gateway):
    import urllib.request
    req = urllib.request.Request(f"{gateway['url']}/v1/chat/completions", method="POST",
                                 data=json.dumps({"model": "n-atlas", "messages": [{"role": "user", "content": "hi"}]}).encode(),
                                 headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    data = json.loads(urllib.request.urlopen(req).read())
    assert data["object"] == "chat.completion" and data["choices"][0]["message"]["content"] == "echo: hi"


def test_eval_harness(gateway, tmp_path):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 100)
    data = tmp_path / "data.jsonl"
    data.write_text((ROOT / "eval/data/smoke.jsonl").read_text(encoding="utf-8") + json.dumps(
        {"task": "asr", "audio": str(wav), "language": "yo", "reference": "Bàwo ni ọjà ṣe ń lọ?"}) + "\n",
        encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "eval/run_eval.py"), "--data", str(data),
                          "--endpoint", f"mock={gateway['url']}", "--endpoint", f"tuned={gateway['url']}#natlas-mt",
                          "--api-key", KEY, "--out", str(tmp_path / "reports")],
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    report = json.loads(next((tmp_path / "reports").glob("*.json")).read_text(encoding="utf-8"))
    summ = report["endpoints"]["mock"]["summary"]
    assert summ["_overall"]["errors"] == 0
    assert summ["lid:yo"]["mean"] == 1.0 and summ["qa:en"]["mean"] > 0
    # FakeTranscriber says "Bawo ni oja se n lo": 5 of the 6 words differ only in tone marks
    assert summ["asr:yo"]["wer"] == 83.3 and summ["asr:yo"]["wer_no_diacritics"] == 0.0
    assert report["endpoints"]["tuned"]["model"] == "natlas-mt"
    assert report["endpoints"]["tuned"]["summary"]["_overall"]["errors"] == 0


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_sdk(gateway):
    env = {**os.environ, "ATLASKIT_BASE_URL": gateway["url"], "ATLASKIT_API_KEY": KEY}
    out = subprocess.run(["node", "--test", str(ROOT / "sdk/js/test/sdk.test.js")], capture_output=True, text=True, env=env, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr

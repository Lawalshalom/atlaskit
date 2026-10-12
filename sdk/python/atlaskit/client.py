"""Zero-dependency Python client for an AtlasKit gateway serving N-ATLaS."""
from __future__ import annotations

import json
import mimetypes
import os
import time
import uuid
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Union

AudioInput = Union[str, os.PathLike, bytes, BinaryIO]
LANGUAGES = {"en": "Nigerian English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}


class AtlasKitError(Exception):
    def __init__(self, message: str, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class ChatResponse:
    text: str
    finish_reason: str | None
    usage: dict[str, Any]
    raw: dict[str, Any] = field(repr=False)

    def __str__(self) -> str:
        return self.text


class AtlasKit:
    """Client for the AtlasKit gateway.

    >>> atlas = AtlasKit("https://<pod-id>-8080.proxy.runpod.net", api_key="...")
    >>> atlas.chat("Kedu ka i mere?", language="ig").text
    """

    def __init__(self, base_url: str | None = None, api_key: str | None = None,
                 timeout: float = 120.0, max_retries: int = 2):
        self.base_url = (base_url or os.environ.get("ATLASKIT_BASE_URL") or "http://localhost:8080").rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key = api_key or os.environ.get("ATLASKIT_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries

    # ------------------------------------------------------------------ http
    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"User-Agent": "atlaskit-python/0.1.0", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        h.update(extra or {})
        return h

    def _open(self, method: str, path: str, data: bytes | None = None,
              headers: dict[str, str] | None = None):
        url = f"{self.base_url}{path}"
        attempt = 0
        while True:
            req = urllib.request.Request(url, data=data, method=method, headers=self._headers(headers))
            try:
                return urllib.request.urlopen(req, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                try:
                    err = json.loads(body).get("error", {})
                    msg, code = err.get("message", body), err.get("code")
                except (json.JSONDecodeError, AttributeError):
                    msg, code = body or exc.reason, None
                if exc.code in (502, 503, 504) and attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                raise AtlasKitError(f"HTTP {exc.code}: {msg}", exc.code, code) from None
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    attempt += 1
                    time.sleep(0.5 * 2 ** attempt)
                    continue
                raise AtlasKitError(f"Cannot reach AtlasKit gateway at {self.base_url}: {exc.reason}") from None

    def _json(self, method: str, path: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else None
        with self._open(method, path, data, headers) as resp:
            return json.loads(resp.read().decode())

    def _multipart(self, path: str, audio: AudioInput, fields: dict[str, str]) -> dict:
        filename, content = _read_audio(audio)
        boundary = uuid.uuid4().hex
        parts = []
        for k, v in fields.items():
            if v is None:
                continue
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
        ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                      f"Content-Type: {ctype}\r\n\r\n").encode() + content + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        with self._open("POST", path, b"".join(parts),
                        {"Content-Type": f"multipart/form-data; boundary={boundary}"}) as resp:
            return json.loads(resp.read().decode())

    # ------------------------------------------------------------------ API
    def health(self) -> dict:
        try:
            return self._json("GET", "/health")
        except AtlasKitError as exc:
            if exc.status == 503:
                return {"status": "degraded"}
            raise

    def models(self) -> list[dict]:
        return self._json("GET", "/v1/models")["data"]

    def chat(self, messages: str | list[dict], *, language: str | None = None,
             system: str | None = None, temperature: float | None = None,
             max_tokens: int | None = None, **params: Any) -> ChatResponse:
        """Send a chat request. ``messages`` may be a plain string for single-turn use.

        ``language`` (en/yo/ha/ig) pins the reply language.
        """
        body = self._chat_body(messages, language, system, temperature, max_tokens, params)
        raw = self._json("POST", "/v1/chat/completions", body)
        choice = raw["choices"][0]
        return ChatResponse(choice["message"]["content"], choice.get("finish_reason"), raw.get("usage") or {}, raw)

    def stream(self, messages: str | list[dict], *, language: str | None = None,
               system: str | None = None, temperature: float | None = None,
               max_tokens: int | None = None, **params: Any) -> Iterator[str]:
        """Yield text deltas as N-ATLaS generates them."""
        body = self._chat_body(messages, language, system, temperature, max_tokens, params)
        body["stream"] = True
        resp = self._open("POST", "/v1/chat/completions", json.dumps(body).encode(),
                          {"Content-Type": "application/json", "Accept": "text/event-stream"})
        with resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        yield delta

    def transcribe(self, audio: AudioInput, language: str) -> dict:
        """Speech to text with the official N-ATLaS ASR model for ``language``."""
        return self._multipart("/v1/audio/transcriptions", audio, {"language": language})

    def voice_chat(self, audio: AudioInput, language: str, system: str | None = None) -> dict:
        """Audio question in, text answer in the same language out."""
        return self._multipart("/v1/atlas/voice-chat", audio, {"language": language, "system": system})

    def translate(self, text: str, target: str, source: str | None = None) -> str:
        return self._json("POST", "/v1/atlas/translate",
                          {"text": text, "target": target, "source": source})["translation"]

    def summarize(self, text: str, language: str | None = None, max_words: int = 80) -> str:
        return self._json("POST", "/v1/atlas/summarize",
                          {"text": text, "language": language, "max_words": max_words})["summary"]

    def detect_language(self, text: str) -> dict:
        return self._json("POST", "/v1/atlas/detect-language", {"text": text})

    def feedback(self, rating: int, language: str | None = None, tag: str = "") -> None:
        self._json("POST", "/v1/atlas/feedback", {"rating": rating, "language": language, "tag": tag})

    def stats(self) -> dict:
        return self._json("GET", "/v1/atlas/stats")

    def conversation(self, language: str | None = None, system: str | None = None) -> "Conversation":
        return Conversation(self, language=language, system=system)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _chat_body(messages, language, system, temperature, max_tokens, params) -> dict:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        messages = list(messages)
        if system:
            messages = [{"role": "system", "content": system}] + [m for m in messages if m.get("role") != "system"]
        body: dict[str, Any] = {"model": "n-atlas", "messages": messages, **params}
        if language:
            body["atlas_language"] = language
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return body


class Conversation:
    """Keeps multi-turn history so callers only pass the new user message."""

    def __init__(self, client: AtlasKit, language: str | None = None, system: str | None = None):
        self.client = client
        self.language = language
        self.messages: list[dict[str, str]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def say(self, text: str, **params: Any) -> str:
        self.messages.append({"role": "user", "content": text})
        reply = self.client.chat(self.messages, language=self.language, **params).text
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def stream(self, text: str, **params: Any) -> Iterator[str]:
        self.messages.append({"role": "user", "content": text})
        parts: list[str] = []
        for delta in self.client.stream(self.messages, language=self.language, **params):
            parts.append(delta)
            yield delta
        self.messages.append({"role": "assistant", "content": "".join(parts)})

    def reset(self) -> None:
        self.messages = [m for m in self.messages if m["role"] == "system"]


def _read_audio(audio: AudioInput) -> tuple[str, bytes]:
    if isinstance(audio, (bytes, bytearray)):
        return "audio.wav", bytes(audio)
    if isinstance(audio, (str, os.PathLike)):
        p = Path(audio)
        return p.name, p.read_bytes()
    if hasattr(audio, "read"):
        return Path(getattr(audio, "name", "audio.wav")).name, audio.read()
    raise TypeError("audio must be a path, bytes or a binary file object")

"""AtlasKit gateway: an OpenAI-compatible HTTP API in front of N-ATLaS.

    vLLM (serving NCAIR1/N-ATLaS)  <--  AtlasKit gateway  <--  SDKs / apps / playground
                                         + N-ATLaS ASR (Whisper fine-tunes)

Run:  uvicorn atlaskit_gateway.app:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:  # pragma: no cover - some sandboxes ship the fork
    import httpx2 as httpx  # type: ignore

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from . import __version__
from .core import (
    LANGUAGES, AtlasError, Settings, build_chat_request, clean_completion, detect_language,
    normalize_language, summary_messages, translation_messages, language_instruction,
    DEFAULT_SYSTEM_PROMPT,
)
from .usage import UsageLog, client_id

PLAYGROUND = Path(os.environ.get("ATLAS_PLAYGROUND_DIR", Path(__file__).resolve().parents[2] / "playground"))


def error(message: str, status: int = 400, code: str = "invalid_request") -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": code, "code": code}}, status_code=status)


def create_app(settings: Settings | None = None, transcriber: Any = None,
               transport: Any = None) -> Starlette:
    """Build the ASGI app. ``transcriber`` and ``transport`` are injectable for tests."""
    s = settings or Settings.from_env()
    usage = UsageLog(s.log_path)
    client = httpx.AsyncClient(base_url=s.upstream_url, timeout=s.request_timeout, transport=transport)

    if transcriber is None and s.enable_asr:
        from .asr import WhisperTranscriber
        transcriber = WhisperTranscriber(device=s.asr_device)

    # ------------------------------------------------------------------ helpers
    def auth(request: Request) -> str:
        key = None
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            key = header[7:].strip()
        key = key or request.headers.get("x-api-key")
        if s.api_keys and key not in s.api_keys:
            raise AtlasError("Invalid or missing API key", 401, "unauthorized")
        return client_id(key, request.client.host if request.client else None)

    async def json_body(request: Request) -> dict:
        try:
            body = await request.json()
        except Exception:
            raise AtlasError("Body must be valid JSON")
        if not isinstance(body, dict):
            raise AtlasError("Body must be a JSON object")
        return body

    async def upstream_chat(body: dict) -> dict:
        try:
            r = await client.post("/chat/completions", json=body)
        except httpx.HTTPError as exc:
            raise AtlasError(f"N-ATLaS backend unreachable: {exc}", 503, "backend_unavailable")
        if r.status_code >= 400:
            raise AtlasError(f"N-ATLaS backend error: {r.text[:500]}", 502, "backend_error")
        return r.json()

    async def complete_text(messages: list[dict], max_tokens: int = 800) -> tuple[str, dict]:
        plan = build_chat_request({"messages": messages, "max_tokens": max_tokens})
        data = await upstream_chat(plan.upstream_body)
        return clean_completion(data["choices"][0]["message"]["content"]), data.get("usage") or {}

    def endpoint(name: str):
        """Decorator: auth, error handling and usage logging for every route."""
        def wrap(fn):
            async def handler(request: Request):
                started = time.perf_counter()
                cid, meta, status = "anonymous", {}, 200
                try:
                    cid = auth(request)
                    resp, meta = await fn(request)
                    status = resp.status_code
                    return resp
                except AtlasError as exc:
                    status = exc.status
                    return error(str(exc), exc.status, exc.code)
                finally:
                    if not meta.get("_streaming"):
                        usage.record(endpoint=name, client=cid, status=status,
                                     latency_ms=round((time.perf_counter() - started) * 1000),
                                     **{k: v for k, v in meta.items() if not k.startswith("_")})
            return handler
        return wrap

    # ------------------------------------------------------------------ routes
    async def health(request: Request):
        upstream_ok = False
        try:
            r = await client.get("/models", timeout=5)
            upstream_ok = r.status_code == 200
        except httpx.HTTPError:
            pass
        return JSONResponse({
            "status": "ok" if upstream_ok else "degraded",
            "version": __version__,
            "model": s.upstream_model,
            "upstream": upstream_ok,
            "asr_enabled": transcriber is not None,
            "asr_loaded": transcriber.loaded() if hasattr(transcriber, "loaded") else [],
        }, status_code=200 if upstream_ok else 503)

    @endpoint("models")
    async def models(request: Request):
        now = int(time.time())
        data = [{"id": "n-atlas", "object": "model", "created": now, "owned_by": "NCAIR / Awarri",
                 "root": s.upstream_model}]
        if transcriber is not None:
            data += [{"id": v["asr_model"], "object": "model", "created": now, "owned_by": "NCAIR",
                      "language": k, "task": "asr"} for k, v in LANGUAGES.items()]
        return JSONResponse({"object": "list", "data": data}), {}

    @endpoint("chat")
    async def chat(request: Request):
        plan = build_chat_request(await json_body(request), s.upstream_model)
        meta = {"language": plan.language, "stream": plan.stream}
        if not plan.stream:
            data = await upstream_chat(plan.upstream_body)
            data["model"] = plan.requested_model
            for choice in data.get("choices", []):
                msg = choice.get("message") or {}
                if isinstance(msg.get("content"), str):
                    msg["content"] = clean_completion(msg["content"])
            u = data.get("usage") or {}
            meta.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"))
            return JSONResponse(data), meta

        cid = auth(request)
        started = time.perf_counter()
        req = client.build_request("POST", "/chat/completions", json=plan.upstream_body)
        try:
            upstream = await client.send(req, stream=True)
        except httpx.HTTPError as exc:
            raise AtlasError(f"N-ATLaS backend unreachable: {exc}", 503, "backend_unavailable")
        if upstream.status_code >= 400:
            text = (await upstream.aread()).decode(errors="replace")
            await upstream.aclose()
            raise AtlasError(f"N-ATLaS backend error: {text[:500]}", 502, "backend_error")

        async def relay():
            tokens = None
            try:
                async for line in upstream.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data:") and '"usage"' in line:
                        try:
                            u = json.loads(line[5:]).get("usage") or {}
                            tokens = u.get("completion_tokens", tokens)
                        except json.JSONDecodeError:
                            pass
                    if line.startswith("data:") and line.strip() != "data: [DONE]":
                        try:
                            chunk = json.loads(line[5:])
                            chunk["model"] = plan.requested_model
                            line = "data: " + json.dumps(chunk, ensure_ascii=False)
                        except json.JSONDecodeError:
                            pass
                    yield line + "\n\n"
            finally:
                await upstream.aclose()
                usage.record(endpoint="chat", client=cid, status=200, stream=True, language=plan.language,
                             completion_tokens=tokens,
                             latency_ms=round((time.perf_counter() - started) * 1000))

        return StreamingResponse(relay(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}), \
            {"_streaming": True}

    async def _transcribe(request: Request) -> tuple[dict, str]:
        if transcriber is None:
            raise AtlasError("ASR is disabled on this deployment (ATLAS_ENABLE_ASR=0)", 501, "asr_disabled")
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise AtlasError("Upload the audio as multipart field `file`")
        lang = normalize_language(form.get("language"), required=True)
        audio = await upload.read()
        if not audio:
            raise AtlasError("Uploaded audio is empty")
        result = await run_in_threadpool(transcriber.transcribe, audio, lang)
        return result, lang

    @endpoint("transcribe")
    async def transcriptions(request: Request):
        result, lang = await _transcribe(request)
        return JSONResponse(result), {"language": lang, "audio_seconds": result.get("duration")}

    @endpoint("translate")
    async def translate(request: Request):
        body = await json_body(request)
        msgs = translation_messages(body.get("text", ""), body.get("target"), body.get("source"))
        text, u = await complete_text(msgs, int(body.get("max_tokens", 800)))
        return JSONResponse({"translation": text, "source": body.get("source"), "target": body.get("target")}), \
            {"language": normalize_language(body.get("target")), "completion_tokens": u.get("completion_tokens")}

    @endpoint("summarize")
    async def summarize(request: Request):
        body = await json_body(request)
        msgs = summary_messages(body.get("text", ""), body.get("language"), int(body.get("max_words", 80)))
        text, u = await complete_text(msgs)
        return JSONResponse({"summary": text}), {"language": body.get("language"),
                                                 "completion_tokens": u.get("completion_tokens")}

    @endpoint("detect_language")
    async def detect(request: Request):
        body = await json_body(request)
        result = detect_language(body.get("text", ""))
        return JSONResponse(result), {"language": result["language"]}

    @endpoint("voice_chat")
    async def voice_chat(request: Request):
        """Audio in, text reply out in the same language: ASR -> N-ATLaS."""
        result, lang = await _transcribe(request)
        form = await request.form()
        system = form.get("system") or DEFAULT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": f"{system}\n{language_instruction(lang)}"},
                    {"role": "user", "content": result["text"]}]
        reply, u = await complete_text(messages)
        return JSONResponse({"transcript": result["text"], "reply": reply, "language": lang,
                             "audio_seconds": result.get("duration")}), \
            {"language": lang, "audio_seconds": result.get("duration"), "completion_tokens": u.get("completion_tokens")}

    @endpoint("feedback")
    async def feedback(request: Request):
        body = await json_body(request)
        rating = body.get("rating")
        if rating not in (-1, 1):
            raise AtlasError("`rating` must be 1 (good) or -1 (bad)")
        return JSONResponse({"ok": True}), {"rating": rating, "language": normalize_language(body.get("language")),
                                            "tag": str(body.get("tag", ""))[:40]}

    async def stats(request: Request):
        try:
            auth(request)
        except AtlasError as exc:
            return error(str(exc), exc.status, exc.code)
        return JSONResponse(usage.stats())

    async def playground(request: Request):
        index = PLAYGROUND / "index.html"
        if not index.exists():
            return Response("Playground not bundled", status_code=404)
        return FileResponse(index)

    routes = [
        Route("/", playground),
        Route("/health", health),
        Route("/v1/models", models),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
        Route("/v1/atlas/translate", translate, methods=["POST"]),
        Route("/v1/atlas/summarize", summarize, methods=["POST"]),
        Route("/v1/atlas/detect-language", detect, methods=["POST"]),
        Route("/v1/atlas/voice-chat", voice_chat, methods=["POST"]),
        Route("/v1/atlas/feedback", feedback, methods=["POST"]),
        Route("/v1/atlas/stats", stats),
    ]
    middleware = [Middleware(CORSMiddleware, allow_origins=list(s.cors_origins), allow_methods=["*"],
                             allow_headers=["*"])]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        yield
        await client.aclose()

    return Starlette(routes=routes, middleware=middleware, lifespan=lifespan)


def _lazy_app():
    return create_app()


class _LazyASGI:
    """Defers construction so ``import atlaskit_gateway.app`` has no side effects."""
    _app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            type(self)._app = _lazy_app()
        await self._app(scope, receive, send)


app = _LazyASGI()

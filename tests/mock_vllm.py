"""A tiny stand-in for vLLM's OpenAI server, used by tests and offline demos.

It does not generate language; it echoes what it received so tests can assert
on exactly what the gateway sent upstream.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LAST_REQUEST: dict = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence
        pass

    def _send(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._send(200, {"object": "list", "data": [{"id": "NCAIR1/N-ATLaS", "object": "model"}]})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LAST_REQUEST.clear()
        LAST_REQUEST.update(body)
        if body.get("model") != "NCAIR1/N-ATLaS":
            return self._send(404, {"error": {"message": "model not found"}})
        last_user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        reply = f"echo: {last_user}"
        if "capital" in last_user.lower():
            reply = "The capital of Nigeria is Abuja."
        if not body.get("stream"):
            return self._send(200, {
                "id": "cmpl-1", "object": "chat.completion", "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": reply + "<|eot_id|>"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": len(reply.split()), "total_tokens": 10 + len(reply.split())},
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in reply.split(" "):
            chunk = {"id": "c", "object": "chat.completion.chunk", "model": body["model"],
                     "choices": [{"index": 0, "delta": {"content": word + " "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        final = {"id": "c", "choices": [], "usage": {"completion_tokens": len(reply.split())}}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


def start(port: int = 0) -> tuple[ThreadingHTTPServer, int]:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


if __name__ == "__main__":
    import sys
    s, p = start(int(sys.argv[1]) if len(sys.argv) > 1 else 8000)
    print(f"mock vLLM on http://127.0.0.1:{p}/v1")
    threading.Event().wait()

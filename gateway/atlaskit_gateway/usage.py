"""Privacy-preserving usage log.

Each request is recorded as one JSON line *without* prompt or completion text:
endpoint, language, status, latency, token counts and a hashed client id. The
log doubles as real-world validation evidence (``/v1/atlas/stats``).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import Counter, defaultdict
from typing import Any


def client_id(api_key: str | None, ip: str | None) -> str:
    raw = api_key or ip or "anonymous"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


class UsageLog:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def record(self, **event: Any) -> None:
        event.setdefault("ts", time.time())
        line = json.dumps(event, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def events(self) -> list[dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def stats(self) -> dict[str, Any]:
        ev = self.events()
        by_endpoint = Counter(e.get("endpoint") for e in ev)
        by_language = Counter(e.get("language") or "auto" for e in ev)
        ok = [e for e in ev if e.get("status", 200) < 400]
        lat = defaultdict(list)
        for e in ok:
            if "latency_ms" in e:
                lat[e.get("endpoint")].append(e["latency_ms"])
        feedback = [e for e in ev if e.get("endpoint") == "feedback"]
        return {
            "total_requests": len(ev),
            "successful_requests": len(ok),
            "unique_clients": len({e.get("client") for e in ev}),
            "by_endpoint": dict(by_endpoint),
            "by_language": dict(by_language),
            "median_latency_ms": {k: sorted(v)[len(v) // 2] for k, v in lat.items() if v},
            "completion_tokens": sum(e.get("completion_tokens") or 0 for e in ev),
            "feedback": {
                "count": len(feedback),
                "positive": sum(1 for e in feedback if e.get("rating", 0) > 0),
                "negative": sum(1 for e in feedback if e.get("rating", 0) < 0),
            },
            "first_seen": min((e["ts"] for e in ev), default=None),
            "last_seen": max((e["ts"] for e in ev), default=None),
        }

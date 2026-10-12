"""Run the gateway locally against the mock model (no GPU) — for UI work and SDK development.

    python examples/dev_server.py   # then open http://localhost:8080
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "gateway"), str(ROOT / "tests")]

import uvicorn  # noqa: E402

import mock_vllm  # noqa: E402
from atlaskit_gateway.app import create_app  # noqa: E402
from atlaskit_gateway.core import Settings  # noqa: E402

_, port = mock_vllm.start()
app = create_app(Settings(upstream_url=f"http://127.0.0.1:{port}/v1", enable_asr=False,
                          log_path=str(ROOT / "logs" / "dev-usage.jsonl")))
uvicorn.run(app, host="127.0.0.1", port=8080, log_level="warning")

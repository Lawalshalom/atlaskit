"""Download N-ATLaS and the four N-ATLaS ASR models into the Hugging Face cache.

The models are gated: accept the terms on each model page with the account that
owns HF_TOKEN first. The ASR repos also contain training checkpoints
(optimizer.pt etc.) that inference never reads, so those are skipped.

    HF_HOME=/workspace/hf-cache python deploy/runpod/download_models.py
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gateway"))
from atlaskit_gateway.core import DEFAULT_MODEL_ID, LANGUAGES  # noqa: E402

REPOS = [DEFAULT_MODEL_ID] + [v["asr_model"] for v in LANGUAGES.values()]
SKIP = ["optimizer.pt", "scheduler.pt", "rng_state*", "training_args.bin", "trainer_state.json"]

for repo in REPOS:
    print(">> downloading", repo, flush=True)
    snapshot_download(repo, ignore_patterns=SKIP)
print("All models downloaded.")

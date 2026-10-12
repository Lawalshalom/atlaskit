# Image for a RunPod Serverless load-balancing endpoint (scale to zero). Built and published to
# ghcr.io/lawalshalom/atlaskit-serverless by .github/workflows/serverless-image.yml.
# The worker needs a network volume holding hf-cache/ (from deploy/runpod/setup.sh) and adapters/.
FROM vllm/vllm-openai:v0.31.0
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gateway gateway
COPY playground playground
COPY deploy deploy
RUN (command -v uv >/dev/null && uv pip install --system --no-cache ./gateway librosa soundfile) \
    || python3 -m pip install --no-cache-dir ./gateway librosa soundfile
ENV ATLAS_PLAYGROUND_DIR=/app/playground PORT=8080 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
EXPOSE 8080
ENTRYPOINT ["bash", "/app/deploy/runpod/serverless/start.sh"]

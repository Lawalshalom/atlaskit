# Optional: bake everything into one image for a RunPod Serverless/Pod template.
# docker build -f deploy/runpod/Dockerfile -t <you>/atlaskit:0.1.0 .
FROM vllm/vllm-openai:v0.9.2
RUN apt-get update -qq && apt-get install -y -qq ffmpeg curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY gateway gateway
COPY sdk/python sdk/python
COPY playground playground
COPY deploy deploy
COPY README.md README.md
RUN pip install --no-cache-dir -e gateway librosa soundfile
ENV HF_HOME=/workspace/hf-cache ATLAS_PLAYGROUND_DIR=/app/playground ATLAS_PYTHON=python3
EXPOSE 8080
ENTRYPOINT ["bash", "deploy/runpod/start.sh"]

FROM ollama/ollama@sha256:020e4134285e2ef4d8fd801234176de3b4faadc992a3eb06c8e66a2f9d4c4ba2

# Docker initializes a named volume from this directory, preserving ownership.
# A numeric identity keeps this wrapper independent of the upstream image's
# passwd implementation.
RUN mkdir -p /home/ollama/.ollama && chown -R 65532:65532 /home/ollama
ENV HOME=/home/ollama OLLAMA_MODELS=/home/ollama/.ollama/models
USER 65532:65532

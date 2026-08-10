# Pinned to 3.9 because that is the interpreter every number in the README was
# produced on. The code also runs on 3.11 and 3.12 — CI proves it on both — but
# the container exists to reproduce the reported results, so it matches.
FROM python:3.9-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # transformers probes for TensorFlow at import; there is none here.
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/artifacts/hf_cache

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch first, from PyTorch's own index. The default PyPI wheel drags
# in the whole CUDA runtime and turns a 1 GB image into a 6 GB one.
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2 \
    && pip install -r requirements-dev.txt

COPY src/ ./src/
COPY experiments/ ./experiments/
COPY tests/ ./tests/
COPY data/ ./data/
COPY config/ ./config/
COPY scripts/ ./scripts/
COPY pyproject.toml Makefile README.md LICENSE ./

RUN chmod +x scripts/*.sh

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -fsS http://localhost:8000/readyz || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

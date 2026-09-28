# ClinVision AI - CPU serving image (FastAPI + Streamlit in one image,
# service chosen at runtime via `command`, see docker-compose.yml).
# GPU training is intentionally out of scope here: train on a CUDA host or
# Colab (requirements-cuda.txt), then mount the .keras weights read-only.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV runtime libs (headless build still needs libGL/libglib).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first (better layer caching than copying code first).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Application code + curated demo subset (5 DICOMs, ~700 KB) + config.
COPY backend ./backend
COPY frontend ./frontend
COPY data/demo ./data/demo
COPY .streamlit ./.streamlit
COPY .env.example ./.env.example

# Mount points (weights read-only, artifacts read-write, see compose).
RUN mkdir -p /app/models /app/artifacts

EXPOSE 8000 8501

# Default = API. The compose file overrides the UI service command.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]

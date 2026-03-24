# STT API - Faster-Whisper + pyannote 화자분리
#
# [빠른 테스트] 1~2분
#   docker build --target test -t stt-api:test .
#   docker run -p 8000:8000 stt-api:test
#   curl http://localhost:8000/health  → {"status":"ok"}
#
# [전체 실행] 20~30분 (첫 빌드)
#   docker compose up -d --build

# ---- test: Docker 동작 확인용 (ML 없음, 1~2분) ----
FROM python:3.12-slim AS test
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn
COPY docker_test_main.py ./main.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- full: 실제 STT API (GPU 런타임) ----
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS full
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates ffmpeg python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY app/ ./app/
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel \
    && python3 -m pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

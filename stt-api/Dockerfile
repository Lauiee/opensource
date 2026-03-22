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

# ---- full: 실제 STT API ----
FROM python:3.12-slim AS full
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY app/ ./app/
RUN pip install -e .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
